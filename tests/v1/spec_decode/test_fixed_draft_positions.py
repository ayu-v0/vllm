# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.v1.spec_decode.llm_base_proposer as proposer_module
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


class _Group:
    def __init__(self, builder: AttentionMetadataBuilder):
        self.builder = builder

    def get_metadata_builder(self) -> AttentionMetadataBuilder:
        return self.builder


class _TritonBuilderSubclass(TritonAttentionMetadataBuilder):
    pass


def _bare_proposer(
    *,
    constant_draft_positions: bool = True,
    max_positions: int = 16,
) -> SpecDecodeBaseProposer:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.constant_draft_positions = constant_draft_positions
    proposer.positions = torch.full(
        (max_positions,),
        999,
        dtype=torch.int64,
    )
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.draft_attn_groups = []
    return proposer


def test_prepare_constant_draft_positions_compacts_and_zeros_padding():
    proposer = _bare_proposer()
    source_positions = torch.tensor([10, 20, 30, 40])
    token_indices_to_sample = torch.tensor([1, 3])
    selected_positions = source_positions[token_indices_to_sample]

    proposer._prepare_constant_draft_positions(
        selected_positions,
        batch_size=2,
        input_batch_size=4,
    )

    torch.testing.assert_close(
        proposer.positions[:4],
        torch.tensor([20, 40, 0, 0]),
    )
    assert proposer.positions[4].item() == 999


def test_prepare_constant_draft_positions_handles_empty_dp_rank():
    proposer = _bare_proposer()

    proposer._prepare_constant_draft_positions(
        torch.empty(0, dtype=torch.int64),
        batch_size=0,
        input_batch_size=4,
    )

    torch.testing.assert_close(
        proposer.positions[:4],
        torch.zeros(4, dtype=torch.int64),
    )


def test_can_reuse_followup_attn_metadata_for_exact_triton_builders():
    proposer = _bare_proposer()
    proposer.draft_attn_groups = [
        _Group(object.__new__(TritonAttentionMetadataBuilder)),
        _Group(object.__new__(TritonAttentionMetadataBuilder)),
    ]

    assert (
        TritonAttentionMetadataBuilder.build_for_drafting
        is AttentionMetadataBuilder.build_for_drafting
    )
    assert proposer._can_reuse_followup_attn_metadata()


def test_cannot_reuse_followup_attn_metadata_when_triton_overrides_drafting(
    monkeypatch,
):
    proposer = _bare_proposer()
    proposer.draft_attn_groups = [
        _Group(object.__new__(TritonAttentionMetadataBuilder))
    ]

    def overridden_build_for_drafting(self, common_attn_metadata, draft_index):
        raise AssertionError("guard must not call build_for_drafting")

    monkeypatch.setattr(
        TritonAttentionMetadataBuilder,
        "build_for_drafting",
        overridden_build_for_drafting,
    )

    assert not proposer._can_reuse_followup_attn_metadata()


def test_cannot_reuse_followup_attn_metadata_without_constant_positions():
    proposer = _bare_proposer(constant_draft_positions=False)
    proposer.draft_attn_groups = [
        _Group(object.__new__(TritonAttentionMetadataBuilder))
    ]

    assert not proposer._can_reuse_followup_attn_metadata()


def test_cannot_reuse_followup_attn_metadata_without_draft_groups():
    proposer = _bare_proposer()

    assert not proposer._can_reuse_followup_attn_metadata()


def test_cannot_reuse_followup_attn_metadata_for_triton_subclass():
    proposer = _bare_proposer()
    proposer.draft_attn_groups = [
        _Group(object.__new__(_TritonBuilderSubclass))
    ]

    assert not proposer._can_reuse_followup_attn_metadata()


def test_cannot_reuse_followup_attn_metadata_for_mixed_triton_builders():
    proposer = _bare_proposer()
    proposer.draft_attn_groups = [
        _Group(object.__new__(TritonAttentionMetadataBuilder)),
        _Group(object.__new__(_TritonBuilderSubclass)),
    ]

    assert not proposer._can_reuse_followup_attn_metadata()


def test_update_positions_dependent_metadata_preserves_normal_behavior(
    monkeypatch,
):
    proposer = _bare_proposer(constant_draft_positions=False)
    proposer.max_model_len = 128
    proposer._slot_mapping_buffer = torch.full(
        (8,),
        -999,
        dtype=torch.int64,
    )

    common_attn_metadata = SimpleNamespace(
        block_table_tensor=torch.tensor([[5, 6], [7, 8]]),
        seq_lens=torch.tensor([4, 8], dtype=torch.int32),
        slot_mapping=torch.tensor([-1, -1], dtype=torch.int64),
        max_seq_len=8,
        _seq_lens_cpu=torch.tensor([4, 8], dtype=torch.int32),
        _num_computed_tokens_cpu=torch.tensor([3, 7], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor(
            [4, 8],
            dtype=torch.int32,
        ),
    )

    def fake_update_kernel(
        *,
        positions_1d,
        block_table_tensor,
        seq_lens,
        block_size,
        max_model_len,
        out_clamped_positions,
        out_slot_mapping,
        input_batch_size,
    ):
        del block_table_tensor, block_size, max_model_len
        batch_size = positions_1d.shape[0]
        out_clamped_positions.copy_(positions_1d + 1)
        out_slot_mapping[:batch_size].copy_(
            torch.tensor([101, 202], dtype=torch.int64)
        )
        out_slot_mapping[batch_size:input_batch_size].fill_(-1)
        seq_lens.add_(1)

    monkeypatch.setattr(
        proposer_module,
        "eagle_step_update_slot_mapping_and_metadata",
        fake_update_kernel,
    )

    updated_positions = proposer._update_positions_dependent_metadata(
        torch.tensor([3, 7], dtype=torch.int64),
        common_attn_metadata,
        batch_size=2,
        input_batch_size=4,
        block_size=16,
    )

    torch.testing.assert_close(updated_positions, torch.tensor([4, 8]))
    torch.testing.assert_close(
        common_attn_metadata.seq_lens,
        torch.tensor([5, 9], dtype=torch.int32),
    )
    torch.testing.assert_close(
        common_attn_metadata.slot_mapping,
        torch.tensor([101, 202]),
    )
    assert common_attn_metadata.max_seq_len == 9
    torch.testing.assert_close(
        common_attn_metadata._seq_lens_cpu,
        torch.tensor([5, 9], dtype=torch.int32),
    )
    torch.testing.assert_close(
        common_attn_metadata._num_computed_tokens_cpu,
        torch.tensor([4, 8], dtype=torch.int32),
    )
    torch.testing.assert_close(
        common_attn_metadata.seq_lens_cpu_upper_bound,
        torch.tensor([5, 9], dtype=torch.int32),
    )
    torch.testing.assert_close(
        proposer._slot_mapping_buffer[2:4],
        torch.tensor([-1, -1]),
    )
