# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

import vllm.v1.spec_decode.llm_base_proposer as proposer_module
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

pytestmark = pytest.mark.skip_global_cleanup


class _Group:
    def __init__(self, builder: AttentionMetadataBuilder):
        self.builder = builder

    def get_metadata_builder(self) -> AttentionMetadataBuilder:
        return self.builder


class _TritonBuilderSubclass(TritonAttentionMetadataBuilder):
    pass


class _FakeCommonMetadata:
    def __init__(self, batch_size: int):
        self.seq_lens = torch.arange(1, batch_size + 1, dtype=torch.int32)
        self.slot_mapping = torch.arange(batch_size, dtype=torch.int64)
        self.block_table_tensor = torch.zeros((batch_size, 1), dtype=torch.int32)
        self.max_seq_len = batch_size
        self._seq_lens_cpu = self.seq_lens.clone()
        self._num_computed_tokens_cpu = self.seq_lens - 1
        self.seq_lens_cpu_upper_bound = self.seq_lens.clone()
        self.num_actual_tokens = batch_size
        self.max_query_len = 1 if batch_size else 0
        self.query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32)
        self.query_start_loc_cpu = self.query_start_loc.clone()

    def batch_size(self) -> int:
        return self.seq_lens.shape[0]


class _RecordingBuilder:
    def __init__(self):
        self.calls = []

    def build_for_drafting(
        self,
        *,
        common_attn_metadata,
        draft_index,
    ):
        record = SimpleNamespace(
            draft_index=draft_index,
            metadata=common_attn_metadata,
            block_table=common_attn_metadata.block_table_tensor.clone(),
            slot_mapping=common_attn_metadata.slot_mapping.clone(),
        )
        self.calls.append(record)
        return record


class _GemmaGroup:
    def __init__(self, gid, layer_name, builder):
        self.kv_cache_group_id = gid
        self.layer_names = [layer_name]
        self.builder = builder

    def get_metadata_builder(self):
        return self.builder


class _GemmaCommonMetadata:
    def __init__(self):
        self.block_table_tensor = torch.full((2, 2), -1)
        self.slot_mapping = torch.full((4,), -1)

    def batch_size(self):
        return 2


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


def _run_propose_harness(
    monkeypatch,
    constant_draft_positions: bool,
    reuse_followup_metadata: bool,
    input_batch_size: int = 4,
    batch_size: int = 2,
    num_speculative_tokens: int = 3,
):
    proposer = _bare_proposer(constant_draft_positions=constant_draft_positions)
    num_input_tokens = 4 if batch_size else 0
    token_indices_to_sample = (
        torch.tensor([1, 3], dtype=torch.int64)
        if batch_size
        else torch.empty(0, dtype=torch.int64)
    )
    common_attn_metadata = _FakeCommonMetadata(batch_size)

    proposer.method = "mtp"
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.parallel_drafting = False
    proposer.supports_mm_inputs = False
    proposer.pass_hidden_states_to_model = False
    proposer.allowed_attn_types = None
    proposer.block_size = 16
    proposer.max_model_len = 128
    proposer.input_ids = torch.zeros(input_batch_size, dtype=torch.int32)
    proposer.hidden_states = torch.zeros((input_batch_size, 1))
    proposer.inputs_embeds = torch.zeros((input_batch_size, 1))
    proposer._slot_mapping_buffer = torch.zeros(input_batch_size, dtype=torch.int64)
    proposer._draft_attn_layer_names = ["layer.0"]
    proposer.arange = torch.arange(input_batch_size + 1, dtype=torch.int32)
    proposer.token_arange_np = np.arange(input_batch_size + 1)
    proposer.vllm_config = mock.MagicMock()
    proposer.positions[:input_batch_size].copy_(
        torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    )

    proposer.set_inputs_first_pass = mock.MagicMock(
        return_value=(
            num_input_tokens,
            token_indices_to_sample,
            common_attn_metadata,
        )
    )
    initial_model_kwargs = {
        "input_ids": proposer.input_ids[:num_input_tokens],
        "positions": proposer.positions[:num_input_tokens],
        "inputs_embeds": None,
    }
    proposer.build_model_inputs_first_pass = mock.MagicMock(
        return_value=(initial_model_kwargs, num_input_tokens)
    )
    proposer._determine_batch_execution_and_padding = mock.MagicMock(
        side_effect=[
            (CUDAGraphMode.NONE, num_input_tokens, None),
            (CUDAGraphMode.NONE, input_batch_size, None),
        ]
    )
    proposer.build_per_group_and_layer_attn_metadata = mock.MagicMock(
        return_value=([], {"layer.0": object()})
    )
    proposer._can_reuse_followup_attn_metadata = mock.MagicMock(
        return_value=reuse_followup_metadata
    )

    def update_positions(
        positions,
        common_attn_metadata,
        batch_size,
        input_batch_size,
        block_size,
    ):
        del common_attn_metadata, block_size
        updated_positions = positions + 1
        proposer.positions[:batch_size].copy_(updated_positions)
        proposer.positions[batch_size:input_batch_size].zero_()
        return updated_positions

    update_mock = mock.MagicMock(side_effect=update_positions)
    proposer._update_positions_dependent_metadata = update_mock
    proposer.model_returns_tuple = mock.MagicMock(return_value=False)
    proposer._greedy_sample = mock.MagicMock(
        side_effect=[
            offset + torch.arange(batch_size, dtype=torch.int64)
            for offset in (101, 201, 301)
        ]
    )

    seen_positions = []
    model_outputs = [
        torch.arange(num_input_tokens, dtype=torch.float32).unsqueeze(1),
        torch.arange(input_batch_size, dtype=torch.float32).unsqueeze(1),
        torch.arange(input_batch_size, dtype=torch.float32).unsqueeze(1),
    ]

    def run_model(**kwargs):
        seen_positions.append(kwargs["positions"].clone())
        return model_outputs[len(seen_positions) - 1]

    proposer.model = mock.MagicMock(side_effect=run_model)
    monkeypatch.setattr(
        proposer_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    result = proposer.propose(
        target_token_ids=torch.arange(num_input_tokens, dtype=torch.int64),
        target_positions=proposer.positions[:num_input_tokens].clone(),
        target_hidden_states=torch.zeros((num_input_tokens, 1)),
        next_token_ids=torch.arange(batch_size, dtype=torch.int64),
        token_indices_to_sample=token_indices_to_sample,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=mock.MagicMock(),
    )
    return (
        proposer,
        result,
        common_attn_metadata,
        seen_positions,
        update_mock,
    )


def test_propose_constant_positions_rebuilds_each_followup_metadata(monkeypatch):
    proposer, result, cad, seen_positions, update_mock = _run_propose_harness(
        monkeypatch,
        constant_draft_positions=True,
        reuse_followup_metadata=False,
    )

    expected_positions = torch.tensor([20, 40, 0, 0], dtype=torch.int64)
    torch.testing.assert_close(proposer.positions[:4], expected_positions)
    for positions in seen_positions[1:]:
        torch.testing.assert_close(positions, expected_positions)
    update_mock.assert_not_called()
    assert result.shape == (2, 3)
    assert cad.max_seq_len == 2
    assert proposer.build_per_group_and_layer_attn_metadata.call_args_list == [
        mock.call(cad),
        mock.call(cad, draft_index=1),
        mock.call(cad, draft_index=2),
    ]
    proposer._can_reuse_followup_attn_metadata.assert_called_once_with()


def test_propose_constant_positions_reuses_followup_metadata(monkeypatch):
    proposer, result, cad, seen_positions, update_mock = _run_propose_harness(
        monkeypatch,
        constant_draft_positions=True,
        reuse_followup_metadata=True,
    )

    expected_positions = torch.tensor([20, 40, 0, 0], dtype=torch.int64)
    torch.testing.assert_close(proposer.positions[:4], expected_positions)
    for positions in seen_positions[1:]:
        torch.testing.assert_close(positions, expected_positions)
    update_mock.assert_not_called()
    assert result.shape == (2, 3)
    assert proposer.build_per_group_and_layer_attn_metadata.call_args_list == [
        mock.call(cad),
        mock.call(cad, draft_index=1),
    ]
    proposer._can_reuse_followup_attn_metadata.assert_called_once_with()


def test_propose_constant_positions_handles_empty_dp_rank(monkeypatch):
    proposer, result, cad, seen_positions, update_mock = _run_propose_harness(
        monkeypatch,
        constant_draft_positions=True,
        reuse_followup_metadata=True,
        input_batch_size=4,
        batch_size=0,
    )

    expected_positions = torch.zeros(4, dtype=torch.int64)
    torch.testing.assert_close(proposer.positions[:4], expected_positions)
    for positions in seen_positions[1:]:
        torch.testing.assert_close(positions, expected_positions)
    update_mock.assert_not_called()
    assert result.shape == (0, 3)
    assert cad.num_actual_tokens == 0
    assert proposer.build_per_group_and_layer_attn_metadata.call_args_list == [
        mock.call(cad),
        mock.call(cad, draft_index=1),
    ]


def test_propose_normal_positions_preserves_updates(monkeypatch):
    proposer, result, cad, _, update_mock = _run_propose_harness(
        monkeypatch,
        constant_draft_positions=False,
        reuse_followup_metadata=False,
    )

    assert result.shape == (2, 3)
    assert update_mock.call_count == 2
    assert proposer.build_per_group_and_layer_attn_metadata.call_args_list == [
        mock.call(cad),
        mock.call(cad, draft_index=1),
        mock.call(cad, draft_index=2),
    ]


def test_propose_k1_returns_before_followup_position_and_metadata_logic(monkeypatch):
    proposer, result, cad, seen_positions, update_mock = _run_propose_harness(
        monkeypatch,
        constant_draft_positions=True,
        reuse_followup_metadata=True,
        num_speculative_tokens=1,
    )

    assert result.shape == (2, 1)
    assert len(seen_positions) == 1
    torch.testing.assert_close(
        proposer.positions[:4],
        torch.tensor([10, 20, 30, 40], dtype=torch.int64),
    )
    update_mock.assert_not_called()
    proposer._can_reuse_followup_attn_metadata.assert_not_called()
    assert proposer.build_per_group_and_layer_attn_metadata.call_args_list == [
        mock.call(cad)
    ]


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
    proposer.draft_attn_groups = [_Group(object.__new__(_TritonBuilderSubclass))]

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
        out_slot_mapping[:batch_size].copy_(torch.tensor([101, 202], dtype=torch.int64))
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


def test_gemma4_metadata_keeps_group_alignment_and_draft_index():
    proposer = object.__new__(Gemma4Proposer)
    builder0 = _RecordingBuilder()
    builder1 = _RecordingBuilder()
    proposer.draft_attn_groups = [
        _GemmaGroup(0, "layer.sliding", builder0),
        _GemmaGroup(1, "layer.full", builder1),
    ]
    proposer._per_group_block_tables = {
        0: torch.tensor([[10, 11], [30, 31], [90, 91]]),
        1: torch.tensor([[20, 21], [40, 41], [80, 81]]),
    }
    proposer._per_group_slot_mappings = {
        0: torch.tensor([100, 300, -1, -1]),
        1: torch.tensor([200, 400, -1, -1]),
    }
    common = _GemmaCommonMetadata()
    original_block_table = common.block_table_tensor.clone()
    original_slot_mapping = common.slot_mapping.clone()

    for draft_index in (1, 2):
        per_group, per_layer = proposer.build_per_group_and_layer_attn_metadata(
            common,
            draft_index=draft_index,
        )
        assert len(per_group) == 2
        assert per_group[0] is builder0.calls[-1]
        assert per_group[1] is builder1.calls[-1]
        assert per_layer["layer.sliding"].draft_index == draft_index
        assert per_layer["layer.full"].draft_index == draft_index
        assert per_layer["layer.sliding"] is builder0.calls[-1]
        assert per_layer["layer.full"] is builder1.calls[-1]
        assert builder0.calls[-1].metadata is not common
        assert builder1.calls[-1].metadata is not common
        assert builder0.calls[-1].metadata is not builder1.calls[-1].metadata
        torch.testing.assert_close(
            common.block_table_tensor,
            original_block_table,
        )
        torch.testing.assert_close(
            common.slot_mapping,
            original_slot_mapping,
        )

    assert [call.draft_index for call in builder0.calls] == [1, 2]
    assert [call.draft_index for call in builder1.calls] == [1, 2]
    for call in builder0.calls:
        torch.testing.assert_close(
            call.block_table,
            torch.tensor([[10, 11], [30, 31]]),
        )
        torch.testing.assert_close(
            call.slot_mapping,
            torch.tensor([100, 300, -1, -1]),
        )
    for call in builder1.calls:
        torch.testing.assert_close(
            call.block_table,
            torch.tensor([[20, 21], [40, 41]]),
        )
        torch.testing.assert_close(
            call.slot_mapping,
            torch.tensor([200, 400, -1, -1]),
        )
