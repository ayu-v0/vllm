# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


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
