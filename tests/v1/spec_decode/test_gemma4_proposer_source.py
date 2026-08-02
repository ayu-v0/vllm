# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

pytestmark = pytest.mark.skip_global_cleanup

ROOT = Path(__file__).parents[3]
GEMMA_SOURCE = ROOT / "vllm" / "v1" / "spec_decode" / "gemma4.py"
BASE_SOURCE = ROOT / "vllm" / "v1" / "spec_decode" / "llm_base_proposer.py"
GPU_RUNNER_SOURCE = ROOT / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
SPECULATIVE_CONFIG_SOURCE = ROOT / "vllm" / "config" / "speculative.py"
SPEC_DECODE_E2E_SOURCE = (
    ROOT / "tests" / "v1" / "e2e" / "spec_decode" / "test_spec_decode.py"
)


def _source_between(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_gemma4_per_group_metadata_keeps_slot_mapping_with_block_table():
    source = GEMMA_SOURCE.read_text(encoding="utf-8")

    assert "self._per_group_slot_mappings" in source
    assert "def set_per_group_attention_metadata(" in source
    assert "cm.slot_mapping = self._per_group_slot_mappings[gid]" in source


def test_gpu_runner_passes_matching_slot_mapping_to_gemma4_proposer():
    source = GPU_RUNNER_SOURCE.read_text(encoding="utf-8")

    assert "set_per_group_attention_metadata(" in source
    assert "kv_cache_gid, cm.block_table_tensor, cm.slot_mapping" in source
    assert "set_per_group_block_table(" not in source


def test_core_base_consumes_constant_draft_position_contract():
    source = BASE_SOURCE.read_text(encoding="utf-8")

    assert "self.constant_draft_positions: bool = False" in source
    assert "def _prepare_constant_draft_positions(" in source
    assert "self.positions[batch_size:input_batch_size].zero_()" in source
    assert "def _update_positions_dependent_metadata(" in source
    assert "if not self.constant_draft_positions:" in source
    assert "def _can_reuse_followup_attn_metadata(" in source
    assert "type(builder) is not TritonAttentionMetadataBuilder" in source
    assert "is not AttentionMetadataBuilder.build_for_drafting" in source
    assert "if not reuse_followup_attn_metadata or token_index == 0:" in source


def test_gemma4_enables_constant_draft_positions():
    source = GEMMA_SOURCE.read_text(encoding="utf-8")
    constructor = _source_between(
        source,
        "class Gemma4Proposer(SpecDecodeBaseProposer):",
        "\n    def set_per_group_block_table(",
    )

    assert "self.constant_draft_positions = True" in constructor


def test_gemma4_e2e_accepts_model_and_parallelism_environment_overrides():
    source = SPEC_DECODE_E2E_SOURCE.read_text(encoding="utf-8")

    assert "def _get_gemma4_mtp_test_case(" in source
    assert "VLLM_TEST_GEMMA4_TARGET_MODEL" in source
    assert "VLLM_TEST_GEMMA4_DRAFT_MODEL" in source
    assert "VLLM_TEST_GEMMA4_TP_SIZE" in source
    assert "VLLM_TEST_GEMMA4_NUM_SPEC_TOKENS" in source
    assert "gemma4-custom" in source
    assert "is_gemma4_mtp = model_setup == _GEMMA4_MTP_TEST_CASE[0]" in source
    assert '_GEMMA4_MTP_TEST_ID == "gemma4-e4b"' in source
    assert "current_platform.device_count() > 1" in source
    assert "_GEMMA4_MTP_RESOURCE_MARK = (" in source


def test_core_gpu_runner_uses_core_gemma4_proposer():
    source = GPU_RUNNER_SOURCE.read_text(encoding="utf-8")
    speculative_config_source = SPECULATIVE_CONFIG_SOURCE.read_text(encoding="utf-8")
    gemma_constructor_branch = _source_between(
        source,
        "elif self.speculative_config.use_gemma4_mtp():",
        'elif self.speculative_config.method == "suffix":',
    )
    propose_method = _source_between(
        source,
        "\n    def propose_draft_token_ids(",
        "\n    def update_config(",
    )
    shared_core_drafter_branch = _source_between(
        propose_method,
        "\n        elif (\n            spec_config.use_eagle()",
        "\n        return draft_token_ids",
    )
    use_gemma4_mtp_method = _source_between(
        speculative_config_source,
        "\n    def use_gemma4_mtp(",
        "\n    def use_eagle(",
    )
    use_eagle_method = _source_between(
        speculative_config_source,
        "\n    def use_eagle(",
        "\n    def use_dflash(",
    )

    assert "self.drafter = Gemma4Proposer(" in gemma_constructor_branch
    assert "spec_config.use_eagle()" in shared_core_drafter_branch
    assert (
        "EagleProposer | DFlashProposer | DraftModelProposer | Gemma4Proposer"
        in shared_core_drafter_branch
    )
    assert "draft_token_ids = self.drafter.propose(" in shared_core_drafter_branch
    assert 'self.method == "mtp"' in use_gemma4_mtp_method
    assert '== "gemma4_mtp"' in use_gemma4_mtp_method
    assert (
        'return self.method in ("eagle", "eagle3", "mtp", "dflash")' in use_eagle_method
    )


if __name__ == "__main__":
    test_gemma4_per_group_metadata_keeps_slot_mapping_with_block_table()
    test_gpu_runner_passes_matching_slot_mapping_to_gemma4_proposer()
    test_core_base_consumes_constant_draft_position_contract()
    test_gemma4_enables_constant_draft_positions()
    test_gemma4_e2e_accepts_model_and_parallelism_environment_overrides()
    test_core_gpu_runner_uses_core_gemma4_proposer()
