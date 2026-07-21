from pathlib import Path


SOURCE = Path(__file__).parents[3] / "vllm" / "v1" / "spec_decode" / "gemma4.py"


def test_gemma4_per_group_metadata_keeps_slot_mapping_with_block_table():
    source = SOURCE.read_text(encoding="utf-8")

    assert "self._per_group_slot_mappings" in source
    assert "def set_per_group_attention_metadata(" in source
    assert "cm.slot_mapping = self._per_group_slot_mappings[gid]" in source


def test_gpu_runner_passes_matching_slot_mapping_to_gemma4_proposer():
    source = (
        Path(__file__).parents[3]
        / "vllm"
        / "v1"
        / "worker"
        / "gpu_model_runner.py"
    ).read_text(encoding="utf-8")

    assert "set_per_group_attention_metadata(" in source
    assert "kv_cache_gid, cm.block_table_tensor, cm.slot_mapping" in source
    assert "set_per_group_block_table(" not in source


if __name__ == "__main__":
    test_gemma4_per_group_metadata_keeps_slot_mapping_with_block_table()
    test_gpu_runner_passes_matching_slot_mapping_to_gemma4_proposer()
