import ast
from pathlib import Path


SOURCE = Path(__file__).parents[2] / "vllm" / "model_executor" / "models" / "gemma4_mtp.py"


def _source_tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def test_gemma4_mtp_weights_mapper_uses_supported_keywords():
    tree = _source_tree()
    unsupported_keywords = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "WeightsMapper":
            continue
        unsupported_keywords.extend(
            kw.arg for kw in node.keywords if kw.arg == "orig_to_new_stacked"
        )

    assert unsupported_keywords == []


def test_gemma4_mtp_load_weights_handles_stacked_gate_up_weights():
    source = SOURCE.read_text(encoding="utf-8")

    assert "stacked_params_mapping" in source
    assert '(".gate_up_proj", ".gate_proj", 0)' in source
    assert '(".gate_up_proj", ".up_proj", 1)' in source
    assert "weight_loader = param.weight_loader" in source
    assert "weight_loader(param, loaded_weight, shard_id)" in source


if __name__ == "__main__":
    test_gemma4_mtp_weights_mapper_uses_supported_keywords()
    test_gemma4_mtp_load_weights_handles_stacked_gate_up_weights()
