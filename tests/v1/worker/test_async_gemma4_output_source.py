import ast
from pathlib import Path


GPU_RUNNER_SOURCE = Path(__file__).parents[3] / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
REJECTION_SAMPLER_SOURCE = Path(__file__).parents[3] / "vllm" / "v1" / "sample" / "rejection_sampler.py"


def _class_source(path: Path, class_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{class_name} was not found in {path}")


def _method_source(path: Path, method_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{method_name} was not found in {path}")


def test_async_output_owns_valid_count_snapshot():
    source = _class_source(GPU_RUNNER_SOURCE, "AsyncGPUModelRunnerOutput")

    assert "valid_sampled_token_count: torch.Tensor | None = None" in source
    assert "self._valid_sampled_token_count = valid_sampled_token_count" in source
    assert "self.valid_sampled_token_count_cpu" in source
    assert "valid_sampled_token_count=self.valid_sampled_token_count_cpu" in source


def test_rejection_parser_uses_count_mask_for_tokens_and_logprobs():
    source = _method_source(REJECTION_SAMPLER_SOURCE, "parse_output")
    module_source = REJECTION_SAMPLER_SOURCE.read_text(encoding="utf-8")

    assert "import numpy as np" in module_source
    assert "valid_sampled_token_count: torch.Tensor | None = None" in source
    assert "count_mask" in source
    assert "np.array_equal" in source
    assert "valid_mask.flatten()" in source
    assert "valid_mask[discard_req_indices] = False" in source


if __name__ == "__main__":
    test_async_output_owns_valid_count_snapshot()
    test_rejection_parser_uses_count_mask_for_tokens_and_logprobs()
