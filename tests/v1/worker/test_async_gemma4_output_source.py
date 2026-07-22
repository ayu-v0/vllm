import ast
from pathlib import Path

import pytest


# This file only parses source text and never allocates accelerator memory.
pytestmark = pytest.mark.skip_global_cleanup


GPU_RUNNER_SOURCE = Path(__file__).parents[3] / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
REJECTION_SAMPLER_SOURCE = Path(__file__).parents[3] / "vllm" / "v1" / "sample" / "rejection_sampler.py"
ENGINE_CORE_SOURCE = Path(__file__).parents[3] / "vllm" / "v1" / "engine" / "core.py"


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


def test_async_output_profiles_existing_copy_wait_without_extra_sync():
    source = _class_source(GPU_RUNNER_SOURCE, "AsyncGPUModelRunnerOutput")

    assert "profile_context: dict[str, float | int] | None = None" in source
    assert "async_output_wait_ms" in source
    assert "Gemma4 MTP async profile: output" in source
    assert source.count("async_copy_ready_event.synchronize()") == 1


def test_engine_core_profiles_scheduler_and_output_boundaries_when_enabled():
    source = ENGINE_CORE_SOURCE.read_text(encoding="utf-8")

    assert "VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE" in source
    assert "Gemma4 MTP async profile: engine" in source
    assert "schedule_ms" in source
    assert "model_wait_ms" in source
    assert "scheduler_update_ms" in source


if __name__ == "__main__":
    test_async_output_owns_valid_count_snapshot()
    test_rejection_parser_uses_count_mask_for_tokens_and_logprobs()
    test_async_output_profiles_existing_copy_wait_without_extra_sync()
    test_engine_core_profiles_scheduler_and_output_boundaries_when_enabled()
