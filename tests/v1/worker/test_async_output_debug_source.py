from pathlib import Path


SOURCE = (
    Path(__file__).parents[3]
    / "vllm"
    / "v1"
    / "worker"
    / "gpu_model_runner.py"
)


def test_gemma4_mtp_async_output_debug_logs_are_guarded():
    source = SOURCE.read_text(encoding="utf-8")

    assert 'getattr(self, "_gemma4_mtp_debug", False)' in source
    for marker in [
        "Gemma4 MTP debug: async get_output synchronize start",
        "Gemma4 MTP debug: async get_output synchronize done",
        "Gemma4 MTP debug: async get_output return",
    ]:
        assert marker in source


if __name__ == "__main__":
    test_gemma4_mtp_async_output_debug_logs_are_guarded()
