# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import pytest
import torch

from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput


# These tests use CPU tensors and mocked CUDA stream objects only. The global
# NPU allocator cleanup is neither needed nor compatible with torch_npu here.
pytestmark = pytest.mark.skip_global_cleanup


class _CopyStream:
    def wait_stream(self, _stream) -> None:
        pass


class _Event:
    def record(self) -> None:
        pass

    def synchronize(self) -> None:
        pass


class _DeviceTensor:
    """CPU test stand-in whose host copy has distinct storage."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def to(self, device: str, *, non_blocking: bool) -> torch.Tensor:
        assert device == "cpu"
        assert non_blocking
        return self.tensor.clone()

    @property
    def shape(self):
        return self.tensor.shape


@pytest.fixture
def fake_async_copy(monkeypatch):
    monkeypatch.setattr(gpu_model_runner.torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(gpu_model_runner.torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(gpu_model_runner.torch, "Event", _Event, raising=False)
    return _CopyStream()


@pytest.mark.parametrize(
    ("sampled", "counts", "expected"),
    [
        ([[101, 0, -1, -1]], [2], [[101, 0]]),
        ([[11, 12, 13, 14]], [4], [[11, 12, 13, 14]]),
        ([[-1, -1, -1, -1]], [0], [[]]),
    ],
)
def test_parse_output_uses_valid_count_without_filtering_token_zero(sampled, counts, expected):
    output, logprobs = RejectionSampler.parse_output(
        torch.tensor(sampled),
        vocab_size=1024,
        valid_sampled_token_count=torch.tensor(counts),
    )

    assert output == expected
    assert logprobs is None


def test_parse_output_rejects_count_placeholder_disagreement():
    with pytest.raises(RuntimeError, match="valid sampled count disagrees"):
        RejectionSampler.parse_output(
            torch.tensor([[101, -1, -1, -1]]),
            vocab_size=1024,
            valid_sampled_token_count=torch.tensor([2]),
        )


def test_parse_output_keeps_logprobs_aligned_for_mixed_rows():
    sampled = torch.tensor(
        [
            [101, 0, -1, -1],
            [11, 12, 13, 14],
            [21, -1, -1, -1],
        ]
    )
    logprobs_tensors = LogprobsTensors(
        logprob_token_ids=torch.arange(12, dtype=torch.int32).reshape(12, 1),
        logprobs=torch.arange(12, dtype=torch.float32).reshape(12, 1),
        selected_token_ranks=torch.arange(12, dtype=torch.int32),
    )

    output, logprobs = RejectionSampler.parse_output(
        sampled,
        vocab_size=1024,
        discard_req_indices=[2],
        logprobs_tensors=logprobs_tensors,
        valid_sampled_token_count=torch.tensor([2, 4, 1]),
    )

    assert output == [[101, 0], [11, 12, 13, 14], []]
    assert logprobs is not None
    assert logprobs.cu_num_generated_tokens == [0, 2, 6, 6]
    assert logprobs.logprob_token_ids.shape[0] == 6
    assert logprobs.logprobs.shape[0] == 6
    assert logprobs.sampled_token_ranks.shape[0] == 6


def _make_model_runner_output(req_id: str) -> ModelRunnerOutput:
    return ModelRunnerOutput(req_ids=[req_id], req_id_to_index={req_id: 0})


def test_async_outputs_own_independent_sampled_token_and_count_snapshots(fake_async_copy):
    sampled_a = _DeviceTensor(torch.tensor([[101, 0, -1, -1]]))
    counts_a = _DeviceTensor(torch.tensor([2]))
    output_a = AsyncGPUModelRunnerOutput(
        _make_model_runner_output("request-a"),
        sampled_a,
        None,
        [],
        fake_async_copy,
        vocab_size=1024,
        valid_sampled_token_count=counts_a,
    )
    sampled_b = _DeviceTensor(torch.tensor([[11, 12, 13, 14]]))
    counts_b = _DeviceTensor(torch.tensor([4]))
    output_b = AsyncGPUModelRunnerOutput(
        _make_model_runner_output("request-b"),
        sampled_b,
        None,
        [],
        fake_async_copy,
        vocab_size=1024,
        valid_sampled_token_count=counts_b,
    )

    sampled_a.tensor.fill_(-1)
    counts_a.tensor.fill_(0)
    sampled_b.tensor.fill_(-1)
    counts_b.tensor.fill_(0)

    result_b = output_b.get_output()
    result_a = output_a.get_output()

    assert result_a.req_ids == ["request-a"]
    assert result_a.sampled_token_ids == [[101, 0]]
    assert result_b.req_ids == ["request-b"]
    assert result_b.sampled_token_ids == [[11, 12, 13, 14]]
