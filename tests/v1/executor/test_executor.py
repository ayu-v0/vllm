# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import (
    ExecutorWithExternalLauncher,
    UniProcExecutor,
)
from vllm.v1.outputs import AsyncModelRunnerOutput


class Mock: ...


class BlockingUniProcWorker:
    def __init__(self, started: Event, release: Event):
        self.started = started
        self.release = release

    def slow_method(self):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test did not release the blocking worker")
        return "worker-result"


class UniProcWorkerFailure(RuntimeError):
    pass


class FailingUniProcWorker:
    def fail(self):
        raise UniProcWorkerFailure("worker failed")


class RecordingUniProcWorker:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def execute_model(self, batch: object):
        self.calls.append(("execute_model", batch))
        return f"execute:{batch}"

    def sample_tokens(self, batch: object):
        self.calls.append(("sample_tokens", batch))
        return f"sample:{batch}"


class BlockingAsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, started: Event, release: Event):
        self.started = started
        self.release = release

    def get_output(self):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test did not release the async output")
        return "async-output"


class AsyncOutputUniProcWorker:
    def __init__(self, output: BlockingAsyncOutput):
        self.output = output
        self.calls: list[str] = []

    def async_method(self):
        self.calls.append("async_method")
        return self.output

    def fast_method(self):
        self.calls.append("fast_method")
        return "fast-result"


class ShutdownRecorder:
    def __init__(self, name: str, calls: list[str]):
        self.name = name
        self.calls = calls

    def shutdown(self, wait: bool = True):
        self.calls.append(self.name)


class ShutdownWorker:
    def __init__(self, calls: list[str]):
        self.calls = calls

    def shutdown(self):
        self.calls.append("worker")


def _bare_uniproc_executor(
    worker: object,
    worker_command_thread: ThreadPoolExecutor,
    async_output_thread: ThreadPoolExecutor | None = None,
) -> UniProcExecutor:
    executor = object.__new__(UniProcExecutor)
    executor.driver_worker = worker
    executor.async_output_thread = async_output_thread
    executor.worker_command_thread = worker_command_thread
    return executor


def _uniproc_gate_config(
    *,
    async_scheduling: bool = True,
    device_type: str = "npu",
    method: str | None = "mtp",
    model_type: str = "gemma4",
):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        device_config=SimpleNamespace(device_type=device_type, device="npu:0"),
        speculative_config=(
            SimpleNamespace(method=method) if method is not None else None
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type)
        ),
    )


def test_supports_async_scheduling_base_executor():
    assert Executor.supports_async_scheduling() is False


def test_supports_async_scheduling_uniproc_executor():
    assert UniProcExecutor.supports_async_scheduling() is True


def test_supports_async_scheduling_executor_with_external_launcher():
    # ExecutorWithExternalLauncher inherits from UniProcExecutor and does not
    # override supports_async_scheduling, so it should return True.
    assert ExecutorWithExternalLauncher.supports_async_scheduling() is True


def test_supports_async_scheduling_multiproc_executor():
    assert MultiprocExecutor.supports_async_scheduling() is True


@pytest.mark.skip_global_cleanup
def test_uniproc_non_block_returns_before_worker_method_finishes():
    started = Event()
    release = Event()
    worker = BlockingUniProcWorker(started, release)
    command_thread = ThreadPoolExecutor(max_workers=1)
    caller_thread = ThreadPoolExecutor(max_workers=1)
    executor = _bare_uniproc_executor(worker, command_thread)

    try:
        submitted_call = caller_thread.submit(
            executor.collective_rpc,
            "slow_method",
            non_block=True,
            single_value=True,
        )
        assert started.wait(timeout=1), "worker method did not start"

        try:
            result_future = submitted_call.result(timeout=0.05)
            assert isinstance(result_future, Future)
            assert not result_future.done()
        finally:
            release.set()

        assert result_future.result(timeout=1) == "worker-result"
    finally:
        release.set()
        caller_thread.shutdown(wait=True, cancel_futures=True)
        command_thread.shutdown(wait=True, cancel_futures=True)


@pytest.mark.skip_global_cleanup
def test_uniproc_non_block_propagates_worker_exception():
    with ThreadPoolExecutor(max_workers=1) as command_thread:
        executor = _bare_uniproc_executor(FailingUniProcWorker(), command_thread)

        result_future = executor.collective_rpc(
            "fail", non_block=True, single_value=True
        )

        with pytest.raises(UniProcWorkerFailure, match="worker failed"):
            result_future.result(timeout=1)


@pytest.mark.skip_global_cleanup
def test_uniproc_non_block_hands_async_output_to_output_thread():
    output_started = Event()
    output_release = Event()
    output = BlockingAsyncOutput(output_started, output_release)
    worker = AsyncOutputUniProcWorker(output)

    with (
        ThreadPoolExecutor(max_workers=1) as command_thread,
        ThreadPoolExecutor(max_workers=1) as output_thread,
    ):
        executor = _bare_uniproc_executor(worker, command_thread, output_thread)

        output_future = executor.collective_rpc(
            "async_method", non_block=True, single_value=True
        )
        assert output_started.wait(timeout=1), "async output did not start"
        assert not output_future.done()

        fast_future = executor.collective_rpc(
            "fast_method", non_block=True, single_value=True
        )
        assert fast_future.result(timeout=1) == "fast-result"
        assert worker.calls == ["async_method", "fast_method"]

        output_release.set()
        assert output_future.result(timeout=1) == "async-output"


@pytest.mark.skip_global_cleanup
def test_uniproc_non_block_preserves_execute_sample_submission_order():
    worker = RecordingUniProcWorker()
    with ThreadPoolExecutor(max_workers=1) as command_thread:
        executor = _bare_uniproc_executor(worker, command_thread)

        futures = [
            executor.execute_model("B1", non_block=True),
            executor.sample_tokens("B1", non_block=True),
            executor.execute_model("B2", non_block=True),
            executor.sample_tokens("B2", non_block=True),
        ]

        assert [future.result(timeout=1) for future in futures] == [
            "execute:B1",
            "sample:B1",
            "execute:B2",
            "sample:B2",
        ]
        assert worker.calls == [
            ("execute_model", "B1"),
            ("sample_tokens", "B1"),
            ("execute_model", "B2"),
            ("sample_tokens", "B2"),
        ]


@pytest.mark.skip_global_cleanup
def test_uniproc_non_block_gate_is_default_off_and_scoped(monkeypatch):
    executor = object.__new__(UniProcExecutor)
    executor.vllm_config = _uniproc_gate_config()

    monkeypatch.delenv("VLLM_ASCEND_GEMMA4_MTP_ASYNC_UNIPROC_SUBMIT", raising=False)
    assert not executor._should_enable_gemma4_mtp_async_uniproc_submit()

    monkeypatch.setenv("VLLM_ASCEND_GEMMA4_MTP_ASYNC_UNIPROC_SUBMIT", "1")
    assert executor._should_enable_gemma4_mtp_async_uniproc_submit()

    for config in (
        _uniproc_gate_config(async_scheduling=False),
        _uniproc_gate_config(device_type="cuda"),
        _uniproc_gate_config(method=None),
        _uniproc_gate_config(method="ngram"),
        _uniproc_gate_config(model_type="qwen3"),
    ):
        executor.vllm_config = config
        assert not executor._should_enable_gemma4_mtp_async_uniproc_submit()

    external_executor = object.__new__(ExecutorWithExternalLauncher)
    external_executor.vllm_config = _uniproc_gate_config()
    assert not external_executor._should_enable_gemma4_mtp_async_uniproc_submit()


@pytest.mark.skip_global_cleanup
def test_uniproc_shutdown_orders_command_output_worker():
    calls: list[str] = []
    executor = object.__new__(UniProcExecutor)
    executor.worker_command_thread = ShutdownRecorder("command", calls)
    executor.async_output_thread = ShutdownRecorder("output", calls)
    executor.driver_worker = ShutdownWorker(calls)

    executor.shutdown()

    assert calls == ["command", "output", "worker"]


class CustomMultiprocExecutor(MultiprocExecutor):
    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator = None,
    ) -> Any | list[Any] | Future[Any | list[Any]]:
        # Drop marker to show that this was run
        with open(".marker", "w"):
            ...
        return super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            non_block,
            unique_reply_rank,
            kv_output_aggregator,
        )


CustomMultiprocExecutorAsync = CustomMultiprocExecutor
MODEL = "Qwen/Qwen3-0.6B"


def test_custom_executor_type_checking():
    with pytest.raises(ValueError):
        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        LLMEngine.from_engine_args(engine_args)
    with pytest.raises(ValueError):
        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        AsyncLLM.from_engine_args(engine_args)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutor,
        "tests.v1.executor.test_executor.CustomMultiprocExecutor",
    ],
)
def test_custom_executor(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutorAsync,
        "tests.v1.executor.test_executor.CustomMultiprocExecutorAsync",
    ],
)
def test_custom_executor_async(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        async def t():
            stream = engine.generate(
                request_id="0", prompt="foo", sampling_params=sampling_params
            )
            async for x in stream:
                ...

        asyncio.run(t())

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)
