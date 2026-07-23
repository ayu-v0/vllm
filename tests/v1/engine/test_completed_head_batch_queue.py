# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

from vllm.v1.engine.core import EngineCore


class _Scheduler:
    def __init__(self, *, has_requests: bool = False) -> None:
        self.has_requests_value = has_requests
        self.has_requests_calls = 0
        self.schedule_calls = 0
        self.update_calls: list[tuple[object, object]] = []

    def has_requests(self) -> bool:
        self.has_requests_calls += 1
        return self.has_requests_value

    def schedule(self):
        self.schedule_calls += 1
        return SimpleNamespace(
            total_num_scheduled_tokens=1,
            pending_structured_output_tokens=False,
        )

    def get_grammar_bitmask(self, scheduler_output):
        return None

    def update_from_output(self, scheduler_output, model_output):
        self.update_calls.append((scheduler_output, model_output))
        return {0: "completed-head"}


def _completed_future(value: object) -> Future:
    future = Future()
    future.set_result(value)
    return future


def _make_engine_core(*, batch_queue_size: int, has_requests: bool = False):
    engine_core = object.__new__(EngineCore)
    engine_core.batch_queue = deque(maxlen=batch_queue_size)
    engine_core.batch_queue_size = batch_queue_size
    engine_core.is_ec_consumer = True
    engine_core.scheduler = _Scheduler(has_requests=has_requests)
    engine_core.aborts_queue = queue.Queue()
    engine_core.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_logging_iteration_details=False)
    )
    engine_core._gemma4_mtp_completed_head_ttft_fix_enabled = True
    engine_core._gemma4_mtp_async_profile_enabled = False
    return engine_core


class _ModelExecutor:
    def execute_model(self, scheduler_output, *, non_block: bool):
        assert non_block
        return _completed_future(None)

    def sample_tokens(self, grammar_output, *, non_block: bool):
        assert non_block
        return _completed_future(object())


def test_completed_head_is_returned_before_new_batch_submission():
    engine_core = _make_engine_core(batch_queue_size=2, has_requests=True)
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    model_output = object()
    engine_core.batch_queue.append(
        (_completed_future(model_output), scheduler_output, _completed_future(None))
    )

    outputs, model_executed = engine_core.step_with_batch_queue()

    assert outputs == {0: "completed-head"}
    assert model_executed
    assert not engine_core.batch_queue
    assert engine_core.scheduler.update_calls == [(scheduler_output, model_output)]
    assert engine_core.scheduler.has_requests_calls == 0


def test_incomplete_head_does_not_enter_completed_head_fast_path():
    engine_core = _make_engine_core(batch_queue_size=3, has_requests=True)
    engine_core.model_executor = _ModelExecutor()
    pending_future = Future()
    engine_core.batch_queue.append(
        (
            pending_future,
            SimpleNamespace(total_num_scheduled_tokens=1),
            _completed_future(None),
        )
    )

    outputs, model_executed = engine_core.step_with_batch_queue()

    assert outputs is None
    assert model_executed
    assert not pending_future.done()
    assert engine_core.scheduler.schedule_calls == 1
    assert len(engine_core.batch_queue) == 2


def test_completed_head_policy_requires_a_single_queued_batch():
    engine_core = _make_engine_core(batch_queue_size=2)
    engine_core.batch_queue.extend((object(), object()))

    assert not engine_core._should_deliver_completed_gemma4_mtp_batch_head()
