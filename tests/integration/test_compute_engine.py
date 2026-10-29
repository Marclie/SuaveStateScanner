"""
Integration tests for LocalThreadEngine.
"""

import time
import pytest
from suave_gui.compute.local_engine import LocalThreadEngine


def simple_compute(x: int, delay: float = 0.05, cancel_token=None) -> int:
    time.sleep(delay)
    if cancel_token and cancel_token.is_set():
        return -1
    return x * 2


def test_local_thread_engine_execution():
    engine = LocalThreadEngine(max_workers=2)
    try:
        handle = engine.submit_task(simple_compute, 21, delay=0.05)
        assert engine.is_busy() or handle.is_done
        res = handle.result(timeout=5.0)
        assert res == 42
        assert not engine.is_busy()
    finally:
        engine.shutdown()


def test_local_thread_engine_cancellation():
    engine = LocalThreadEngine(max_workers=2)
    try:
        handle = engine.submit_task(simple_compute, 21, delay=0.2)
        engine.abort_current()
        res = handle.result(timeout=5.0)
        assert res == -1
    finally:
        engine.shutdown()
