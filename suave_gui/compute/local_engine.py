"""
Local asynchronous compute engine utilizing Python ThreadPoolExecutor.
Zero-dependency, low-latency execution for interactive desktop workloads.
"""

from __future__ import annotations
import concurrent.futures
import threading
import uuid
from typing import Any, Callable, Optional
from suave_gui.compute.base import ComputeEngine, TaskHandle


class LocalThreadEngine(ComputeEngine):
    """
    Executes tasks in background threads, allowing seamless non-blocking GUI updates
    while sharing in-memory NumPy arrays without serialization overhead.
    """

    def __init__(self, max_workers: int = 2):
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="SuaveWorker"
        )
        self._current_handle: Optional[TaskHandle] = None
        self._lock = threading.Lock()

    def submit_task(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> TaskHandle:
        with self._lock:
            if self._current_handle and not self._current_handle.is_done:
                raise RuntimeError("Another calculation is already running. Abort or wait.")

            task_id = str(uuid.uuid4())
            cancel_token = threading.Event()
            # If the callable accepts cancel_token, pass it in kwargs
            kwargs["cancel_token"] = cancel_token

            future = self._executor.submit(func, *args, **kwargs)
            handle = TaskHandle(task_id=task_id, cancel_token=cancel_token, _future=future)
            self._current_handle = handle
            return handle

    def is_busy(self) -> bool:
        with self._lock:
            return self._current_handle is not None and not self._current_handle.is_done

    def abort_current(self) -> None:
        with self._lock:
            if self._current_handle:
                self._current_handle.cancel()

    def shutdown(self) -> None:
        with self._lock:
            if self._current_handle:
                self._current_handle.cancel()
            self._executor.shutdown(wait=False)
