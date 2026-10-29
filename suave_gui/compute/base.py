"""
Compute Engine interface and task abstractions.
Provides a unified API for dispatching background scientific calculations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import threading
from typing import Any, Callable, Optional


@dataclass
class TaskHandle:
    """Represents a background computation task."""
    task_id: str
    cancel_token: threading.Event = field(default_factory=threading.Event)
    _future: Optional[Any] = None

    def cancel(self) -> None:
        self.cancel_token.set()
        if self._future and hasattr(self._future, "cancel"):
            try:
                self._future.cancel()
            except Exception:
                pass

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_token.is_set()

    @property
    def is_done(self) -> bool:
        if self._future is None:
            return True
        if hasattr(self._future, "done"):
            return self._future.done()
        return True

    def result(self, timeout: Optional[float] = None) -> Any:
        if self._future is None:
            return None
        if hasattr(self._future, "result"):
            return self._future.result(timeout=timeout)
        return self._future


class ComputeEngine(ABC):
    """Abstract Base Class for scientific compute engines."""

    @abstractmethod
    def submit_task(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> TaskHandle:
        """Submit a callable for asynchronous execution."""
        pass

    @abstractmethod
    def is_busy(self) -> bool:
        """Check if an asynchronous task is currently running."""
        pass

    @abstractmethod
    def abort_current(self) -> None:
        """Signal cancellation to the active calculation."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanly terminate worker threads/processes and release resources."""
        pass
