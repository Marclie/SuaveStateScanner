"""
Session state manager for Suave GUI.
Directly binds to the backend SuaveStateScanner instance, eliminating duplicate state.
"""

from __future__ import annotations
import threading
from typing import Any, Dict, Optional
from suave_gui.core.dataset import StateDataset
from suave_gui.core.history import HistoryManager


class SessionContext:
    """Encapsulates all runtime state for a single user/client session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.suave: Optional[Any] = None
        self.dataset: Optional[StateDataset] = None
        self.history: HistoryManager = HistoryManager(max_depth=25)
        self.sweep_thread: Optional[threading.Thread] = None
        self.sweep_telemetry: Dict[str, Any] = {
            "running": False,
            "updated": False,
            "done": False,
            "state": 0,
            "sweep": 0,
            "total": 0,
            "delMax": float("inf"),
            "direction": "forward",
            "message": "",
        }
        self.selected_property: int = 0
        self.selected_state: Optional[int] = None
        self.lock = threading.Lock()

    def set_scanner(self, suave: Any) -> None:
        """Bind session directly to the backend SuaveStateScanner instance."""
        with self.lock:
            self.suave = suave
            self.dataset = StateDataset(
                reaction_coordinates=suave.allPnts,
                energies=suave.E,
                properties=suave.P,
                filename=getattr(suave, "infile", "memory.csv"),
                suave_instance=suave,
            )
            self.history.reset(self.dataset.create_snapshot("Loaded " + getattr(suave, "infile", "dataset")))
            self.selected_property = 0
            self.selected_state = None
            self.sweep_telemetry["running"] = False
            self.sweep_telemetry["done"] = False
            self.sweep_telemetry["updated"] = False

    def abort_sweep(self) -> None:
        """Halt running calculations on the backend scanner."""
        with self.lock:
            if self.suave:
                self.suave.halt = True
            self.sweep_telemetry["running"] = False
            self.sweep_telemetry["done"] = True


class SessionRegistry:
    """Thread-safe registry mapping session IDs to SessionContext instances."""

    def __init__(self):
        self._sessions: Dict[str, SessionContext] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionContext:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionContext(session_id=session_id)
            return self._sessions[session_id]

    def remove(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].abort_sweep()
                del self._sessions[session_id]
