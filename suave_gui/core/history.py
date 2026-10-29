"""
History and Undo/Redo manager implementing the Memento pattern.
Provides non-destructive state rollbacks for manual interventions and automated sweeps.
"""

from __future__ import annotations
from typing import List, Optional
from suave_gui.core.dataset import StateSnapshot


class HistoryManager:
    """
    Manages undo and redo stacks of StateSnapshots.
    Enforces maximum history depth to prevent unbounded memory growth.
    """

    def __init__(self, max_depth: int = 25):
        self._max_depth = max(1, max_depth)
        self._undo_stack: List[StateSnapshot] = []
        self._redo_stack: List[StateSnapshot] = []
        self._current_snapshot: Optional[StateSnapshot] = None

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_count(self) -> int:
        return len(self._undo_stack)

    @property
    def redo_count(self) -> int:
        return len(self._redo_stack)

    @property
    def current(self) -> Optional[StateSnapshot]:
        return self._current_snapshot

    def reset(self, initial_snapshot: StateSnapshot) -> None:
        """Initialize the history manager with the starting dataset snapshot."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._current_snapshot = initial_snapshot

    def push(self, previous_snapshot: StateSnapshot, new_snapshot: Optional[StateSnapshot] = None) -> None:
        """
        Record a state transition: pushes previous_snapshot onto the undo stack and clears redo.
        """
        if len(self._undo_stack) >= self._max_depth:
            self._undo_stack.pop(0)  # Evict oldest snapshot

        self._undo_stack.append(previous_snapshot)
        self._redo_stack.clear()
        self._current_snapshot = new_snapshot or previous_snapshot

    def undo(self, current_live_snapshot: Optional[StateSnapshot] = None) -> Optional[StateSnapshot]:
        """
        Roll back one state. Moves current state to redo stack and returns previous snapshot.
        """
        if not self._undo_stack:
            return None

        # If a live snapshot is provided or recorded, push to redo
        live = current_live_snapshot or self._current_snapshot
        if live is not None:
            if len(self._redo_stack) >= self._max_depth:
                self._redo_stack.pop(0)
            self._redo_stack.append(live)

        target = self._undo_stack.pop()
        self._current_snapshot = target
        return target

    def redo(self, current_live_snapshot: Optional[StateSnapshot] = None) -> Optional[StateSnapshot]:
        """
        Roll forward one state. Moves current state to undo stack and returns next snapshot.
        """
        if not self._redo_stack:
            return None

        live = current_live_snapshot or self._current_snapshot
        if live is not None:
            if len(self._undo_stack) >= self._max_depth:
                self._undo_stack.pop(0)
            self._undo_stack.append(live)

        target = self._redo_stack.pop()
        self._current_snapshot = target
        return target

    def clear(self) -> None:
        """Empty both undo and redo buffers."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._current_snapshot = None
