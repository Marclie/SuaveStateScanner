"""
Unit tests for HistoryManager.
"""

import numpy as np
from suave_gui.core.dataset import StateDataset
from suave_gui.core.history import HistoryManager


def test_history_push_undo_redo():
    rc = np.array([1.0, 2.0])
    e = np.array([[1.0, 2.0], [3.0, 4.0]])
    p = np.zeros((2, 2, 1))
    dataset = StateDataset(rc, e, p)

    history = HistoryManager(max_depth=5)
    snap0 = dataset.create_snapshot("Initial")
    history.reset(snap0)

    assert not history.can_undo
    assert not history.can_redo

    # Perform mutation 1
    dataset.apply_swap(0, 1, 0, 2)
    snap1 = dataset.create_snapshot("Swap 0-1")
    history.push(snap0, snap1)

    assert history.can_undo
    assert not history.can_redo

    # Undo
    undone = history.undo(current_live_snapshot=dataset.create_snapshot("Live"))
    assert undone is not None
    assert undone.description == "Initial"
    dataset.restore_snapshot(undone)
    assert dataset.energies[0, 0] == 1.0
    assert history.can_redo

    # Redo
    redone = history.redo(current_live_snapshot=dataset.create_snapshot("Live"))
    assert redone is not None
    dataset.restore_snapshot(redone)
    assert dataset.energies[0, 0] == 3.0


def test_history_max_depth():
    history = HistoryManager(max_depth=3)
    dataset = StateDataset(np.array([1.0]), np.array([[1.0]]), np.zeros((1, 1, 1)))

    for i in range(10):
        dataset.energies[0, 0] = float(i)
        history.push(dataset.create_snapshot(f"Step {i}"))

    assert history.undo_count == 3
