"""
Unit tests for StateDataset and StateSnapshot.
"""

import numpy as np
import pytest
from suave_gui.core.dataset import StateDataset, StateSnapshot


@pytest.fixture
def sample_dataset():
    rc = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
    energies = np.array([
        [-110.0, -109.8, -109.5, -109.2, -108.9],
        [-108.0, -107.8, -107.5, -107.2, -106.9],
        [-105.0, -104.8, -104.5, -104.2, -103.9],
    ])
    # 3 states, 5 points, 2 properties
    properties = np.zeros((3, 5, 2))
    properties[0, :, 0] = 1.0
    properties[1, :, 0] = 2.0
    properties[2, :, 0] = 3.0
    return StateDataset(rc, energies, properties, filename="test.csv")


def test_dataset_initialization(sample_dataset):
    assert sample_dataset.num_states == 3
    assert sample_dataset.num_points == 5
    assert sample_dataset.num_properties == 2
    assert np.isclose(sample_dataset.min_step_size, 0.1)
    assert not sample_dataset.has_missing


def test_dataset_dimension_validation():
    rc = np.array([0.9, 1.0])
    energies = np.array([[-110.0]])  # Shape (1, 1) mismatch with rc shape (2,)
    props = np.zeros((1, 1, 1))

    with pytest.raises(ValueError):
        StateDataset(rc, energies, props)


def test_apply_swap(sample_dataset):
    initial_e0 = sample_dataset.energies[0, 1].copy()
    initial_e1 = sample_dataset.energies[1, 1].copy()

    sample_dataset.apply_swap(0, 1, point_start=1, point_end=3)

    # Point 0 should be unchanged
    assert sample_dataset.energies[0, 0] == -110.0
    # Point 1 should be swapped
    assert sample_dataset.energies[0, 1] == initial_e1
    assert sample_dataset.energies[1, 1] == initial_e0
    # Point 3 should be unchanged
    assert sample_dataset.energies[0, 3] == -109.2


def test_remove_state(sample_dataset):
    sample_dataset.remove_state_range(0, point_start=1, point_end=3)
    assert sample_dataset.has_missing
    assert np.isnan(sample_dataset.energies[0, 1])
    assert np.isnan(sample_dataset.energies[0, 2])
    assert not np.isnan(sample_dataset.energies[0, 0])


def test_snapshot_restore(sample_dataset):
    snap = sample_dataset.create_snapshot("Initial")
    sample_dataset.apply_swap(0, 2, point_start=0, point_end=5)
    assert sample_dataset.energies[0, 0] == -105.0

    sample_dataset.restore_snapshot(snap)
    assert sample_dataset.energies[0, 0] == -110.0


def test_sort_energies(sample_dataset):
    # Invert energies at point 0
    sample_dataset.energies[0, 0] = 50.0
    sample_dataset.sort_energies()
    # At point 0, state 0 should now have lowest energy (-108.0)
    assert sample_dataset.energies[0, 0] == -108.0
    assert sample_dataset.energies[2, 0] == 50.0
