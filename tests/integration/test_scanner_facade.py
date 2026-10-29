"""
Integration tests for SuaveFacade against input.csv.
"""

import threading
import numpy as np
import pytest
from suave_gui.core.parser import DataParserService
from suave_gui.core.scanner_facade import SuaveFacade, SweepProgress


@pytest.fixture
def input_dataset():
    return DataParserService.parse_csv_stream("input.csv", num_states=9, filename="input.csv")


def test_facade_initialization(input_dataset):
    facade = SuaveFacade(input_dataset)
    assert facade.numStates == 9
    assert facade.numPoints == 100
    assert facade.numProps == 8
    assert facade.E.shape == (9, 100)
    assert facade.P.shape == (9, 100, 8)


def test_facade_single_sweep(input_dataset):
    facade = SuaveFacade(input_dataset)
    # Configure tight parameters for fast test
    facade.configure({"width": 3, "orders": [1], "pntBounds": [0, 50], "stateBounds": [0, 4]})

    progress = facade.execute_sweeps(num_sweeps=1, direction="forward")
    assert isinstance(progress, SweepProgress)
    assert progress.current_sweep == 1
    assert not progress.aborted

    # Sync to dataset
    facade.sync_to_dataset(input_dataset)
    assert input_dataset.energies.shape == (9, 100)


def test_facade_cancellation(input_dataset):
    facade = SuaveFacade(input_dataset)
    cancel_token = threading.Event()
    cancel_token.set()  # Pre-cancel

    progress = facade.execute_sweeps(num_sweeps=5, direction="forward", cancel_token=cancel_token)
    assert progress.aborted
    assert "aborted" in progress.message.lower()
