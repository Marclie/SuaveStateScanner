"""
Numerical Equivalence verification test suite.
Asserts that SuaveFacade produces bit-for-bit / epsilon-identical results to SuaveStateScanner.
"""

import json
import numpy as np
import pytest
from suave_state_scanner import SuaveStateScanner
from suave_gui.core.parser import DataParserService
from suave_gui.core.scanner_facade import SuaveFacade


def test_single_sweep_numerical_equivalence():
    with open("suave.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # 1. Run legacy scanner for 1 sweep with bounded points for fast execution
    legacy = SuaveStateScanner("input.csv", "output_test_legacy.csv", 9, "suave.json")
    legacy.width = 3
    legacy.orders = [1]
    legacy.pntBounds = [0, 40]
    legacy.stateBounds = [0, 4]

    last_e, last_p = legacy.prepareSweep()
    for state in range(legacy.numStates):
        legacy.sweepState(state, 1, backward=False)
    del_max_legacy = legacy.analyzeSweep(last_e, last_p)

    # 2. Run facade scanner with exact identical configuration
    dataset = DataParserService.parse_csv_stream("input.csv", num_states=9, filename="input.csv")
    facade = SuaveFacade(dataset, config=config)
    facade.width = 3
    facade.orders = [1]
    facade.pntBounds = [0, 40]
    facade.stateBounds = [0, 4]

    res_facade = facade.execute_sweeps(num_sweeps=1, direction="forward")

    # 3. Verify identical results
    assert np.allclose(legacy.E, facade.E, atol=1e-12, equal_nan=True)
    assert np.allclose(legacy.P, facade.P, atol=1e-12, equal_nan=True)
    assert np.isclose(del_max_legacy, res_facade.del_max, atol=1e-12)
