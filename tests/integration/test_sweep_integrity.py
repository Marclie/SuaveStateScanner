"""
Test suite for sweep integrity and state consistency.
Verifies that sweep finalization (analyzeSweep, sort restoration) does not
revert, scramble, or corrupt points solved during sweeps.
"""

import copy
import io
import sys
import numpy as np
import pytest
from suave_state_scanner import SuaveStateScanner


def test_analyze_sweep_preserves_solved_points():
    """Ensure analyzeSweep preserves the exact solved points without reverting them."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        s = SuaveStateScanner("input.csv", "output_test.csv", 9, "suave.json")
        s.pntBounds = [0, 30]
        s.stateBounds = [0, 4]

        # Invert states 0 and 1 before prepareSweep to test non-ascending input order
        s.E[[0, 1]] = s.E[[1, 0]]
        s.P[[0, 1]] = s.P[[1, 0]]

        last_e, last_p = s.prepareSweep()
        for st in range(s.numStates):
            s.sweepState(st, 1, backward=False)

        before_analyze = copy.deepcopy(s.E)
        del_max = s.analyzeSweep(last_e, last_p)
        after_analyze = copy.deepcopy(s.E)
    finally:
        sys.stdout = old_stdout

    # analyzeSweep must NOT change s.E away from the post-sweep array
    assert np.allclose(after_analyze, before_analyze, atol=1e-12)


def test_redundant_swaps_disabled_locks_earlier_states():
    """Verify that when redundantSwaps is False, earlier states are not modified by later states."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        s = SuaveStateScanner("input.csv", "output_test.csv", 9, "suave.json")
        s.pntBounds = [0, 30]
        s.stateBounds = [0, 4]
        s.redundantSwaps = False

        last_e, last_p = s.prepareSweep()
        history = []
        for st in range(s.numStates):
            s.sweepState(st, 1, backward=False)
            history.append(copy.deepcopy(s.E))
    finally:
        sys.stdout = old_stdout

    # For any state i, its values must not change when state j (j > i) is swept
    for target_state in range(s.numStates):
        for sweeping_state in range(target_state + 1, s.numStates):
            assert np.array_equal(
                history[sweeping_state - 1][target_state],
                history[sweeping_state][target_state],
            ), f"State {target_state} was modified by sweeping state {sweeping_state} even though redundantSwaps=False"
