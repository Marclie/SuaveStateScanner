"""Fast workflow-format tests; no PySCF installation or SCF calculation needed."""

import numpy as np
import pytest

from examples import pyscf_h2
from examples.pyscf_h2 import (
    BASIS, COLUMNS, NUM_AOS, NUM_STATES, NUM_EXCITED_STATES, NUM_TRANSITIONS,
    SORT_PROPERTIES, X_SLICE, Y_SLICE,
    _canonical_amplitudes, _transition_features, generate_pec, reaction_mesh,
    shuffled_labels, write_pec,
)
from scipy.optimize import linear_sum_assignment
from suave_gui.core.parser import DataParserService


@pytest.mark.parametrize("coordinates", [[], [0.0], [-0.4], [np.nan], [np.inf],
                                        [0.5, 0.4], [0.4, 0.4], [[0.4, 0.5]]])
def test_invalid_coordinates_fail_before_importing_pyscf(coordinates):
    with pytest.raises(ValueError, match="finite, positive and strictly increasing"):
        generate_pec(coordinates)


def test_excess_roots_fail_before_importing_pyscf(monkeypatch):
    monkeypatch.setattr(pyscf_h2, "NUM_EXCITED_STATES", NUM_TRANSITIONS + 1)
    with pytest.raises(ValueError, match="supports at most 6 spin-conserving TDDFT roots, not 7"):
        generate_pec([0.4])


def test_reaction_mesh_has_43_points_and_uniform_spacing():
    mesh = reaction_mesh()
    assert len(mesh) == 43
    assert mesh[0] == 0.40
    assert mesh[-1] == 2.50
    assert np.all(np.diff(mesh) > 0)
    np.testing.assert_allclose(np.diff(mesh), 0.05, rtol=0, atol=1e-12)


def test_native_csv_round_trip_and_whole_record_shuffling(tmp_path):
    num_points = len(reaction_mesh())
    table = np.arange(num_points * NUM_STATES * len(COLUMNS), dtype=float).reshape(num_points, NUM_STATES, len(COLUMNS))
    table[:, :, 0] = reaction_mesh()[:, None]
    shuffled = shuffled_labels(table)
    np.testing.assert_array_equal(table[:2], shuffled[:2])
    np.testing.assert_array_equal(table[:, 0], shuffled[:, 0])  # Ground is never shuffled.
    assert not np.array_equal(table[2:], shuffled[2:])
    np.testing.assert_array_equal(shuffled, shuffled_labels(table))
    for original, reordered in zip(table, shuffled):
        cost = np.max(np.abs(reordered[:, None] - original[None, :]), axis=2)
        rows, columns = linear_sum_assignment(cost)
        assert np.all(cost[rows, columns] == 0)
    path = tmp_path / "input.csv"
    write_pec(path, shuffled)
    dataset = DataParserService.parse_csv_stream(str(path), num_states=NUM_STATES)
    np.testing.assert_array_equal(dataset.reaction_coordinates, reaction_mesh())
    np.testing.assert_array_equal(dataset.energies, shuffled[:, :, 1].T)
    np.testing.assert_array_equal(dataset.properties, shuffled[:, :, 2:].transpose(1, 0, 2))
    assert BASIS == "6-31g"
    assert NUM_AOS == 4
    assert NUM_STATES == 7
    assert NUM_EXCITED_STATES == NUM_TRANSITIONS == 6
    assert len(COLUMNS) == 9
    assert len(dataset.properties[0, 0]) == 7
    assert list(COLUMNS) == [
        "bond_length_angstrom",
        "total_energy_hartree",
        "excitation_energy_hartree",
        "oscillator_strength_length_gauge",
        "spin_transition_character",
        "attachment_weight_ao_0",
        "attachment_weight_ao_1",
        "attachment_weight_ao_2",
        "attachment_weight_ao_3",
    ]
    selected = {COLUMNS[index + 2] for index in SORT_PROPERTIES}
    assert "excitation_energy_hartree" in selected
    assert "oscillator_strength_length_gauge" in selected
    assert "spin_transition_character" in selected
    assert "attachment_weight_ao_3" in selected
    assert "uks_energy_hartree" not in selected
    assert "uks_spin_squared" not in selected
    assert "dominant_x_abs" not in selected
    assert "x_norm_squared" not in selected
    assert not any(name.startswith(("x_alpha_", "x_beta_", "y_alpha_", "y_beta_", "dominant_x_virtual"))
                   for name in selected)


def test_multiorbital_amplitudes_invariant_under_global_state_phase():
    x = [np.array([[0.1, -0.8, 0.2]]), np.array([[0.2, 0.8, -0.1]])]
    y = [np.array([[0.01, -0.2, 0.02]]), np.array([[0.02, 0.2, -0.01]])]
    reference = _canonical_amplitudes(x, y)
    transformed = _canonical_amplitudes([-channel for channel in x], [-channel for channel in y])
    for actual, expected in zip(transformed, reference):
        np.testing.assert_array_equal(actual, expected)
    assert reference[0][1] > 0
    assert reference[0][4] < 0  # Preserve relative alpha/beta signs.


def test_zero_alpha_amplitude_uses_beta_phase_anchor():
    x, y = _canonical_amplitudes([np.zeros((1, 3)), -np.ones((1, 3))],
                                [np.zeros((1, 3)), np.zeros((1, 3))])
    np.testing.assert_array_equal(x, [0, 0, 0, 1, 1, 1])
    assert not y.any()


def test_all_zero_amplitudes_cannot_define_excited_state_phase():
    with pytest.raises(RuntimeError, match="all X amplitudes are zero"):
        _canonical_amplitudes([np.zeros((1, 1)), np.zeros((1, 1))],
                              [np.zeros((1, 1)), np.zeros((1, 1))])


def test_transition_features_invariant_under_mo_phases_and_virtual_rotations():
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(4, 4))
    overlap = matrix.T @ matrix + np.eye(4)
    values, vectors = np.linalg.eigh(overlap)
    inv_sqrt = (vectors / np.sqrt(values)) @ vectors.T
    coeff = inv_sqrt @ np.linalg.qr(rng.normal(size=(4, 4)))[0]
    occ = np.array([1, 0, 0, 0])
    x = [np.array([[0.3, 0.6, -0.1]]), np.array([[-0.2, -0.5, 0.3]])]
    y = [channel / 10 for channel in x]
    expected = _transition_features(x, y, [coeff, coeff], [occ, occ], overlap)
    phases = np.array([-1, 1, -1, 1])
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    changed_coeff = coeff * phases
    changed_coeff[:, 1:] = changed_coeff[:, 1:] @ rotation
    changed_x = [(channel * phases[0] * phases[1:]) @ rotation for channel in x]
    changed_y = [(channel * phases[0] * phases[1:]) @ rotation for channel in y]
    actual = _transition_features(changed_x, changed_y, [changed_coeff, changed_coeff], [occ, occ], overlap)
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, atol=1e-12)
    np.testing.assert_allclose(actual[1].sum(), 1, atol=1e-12)
    assert abs(actual[0]) <= 1


@pytest.mark.parametrize("relative_sign", [-1, 1])
def test_transition_character_distinguishes_spin_channels(relative_sign):
    x = [np.array([[0.8, 0.2]]), relative_sign * np.array([[0.8, 0.2]])]
    y = [channel / 10 for channel in x]
    character, weights = _transition_features(x, y, [np.eye(3)] * 2,
                                              [np.array([1, 0, 0])] * 2, np.eye(3))
    np.testing.assert_allclose(character, relative_sign)
    np.testing.assert_allclose(weights.sum(), 1)