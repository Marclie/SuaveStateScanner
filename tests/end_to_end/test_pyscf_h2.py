"""UKS/B3LYP/6-31G -> seven-state CSV -> scanner CLI -> full-property CSV.

PySCF calculations are shared by native/shuffled cases; skip without PySCF.
Select with ``-m pyscf``. The assignment oracle is a continuity regression,
not proof of exact physical identities at degeneracies or avoided crossings.
"""

import io
import json
import shutil

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from examples.pyscf_h2 import (
    ATTACHMENT_SLICE, BASIS, COLUMNS, FUNCTIONAL, NUM_AOS, NUM_EXCITED_STATES, NUM_STATES,
    NUM_TRANSITIONS, NUM_VIRTUAL, REFERENCE_ENERGY_COLUMN, REFERENCE_SPIN_COLUMN,
    X_SLICE, Y_SLICE, _transition_features, reaction_mesh, run_workflow,
    shuffled_labels, sort_with_scanner, write_pec,
)
from suave_gui.core.parser import DataParserService


pytestmark = pytest.mark.pyscf
# Independent continuity descriptor: excitation energy, oscillator strength,
# spin-transition character and AO attachment weights. No native root index,
# MO ID or raw amplitude information is retained.
CONTINUITY_COLUMNS = [0, 1, 2, *range(3, len(COLUMNS) - 2)]


@pytest.fixture(scope="module")
def calculations(tmp_path_factory):
    pytest.importorskip("pyscf", minversion="2.9", reason="Install requirements-pyscf.txt for ab initio workflow tests.")
    directory = tmp_path_factory.mktemp("pyscf-h2-native")
    checkpoint = run_workflow(directory)
    raw = np.genfromtxt(directory / "raw.csv").reshape(len(reaction_mesh()), NUM_STATES, len(COLUMNS))
    return directory, raw, checkpoint


@pytest.fixture(scope="module", params=[False, True], ids=["native-roots", "shuffled-roots"])
def workflow(request, calculations, tmp_path_factory):
    if not request.param:
        return calculations
    native_dir, raw, _ = calculations
    directory = tmp_path_factory.mktemp("pyscf-h2-shuffled")
    write_pec(directory / "raw.csv", raw)
    write_pec(directory / "input.csv", shuffled_labels(raw))
    metadata = json.loads((native_dir / "metadata.json").read_text())
    metadata.update(shuffled=True, shuffle_seed=7)
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    shutil.copyfile(native_dir / "orbitals.npz", directory / "orbitals.npz")
    checkpoint = sort_with_scanner(directory / "input.csv", directory)
    return directory, raw, checkpoint


def _continuity_reference(raw):
    """Nearest-neighbor energy/property assignment, independent of scanner stencils."""
    reference = raw.copy()
    for point in range(1, len(raw)):
        previous = reference[point - 1, 1:][:, CONTINUITY_COLUMNS]
        current = raw[point, 1:][:, CONTINUITY_COLUMNS]
        cost = np.sum((previous[:, None, :] - current[None, :, :])**2, axis=2)
        _, order = linear_sum_assignment(cost)
        reference[point, 1:] = raw[point, 1 + order]
    return reference


def _roughness(table):
    # Dimensionless regression score; deliberately not called a physical norm.
    return np.sum(np.diff(table[:, :, CONTINUITY_COLUMNS], axis=0)**2)


def test_generated_tddft_properties_and_ground_state(workflow):
    directory, raw, _ = workflow
    np.testing.assert_allclose(raw[:, :, 0], np.repeat(reaction_mesh()[:, None], NUM_STATES, axis=1))
    assert raw.shape == (43, 7, 9)
    assert np.all(np.isfinite(raw))
    assert np.all(raw[:, 1:, 2] > 0)
    assert np.all(np.diff(raw[:, 1:, 2], axis=1) >= -1e-10)
    np.testing.assert_array_equal(raw[:, 0, 2:], 0)
    np.testing.assert_allclose(raw[:, 1:, 5:].sum(axis=2), 1, atol=1e-9)
    assert np.all(raw[:, 1:, 5:] >= 0)
    assert np.all(raw[:, :, 3] >= -1e-12)
    minimum = np.argmin(raw[:, 0, 1])
    assert 0 < minimum < len(raw) - 1
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["basis"] == BASIS == "6-31g"
    assert metadata["functional"] == FUNCTIONAL == "b3lyp"
    assert metadata["method"] == "UKS/TDDFT"
    assert metadata["num_points"] == 43
    assert metadata["mesh_step_angstrom"] == 0.05
    assert metadata["num_states"] == NUM_STATES == 7
    assert metadata["num_excited_states"] == NUM_EXCITED_STATES == NUM_TRANSITIONS == 6
    assert metadata["complete_spin_conserving_root_set"] is True
    assert metadata["includes_ground_state"] is True
    assert metadata["columns"] == list(COLUMNS)


def test_full_amplitudes_reproduce_saved_transition_features(workflow):
    directory, raw, _ = workflow
    with np.load(directory / "orbitals.npz", allow_pickle=False) as orbitals:
        assert orbitals["mo_coeff"].shape == (43, 2, NUM_AOS, NUM_AOS)
        assert orbitals["mo_occ"].shape == (43, 2, NUM_AOS)
        np.testing.assert_allclose(orbitals["bond_lengths"], reaction_mesh())
        assert raw.shape[2] == 9
        for point in [0, 15, 30, 42]:
            for row in raw[point, 1:]:
                np.testing.assert_allclose(row[5:].sum(), 1, atol=1e-9)
                assert np.all(row[5:] >= 0)


def test_scanner_continuity_and_whole_record_conservation(workflow):
    directory, raw, checkpoint = workflow
    dataset = DataParserService.parse_csv_stream(str(checkpoint), num_states=NUM_STATES)
    saved = np.genfromtxt(checkpoint).reshape(raw.shape)
    # Compare with independent energy+property assignment, not energy ordering.
    # Root crossings need not give monotonic PECs, even with the complete root set.
    reference = _continuity_reference(raw)
    np.testing.assert_allclose(saved, reference, rtol=0, atol=2e-12)
    np.testing.assert_allclose(saved[:, 0], raw[:, 0], rtol=0, atol=2e-12)
    for original, reordered in zip(raw, saved):
        cost = np.max(np.abs(reordered[:, None] - original[None, :]), axis=2)
        rows, columns = linear_sum_assignment(cost)
        assert np.all(cost[rows, columns] <= 2e-12)
    input_table = np.genfromtxt(directory / "input.csv").reshape(raw.shape)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert _roughness(saved) < _roughness(input_table)
    if metadata["shuffled"]:
        assert not np.allclose(input_table[:, 1:, 1:], reference[:, 1:, 1:])
        assert _roughness(saved) < 0.1 * _roughness(input_table)
        np.testing.assert_array_equal(input_table[:2], raw[:2])
    # There are real crossings: correct property tracking is not the raw root order.
    assert not np.allclose(saved[:, 1:, 1:], raw[:, 1:, 1:])
    energy_output = np.genfromtxt(directory / "energies.csv")
    assert energy_output.shape == (NUM_STATES + 1, len(reaction_mesh()))
    np.testing.assert_allclose(energy_output[0], reaction_mesh(), atol=1e-12)
    np.testing.assert_allclose(energy_output[1:], dataset.energies, atol=1e-12)
    exported = DataParserService.export_checkpoint_csv(dataset)
    reparsed = DataParserService.parse_csv_stream(io.StringIO(exported), num_states=NUM_STATES)
    np.testing.assert_allclose(reparsed.energies, dataset.energies, atol=1e-12)
    np.testing.assert_allclose(reparsed.properties, dataset.properties, atol=1e-12)