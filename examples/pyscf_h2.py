"""Generate H2 UKS/B3LYP/6-31G TDDFT curves and run the scanner CLI.

Run from the repository root with ``python -m examples.pyscf_h2``. PySCF is
optional; there are seven states per geometry: the ground state and all six
positive spin-conserving full TDDFT roots (not TDA).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


BASIS = "6-31g"
FUNCTIONAL = "b3lyp"
NUM_EXCITED_STATES = 6
NUM_STATES = NUM_EXCITED_STATES + 1
NUM_AOS = 4  # Two 6-31G functions on each hydrogen; STO-3G allows only two roots.
NUM_VIRTUAL = NUM_AOS - 1  # One occupied orbital per spin.
NUM_TRANSITIONS = 2 * NUM_VIRTUAL
COLUMNS = (
    "bond_length_angstrom", "total_energy_hartree", "excitation_energy_hartree",
    "oscillator_strength_length_gauge", "spin_transition_character",
    *(f"attachment_weight_ao_{ao}" for ao in range(NUM_AOS)),
)
# The full TDDFT amplitudes are still computed at runtime for the underlying
# reference data, but the workflow outputs only the smooth continuity metrics.
X_SLICE = slice(0, 0)
Y_SLICE = slice(0, 0)
ATTACHMENT_SLICE = slice(COLUMNS.index("attachment_weight_ao_0"), len(COLUMNS))
REFERENCE_ENERGY_COLUMN = COLUMNS.index("total_energy_hartree")
REFERENCE_SPIN_COLUMN = COLUMNS.index("total_energy_hartree")
# propList indexes columns AFTER coordinate and total energy.
SORT_PROPERTIES = [COLUMNS.index(name) - 2 for name in (
    "excitation_energy_hartree", "oscillator_strength_length_gauge",
    "spin_transition_character",
    *(f"attachment_weight_ao_{ao}" for ao in range(NUM_AOS)),
)]


def reaction_mesh() -> np.ndarray:
    """43 points from 0.40 to 2.50 Angstrom inclusive, uniformly spaced by 0.05."""
    return np.arange(40, 251, 5, dtype=float) / 100.0


def _canonical_amplitudes(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Flatten alpha then beta channels and fix the overall real TD state sign.

    Choose the largest alpha X coefficient positive (fall back to beta). This
    does NOT align the underlying MOs across geometries: signed coefficients and
    their energy-ordered MO indices are diagnostic, not continuity features.
    """
    x_flat = np.concatenate([np.asarray(channel, dtype=float).ravel() for channel in x])
    y_flat = np.concatenate([np.asarray(channel, dtype=float).ravel() for channel in y])
    if x_flat.shape != y_flat.shape or not x_flat.size:
        raise ValueError("X and Y must have identical, nonempty spin-channel shapes.")
    alpha_size = np.asarray(x[0]).size
    anchor = x_flat[:alpha_size]
    if not anchor.size or np.max(np.abs(anchor)) < 1e-12:
        anchor = x_flat
    sign = np.sign(anchor[np.argmax(np.abs(anchor))])
    if not sign:
        raise RuntimeError("Cannot fix the TDDFT phase: all X amplitudes are zero.")
    return x_flat * sign, y_flat * sign


def _transition_features(x, y, mo_coeff, mo_occ, overlap) -> tuple[float, np.ndarray]:
    """MO-phase/permutation-invariant spin character and AO attachment weights.

    In the symmetric orthonormal AO basis, D_sigma = S**(1/2) C_occ
    (X_sigma+Y_sigma) C_virt.T S**(1/2). The normalized alpha/beta overlap
    is triplet-like (-1) or singlet-like (+1) on a spin-pure reference, not an
    excited-state <S^2>. Diag(sum_sigma D_sigma.T D_sigma), normalized to one,
    describes where the transition's virtual/attachment character resides.
    """
    values, vectors = np.linalg.eigh(overlap)
    if np.any(values <= 0):
        raise RuntimeError("AO overlap is not positive definite.")
    sqrt_overlap = (vectors * np.sqrt(values)) @ vectors.T
    densities = []
    for xs, ys, coeff, occ in zip(x, y, mo_coeff, mo_occ):
        occupied = coeff[:, occ > 0]
        virtual = coeff[:, occ == 0]
        densities.append(sqrt_overlap @ occupied @ (xs + ys) @ virtual.T @ sqrt_overlap)
    norm = sum(np.sum(d**2) for d in densities)
    if norm < 1e-14:
        raise RuntimeError("Excited-state transition density has zero norm.")
    character = 2 * np.sum(densities[0] * densities[1]) / norm
    attachment = sum(np.sum(d**2, axis=0) for d in densities) / norm
    return float(character), attachment


def _localized_guess(mf) -> np.ndarray:
    """Positive semidefinite alpha-on-H1/beta-on-H2 atomic density, one e-/spin."""
    atomic = mf.get_init_guess(key="atom").sum(axis=0)
    overlap = mf.get_ovlp()
    guess = []
    for atom in range(2):
        start, stop = mf.mol.aoslice_by_atom()[atom, 2:]
        density = np.zeros_like(atomic)
        density[start:stop, start:stop] = atomic[start:stop, start:stop]
        guess.append(density / np.einsum("ij,ji", density, overlap))
    return np.asarray(guess)


def generate_pec(coordinates=None, *, orbital_path: Path | None = None) -> np.ndarray:
    """Return (points, 7 states, 8 smooth columns) in ground + native TDDFT order.

    Only the smooth continuity features are retained in the exported workflow:
    excitation energy, oscillator strength and AO attachment weights. The full
    TDDFT amplitude diagnostics remain available internally for analysis, but they
    are not stored in the exported state-curve data.
    """
    coordinates = reaction_mesh() if coordinates is None else np.asarray(coordinates, dtype=float)
    if (coordinates.ndim != 1 or not coordinates.size
            or not np.all(np.isfinite(coordinates)) or np.any(coordinates <= 0)
            or np.any(np.diff(coordinates) <= 0)):
        raise ValueError("Bond lengths must be finite, positive and strictly increasing.")

    if NUM_EXCITED_STATES > NUM_TRANSITIONS:
        raise ValueError(f"H2/{BASIS} supports at most {NUM_TRANSITIONS} spin-conserving "
                         f"TDDFT roots, not {NUM_EXCITED_STATES}.")

    from pyscf import dft, gto, lib, tdscf

    rows, coeffs, occupations, orbital_energies, overlaps = [], [], [], [], []
    with lib.with_omp_threads(1):
        for distance in coordinates:
            # C2v does not forbid alpha-on-H1/beta-on-H2 spin symmetry breaking
            # along the z axis. All orbitals are sigma-type in this s-only basis.
            mol = gto.M(
                atom=[("H", (0, 0, 0)), ("H", (0, 0, float(distance)))],
                unit="Angstrom", basis=BASIS, charge=0, spin=0,
                symmetry="C2v", verbose=0,
            )
            if mol.nao_nr() != NUM_AOS or mol.nelec != (1, 1):
                raise RuntimeError(f"Unexpected H2/{BASIS} orbital dimensions.")
            mf = dft.UKS(mol)
            mf.xc = FUNCTIONAL
            mf.grids.level = 3
            mf.conv_tol = 1e-10
            mf.max_cycle = 200
            mf.kernel(dm0=_localized_guess(mf))
            if not mf.converged:
                raise RuntimeError(f"UKS did not converge at R={distance:.2f} Angstrom.")
            _, _, stable, _ = mf.stability(return_status=True)
            if not stable:
                raise RuntimeError(f"Internally unstable UKS at R={distance:.2f} Angstrom.")

            td = tdscf.TDDFT(mf)
            td.nstates = NUM_EXCITED_STATES
            td.conv_tol = 1e-9
            td.max_cycle = 200
            # Span all six spin-conserving single excitations for the complete
            # 12-dimensional full-response problem. Y starts zero, but the full
            # TDDFT solver produces nonzero de-excitations (not TDA).
            td.kernel(x0=np.c_[np.eye(NUM_TRANSITIONS), np.zeros((NUM_TRANSITIONS, NUM_TRANSITIONS))])
            if (len(td.e) != NUM_EXCITED_STATES or len(td.xy) != NUM_EXCITED_STATES
                    or not np.all(td.converged) or not np.all(np.isfinite(td.e))
                    or np.iscomplexobj(td.e) or np.any(td.e <= 0)):
                raise RuntimeError(f"Expected {NUM_EXCITED_STATES} positive converged TDDFT roots "
                                   f"at R={distance:.2f} Angstrom.")

            strengths = td.oscillator_strength(gauge="length")
            dipoles = td.transition_dipole()
            spin_squared = mf.spin_square()[0]
            overlap = mf.get_ovlp()
            ground = np.zeros(len(COLUMNS))
            ground[0:2] = [distance, mf.e_tot]
            point = [ground]
            for state, (x, y) in enumerate(td.xy):
                character, attachment = _transition_features(x, y, mf.mo_coeff, mf.mo_occ, overlap)
                point.append([
                    distance,
                    mf.e_tot + td.e[state],
                    td.e[state],
                    strengths[state],
                    character,
                    *attachment,
                ])
            rows.append(point)
            coeffs.append(mf.mo_coeff)
            occupations.append(mf.mo_occ)
            orbital_energies.append(mf.mo_energy)
            overlaps.append(overlap)

    table = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(table)):
        raise RuntimeError("Nonfinite TDDFT properties in generated PEC.")
    if orbital_path is not None:
        orbital_path = Path(orbital_path)
        orbital_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(orbital_path, bond_lengths=coordinates, mo_coeff=np.asarray(coeffs),
                            mo_occ=np.asarray(occupations), mo_energy=np.asarray(orbital_energies),
                            ao_overlap=np.asarray(overlaps), ao_labels=np.asarray(mol.ao_labels()))
    return table


def shuffled_labels(table: np.ndarray, seed: int = 7) -> np.ndarray:
    """Shuffle complete excited-state records; keep ground and two anchor points."""
    shuffled = table.copy()
    rng = np.random.default_rng(seed)
    for point in range(2, len(table)):
        shuffled[point, 1:] = table[point, 1 + rng.permutation(NUM_EXCITED_STATES)]
    return shuffled


def write_pec(path: Path, table: np.ndarray) -> None:
    """Write whitespace-delimited, comment-headed CSV understood by both parsers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, table.reshape(-1, len(COLUMNS)), fmt="%.16e", header=" ".join(COLUMNS))


def sort_with_scanner(input_path: Path, output_dir: Path) -> Path:
    """Exercise the real CLI, isolating its checkpoint/temp files in output_dir."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "suave.json"
    config_path.write_text(json.dumps({
        "interactive": False, "maxiter": 20, "orders": [1], "width": 3,
        "futurePnts": 0, "maxPan": 1, "propList": SORT_PROPERTIES,
        "stateBounds": [1, NUM_STATES],  # Ground state is known; sort only excitations.
        "sweepBack": False, "redundantSwaps": True,
        "makePos": False, "normalize": False, "nthreads": 1,
        "interpolate": False, "keepInterp": False, "doShuffle": False,
    }, indent=2) + "\n", encoding="utf-8")
    scanner_script = Path(__file__).resolve().parents[1] / "suave_state_scanner.py"
    log_path = output_dir / "scanner.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run([
            sys.executable, str(scanner_script), "-c", str(config_path),
            str(Path(input_path).resolve()), str(output_dir / "energies.csv"), str(NUM_STATES),
        ], cwd=output_dir, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    if "%%%%%%%%%%%%%%%%%%%% CONVERGED" not in log_path.read_text(encoding="utf-8"):
        raise RuntimeError(f"Scanner did not converge; inspect {log_path}.")
    return output_dir / "checkpoint.csv"


def run_workflow(output_dir: Path, shuffle: bool = False, seed: int = 7) -> Path:
    """Save raw calculations, optional shuffled input, orbital archive and sorting."""
    import pyscf
    from pyscf.dft import libxc

    output_dir = Path(output_dir).resolve()
    raw = generate_pec(orbital_path=output_dir / "orbitals.npz")
    write_pec(output_dir / "raw.csv", raw)
    write_pec(output_dir / "input.csv", shuffled_labels(raw, seed) if shuffle else raw)
    metadata = {
        "system": "H2", "basis": BASIS, "functional": FUNCTIONAL, "method": "UKS/TDDFT",
        "pyscf_version": pyscf.__version__, "libxc_version": libxc.__version__,
        "functional_components": repr(libxc.parse_xc(FUNCTIONAL)), "grid_level": 3,
        "symmetry": "C2v", "num_states": NUM_STATES, "num_excited_states": NUM_EXCITED_STATES,
        "includes_ground_state": True, "ground_state_index": 0,
        "num_points": len(raw), "bond_lengths_angstrom": raw[:, 0, 0].tolist(),
        "mesh_step_angstrom": 0.05, "num_spin_conserving_transitions": NUM_TRANSITIONS,
        "complete_spin_conserving_root_set": NUM_EXCITED_STATES == NUM_TRANSITIONS,
        "columns": list(COLUMNS), "sort_property_indices": SORT_PROPERTIES,
        "shuffled": shuffle, "shuffle_seed": seed if shuffle else None,
        "scf_convergence_tolerance": 1e-10, "tddft_convergence_tolerance": 1e-9,
        "amplitude_normalization": "excited roots: sum(X**2 - Y**2) = 1; ground: X=Y=0",
        "phase_convention": "largest alpha X positive, beta fallback; MOs in native energy order",
        "orbital_archive": "orbitals.npz: MO coefficients/occupations/energies, AO overlap/labels",
        "ground_transition_properties": "zero; dominant MO/spin identifiers -1 (not applicable)",
        "reference": "localized alpha/beta atomic guess; internal UKS stability checked",
        "energy_definition": "UKS total energy (including nuclear repulsion) + TDDFT excitation energy",
        "spin_transition_character": "normalized alpha/beta transition-density overlap, NOT excited-state S^2",
        "caveat": "Complete six-root spin-conserving set; no spin-flip or double excitations; broken-symmetry UKS is spin contaminated; 6-31G has no polarization functions on H.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return sort_with_scanner(output_dir / "input.csv", output_dir)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pyscf_h2"))
    parser.add_argument("--shuffle", action="store_true", help="Challenge the scanner with shuffled excited roots.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    checkpoint = run_workflow(args.output_dir, shuffle=args.shuffle, seed=args.seed)
    print(f"Generated {len(reaction_mesh())} H2/{BASIS} geometries with UKS/{FUNCTIONAL.upper()} TDDFT: "
          f"ground + {NUM_EXCITED_STATES} excited states.\nSorted energies and properties: {checkpoint}")


if __name__ == "__main__":
    main()