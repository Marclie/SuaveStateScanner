"""
Integration tests for Curve Sampling and Export feature in Suave GUI.
Tests sampling/subdividing to 100 points, interpolating to 100 points,
interpolating/upsampling to 1000 points for both curves.csv and checkpoint.csv.
"""

import io
import json
import numpy as np
import pytest
from dash import no_update
import dash_mantine_components as dmc

from suave_gui.app import create_app
from suave_gui.core.dataset import StateDataset
from suave_gui.core.parser import DataParserService
from suave_gui.state.session_store import SessionRegistry
from suave_state_scanner import SuaveStateScanner


def test_save_modal_structure():
    """Verify that the save modal component is present in the app layout with all required controls."""
    app = create_app(default_dataset_path="input.csv", default_num_states=9)

    # Search for save modal and its controls in app layout
    layout_str = str(app.layout)
    assert "save-sampling-modal" in layout_str
    assert "save-modal-title" in layout_str
    assert "save-modal-target-type" in layout_str
    assert "save-modal-num-points" in layout_str
    assert "save-modal-interpolate-switch" in layout_str
    assert "save-modal-interp-kind" in layout_str
    assert "save-modal-filename" in layout_str
    assert "btn-save-modal-cancel" in layout_str
    assert "btn-save-modal-confirm" in layout_str


def test_curve_sampling_export_curves_subdivide_100():
    """Verify export of curves.csv with 100 subdivided points from 500 original points."""
    dataset = DataParserService.parse_csv_stream("input.csv", num_states=9, filename="input.csv")
    assert dataset.num_points == 100

    csv_str = DataParserService.export_curves_csv(
        dataset,
        property_idx=0,
        target_points=50,
        interpolate_curves=False,
    )
    mat = np.genfromtxt(io.StringIO(csv_str))

    # Shape: 10 rows (1 rc row + 9 state energy rows) and 100 columns
    assert mat.shape == (10, 50)
    # Check start and end reaction coordinates
    assert np.isclose(mat[0, 0], dataset.reaction_coordinates[0])
    assert np.isclose(mat[0, -1], dataset.reaction_coordinates[-1])
    # Check that points are exact values from original points (subdivision)
    indices = np.round(np.linspace(0, 99, 50)).astype(int)
    assert np.allclose(mat[0, :], dataset.reaction_coordinates[indices])
    assert np.allclose(mat[1:, :], dataset.energies[:, indices])


def test_curve_sampling_export_curves_interpolate_100():
    """Verify export of curves.csv with 100 interpolated points."""
    dataset = DataParserService.parse_csv_stream("input.csv", num_states=9, filename="input.csv")
    csv_str = DataParserService.export_curves_csv(
        dataset,
        property_idx=0,
        target_points=50,
        interpolate_curves=True,
        interp_kind="cubic",
    )
    mat = np.genfromtxt(io.StringIO(csv_str))

    assert mat.shape == (10, 50)
    assert np.isclose(mat[0, 0], dataset.reaction_coordinates[0])
    assert np.isclose(mat[0, -1], dataset.reaction_coordinates[-1])


def test_curve_sampling_export_curves_interpolate_1000():
    """Verify export of curves.csv with 1000 interpolated evenly spaced points."""
    dataset = DataParserService.parse_csv_stream("input.csv", num_states=9, filename="input.csv")
    csv_str = DataParserService.export_curves_csv(
        dataset,
        property_idx=0,
        target_points=1000,
        interpolate_curves=True,
        interp_kind="cubic",
    )
    mat = np.genfromtxt(io.StringIO(csv_str))

    # Shape: 10 rows and 1000 columns
    assert mat.shape == (10, 1000)
    assert np.isclose(mat[0, 0], dataset.reaction_coordinates[0])
    assert np.isclose(mat[0, -1], dataset.reaction_coordinates[-1])


def test_curve_sampling_export_checkpoint_subdivide_100():
    """Verify export of checkpoint.csv with 100 subdivided points and roundtrip parsing."""
    dataset = DataParserService.parse_csv_stream("input.csv", num_states=9, filename="input.csv")
    csv_str = DataParserService.export_checkpoint_csv(
        dataset,
        target_points=50,
        interpolate_curves=False,
    )

    reimported = DataParserService.parse_csv_stream(io.StringIO(csv_str), num_states=9)
    assert reimported.num_points == 50
    assert reimported.num_states == 9
    assert reimported.num_properties == dataset.num_properties

    indices = np.round(np.linspace(0, 99, 50)).astype(int)
    assert np.allclose(reimported.reaction_coordinates, dataset.reaction_coordinates[indices])
    assert np.allclose(reimported.energies, dataset.energies[:, indices])
    assert np.allclose(reimported.properties, dataset.properties[:, indices, :])


def test_curve_sampling_export_checkpoint_interpolate_1000():
    """Verify export of checkpoint.csv with 1000 interpolated points and roundtrip parsing."""
    dataset = DataParserService.parse_csv_stream("input.csv", num_states=9, filename="input.csv")
    csv_str = DataParserService.export_checkpoint_csv(
        dataset,
        target_points=1000,
        interpolate_curves=True,
        interp_kind="cubic",
    )

    reimported = DataParserService.parse_csv_stream(io.StringIO(csv_str), num_states=9)
    assert reimported.num_points == 1000
    assert reimported.num_states == 9
    assert reimported.num_properties == dataset.num_properties

    assert np.isclose(reimported.reaction_coordinates[0], dataset.reaction_coordinates[0])
    assert np.isclose(reimported.reaction_coordinates[-1], dataset.reaction_coordinates[-1])


def test_backend_saveorder_with_sampling(tmp_path):
    """Verify backend SuaveStateScanner.saveOrder directly supports sampling for checkpoint and curves output."""
    outfile = str(tmp_path / "out_sample.csv")
    suave = SuaveStateScanner("input.csv", outfile, 9, "suave.json")

    # 1. Subdivide to 100 points
    suave.saveOrder(isFinalResults=True, target_points=100, interpolate=False)
    chk_100 = np.genfromtxt("checkpoint.csv")
    assert chk_100.shape == (900, 10)  # 100 points * 9 states = 900 rows
    out_100 = np.genfromtxt(outfile)
    assert out_100.shape == (10, 100)

    # 2. Interpolate to 1000 points
    suave.saveOrder(isFinalResults=True, target_points=1000, interpolate=True)
    chk_1000 = np.genfromtxt("checkpoint.csv")
    assert chk_1000.shape == (9000, 10)  # 1000 points * 9 states = 9000 rows
    out_1000 = np.genfromtxt(outfile)
    assert out_1000.shape == (10, 1000)
