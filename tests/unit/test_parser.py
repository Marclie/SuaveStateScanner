"""
Unit tests for DataParserService.
"""

import io
import os
import numpy as np
import pytest
from suave_gui.core.parser import DataParserService


def test_inspect_file(tmp_path):
    csv_content = """# Comment line
1.0 10.0 0.1 0.2
1.0 12.0 0.3 0.4
2.0 10.1 0.1 0.2
2.0 12.1 0.3 0.4
"""
    test_file = tmp_path / "test.csv"
    test_file.write_text(csv_content, encoding="utf-8")

    info = DataParserService.inspect_file(str(test_file))
    assert info["total_rows"] == 4
    assert info["num_columns"] == 4
    assert info["num_properties"] == 2
    assert 2 in info["plausible_states"]
    assert 4 in info["plausible_states"]
    assert len(info["preview_rows"]) == 4


def test_parse_csv_stream():
    csv_data = """
1.0 -10.0 1.0 2.0
1.0 -8.0  3.0 4.0
2.0 -9.9  1.0 2.0
2.0 -7.9  3.0 4.0
"""
    stream = io.StringIO(csv_data.strip())
    dataset = DataParserService.parse_csv_stream(stream, num_states=2, filename="sample.csv")

    assert dataset.num_states == 2
    assert dataset.num_points == 2
    assert dataset.num_properties == 2
    assert np.array_equal(dataset.reaction_coordinates, np.array([1.0, 2.0]))
    assert dataset.energies[0, 0] == -10.0
    assert dataset.energies[1, 0] == -8.0
    assert dataset.energies[0, 1] == -9.9
    assert dataset.energies[1, 1] == -7.9


def test_parse_invalid_state_count():
    csv_data = """
1.0 -10.0 1.0
1.0 -8.0  3.0
2.0 -9.9  1.0
"""
    stream = io.StringIO(csv_data.strip())
    with pytest.raises(ValueError, match="cannot be evenly divided"):
        DataParserService.parse_csv_stream(stream, num_states=2)


def test_parse_real_input_csv():
    # Test with the actual input.csv in repo root
    input_path = "input.csv"
    if os.path.exists(input_path):
        dataset = DataParserService.parse_csv_stream(input_path, num_states=9, filename="input.csv")
        assert dataset.num_states == 9
        assert dataset.num_points == 100
        assert dataset.num_properties == 8
        assert dataset.reaction_coordinates[0] == 0.9000
        assert np.isclose(dataset.reaction_coordinates[1] - dataset.reaction_coordinates[0], 0.005)


def test_export_and_reimport():
    csv_data = """
0.90 -110.0 1.5 2.5
0.90 -108.0 3.5 4.5
0.91 -109.9 1.6 2.6
0.91 -107.9 3.6 4.6
"""
    dataset = DataParserService.parse_csv_stream(io.StringIO(csv_data.strip()), num_states=2)
    checkpoint_csv = DataParserService.export_checkpoint_csv(dataset)

    # Reimport exported checkpoint
    reimported = DataParserService.parse_csv_stream(io.StringIO(checkpoint_csv), num_states=2)
    assert np.allclose(dataset.energies, reimported.energies)
    assert np.allclose(dataset.properties, reimported.properties)
    assert np.allclose(dataset.reaction_coordinates, reimported.reaction_coordinates)


def test_export_curves_with_resampling():
    input_path = "input.csv"
    if os.path.exists(input_path):
        dataset = DataParserService.parse_csv_stream(input_path, num_states=9)

        # 100 points subdivide
        curves_100 = DataParserService.export_curves_csv(dataset, target_points=100, interpolate_curves=False)
        mat_100 = np.genfromtxt(io.StringIO(curves_100))
        assert mat_100.shape == (10, 100)

        # 1000 points interpolate
        curves_1000 = DataParserService.export_curves_csv(dataset, target_points=1000, interpolate_curves=True)
        mat_1000 = np.genfromtxt(io.StringIO(curves_1000))
        assert mat_1000.shape == (10, 1000)


def test_export_checkpoint_with_resampling():
    input_path = "input.csv"
    if os.path.exists(input_path):
        dataset = DataParserService.parse_csv_stream(input_path, num_states=9)

        # 100 points subdivide
        ckpt_100 = DataParserService.export_checkpoint_csv(dataset, target_points=100, interpolate_curves=False)
        reimported_100 = DataParserService.parse_csv_stream(io.StringIO(ckpt_100), num_states=9)
        assert reimported_100.num_points == 100

        # 1000 points interpolate
        ckpt_1000 = DataParserService.export_checkpoint_csv(dataset, target_points=1000, interpolate_curves=True)
        reimported_1000 = DataParserService.parse_csv_stream(io.StringIO(ckpt_1000), num_states=9)
        assert reimported_1000.num_points == 1000
