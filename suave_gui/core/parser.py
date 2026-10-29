"""
File parsing, validation, and export service for electronic structure trajectories.
"""

from __future__ import annotations
import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from suave_gui.core.dataset import StateDataset


class DataParserService:
    """
    Parses and serializes multi-state curve files.
    Format specification:
      rc1 energy1 feature1.1 feature1.2 ...
      rc1 energy2 feature2.1 feature2.2 ...
      ...
      rc2 energy1 feature1.1 feature1.2 ...
    """

    @classmethod
    def inspect_file(
        cls, file_or_stream: Union[str, io.StringIO, io.BytesIO], max_preview_rows: int = 5
    ) -> Dict[str, Any]:
        """
        Inspect a file without full array reshaping to detect rows, columns, and candidate state divisors.
        """
        if isinstance(file_or_stream, str):
            if not os.path.isfile(file_or_stream):
                raise FileNotFoundError(f"File not found: {file_or_stream}")
            with open(file_or_stream, "r", encoding="utf-8") as f:
                content = f.read()
        elif isinstance(file_or_stream, io.BytesIO):
            content = file_or_stream.getvalue().decode("utf-8", errors="replace")
        elif isinstance(file_or_stream, io.StringIO):
            content = file_or_stream.getvalue()
        else:
            raise TypeError(f"Unsupported input type: {type(file_or_stream)}")

        lines = [line.strip() for line in content.strip().splitlines() if line.strip() and not line.strip().startswith("#")]
        if not lines:
            raise ValueError("Input file is empty or contains only comments.")

        total_rows = len(lines)
        first_row_tokens = lines[0].split()
        num_columns = len(first_row_tokens)

        # Detect plausible divisors for state count (2 <= num_states <= 50)
        plausible_states = [s for s in range(2, 51) if total_rows % s == 0]

        preview_rows = []
        for line in lines[:max_preview_rows]:
            try:
                row_vals = [float(token) for token in line.split()]
                preview_rows.append(row_vals)
            except ValueError:
                preview_rows.append(line.split())

        return {
            "total_rows": total_rows,
            "num_columns": num_columns,
            "num_properties": max(0, num_columns - 2),
            "plausible_states": plausible_states,
            "preview_rows": preview_rows,
        }

    @classmethod
    def parse_csv_stream(
        cls,
        stream_or_path: Union[str, io.StringIO, io.BytesIO],
        num_states: int,
        filename: str = "dataset.csv",
    ) -> StateDataset:
        """
        Parse raw numeric matrix into a verified StateDataset.
        """
        if num_states <= 0:
            raise ValueError(f"State count must be a positive integer, got {num_states}")

        # Load numerical matrix with support for inf and nan
        raw_matrix = np.genfromtxt(stream_or_path)

        if raw_matrix.ndim == 1:
            raw_matrix = raw_matrix.reshape(1, -1)

        if raw_matrix.ndim != 2:
            raise ValueError(f"Expected 2D numerical table, got dimensions {raw_matrix.ndim}")

        total_rows, num_columns = raw_matrix.shape

        if num_columns < 2:
            raise ValueError(
                f"Input requires at least 2 columns (reaction_coord, energy). Found {num_columns}."
            )

        if total_rows % num_states != 0:
            raise ValueError(
                f"Total rows ({total_rows}) cannot be evenly divided by state count ({num_states}). "
                f"Remainder is {total_rows % num_states}."
            )

        num_points = total_rows // num_states

        # Reshape: (num_points, num_states, num_columns)
        curves = raw_matrix.reshape((num_points, num_states, num_columns))
        # Swap axes to: (num_states, num_points, num_columns)
        curves = np.swapaxes(curves, 0, 1)

        reaction_coords = curves[0, :, 0]
        energies = curves[:, :, 1]
        properties = curves[:, :, 2:]

        return StateDataset(
            reaction_coordinates=reaction_coords,
            energies=energies,
            properties=properties,
            filename=filename,
            metadata={"raw_rows": total_rows, "num_columns": num_columns},
        )

    @classmethod
    def export_curves_csv(
        cls,
        dataset: StateDataset,
        property_idx: int = 0,
        filepath: Optional[str] = None,
    ) -> Optional[str]:
        """
        Export final curve ordering.
        Row 0: Reaction Coordinates
        Rows 1..num_states: Values for each state along reaction coordinates.
        """
        num_points = dataset.num_points
        curves_list = []

        if property_idx == 0:
            for pnt in range(num_points):
                curves_list.append(dataset.energies[:, pnt])
        else:
            p_idx = property_idx - 1
            for pnt in range(num_points):
                curves_list.append(dataset.properties[:, pnt, p_idx])

        stacked = np.stack(curves_list, axis=1)  # shape: (num_states, num_points)
        output_matrix = np.insert(stacked, 0, dataset.reaction_coordinates, axis=0)

        if filepath:
            np.savetxt(filepath, output_matrix, fmt="%20.12f")
            return None
        else:
            buffer = io.StringIO()
            np.savetxt(buffer, output_matrix, fmt="%20.12f")
            return buffer.getvalue()

    @classmethod
    def export_checkpoint_csv(
        cls, dataset: StateDataset, filepath: Optional[str] = None
    ) -> Optional[str]:
        """
        Export full dataset in native input format:
        pnt*num_states + state: [rc, energy, prop1, prop2, ...]
        """
        num_points = dataset.num_points
        num_states = dataset.num_states
        num_props = dataset.num_properties
        total_rows = num_points * num_states
        total_cols = num_props + 2

        output_matrix = np.zeros((total_rows, total_cols), dtype=np.float64)

        for pnt in range(num_points):
            rc = dataset.reaction_coordinates[pnt]
            for state in range(num_states):
                row_idx = pnt * num_states + state
                output_matrix[row_idx, 0] = rc
                output_matrix[row_idx, 1] = dataset.energies[state, pnt]
                for p in range(num_props):
                    output_matrix[row_idx, p + 2] = dataset.properties[state, pnt, p]

        if filepath:
            np.savetxt(filepath, output_matrix, fmt="%20.12f")
            return None
        else:
            buffer = io.StringIO()
            np.savetxt(buffer, output_matrix, fmt="%20.12f")
            return buffer.getvalue()
