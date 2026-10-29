"""
Domain models for potential energy curves and electronic structure properties.
Encapsulates state data, immutable snapshots, and safe mutation primitives.
Synchronized directly with backend SuaveStateScanner data structures without copying.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import interpolate


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable snapshot of the dataset matrices for undo/redo tracking."""
    energies: np.ndarray
    properties: np.ndarray
    description: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        # Ensure arrays are read-only copies
        if not self.energies.flags.writeable:
            object.__setattr__(self, "energies", self.energies.copy())
        if not self.properties.flags.writeable:
            object.__setattr__(self, "properties", self.properties.copy())
        self.energies.flags.writeable = False
        self.properties.flags.writeable = False


class StateDataset:
    """
    Primary container representing multi-state curve trajectories.
    Energies (E) have shape: (num_states, num_points)
    Properties (P) have shape: (num_states, num_points, num_properties)
    Reaction coordinates have shape: (num_points,)

    When initialized with suave_instance, directly references the backend
    SuaveStateScanner arrays without copying, guaranteeing 100% synchronization.
    """

    def __init__(
        self,
        reaction_coordinates: np.ndarray,
        energies: np.ndarray,
        properties: np.ndarray,
        filename: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        suave_instance: Optional[Any] = None,
    ):
        self.suave = suave_instance
        # Directly reference backend arrays if provided (no .copy())
        self._rc = np.asarray(reaction_coordinates, dtype=np.float64)
        self._e = np.asarray(energies, dtype=np.float64)
        self._p = np.asarray(properties, dtype=np.float64)
        self._filename = filename
        self._metadata = metadata or {}

        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        """Validate tensor shapes and consistency."""
        if self._rc.ndim != 1:
            raise ValueError(f"Reaction coordinates must be 1D, got shape {self._rc.shape}")
        if self._e.ndim != 2:
            raise ValueError(f"Energies must be 2D (states, points), got shape {self._e.shape}")
        if self._e.shape[1] != self._rc.shape[0]:
            raise ValueError(
                f"Mismatch: {self._rc.shape[0]} points in coordinates, but energies have {self._e.shape[1]} columns"
            )
        if self._p.ndim != 3:
            raise ValueError(
                f"Properties must be 3D (states, points, properties), got shape {self._p.shape}"
            )
        if self._p.shape[:2] != self._e.shape:
            raise ValueError(
                f"Shape mismatch: Energies {self._e.shape} vs Properties {self._p.shape[:2]}"
            )

    @property
    def num_states(self) -> int:
        if self.suave is not None and hasattr(self.suave, "numStates"):
            return self.suave.numStates
        return self.energies.shape[0]

    @property
    def num_points(self) -> int:
        if self.suave is not None and hasattr(self.suave, "numPoints"):
            return self.suave.numPoints
        return self.energies.shape[1]

    @property
    def num_properties(self) -> int:
        if self.suave is not None and hasattr(self.suave, "numProps"):
            return self.suave.numProps
        return self.properties.shape[2]

    @property
    def reaction_coordinates(self) -> np.ndarray:
        if self.suave is not None and hasattr(self.suave, "allPnts"):
            return self.suave.allPnts
        return self._rc

    @reaction_coordinates.setter
    def reaction_coordinates(self, value: np.ndarray) -> None:
        val_arr = np.asarray(value, dtype=np.float64)
        if self.suave is not None and hasattr(self.suave, "allPnts"):
            self.suave.allPnts = val_arr
        self._rc = val_arr

    @property
    def energies(self) -> np.ndarray:
        if self.suave is not None and hasattr(self.suave, "E"):
            return self.suave.E
        return self._e

    @energies.setter
    def energies(self, value: np.ndarray) -> None:
        val_arr = np.asarray(value, dtype=np.float64)
        if val_arr.shape != self.energies.shape:
            raise ValueError(f"Cannot assign energies of shape {val_arr.shape} to shape {self.energies.shape}")
        if self.suave is not None and hasattr(self.suave, "E"):
            self.suave.E = val_arr
        self._e = val_arr

    @property
    def properties(self) -> np.ndarray:
        if self.suave is not None and hasattr(self.suave, "P"):
            return self.suave.P
        return self._p

    @properties.setter
    def properties(self, value: np.ndarray) -> None:
        val_arr = np.asarray(value, dtype=np.float64)
        if val_arr.shape != self.properties.shape:
            raise ValueError(f"Cannot assign properties of shape {val_arr.shape} to shape {self.properties.shape}")
        if self.suave is not None and hasattr(self.suave, "P"):
            self.suave.P = val_arr
        self._p = val_arr

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def min_step_size(self) -> float:
        """Find the smallest step size between consecutive reaction coordinates."""
        rc = self.reaction_coordinates
        if self.num_points <= 1:
            return 0.0
        diffs = np.diff(rc)
        positive_diffs = diffs[diffs > 0]
        return float(np.min(positive_diffs)) if len(positive_diffs) > 0 else float(np.min(np.abs(diffs)))

    @property
    def has_missing(self) -> bool:
        """Check whether energies or properties contain NaN or Inf values."""
        e = self.energies
        p = self.properties
        return (
            bool(np.isnan(e).any())
            or bool(np.isinf(e).any())
            or bool(np.isnan(p).any())
            or bool(np.isinf(p).any())
        )

    def create_snapshot(self, description: str = "") -> StateSnapshot:
        """Create an immutable snapshot of current state for undo/redo."""
        import time
        return StateSnapshot(
            energies=self.energies.copy(),
            properties=self.properties.copy(),
            description=description,
            timestamp=time.time(),
        )

    def restore_snapshot(self, snapshot: StateSnapshot) -> None:
        """Restore state from a snapshot directly into the underlying arrays in-place."""
        if self.suave is not None and hasattr(self.suave, "E"):
            self.suave.E[:] = snapshot.energies
            self.suave.P[:] = snapshot.properties
        self._e[:] = snapshot.energies
        self._p[:] = snapshot.properties

    def clone(self) -> StateDataset:
        """Return a copy of this dataset."""
        return StateDataset(
            reaction_coordinates=self.reaction_coordinates.copy(),
            energies=self.energies.copy(),
            properties=self.properties.copy(),
            filename=self._filename,
            metadata=dict(self._metadata),
        )

    def get_curve(self, state_idx: int, property_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract (x, y) coordinates for a given state and property.
        property_idx = 0 represents energy.
        property_idx > 0 represents property at index (property_idx - 1).
        """
        if not (0 <= state_idx < self.num_states):
            raise IndexError(f"State index {state_idx} out of range [0, {self.num_states})")

        x = self.reaction_coordinates
        if property_idx == 0:
            y = self.energies[state_idx, :]
        else:
            p_idx = property_idx - 1
            if not (0 <= p_idx < self.num_properties):
                raise IndexError(f"Property index {property_idx} out of range [0, {self.num_properties}]")
            y = self.properties[state_idx, :, p_idx]
        return x, y

    def apply_swap(
        self, state_a: int, state_b: int, point_start: int = 0, point_end: Optional[int] = None
    ) -> None:
        """
        Swap energies and properties of state_a and state_b within point range [point_start, point_end).
        Directly mutates the in-memory array (and thus the backend calculation structure).
        """
        if not (0 <= state_a < self.num_states and 0 <= state_b < self.num_states):
            raise IndexError(f"Invalid state indices: {state_a}, {state_b}")

        if point_end is None:
            point_end = self.num_points

        p_start = max(0, point_start)
        p_end = min(self.num_points, point_end)

        if p_start >= p_end:
            raise ValueError(f"Invalid point range [{p_start}, {p_end})")

        idx_slice = slice(p_start, p_end)
        e = self.energies
        p = self.properties
        e[[state_a, state_b], idx_slice] = e[[state_b, state_a], idx_slice]
        p[[state_a, state_b], idx_slice] = p[[state_b, state_a], idx_slice]
        if e is not self._e:
            self._e[[state_a, state_b], idx_slice] = self._e[[state_b, state_a], idx_slice]
        if p is not self._p:
            self._p[[state_a, state_b], idx_slice] = self._p[[state_b, state_a], idx_slice]

    def remove_state_range(
        self, state_idx: int, point_start: int = 0, point_end: Optional[int] = None, prop_indices: Optional[List[int]] = None
    ) -> None:
        """
        Set energy and/or properties to NaN for a state across a point range.
        Directly mutates the in-memory array.
        """
        if not (0 <= state_idx < self.num_states):
            raise IndexError(f"Invalid state index: {state_idx}")

        if point_end is None:
            point_end = self.num_points

        p_start = max(0, point_start)
        p_end = min(self.num_points, point_end)
        idx_slice = slice(p_start, p_end)

        e = self.energies
        p = self.properties
        e[state_idx, idx_slice] = np.nan
        if prop_indices is None:
            p[state_idx, idx_slice, :] = np.nan
        else:
            p[state_idx, idx_slice, prop_indices] = np.nan

        if e is not self._e:
            self._e[state_idx, idx_slice] = np.nan
        if p is not self._p:
            if prop_indices is None:
                self._p[state_idx, idx_slice, :] = np.nan
            else:
                self._p[state_idx, idx_slice, prop_indices] = np.nan

        if self.suave is not None:
            self.suave.hasMissing = True

    def shuffle_points(self, point_start: int = 0, point_end: Optional[int] = None) -> None:
        """
        Randomly permute states ONLY for points in the sliding-window range [point_start, point_end).
        Points outside this sliding-window region remain unmodified.
        Directly mutates the in-memory backend data structure.
        """
        if point_end is None:
            point_end = self.num_points
        p_start = max(0, point_start)
        p_end = min(self.num_points, point_end)
        e = self.energies
        p = self.properties
        for pnt in range(p_start, p_end):
            perm = np.random.permutation(self.num_states)
            e[:, pnt] = e[perm, pnt]
            p[:, pnt, :] = p[perm, pnt, :]
            if e is not self._e:
                self._e[:, pnt] = e[:, pnt]
            if p is not self._p:
                self._p[:, pnt, :] = p[:, pnt, :]

    def shuffle_energies(self) -> None:
        """Permute state order across all points."""
        self.shuffle_points(0, self.num_points)

    def sort_energies(self, sort_by_last_point: bool = False) -> None:
        """
        Sort states monotonically by energy at each point, reordering properties simultaneously.
        """
        e = self.energies
        p = self.properties
        if sort_by_last_point:
            order = np.argsort(e[:, -1])
            e[:] = e[order, :]
            p[:] = p[order, :, :]
        else:
            for pnt in range(self.num_points):
                order = np.argsort(e[:, pnt])
                e[:, pnt] = e[order, pnt]
                p[:, pnt, :] = p[order, pnt, :]
        if e is not self._e:
            self._e[:] = e
        if p is not self._p:
            self._p[:] = p

    def interpolate_missing(self, kind: str = "cubic") -> None:
        """
        Interpolate missing NaN / Inf values across the reaction coordinate for each state.
        """
        e = self.energies
        p = self.properties
        for state in range(self.num_states):
            e_curve = e[state, :]
            valid_e = np.where(np.isfinite(e_curve))[0]
            if len(valid_e) >= 2 and len(valid_e) < self.num_points:
                interp_kind = kind if len(valid_e) >= 4 and kind == "cubic" else "linear"
                f_e = interpolate.interp1d(
                    valid_e, e_curve[valid_e], kind=interp_kind, fill_value="extrapolate"
                )
                e[state, :] = f_e(np.arange(self.num_points))

            for prop in range(self.num_properties):
                p_curve = p[state, :, prop]
                valid_p = np.where(np.isfinite(p_curve))[0]
                if len(valid_p) >= 2 and len(valid_p) < self.num_points:
                    interp_kind = kind if len(valid_p) >= 4 and kind == "cubic" else "linear"
                    f_p = interpolate.interp1d(
                        valid_p, p_curve[valid_p], kind=interp_kind, fill_value="extrapolate"
                    )
                    p[state, :, prop] = f_p(np.arange(self.num_points))
        if e is not self._e:
            self._e[:] = e
        if p is not self._p:
            self._p[:] = p
