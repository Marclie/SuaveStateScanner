"""
Algorithmic Facade encapsulating the Suave state scanning and continuity engine.
Provides a clean, headless API over Numba-accelerated finite difference sweeps.
"""

from __future__ import annotations
import copy
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from suave_state_scanner import SuaveStateScanner
from suave_metric import getMetric
from suave_gui.core.dataset import StateDataset


@dataclass
class SweepProgress:
    """Progress status for long-running algorithmic sweeps."""
    current_sweep: int
    total_sweeps: int
    del_max: float
    converged: bool
    aborted: bool
    message: str = ""


class SuaveFacade(SuaveStateScanner):
    """
    Decoupled facade for SuaveStateScanner that operates directly on in-memory StateDatasets.
    Eliminates dependency on CLI arguments, global state, and direct disk files.
    """

    def __init__(
        self,
        dataset: Optional[StateDataset] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        # Initialize basic attributes without triggering CLI/disk parsing
        self.infile = dataset.filename if dataset else "memory.csv"
        self.outfile = "output.csv"
        self.numStates = dataset.num_states if dataset else 0
        self.configPath = None
        self.getMetric = getMetric

        # Default configuration
        self.printVar = 0
        self.maxiter = 1000
        self.tol = 1e-12
        self.orders = [1]
        self.width = 5
        self.futurePnts = 0
        self.maxPan = None
        self.stateBounds = None
        self.pntBounds = None
        self.propList = None
        self.ignoreProps = False
        self.sweepBack = False
        self.eBounds = None
        self.eWidth = None
        self.interpolate = False
        self.sortLast = False
        self.keepInterp = False
        self.nthreads = 1
        self.makePos = False
        self.normalize = False
        self.doShuffle = False
        self.maxStateRepeat = -1
        self.redundantSwaps = True
        self.interactive = False
        self.halt = False
        self.hasMissing = False
        self.stateMap = None

        # Finite difference working parameters
        self.center = None
        self.validPnts = None
        self.backwards = None
        self.bounds = None
        self.setDiff = None
        self.combinedStencils = None

        if config:
            self.configure(config)

        if dataset:
            self.load_dataset(dataset)

    def load_dataset(self, dataset: StateDataset) -> None:
        """Load in-memory dataset arrays into the scanner engine."""
        self.numStates = dataset.num_states
        self.numPoints = dataset.num_points
        self.numProps = dataset.num_properties

        self.E = dataset.energies
        self.P = dataset.properties
        self.allPnts = dataset.reaction_coordinates
        self.minh = dataset.min_step_size
        dataset.suave = self

        self.pntBounds = self.pntBounds or [0, self.numPoints]
        self.stateBounds = self.stateBounds or [0, self.numStates]
        self.propList = self.propList or [i for i in range(self.numProps)]
        self.stateRepeatList = np.zeros(self.numStates)

        self.checkMissing()

    def sync_to_dataset(self, dataset: StateDataset) -> None:
        """Propagate current scanner state back to a StateDataset instance."""
        dataset.suave = self
        dataset.energies = self.E
        dataset.properties = self.P

    def configure(self, config_dict: Dict[str, Any]) -> None:
        """Update algorithm parameters dynamically."""
        for key, val in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, val)

        # Enforce constraints
        if self.width <= 0 or (self.orders and self.width <= max(self.orders)):
            self.width = max(self.orders) + 3

    def execute_sweeps(
        self,
        num_sweeps: int = 1,
        direction: str = "forward",  # 'forward', 'backward', 'alternate'
        sort_last: Optional[bool] = None,
        progress_callback: Optional[Callable[[SweepProgress], None]] = None,
        cancel_token: Optional[threading.Event] = None,
    ) -> SweepProgress:
        """
        Execute iterative sweeps to reorder states based on property continuity.
        Supports cancellation and per-iteration progress callbacks.
        """
        self.halt = False
        is_back = (direction == "backward")
        is_alternate = (direction == "alternate")

        if sort_last is not None:
            self.sortLast = sort_last

        total_iterations = num_sweeps * (2 if is_alternate else 1)
        current_iter = 0
        del_max = float("inf")
        converged = False

        for sweep_idx in range(1, total_iterations + 1):
            if (cancel_token and cancel_token.is_set()) or self.halt:
                self.halt = True
                status = SweepProgress(
                    current_sweep=sweep_idx - 1,
                    total_sweeps=total_iterations,
                    del_max=del_max,
                    converged=False,
                    aborted=True,
                    message="Sweep aborted by user.",
                )
                if progress_callback:
                    progress_callback(status)
                return status

            last_e, last_p = self.prepareSweep()
            current_iter += 1

            # Sweep each state
            for state in range(self.numStates):
                if (cancel_token and cancel_token.is_set()) or self.halt:
                    break
                self.sweepState(state, current_iter, backward=is_back)

            del_max = float(self.analyzeSweep(last_e, last_p))

            if is_alternate:
                is_back = not is_back
                self.sortLast = is_back
                self.sortEnergies()

            if del_max < self.tol:
                converged = True
                self.sortEnergies()
                break

            if progress_callback:
                progress_callback(
                    SweepProgress(
                        current_sweep=sweep_idx,
                        total_sweeps=total_iterations,
                        del_max=del_max,
                        converged=converged,
                        aborted=False,
                        message=f"Sweep {sweep_idx}/{total_iterations} completed. Δmax = {del_max:.4e}",
                    )
                )

        if is_alternate and not converged:
            self.sortLast = self.sortLast
            self.sortEnergies()

        return SweepProgress(
            current_sweep=current_iter,
            total_sweeps=total_iterations,
            del_max=del_max,
            converged=converged,
            aborted=False,
            message="Converged" if converged else "Max sweeps reached.",
        )
