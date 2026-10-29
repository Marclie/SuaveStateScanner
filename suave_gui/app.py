"""
Main Dash Application Factory and Callback Orchestrator for SuaveStateScanner.
Provides single-node, real-time interactive plotting directly synchronized with backend data structures.
"""

from __future__ import annotations
import base64
import io
import json
import os
import threading
from typing import Any, Dict, Optional, Tuple

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_mantine_components as dmc
import numpy as np

from suave_state_scanner import SuaveStateScanner
from suave_gui.core.dataset import StateDataset
from suave_gui.core.parser import DataParserService
from suave_gui.components.plot_canvas import PlotCanvasBuilder
from suave_gui.components.file_modal import create_file_modal
from suave_gui.components.sidebar_controls import create_sidebar_controls
from suave_gui.components.status_bar import create_status_bar
from suave_gui.state.session_store import SessionRegistry, SessionContext


def create_app(
    suave_instance: Optional[SuaveStateScanner] = None,
    default_dataset_path: Optional[str] = "input.csv",
    default_num_states: int = 9,
    default_config_path: Optional[str] = "suave.json",
) -> dash.Dash:
    """Creates and configures the single-node Dash application instance."""

    session_registry = SessionRegistry()
    default_session = session_registry.get_or_create("default_user")

    # 1. Direct Backend Binding: Bind to provided SuaveStateScanner instance
    if suave_instance is not None:
        default_session.set_scanner(suave_instance)
    elif default_dataset_path and os.path.exists(default_dataset_path):
        try:
            # Instantiate the backend SuaveStateScanner directly
            scanner = SuaveStateScanner(
                default_dataset_path, "output.csv", default_num_states, default_config_path
            )
            default_session.set_scanner(scanner)
        except Exception as e:
            print(f"[SuaveGUI] Warning: Could not initialize backend from '{default_dataset_path}': {e}")

    app = dash.Dash(
        __name__,
        title="Suave State Scanner",
        suppress_callback_exceptions=True,
    )

    suave = default_session.suave
    num_states = suave.numStates if suave else default_num_states
    num_props = suave.numProps if suave else 8
    max_pnt_idx = (suave.numPoints - 1) if suave else 100

    initial_fig = (
        PlotCanvasBuilder.build_figure(default_session.dataset)
        if default_session.dataset
        else PlotCanvasBuilder.build_empty_figure()
    )

    app.layout = dmc.MantineProvider(
        forceColorScheme="dark",
        children=dmc.Box(
            style={
                "height": "100vh",
                "display": "flex",
                "flexDirection": "column",
                "backgroundColor": "#09090B",
                "color": "#F4F4F5",
                "overflow": "hidden",
            },
            children=[
                # Hidden Stores & Interval for live updates during sweeps
                dcc.Store(id="session-id", data="default_user"),
                dcc.Store(id="active-dataset-meta"),
                dcc.Interval(id="poll-interval", interval=200, disabled=True),
                dcc.Download(id="download-curves-csv"),
                dcc.Download(id="download-checkpoint-csv"),

                # Data Loader Modal
                create_file_modal(),

                # Header Navigation
                dmc.Box(
                    style={
                        "height": "52px",
                        "padding": "0 16px",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                        "backgroundColor": "#18181B",
                        "borderBottom": "1px solid #27272A",
                    },
                    children=[
                        dmc.Group(gap="sm", children=[
                            dmc.Text("SuaveStateScanner", fw=700, size="lg"),
                            dmc.Badge("Single-Node", color="blue", variant="filled", size="sm"),
                        ]),
                        dmc.Group(gap="xs", children=[
                            dmc.Button(
                                "Load Dataset",
                                id="btn-open-load-modal",
                                variant="light",
                                color="blue",
                                size="xs",
                            ),
                            dmc.Button(
                                "Undo",
                                id="btn-undo",
                                variant="subtle",
                                color="gray",
                                size="xs",
                            ),
                            dmc.Button(
                                "Redo",
                                id="btn-redo",
                                variant="subtle",
                                color="gray",
                                size="xs",
                            ),
                            dmc.Button(
                                "Save Curves CSV",
                                id="btn-save-curves",
                                color="green",
                                size="xs",
                            ),
                            dmc.Button(
                                "Save Checkpoint CSV",
                                id="btn-save-checkpoint",
                                color="teal",
                                size="xs",
                            ),
                        ]),
                    ],
                ),

                # Main Content: Plot Viewport (Left) + Control Drawer (Right)
                dmc.Box(
                    style={"flex": 1, "display": "flex", "overflow": "hidden"},
                    children=[
                        # Left Viewport
                        dmc.Box(
                            style={
                                "flex": 1,
                                "padding": "12px",
                                "display": "flex",
                                "flexDirection": "column",
                                "overflow": "hidden",
                            },
                            children=[
                                dcc.Graph(
                                    id="main-curve-plot",
                                    figure=initial_fig,
                                    style={"flex": 1, "width": "100%", "minHeight": "400px"},
                                    config={
                                        "displayModeBar": True,
                                        "scrollZoom": True,
                                        "toImageButtonOptions": {"format": "svg", "filename": "suave_plot"},
                                    },
                                ),
                                # Sliding window range selectors
                                dmc.Paper(
                                    p="xs",
                                    radius="sm",
                                    withBorder=True,
                                    style={"backgroundColor": "#18181B", "marginTop": "8px"},
                                    children=[
                                        dmc.Group([
                                            dmc.Box(style={"flex": 1}, children=[
                                                dmc.Text("Reaction Coordinate Window (Point Range)", size="xs", c="dimmed"),
                                                dcc.RangeSlider(
                                                    id="slider-point-range",
                                                    min=0,
                                                    max=max_pnt_idx,
                                                    value=[0, max_pnt_idx],
                                                    allowCross=False,
                                                    tooltip={"always_visible": False, "placement": "top"},
                                                ),
                                            ]),
                                            dmc.Box(style={"flex": 1}, children=[
                                                dmc.Text("State Index Filter Window", size="xs", c="dimmed"),
                                                dcc.RangeSlider(
                                                    id="slider-state-range",
                                                    min=0,
                                                    max=num_states,
                                                    value=[0, num_states],
                                                    allowCross=False,
                                                    tooltip={"always_visible": False, "placement": "top"},
                                                ),
                                            ]),
                                        ]),
                                    ],
                                ),
                            ],
                        ),

                        # Right Controls Drawer: Instantiated with full properties and states
                        create_sidebar_controls(num_states=num_states, num_props=num_props),
                    ],
                ),

                # Persistent Status Bar Footer
                create_status_bar(),
            ],
        ),
    )

    _register_callbacks(app, session_registry)
    return app


def _register_callbacks(app: dash.Dash, session_registry: SessionRegistry) -> None:
    """Registers reactive callback graph for single-node execution."""

    # 1. Toggle File Modal
    @app.callback(
        Output("file-load-modal", "opened"),
        [
            Input("btn-open-load-modal", "n_clicks"),
            Input("btn-modal-cancel", "n_clicks"),
        ],
        State("file-load-modal", "opened"),
        prevent_initial_call=True,
    )
    def toggle_file_modal(open_clicks, cancel_clicks, is_opened):
        return not is_opened

    # 2. Inspect File on Upload / Path input
    @app.callback(
        Output("modal-inspection-alert", "children"),
        [
            Input("modal-file-upload", "contents"),
            Input("modal-input-filepath", "value"),
        ],
        State("modal-file-upload", "filename"),
        prevent_initial_call=True,
    )
    def inspect_input_data(upload_contents, file_path, upload_filename):
        target = None
        if upload_contents:
            try:
                _, content_string = upload_contents.split(",")
                decoded = base64.b64decode(content_string)
                target = io.StringIO(decoded.decode("utf-8", errors="replace"))
            except Exception as e:
                return dmc.Alert(f"Error decoding upload: {e}", color="red")
        elif file_path and os.path.exists(file_path):
            target = file_path

        if not target:
            return None

        try:
            info = DataParserService.inspect_file(target)
            return dmc.Alert(
                f"File Inspected: {info['total_rows']} total rows, {info['num_columns']} columns "
                f"({info['num_properties']} properties). Plausible state counts: {info['plausible_states']}",
                color="blue",
            )
        except Exception as e:
            return dmc.Alert(f"Inspection error: {e}", color="yellow")

    # 3. Load Dataset Confirm (Instantiates backend SuaveStateScanner)
    @app.callback(
        [
            Output("file-load-modal", "opened", allow_duplicate=True),
            Output("main-curve-plot", "figure", allow_duplicate=True),
            Output("selector-property", "data"),
            Output("selector-property", "value"),
            Output("selector-highlight-state", "data"),
            Output("selector-highlight-state", "value"),
            Output("slider-point-range", "max"),
            Output("slider-point-range", "value"),
            Output("slider-state-range", "max"),
            Output("slider-state-range", "value"),
            Output("text-status-message", "children", allow_duplicate=True),
            Output("text-dataset-summary", "children"),
            Output("active-dataset-meta", "data"),
        ],
        Input("btn-modal-confirm", "n_clicks"),
        [
            State("modal-file-upload", "contents"),
            State("modal-file-upload", "filename"),
            State("modal-input-filepath", "value"),
            State("modal-num-states", "value"),
            State("modal-config-path", "value"),
            State("session-id", "data"),
        ],
        prevent_initial_call=True,
    )
    def load_dataset_callback(
        n_clicks, upload_contents, upload_filename, filepath, num_states, config_path, session_id
    ):
        if not n_clicks:
            return (no_update,) * 13

        session = session_registry.get_or_create(session_id)
        temp_input_path = filepath
        fname = "uploaded_dataset.csv"

        if upload_contents:
            _, content_string = upload_contents.split(",")
            decoded = base64.b64decode(content_string)
            temp_input_path = "temp_uploaded.csv"
            with open(temp_input_path, "wb") as f:
                f.write(decoded)
            fname = upload_filename or "uploaded.csv"
        elif filepath and os.path.exists(filepath):
            temp_input_path = filepath
            fname = os.path.basename(filepath)
        else:
            return (
                no_update, no_update, no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                "Error: No valid file specified or file does not exist.",
                no_update, no_update,
            )

        try:
            num_states_int = int(num_states) if num_states else 9
            # Direct backend instantiation: create SuaveStateScanner
            scanner = SuaveStateScanner(
                infile=temp_input_path,
                outfile="output.csv",
                numStates=num_states_int,
                configPath=config_path if (config_path and os.path.exists(config_path)) else None,
            )
            session.set_scanner(scanner)
            ds = session.dataset

            fig = PlotCanvasBuilder.build_figure(ds, property_idx=0)

            # Full list of all properties
            prop_data = [{"label": "Energy", "value": "0"}] + [
                {"label": f"Property {i}", "value": str(i)}
                for i in range(1, ds.num_properties + 1)
            ]

            # Complete list of all states
            state_data = [{"label": "All States", "value": "all"}] + [
                {"label": f"State {i}", "value": str(i)} for i in range(ds.num_states)
            ]

            status_msg = f"Loaded {fname} ({ds.num_states} states, {ds.num_points} points)."
            summary = f"{fname} | {ds.num_states} States | {ds.num_points} Points"
            meta = {"filename": fname, "num_states": ds.num_states, "num_points": ds.num_points}

            return (
                False,  # close modal
                fig,
                prop_data,
                "0",
                state_data,
                "all",
                ds.num_points - 1,
                [0, ds.num_points - 1],
                ds.num_states,
                [0, ds.num_states],
                status_msg,
                summary,
                meta,
            )
        except Exception as e:
            return (
                True,  # keep modal open
                no_update, no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                f"Failed to load dataset: {e}",
                no_update, no_update,
            )

    # 4. Property or State Selection Changes
    @app.callback(
        Output("main-curve-plot", "figure", allow_duplicate=True),
        [
            Input("selector-property", "value"),
            Input("selector-highlight-state", "value"),
            Input("slider-point-range", "value"),
            Input("slider-state-range", "value"),
        ],
        State("session-id", "data"),
        prevent_initial_call=True,
    )
    def update_plot_view(prop_val, highlight_val, point_range, state_range, session_id):
        session = session_registry.get_or_create(session_id)
        if not session.dataset:
            return no_update

        prop_idx = int(prop_val) if prop_val else 0
        hl_state = int(highlight_val) if (highlight_val and highlight_val != "all") else None
        p_bounds = (int(point_range[0]), int(point_range[1]) + 1) if point_range else None
        s_bounds = (int(state_range[0]), int(state_range[1])) if state_range else None

        return PlotCanvasBuilder.build_figure(
            session.dataset,
            property_idx=prop_idx,
            highlight_state=hl_state,
            point_bounds=p_bounds,
            state_bounds=s_bounds,
        )

    # 5. Manual State Interventions: Sliding-window Shuffle, Swap, Reset, Remove State, Undo, Redo
    @app.callback(
        [
            Output("main-curve-plot", "figure", allow_duplicate=True),
            Output("text-status-message", "children", allow_duplicate=True),
        ],
        [
            Input("btn-shuffle-values", "n_clicks"),
            Input("btn-apply-swap", "n_clicks"),
            Input("btn-reset-initial", "n_clicks"),
            Input("btn-remove-state", "n_clicks"),
            Input("btn-undo", "n_clicks"),
            Input("btn-redo", "n_clicks"),
        ],
        [
            State("input-swap-state-1", "value"),
            State("input-swap-state-2", "value"),
            State("input-remove-state", "value"),
            State("slider-point-range", "value"),
            State("selector-property", "value"),
            State("selector-highlight-state", "value"),
            State("session-id", "data"),
        ],
        prevent_initial_call=True,
    )
    def handle_manual_actions(
        shuffle_clicks, swap_clicks, reset_clicks, remove_clicks, undo_clicks, redo_clicks,
        swap_s1, swap_s2, remove_s, point_range, prop_val, highlight_val, session_id
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        session = session_registry.get_or_create(session_id)
        if not session.dataset or not session.suave:
            return no_update, "No dataset loaded."

        prop_idx = int(prop_val) if prop_val else 0
        hl_state = int(highlight_val) if (highlight_val and highlight_val != "all") else None
        p_start = max(0, int(point_range[0])) if point_range else 0
        p_end = min(session.suave.numPoints, int(point_range[1]) + 1) if point_range else session.suave.numPoints

        # Record undo snapshot before mutation
        prev_snap = session.dataset.create_snapshot("Before " + trigger_id)
        msg = ""

        # Requirement 3: Shuffle ONLY within selected sliding-window region
        if trigger_id == "btn-shuffle-values":
            session.dataset.shuffle_points(p_start, p_end)
            session.history.push(prev_snap, session.dataset.create_snapshot(f"Shuffled Points [{p_start}:{p_end}]"))
            msg = f"Shuffled states for points {p_start} to {p_end - 1} within the sliding window."

        elif trigger_id == "btn-apply-swap":
            s1 = int(swap_s1) if swap_s1 is not None else 0
            s2 = int(swap_s2) if swap_s2 is not None else 1
            session.dataset.apply_swap(s1, s2, p_start, p_end)
            session.history.push(prev_snap, session.dataset.create_snapshot(f"Swapped {s1} and {s2}"))
            msg = f"Swapped State {s1} and State {s2} across points [{p_start}, {p_end - 1}]."

        elif trigger_id == "btn-reset-initial":
            if session.history.current:
                session.dataset.restore_snapshot(session.history.current)
                session.history.clear()
                session.history.reset(session.dataset.create_snapshot("Reset to Initial"))
                msg = "Reset dataset to initial state."

        elif trigger_id == "btn-remove-state":
            rs = int(remove_s) if remove_s is not None else 0
            session.dataset.remove_state_range(rs, p_start, p_end)
            session.history.push(prev_snap, session.dataset.create_snapshot(f"Removed State {rs}"))
            msg = f"Removed State {rs} data across points [{p_start}, {p_end - 1}]."

        elif trigger_id == "btn-undo":
            snap = session.history.undo(session.dataset.create_snapshot("Live"))
            if snap:
                session.dataset.restore_snapshot(snap)
                msg = f"Undo applied: {snap.description}"
            else:
                msg = "Undo buffer empty."

        elif trigger_id == "btn-redo":
            snap = session.history.redo(session.dataset.create_snapshot("Live"))
            if snap:
                session.dataset.restore_snapshot(snap)
                msg = f"Redo applied: {snap.description}"
            else:
                msg = "Redo buffer empty."

        fig = PlotCanvasBuilder.build_figure(session.dataset, property_idx=prop_idx, highlight_state=hl_state)
        return fig, msg

    # 6. Requirement 4: Trigger Forward, Backward, and Alternate Sweeps
    @app.callback(
        [
            Output("poll-interval", "disabled"),
            Output("badge-engine-status", "children"),
            Output("badge-engine-status", "color"),
            Output("text-status-message", "children", allow_duplicate=True),
        ],
        [
            Input("btn-sweep-forward", "n_clicks"),
            Input("btn-sweep-backward", "n_clicks"),
            Input("btn-sweep-alternate", "n_clicks"),
        ],
        [
            State("input-num-sweeps", "value"),
            State("input-stencil-width", "value"),
            State("input-deriv-order", "value"),
            State("switch-alternate", "checked"),
            State("switch-sort-last", "checked"),
            State("switch-interpolate", "checked"),
            State("switch-redundant", "checked"),
            State("input-max-pan", "value"),
            State("input-energy-width", "value"),
            State("slider-point-range", "value"),
            State("slider-state-range", "value"),
            State("session-id", "data"),
        ],
        prevent_initial_call=True,
    )
    def start_sweep_callback(
        fwd_clicks, bwd_clicks, alt_clicks,
        num_sweeps, width, order, alternate, sort_last, interpolate, redundant,
        max_pan, e_width, point_range, state_range, session_id
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        session = session_registry.get_or_create(session_id)
        if not session.dataset or not session.suave:
            return no_update, no_update, no_update, "No dataset loaded to sweep."

        if session.sweep_telemetry["running"]:
            return no_update, no_update, no_update, "Sweep calculation already in progress."

        direction = "forward"
        if trigger_id == "btn-sweep-backward":
            direction = "backward"
        elif trigger_id == "btn-sweep-alternate" or alternate:
            direction = "alternate"

        p_bounds = [int(point_range[0]), int(point_range[1]) + 1] if point_range else [0, session.suave.numPoints]
        s_bounds = [int(state_range[0]), int(state_range[1])] if state_range else [0, session.suave.numStates]

        # Directly configure the backend SuaveStateScanner
        suave = session.suave
        suave.width = int(width) if width else 5
        suave.orders = [int(order)] if order else [1]
        suave.sortLast = bool(sort_last)
        suave.interpolate = bool(interpolate)
        suave.redundantSwaps = bool(redundant)
        suave.maxPan = int(max_pan) if max_pan else None
        suave.energyWidth = float(e_width) if e_width else None
        suave.pntBounds = p_bounds
        suave.stateBounds = s_bounds
        suave.halt = False

        # Save snapshot for undo
        session.history.push(session.dataset.create_snapshot(f"Before {direction} sweep"))

        sweeps_count = int(num_sweeps) if num_sweeps else 1
        is_back = (direction == "backward")
        is_alt = (direction == "alternate")
        total_iterations = sweeps_count * (2 if is_alt else 1)

        with session.lock:
            session.sweep_telemetry = {
                "running": True,
                "updated": False,
                "done": False,
                "state": 0,
                "sweep": 0,
                "total": total_iterations,
                "delMax": float("inf"),
                "direction": direction,
                "message": f"Starting {direction} sweep...",
            }

        def sweep_worker():
            """Background worker executing sweeps directly on backend SuaveStateScanner."""
            nonlocal is_back
            current_sweep = 0

            for i in range(total_iterations):
                if suave.halt:
                    break
                lastEvals, lastPvals = suave.prepareSweep()
                current_sweep += 1

                for state in range(suave.numStates):
                    if suave.halt:
                        break
                    suave.sweepState(state, current_sweep, backward=is_back)
                    # Notify UI that points have been updated after each state sweep
                    with session.lock:
                        session.sweep_telemetry["state"] = state
                        session.sweep_telemetry["sweep"] = current_sweep
                        session.sweep_telemetry["updated"] = True

                delMax = suave.analyzeSweep(lastEvals, lastPvals)
                with session.lock:
                    session.sweep_telemetry["delMax"] = float(delMax)
                    session.sweep_telemetry["updated"] = True

                if is_alt:
                    is_back = not is_back
                    suave.sortLast = is_back
                    suave.sortEnergies()
                    with session.lock:
                        session.sweep_telemetry["updated"] = True

                if delMax < suave.tol:
                    suave.sortEnergies()
                    with session.lock:
                        session.sweep_telemetry["updated"] = True
                    break

            with session.lock:
                session.sweep_telemetry["running"] = False
                session.sweep_telemetry["done"] = True

        worker_thread = threading.Thread(target=sweep_worker, name="SingleNodeSweepWorker", daemon=True)
        session.sweep_thread = worker_thread
        worker_thread.start()

        return False, "Sweeping...", "orange", f"Running {direction} sweep ({total_iterations} iterations)..."

    # 7. Abort Sweep
    @app.callback(
        Output("text-status-message", "children", allow_duplicate=True),
        Input("btn-abort-sweep", "n_clicks"),
        State("session-id", "data"),
        prevent_initial_call=True,
    )
    def abort_sweep_callback(n_clicks, session_id):
        if not n_clicks:
            return no_update
        session = session_registry.get_or_create(session_id)
        session.abort_sweep()
        return "Calculation aborted by user."

    # 8. Requirement 4: Poll Interval updates plot continuously in real time during sweep
    @app.callback(
        [
            Output("poll-interval", "disabled", allow_duplicate=True),
            Output("badge-engine-status", "children", allow_duplicate=True),
            Output("badge-engine-status", "color", allow_duplicate=True),
            Output("main-curve-plot", "figure", allow_duplicate=True),
            Output("text-status-message", "children", allow_duplicate=True),
            Output("text-convergence-telemetry", "children"),
        ],
        Input("poll-interval", "n_intervals"),
        [
            State("selector-property", "value"),
            State("selector-highlight-state", "value"),
            State("session-id", "data"),
        ],
        prevent_initial_call=True,
    )
    def poll_sweep_task(n_intervals, prop_val, highlight_val, session_id):
        session = session_registry.get_or_create(session_id)
        telem = session.sweep_telemetry

        if not telem["running"] and not telem["done"]:
            return True, "Ready", "green", no_update, no_update, no_update

        prop_idx = int(prop_val) if prop_val else 0
        hl_state = int(highlight_val) if (highlight_val and highlight_val != "all") else None

        # Live visual tracking: highlight active state and render vertical red scan line
        active_state = getattr(session.suave, "currentState", None)
        if active_state is None:
            active_state = telem.get("state", 0)
        current_pnt = getattr(session.suave, "currentPoint", None)

        effective_hl = hl_state if hl_state is not None else (active_state if telem["running"] else None)
        effective_scan_pnt = current_pnt if telem["running"] else None

        # Build figure directly from backend data with scan indicator and active state highlight
        fig = PlotCanvasBuilder.build_figure(
            session.dataset,
            property_idx=prop_idx,
            highlight_state=effective_hl,
            scan_point=effective_scan_pnt,
        )

        del_max_val = telem["delMax"]
        del_str = f"Δmax: {del_max_val:.4e}" if (del_max_val < float("inf")) else "Δmax: in progress"

        if telem["done"]:
            # Sweep completed or aborted - clear scan indicator
            with session.lock:
                telem["done"] = False
                telem["running"] = False
            final_msg = (
                f"Sweep completed ({telem['sweep']}/{telem['total']}). {del_str}"
                if not getattr(session.suave, "halt", False)
                else "Sweep aborted."
            )
            # Final clean figure without scan line
            fig = PlotCanvasBuilder.build_figure(session.dataset, property_idx=prop_idx, highlight_state=hl_state)
            session.history.push(session.dataset.create_snapshot("After sweep completion"))
            return True, "Ready", "green", fig, final_msg, del_str

        # Ongoing sweep: update plot continuously
        pnt_str = f", Point {current_pnt}/{session.suave.numPoints}" if current_pnt is not None else ""
        status_txt = (
            f"Sweeping ({telem['direction']}): Iteration {telem['sweep']}/{telem['total']}, "
            f"State {active_state}/{session.suave.numStates}{pnt_str} | Points updated live"
        )
        return False, "Sweeping...", "orange", fig, status_txt, del_str

    # 9. CSV Downloads
    @app.callback(
        Output("download-curves-csv", "data"),
        Input("btn-save-curves", "n_clicks"),
        [
            State("selector-property", "value"),
            State("session-id", "data"),
        ],
        prevent_initial_call=True,
    )
    def download_curves_csv(n_clicks, prop_val, session_id):
        if not n_clicks:
            return no_update
        session = session_registry.get_or_create(session_id)
        if not session.dataset:
            return no_update
        prop_idx = int(prop_val) if prop_val else 0
        csv_string = DataParserService.export_curves_csv(session.dataset, property_idx=prop_idx)
        return dict(content=csv_string, filename="curves_output.csv")

    @app.callback(
        Output("download-checkpoint-csv", "data"),
        Input("btn-save-checkpoint", "n_clicks"),
        State("session-id", "data"),
        prevent_initial_call=True,
    )
    def download_checkpoint_csv(n_clicks, session_id):
        if not n_clicks:
            return no_update
        session = session_registry.get_or_create(session_id)
        if not session.dataset:
            return no_update
        csv_string = DataParserService.export_checkpoint_csv(session.dataset)
        return dict(content=csv_string, filename="checkpoint.csv")
