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
from suave_gui.components.save_modal import create_save_modal
from suave_gui.components.sidebar_controls import create_sidebar_controls
from suave_gui.components.status_bar import create_status_bar
from suave_gui.components.tooltip_helper import create_tooltip
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
                # Curve Sampling & Save Modal
                create_save_modal(),

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
                            create_tooltip(
                                dmc.Text("SuaveStateScanner", fw=700, size="lg"),
                                title="SuaveStateScanner",
                                description=(
                                    "A specialized tool for excited electronic state labeling, continuity enforcement, "
                                    "and avoided-crossing resolution along reaction coordinates using finite differences."
                                ),
                                position="bottom",
                            ),
                            create_tooltip(
                                dmc.Badge("Single-Node", color="blue", variant="filled", size="sm"),
                                title="Single-Node Architecture",
                                description="Executes JIT-compiled Numba finite difference sweeps directly in local background threads without external servers.",
                                position="bottom",
                            ),
                        ]),
                        dmc.Group(gap="xs", children=[
                            create_tooltip(
                                dmc.Button(
                                    "Load Dataset",
                                    id="btn-open-load-modal",
                                    variant="light",
                                    color="blue",
                                    size="xs",
                                ),
                                title="Load Dataset Matrix",
                                description="Upload a CSV matrix or select a server-local file, set the number of states, and load the dataset into the workspace.",
                                config_name="infile / numStates",
                                position="bottom",
                            ),
                            create_tooltip(
                                dmc.Button(
                                    "Undo",
                                    id="btn-undo",
                                    variant="subtle",
                                    color="gray",
                                    size="xs",
                                ),
                                title="Undo Last Action",
                                description="Reverts the most recent manual state swap, point shuffle, state removal, or automated reordering sweep.",
                                position="bottom",
                            ),
                            create_tooltip(
                                dmc.Button(
                                    "Redo",
                                    id="btn-redo",
                                    variant="subtle",
                                    color="gray",
                                    size="xs",
                                ),
                                title="Redo Next Action",
                                description="Re-applies the previously undone action from the history stack.",
                                position="bottom",
                            ),
                            create_tooltip(
                                dmc.Button(
                                    "Save Curves CSV",
                                    id="btn-save-curves",
                                    color="green",
                                    size="xs",
                                ),
                                title="Export Curves CSV",
                                description="Export reordered state energy and property curves as CSV, with optional point sampling or spline interpolation.",
                                config_name="outfile / keepInterp",
                                position="bottom",
                            ),
                            create_tooltip(
                                dmc.Button(
                                    "Save Checkpoint CSV",
                                    id="btn-save-checkpoint",
                                    color="teal",
                                    size="xs",
                                ),
                                title="Export Checkpoint CSV",
                                description="Export the full state matrix to checkpoint.csv to allow resuming scans or external processing.",
                                position="bottom",
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
                                # Active Inspection & Range Presets Tray - Single Row Integrated Layout
                                dmc.Paper(
                                    p="xs",
                                    radius="sm",
                                    withBorder=True,
                                    style={"backgroundColor": "#18181B", "marginTop": "6px", "width": "100%"},
                                    children=[
                                        dmc.SimpleGrid(
                                            cols=2,
                                            spacing="xl",
                                            style={"width": "100%"},
                                            children=[
                                                # Column 1: Reaction Coordinate Section (Slider on Left, Presets on Right)
                                                dmc.Box(
                                                    style={
                                                        "width": "100%",
                                                        "display": "flex",
                                                        "flexDirection": "row",
                                                        "alignItems": "flex-end",
                                                        "gap": "12px",
                                                    },
                                                    children=[
                                                        # Slider Container
                                                        create_tooltip(
                                                            dmc.Box(
                                                                className="full-width-slider-box",
                                                                style={"flex": 1, "minWidth": "200px"},
                                                                children=[
                                                                    dmc.Group(
                                                                        justify="space-between",
                                                                        align="center",
                                                                        style={"marginBottom": "2px", "width": "100%"},
                                                                        children=[
                                                                            dmc.Text(
                                                                                "Reaction Coordinate Window",
                                                                                size="xs",
                                                                                fw=700,
                                                                                c="#FAFAFA",
                                                                            ),
                                                                            dmc.Badge(
                                                                                id="badge-coord-readout",
                                                                                children="Points: 0 – 100",
                                                                                size="xs",
                                                                                color="blue",
                                                                                variant="light",
                                                                            ),
                                                                        ],
                                                                    ),
                                                                    dcc.RangeSlider(
                                                                        id="slider-point-range",
                                                                        min=0,
                                                                        max=max_pnt_idx,
                                                                        value=[0, max_pnt_idx],
                                                                        allowCross=False,
                                                                        tooltip={"always_visible": False, "placement": "top"},
                                                                        className="custom-range-slider",
                                                                    ),
                                                                ],
                                                            ),
                                                            title="Coordinate Point Range (pntBounds)",
                                                            description=(
                                                                "Restricts manual state manipulations (swaps, shuffles, removals) and automated sweeps "
                                                                "to this subset of reaction coordinate indices [xmin, xmax]. Segmenting complex trajectories "
                                                                "into smaller batches makes difficult curve crossings much easier to converge."
                                                            ),
                                                            config_name="pntBounds",
                                                            default_val="Full coordinate range",
                                                            style={"flex": 1},
                                                            position="top",
                                                        ),
                                                        # Coordinate Presets (Right next to the slider)
                                                        dmc.Group(
                                                            gap="4px",
                                                            wrap="nowrap",
                                                            align="center",
                                                            style={"flexShrink": 0, "paddingBottom": "4px"},
                                                            children=[
                                                                create_tooltip(
                                                                    dmc.Button(
                                                                        "Full",
                                                                        id="btn-preset-full",
                                                                        size="xs",
                                                                        variant="light",
                                                                        color="blue",
                                                                        className="preset-button",
                                                                    ),
                                                                    title="Full Trajectory Preset",
                                                                    description="Expands the active reaction coordinate window to span all points (0 to max).",
                                                                    position="top",
                                                                ),
                                                                create_tooltip(
                                                                    dmc.Button(
                                                                        "Left",
                                                                        id="btn-preset-left",
                                                                        size="xs",
                                                                        variant="subtle",
                                                                        color="gray",
                                                                        className="preset-button",
                                                                    ),
                                                                    title="Left Half Window Preset",
                                                                    description="Focuses the coordinate window on the first half of the trajectory [0, 50%].",
                                                                    position="top",
                                                                ),
                                                                create_tooltip(
                                                                    dmc.Button(
                                                                        "Right",
                                                                        id="btn-preset-right",
                                                                        size="xs",
                                                                        variant="subtle",
                                                                        color="gray",
                                                                        className="preset-button",
                                                                    ),
                                                                    title="Right Half Window Preset",
                                                                    description="Focuses the coordinate window on the second half of the trajectory [50%, 100%].",
                                                                    position="top",
                                                                ),
                                                                create_tooltip(
                                                                    dmc.Button(
                                                                        "Avoided Crossing",
                                                                        id="btn-preset-crossing",
                                                                        size="xs",
                                                                        variant="light",
                                                                        color="grape",
                                                                        className="preset-button",
                                                                    ),
                                                                    title="Auto-Focus Avoided Crossing",
                                                                    description="Detects the narrowest energy gap between adjacent electronic states and centers the window right on the avoided crossing.",
                                                                    position="top",
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),

                                                # Column 2: State Filter Section (Slider on Left, Presets on Right)
                                                dmc.Box(
                                                    style={
                                                        "width": "100%",
                                                        "display": "flex",
                                                        "flexDirection": "row",
                                                        "alignItems": "flex-end",
                                                        "gap": "12px",
                                                    },
                                                    children=[
                                                        # Slider Container
                                                        create_tooltip(
                                                            dmc.Box(
                                                                className="full-width-slider-box",
                                                                style={"flex": 1, "minWidth": "160px"},
                                                                children=[
                                                                    dmc.Group(
                                                                        justify="space-between",
                                                                        align="center",
                                                                        style={"marginBottom": "2px", "width": "100%"},
                                                                        children=[
                                                                            dmc.Text(
                                                                                "State Index Filter Window",
                                                                                size="xs",
                                                                                fw=700,
                                                                                c="#FAFAFA",
                                                                            ),
                                                                            dmc.Badge(
                                                                                id="badge-state-readout",
                                                                                children=f"Active: S0 – S{num_states - 1} ({num_states} States)",
                                                                                size="xs",
                                                                                color="violet",
                                                                                variant="light",
                                                                            ),
                                                                        ],
                                                                    ),
                                                                    dcc.RangeSlider(
                                                                        id="slider-state-range",
                                                                        min=0,
                                                                        max=num_states,
                                                                        value=[0, num_states],
                                                                        allowCross=False,
                                                                        tooltip={"always_visible": False, "placement": "top"},
                                                                        className="custom-range-slider",
                                                                    ),
                                                                ],
                                                            ),
                                                            title="State Index Filter Range (stateBounds)",
                                                            description=(
                                                                "Specifies inclusive lower and upper index bounds [statemin, statemax] identifying electronic "
                                                                "states to reorder. Allows isolating specific manifolds (e.g. low-lying valence states) while "
                                                                "leaving higher-lying Rydberg states untouched."
                                                            ),
                                                            config_name="stateBounds",
                                                            default_val="All electronic states",
                                                            style={"flex": 1},
                                                            position="top",
                                                        ),
                                                        # State Presets (Right next to the slider)
                                                        dmc.Group(
                                                            gap="4px",
                                                            wrap="nowrap",
                                                            align="center",
                                                            style={"flexShrink": 0, "paddingBottom": "4px"},
                                                            children=[
                                                                create_tooltip(
                                                                    dmc.Button(
                                                                        "All States",
                                                                        id="btn-preset-all-states",
                                                                        size="xs",
                                                                        variant="light",
                                                                        color="blue",
                                                                        className="preset-button",
                                                                    ),
                                                                    title="All States Preset",
                                                                    description="Includes all electronic states (0 to numStates) in optimization and filtering.",
                                                                    position="top",
                                                                ),
                                                                create_tooltip(
                                                                    dmc.Button(
                                                                        "Low States (S0–S3)",
                                                                        id="btn-preset-low-states",
                                                                        size="xs",
                                                                        variant="subtle",
                                                                        color="gray",
                                                                        className="preset-button",
                                                                    ),
                                                                    title="Low-Lying States Preset",
                                                                    description="Restricts analysis to low-lying electronic states (states 0 to 4), excluding higher Rydberg states.",
                                                                    position="top",
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
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
            Output("select-prop-list", "data"),
            Output("select-prop-list", "value"),
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
            return (no_update,) * 15

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
                no_update, no_update, no_update, no_update,
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

            prop_filter_data = [{"label": f"Property {i}", "value": str(i)} for i in range(ds.num_properties)]
            prop_filter_default = [str(i) for i in range(ds.num_properties)]

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
                prop_filter_data,
                prop_filter_default,
            )
        except Exception as e:
            return (
                True,  # keep modal open
                no_update, no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                f"Failed to load dataset: {e}",
                no_update, no_update, no_update, no_update,
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
            State("input-future-pnts", "value"),
            State("switch-alternate", "checked"),
            State("switch-sort-last", "checked"),
            State("switch-interpolate", "checked"),
            State("switch-redundant", "checked"),
            State("switch-make-pos", "checked"),
            State("switch-normalize", "checked"),
            State("input-max-pan", "value"),
            State("input-energy-width", "value"),
            State("input-max-state-repeat", "value"),
            State("select-prop-list", "value"),
            State("slider-point-range", "value"),
            State("slider-state-range", "value"),
            State("session-id", "data"),
        ],
        prevent_initial_call=True,
    )
    def start_sweep_callback(
        fwd_clicks, bwd_clicks, alt_clicks,
        num_sweeps, width, order, future_pnts, alternate, sort_last, interpolate, redundant,
        make_pos, normalize, max_pan, e_width, max_state_repeat, prop_list,
        point_range, state_range, session_id
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
        if isinstance(order, list):
            suave.orders = [int(o) for o in order] if order else [1]
        elif order is not None:
            suave.orders = [int(order)]
        else:
            suave.orders = [1]
        suave.futurePnts = int(future_pnts) if future_pnts is not None else 0
        suave.sortLast = bool(sort_last)
        suave.interpolate = bool(interpolate)
        suave.redundantSwaps = bool(redundant)
        suave.makePos = bool(make_pos)
        suave.normalize = bool(normalize)
        suave.maxPan = int(max_pan) if max_pan else None
        suave.eWidth = float(e_width) if e_width else None
        suave.energyWidth = float(e_width) if e_width else None
        suave.maxStateRepeat = int(max_state_repeat) if max_state_repeat is not None else -1
        if prop_list:
            suave.propList = [int(p) for p in prop_list]
        else:
            suave.propList = [i for i in range(suave.numProps)]
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

            if is_alt:
                suave.sortLast = bool(sort_last)
                suave.sortEnergies()

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

    # 9. Curve Sampling & CSV Exports Modal and Downloads
    @app.callback(
        [
            Output("save-sampling-modal", "opened"),
            Output("save-modal-target-type", "value", allow_duplicate=True),
            Output("save-modal-num-points", "value"),
            Output("save-modal-info-text", "children"),
        ],
        [
            Input("btn-save-curves", "n_clicks"),
            Input("btn-save-checkpoint", "n_clicks"),
            Input("btn-save-modal-cancel", "n_clicks"),
            Input("btn-save-modal-confirm", "n_clicks"),
        ],
        [
            State("save-sampling-modal", "opened"),
            State("session-id", "data"),
        ],
        prevent_initial_call=True,
    )
    def manage_save_modal(btn_curves, btn_ckpt, btn_cancel, btn_confirm, is_opened, session_id):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        session = session_registry.get_or_create(session_id)
        num_pts = session.dataset.num_points if (session and session.dataset) else 100
        num_states = session.dataset.num_states if (session and session.dataset) else 9

        if trigger_id == "btn-save-curves":
            info_msg = (
                f"Current dataset contains {num_pts} points across {num_states} states. "
                "Specify how many points to sample in the exported curves CSV."
            )
            return True, "curves", num_pts, info_msg
        elif trigger_id == "btn-save-checkpoint":
            info_msg = (
                f"Current dataset contains {num_pts} points across {num_states} states. "
                "Specify how many points to sample in the exported checkpoint CSV."
            )
            return True, "checkpoint", num_pts, info_msg
        elif trigger_id in ("btn-save-modal-cancel", "btn-save-modal-confirm"):
            return False, no_update, no_update, no_update

        return is_opened, no_update, no_update, no_update

    @app.callback(
        [
            Output("save-modal-title", "children"),
            Output("save-modal-filename", "value"),
        ],
        Input("save-modal-target-type", "value"),
        prevent_initial_call=True,
    )
    def toggle_save_target_type(target_type):
        if target_type == "checkpoint":
            return "Save Checkpoint CSV", "checkpoint.csv"
        return "Save Curves CSV", "curves.csv"

    @app.callback(
        Output("save-modal-interp-kind", "disabled"),
        Input("save-modal-interpolate-switch", "checked"),
    )
    def toggle_interp_kind_disabled(is_interpolate):
        return not bool(is_interpolate)

    @app.callback(
        Output("save-modal-feedback", "children"),
        [
            Input("save-modal-num-points", "value"),
            Input("save-modal-interpolate-switch", "checked"),
            Input("save-modal-interp-kind", "value"),
            Input("save-modal-target-type", "value"),
        ],
        State("session-id", "data"),
    )
    def update_save_modal_feedback(num_points, interpolate, interp_kind, target_type, session_id):
        session = session_registry.get_or_create(session_id)
        cur_pts = session.dataset.num_points if (session and session.dataset) else 500
        if not num_points or int(num_points) <= 0:
            return dmc.Alert("Please enter a valid positive number of points.", color="red")

        n_pts = int(num_points)
        target_name = "Curves CSV" if target_type == "curves" else "Checkpoint CSV"
        interp_name = "Cubic Spline" if interp_kind == "cubic" else "Linear"

        if interpolate:
            if n_pts == cur_pts:
                return dmc.Alert(
                    f"Interpolation enabled: Will fit {interp_name} curves and sample {n_pts} evenly spaced points across reaction coordinates for {target_name}.",
                    color="indigo",
                )
            elif n_pts < cur_pts:
                return dmc.Alert(
                    f"Downsampling with interpolation: Will fit {interp_name} curves and sample {n_pts} evenly spaced points across the curve (original: {cur_pts} points).",
                    color="indigo",
                )
            else:
                return dmc.Alert(
                    f"Upsampling with interpolation: Will fit {interp_name} curves and sample {n_pts} evenly spaced points across the curve (original: {cur_pts} points).",
                    color="blue",
                )
        else:
            if n_pts == cur_pts:
                return dmc.Alert(
                    f"Subdivision: Keeping all {cur_pts} original points without interpolation for {target_name}.",
                    color="teal",
                )
            elif n_pts < cur_pts:
                return dmc.Alert(
                    f"Subdividing points: Will select {n_pts} equally spaced points from existing {cur_pts} points without interpolation.",
                    color="teal",
                )
            else:
                return dmc.Alert(
                    f"Notice: Cannot subdivide to {n_pts} points because it exceeds the current {cur_pts} points. "
                    f"Interpolation ({interp_name}) will be automatically applied to sample {n_pts} evenly spaced points.",
                    color="yellow",
                )

    @app.callback(
        [
            Output("download-curves-csv", "data"),
            Output("download-checkpoint-csv", "data"),
        ],
        Input("btn-save-modal-confirm", "n_clicks"),
        [
            State("save-modal-target-type", "value"),
            State("save-modal-num-points", "value"),
            State("save-modal-interpolate-switch", "checked"),
            State("save-modal-interp-kind", "value"),
            State("save-modal-filename", "value"),
            State("selector-property", "value"),
            State("session-id", "data"),
        ],
        prevent_initial_call=True,
    )
    def handle_save_export(
        n_clicks,
        target_type,
        num_points,
        interpolate,
        interp_kind,
        filename,
        prop_val,
        session_id,
    ):
        if not n_clicks:
            return no_update, no_update

        session = session_registry.get_or_create(session_id)
        if not session.dataset:
            return no_update, no_update

        cur_pts = session.dataset.num_points
        target_pts = int(num_points) if num_points else cur_pts
        interp = bool(interpolate)
        if target_pts > cur_pts:
            interp = True
        kind = str(interp_kind) if interp_kind else "cubic"

        if target_type == "curves":
            prop_idx = int(prop_val) if prop_val else 0
            csv_string = DataParserService.export_curves_csv(
                session.dataset,
                property_idx=prop_idx,
                target_points=target_pts,
                interpolate_curves=interp,
                interp_kind=kind,
            )
            out_filename = filename.strip() if filename and filename.strip() else "curves.csv"
            return dict(content=csv_string, filename=out_filename), no_update
        else:
            csv_string = DataParserService.export_checkpoint_csv(
                session.dataset,
                target_points=target_pts,
                interpolate_curves=interp,
                interp_kind=kind,
            )
            out_filename = filename.strip() if filename and filename.strip() else "checkpoint.csv"
            return no_update, dict(content=csv_string, filename=out_filename)

    # 13. Live Numerical & Physical Readout Badges
    @app.callback(
        [
            Output("badge-coord-readout", "children"),
            Output("badge-state-readout", "children"),
        ],
        [
            Input("slider-point-range", "value"),
            Input("slider-state-range", "value"),
        ],
        State("session-id", "data"),
    )
    def update_readouts(point_range, state_range, session_id):
        session = session_registry.get_or_create(session_id)
        coord_text = "Coordinates: N/A"
        state_text = "States: N/A"

        if point_range:
            p_min = int(point_range[0])
            p_max = int(point_range[1])
            pts_count = p_max - p_min + 1

            if session.dataset and session.dataset.reaction_coordinates is not None:
                rc = session.dataset.reaction_coordinates
                max_idx = len(rc) - 1
                p_min_clamped = min(max_idx, max(0, p_min))
                p_max_clamped = min(max_idx, max(0, p_max))
                r_min = float(rc[p_min_clamped])
                r_max = float(rc[p_max_clamped])
                coord_text = f"Points: {p_min} – {p_max} | R: {r_min:.3f} – {r_max:.3f} ({pts_count} pts)"
            else:
                coord_text = f"Points: {p_min} – {p_max} ({pts_count} pts)"

        if state_range:
            s_min = int(state_range[0])
            s_max = int(state_range[1])
            total_states = session.dataset.num_states if session.dataset else 9
            s_count = max(0, s_max - s_min)
            high_idx = max(s_min, s_max - 1)
            state_text = f"Active: S{s_min} – S{high_idx} ({s_count} of {total_states} States)"

        return coord_text, state_text

    # 14. One-Click Slider Range Presets (Full, Left/Right Half, Zoom Avoided Crossing, State Presets)
    @app.callback(
        [
            Output("slider-point-range", "value", allow_duplicate=True),
            Output("slider-state-range", "value", allow_duplicate=True),
            Output("text-status-message", "children", allow_duplicate=True),
        ],
        [
            Input("btn-preset-full", "n_clicks"),
            Input("btn-preset-left", "n_clicks"),
            Input("btn-preset-right", "n_clicks"),
            Input("btn-preset-crossing", "n_clicks"),
            Input("btn-preset-all-states", "n_clicks"),
            Input("btn-preset-low-states", "n_clicks"),
        ],
        [
            State("slider-point-range", "max"),
            State("slider-state-range", "max"),
            State("session-id", "data"),
        ],
        prevent_initial_call=True,
    )
    def apply_slider_presets(
        full_clicks, left_clicks, right_clicks, crossing_clicks,
        all_states_clicks, low_states_clicks,
        max_pnt, max_state, session_id
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        session = session_registry.get_or_create(session_id)
        total_pnts = (int(max_pnt) + 1) if max_pnt is not None else 100
        total_states = int(max_state) if max_state is not None else 9

        if trigger_id == "btn-preset-full":
            return [0, total_pnts - 1], no_update, f"Preset applied: Full Trajectory [0, {total_pnts - 1}]."

        elif trigger_id == "btn-preset-left":
            mid = max(1, (total_pnts - 1) // 2)
            return [0, mid], no_update, f"Preset applied: Left Half [0, {mid}]."

        elif trigger_id == "btn-preset-right":
            mid = max(0, (total_pnts - 1) // 2)
            return [mid, total_pnts - 1], no_update, f"Preset applied: Right Half [{mid}, {total_pnts - 1}]."

        elif trigger_id == "btn-preset-crossing":
            if session.dataset and session.dataset.energies is not None:
                E = session.dataset.energies
                num_pts = E.shape[1]
                min_gaps = []
                for p in range(num_pts):
                    sorted_e = np.sort(E[:, p])
                    gaps = np.diff(sorted_e)
                    pos_gaps = gaps[gaps > 1e-6]
                    min_gaps.append(float(np.min(pos_gaps)) if len(pos_gaps) > 0 else float(np.min(gaps)))

                crossing_p = int(np.argmin(min_gaps))
                half_window = max(8, int(num_pts * 0.15))
                p_start = max(0, crossing_p - half_window)
                p_end = min(num_pts - 1, crossing_p + half_window)
                rc_val = (
                    float(session.dataset.reaction_coordinates[crossing_p])
                    if session.dataset.reaction_coordinates is not None
                    else crossing_p
                )
                return (
                    [p_start, p_end],
                    no_update,
                    f"Preset: Zoomed on narrowest energy gap at point {crossing_p} (R ≈ {rc_val:.3f}, ΔE ≈ {min_gaps[crossing_p]:.4e}).",
                )
            else:
                return no_update, no_update, "No dataset loaded to detect avoided crossings."

        elif trigger_id == "btn-preset-all-states":
            return no_update, [0, total_states], f"Preset applied: All States [0, {total_states}]."

        elif trigger_id == "btn-preset-low-states":
            low_cut = min(4, total_states)
            return no_update, [0, low_cut], f"Preset applied: Low-Lying States [0, {low_cut}]."

        return no_update, no_update, no_update
