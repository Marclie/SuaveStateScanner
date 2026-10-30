"""
Save modal dialog supporting curve sampling, subdivision, interpolation,
and exporting to curves.csv or checkpoint.csv, enriched with descriptive tooltips.
"""

import dash_mantine_components as dmc
from dash import html
from suave_gui.components.tooltip_helper import create_tooltip


def create_save_modal() -> dmc.Modal:
    """Creates the curve sampling and export modal dialog with detailed tooltips."""
    return dmc.Modal(
        id="save-sampling-modal",
        title=dmc.Text("Save Curves CSV", fw=700, size="lg", id="save-modal-title"),
        opened=False,
        size="md",
        children=[
            dmc.Stack([
                dmc.Text(
                    "Choose how many points to sample along the curve when saving. "
                    "You can keep equally spaced subdivided points or interpolate curves smoothly.",
                    id="save-modal-info-text",
                    size="sm",
                    c="dimmed",
                ),

                # Format selection: Curves CSV vs Checkpoint CSV
                create_tooltip(
                    dmc.SegmentedControl(
                        id="save-modal-target-type",
                        data=[
                            {"label": "Curves CSV (curves.csv)", "value": "curves"},
                            {"label": "Checkpoint CSV (checkpoint.csv)", "value": "checkpoint"},
                        ],
                        value="curves",
                        fullWidth=True,
                    ),
                    title="Export File Format",
                    description=(
                        "Curves CSV exports reaction coordinates and columns for each state's energy and properties. "
                        "Checkpoint CSV exports the full sequential matrix format used to restart SuaveStateScanner."
                    ),
                    position="top",
                ),

                # Target points count
                create_tooltip(
                    dmc.NumberInput(
                        id="save-modal-num-points",
                        label="Sample Points",
                        description="Total points to sample along the reaction coordinate",
                        value=100,
                        min=2,
                        step=1,
                    ),
                    title="Exported Sample Points",
                    description=(
                        "Specifies the number of reaction coordinate points written to the output file. "
                        "When interpolation is off, existing grid points are subdivided; when interpolation is on, "
                        "curves are resampled smoothly at N points."
                    ),
                    position="top",
                ),

                # Interpolation toggle
                create_tooltip(
                    dmc.Switch(
                        id="save-modal-interpolate-switch",
                        label="Interpolate curves",
                        description="When disabled, subdivides existing points. When enabled, fits spline and samples N points evenly.",
                        checked=False,
                    ),
                    title="Spline Interpolation on Export",
                    description=(
                        "Whether to keep interpolated missing points or resample curves smoothly via splines upon saving. "
                        "Corresponds to keepInterp in README: when enabled, missing or coarse points are replaced with "
                        "spline-interpolated values in the saved file."
                    ),
                    config_name="keepInterp",
                    default_val="False",
                    position="top",
                ),

                # Interpolation kind (spline vs linear)
                create_tooltip(
                    dmc.Select(
                        id="save-modal-interp-kind",
                        label="Interpolation Method",
                        description="Spline method used when interpolation is enabled",
                        data=[
                            {"label": "Cubic Spline", "value": "cubic"},
                            {"label": "Linear", "value": "linear"},
                        ],
                        value="cubic",
                        disabled=True,
                    ),
                    title="Interpolation Function",
                    description=(
                        "Algorithm used for curve resampling: Cubic Spline provides smooth curvature through points; "
                        "Linear connects neighboring points with straight segments."
                    ),
                    default_val="Cubic Spline",
                    position="top",
                ),

                # Filename
                create_tooltip(
                    dmc.TextInput(
                        id="save-modal-filename",
                        label="Download Filename",
                        value="curves.csv",
                    ),
                    title="Output CSV Filename",
                    description="Name of the file to be generated and downloaded in your browser.",
                    default_val="curves.csv",
                    position="top",
                ),

                # Informative preview / feedback alert
                html.Div(id="save-modal-feedback"),

                # Action Buttons
                dmc.Group(
                    justify="flex-end",
                    children=[
                        create_tooltip(
                            dmc.Button(
                                "Cancel",
                                id="btn-save-modal-cancel",
                                variant="subtle",
                                color="gray",
                            ),
                            title="Cancel",
                            description="Dismiss the save dialog without generating any file.",
                            position="top",
                        ),
                        create_tooltip(
                            dmc.Button(
                                "Save & Download",
                                id="btn-save-modal-confirm",
                                color="green",
                            ),
                            title="Generate & Download CSV",
                            description="Executes the export with the requested sampling and downloads the CSV file.",
                            position="top",
                        ),
                    ],
                ),
            ])
        ],
    )
