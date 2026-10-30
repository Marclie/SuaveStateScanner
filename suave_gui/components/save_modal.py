"""
Save modal dialog supporting curve sampling, subdivision, interpolation,
and exporting to curves.csv or checkpoint.csv.
"""

import dash_mantine_components as dmc
from dash import html


def create_save_modal() -> dmc.Modal:
    """Creates the curve sampling and export modal dialog."""
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
                dmc.SegmentedControl(
                    id="save-modal-target-type",
                    data=[
                        {"label": "Curves CSV (curves.csv)", "value": "curves"},
                        {"label": "Checkpoint CSV (checkpoint.csv)", "value": "checkpoint"},
                    ],
                    value="curves",
                    fullWidth=True,
                ),

                # Target points count
                dmc.NumberInput(
                    id="save-modal-num-points",
                    label="Sample Points",
                    description="Total points to sample along the reaction coordinate",
                    value=100,
                    min=2,
                    step=1,
                ),

                # Interpolation toggle
                dmc.Switch(
                    id="save-modal-interpolate-switch",
                    label="Interpolate curves",
                    description="When disabled, subdivides existing points to keep N points. When enabled, fits spline and samples N points evenly.",
                    checked=False,
                ),

                # Interpolation kind (spline vs linear)
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

                # Filename
                dmc.TextInput(
                    id="save-modal-filename",
                    label="Download Filename",
                    value="curves.csv",
                ),

                # Informative preview / feedback alert
                html.Div(id="save-modal-feedback"),

                # Action Buttons
                dmc.Group(
                    justify="flex-end",
                    children=[
                        dmc.Button(
                            "Cancel",
                            id="btn-save-modal-cancel",
                            variant="subtle",
                            color="gray",
                        ),
                        dmc.Button(
                            "Save & Download",
                            id="btn-save-modal-confirm",
                            color="green",
                        ),
                    ],
                ),
            ])
        ],
    )
