"""
Bottom status bar component.
Renders real-time telemetry, convergence metrics, and execution feedback,
equipped with descriptive hover tooltips.
"""

import dash_mantine_components as dmc
from dash import html
from suave_gui.components.tooltip_helper import create_tooltip


def create_status_bar() -> dmc.Box:
    """Creates the persistent footer status bar with explanatory tooltips."""
    return dmc.Box(
        style={
            "height": "32px",
            "padding": "0 16px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "backgroundColor": "#09090B",
            "borderTop": "1px solid #27272A",
            "fontSize": "12px",
            "color": "#A1A1AA",
        },
        children=[
            dmc.Group(gap="sm", children=[
                create_tooltip(
                    dmc.Badge("Ready", id="badge-engine-status", color="green", size="sm", variant="dot"),
                    title="Compute Engine Status",
                    description="Indicates whether the background calculation worker is Ready, actively Sweeping, or Aborted.",
                    position="top",
                ),
                create_tooltip(
                    html.Span(id="text-status-message", children="System initialized. Load dataset to begin."),
                    title="System Status & Action Log",
                    description="Displays notifications, live progress updates, and results for the latest user action or sweep.",
                    position="top",
                ),
            ]),
            dmc.Group(gap="md", children=[
                create_tooltip(
                    html.Span(id="text-convergence-telemetry", children="Δmax: N/A"),
                    title="Convergence Telemetry (Δmax)",
                    description=(
                        "Maximum change in the finite-difference continuity metric evaluated across all points in the latest sweep pass. "
                        "When Δmax drops below 1e-12, the reordering scan is deemed converged."
                    ),
                    position="top",
                ),
                create_tooltip(
                    dmc.Badge("Local Engine", id="badge-compute-backend", color="gray", size="sm", variant="outline"),
                    title="Execution Architecture",
                    description="Numba JIT-accelerated backend running asynchronously inside a Python ThreadPoolExecutor on the local machine.",
                    position="top",
                ),
                create_tooltip(
                    html.Span(id="text-dataset-summary", children="No dataset loaded"),
                    title="Active Dataset Statistics",
                    description="Displays the loaded file name, number of reaction coordinate points, electronic states, and property dimensions.",
                    position="top",
                ),
            ]),
        ],
    )
