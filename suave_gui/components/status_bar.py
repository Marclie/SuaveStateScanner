"""
Bottom status bar component.
Renders real-time telemetry, convergence metrics, and execution feedback.
"""

import dash_mantine_components as dmc
from dash import html


def create_status_bar() -> dmc.Box:
    """Creates the persistent footer status bar."""
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
                dmc.Badge("Ready", id="badge-engine-status", color="green", size="sm", variant="dot"),
                html.Span(id="text-status-message", children="System initialized. Load dataset to begin."),
            ]),
            dmc.Group(gap="md", children=[
                html.Span(id="text-convergence-telemetry", children="Δmax: N/A"),
                dmc.Badge("Local Engine", id="badge-compute-backend", color="gray", size="sm", variant="outline"),
                html.Span(id="text-dataset-summary", children="No dataset loaded"),
            ]),
        ],
    )
