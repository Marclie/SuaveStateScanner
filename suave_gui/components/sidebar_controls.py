"""
Sidebar controls component.
Houses modular accordion docks for display selection, manual interventions,
automated continuity sweeps, and history recovery.
"""

from typing import List, Dict, Any, Optional
import dash_mantine_components as dmc
from dash import html, dcc


def create_sidebar_controls(num_states: int = 9, num_props: int = 8) -> dmc.Box:
    """
    Constructs the sidebar controls drawer.
    Dynamically populates all available properties and the complete list of states.
    """
    # Build complete list of property options: Energy (0) and Property 1..N
    prop_data = [{"label": "Energy", "value": "0"}] + [
        {"label": f"Property {i}", "value": str(i)}
        for i in range(1, max(1, num_props + 1))
    ]

    # Build complete list of state options: All States and State 0..S-1
    state_data = [{"label": "All States", "value": "all"}] + [
        {"label": f"State {i}", "value": str(i)}
        for i in range(max(1, num_states))
    ]

    return dmc.Box(
        style={
            "width": "360px",
            "minWidth": "340px",
            "padding": "12px",
            "overflowY": "auto",
            "backgroundColor": "#18181B",
            "borderLeft": "1px solid #27272A",
            "height": "100%",
        },
        children=[
            dmc.Accordion(
                multiple=True,
                value=["display", "manual", "reorder"],
                children=[
                    # 1. Active Display & Targets
                    dmc.AccordionItem([
                        dmc.AccordionControl(
                            dmc.Text("Active Display & Targets", fw=600, size="sm")
                        ),
                        dmc.AccordionPanel([
                            dmc.Stack(gap="xs", children=[
                                dmc.Select(
                                    id="selector-property",
                                    label="Target Property to Plot",
                                    description="Select and plot Energy or any property curve",
                                    value="0",
                                    data=prop_data,
                                ),
                                dmc.Select(
                                    id="selector-highlight-state",
                                    label="Available States / Highlight",
                                    description="Full list of states or view all simultaneously",
                                    value="all",
                                    data=state_data,
                                ),
                            ]),
                        ]),
                    ], value="display"),

                    # 2. Manual State Swapping & Interventions
                    dmc.AccordionItem([
                        dmc.AccordionControl(
                            dmc.Text("Manual State Interventions", fw=600, size="sm")
                        ),
                        dmc.AccordionPanel([
                            dmc.Stack(gap="xs", children=[
                                dmc.Group([
                                    dmc.NumberInput(
                                        id="input-swap-state-1",
                                        label="State 1",
                                        value=0,
                                        min=0,
                                        max=max(0, num_states - 1),
                                        step=1,
                                        style={"flex": 1},
                                    ),
                                    dmc.NumberInput(
                                        id="input-swap-state-2",
                                        label="State 2",
                                        value=min(1, max(0, num_states - 1)),
                                        min=0,
                                        max=max(0, num_states - 1),
                                        step=1,
                                        style={"flex": 1},
                                    ),
                                ]),
                                dmc.Button(
                                    "Swap States",
                                    id="btn-apply-swap",
                                    color="orange",
                                    fullWidth=True,
                                    variant="light",
                                ),
                                dmc.Group([
                                    dmc.Button(
                                        "Shuffle Points in Window",
                                        id="btn-shuffle-values",
                                        color="yellow",
                                        variant="subtle",
                                        style={"flex": 1},
                                    ),
                                    dmc.Button(
                                        "Reset Initial",
                                        id="btn-reset-initial",
                                        color="red",
                                        variant="subtle",
                                        style={"flex": 1},
                                    ),
                                ]),
                                dmc.Divider(my="xs"),
                                dmc.Group([
                                    dmc.NumberInput(
                                        id="input-remove-state",
                                        label="Remove State",
                                        value=0,
                                        min=0,
                                        max=max(0, num_states - 1),
                                        step=1,
                                        style={"flex": 1},
                                    ),
                                    dmc.Button(
                                        "Remove State Data",
                                        id="btn-remove-state",
                                        color="red",
                                        variant="outline",
                                        style={"flex": 1, "marginTop": "24px"},
                                    ),
                                ]),
                            ]),
                        ]),
                    ], value="manual"),

                    # 3. Automated Reordering Sweeps
                    dmc.AccordionItem([
                        dmc.AccordionControl(
                            dmc.Text("Automated Reordering Sweeps", fw=600, size="sm")
                        ),
                        dmc.AccordionPanel([
                            dmc.Stack(gap="xs", children=[
                                dmc.Group([
                                    dmc.NumberInput(
                                        id="input-num-sweeps",
                                        label="Sweeps",
                                        value=1,
                                        min=1,
                                        max=100,
                                        style={"flex": 1},
                                    ),
                                    dmc.NumberInput(
                                        id="input-stencil-width",
                                        label="Stencil Width",
                                        value=5,
                                        min=3,
                                        max=50,
                                        style={"flex": 1},
                                    ),
                                    dmc.NumberInput(
                                        id="input-deriv-order",
                                        label="Order",
                                        value=1,
                                        min=1,
                                        max=10,
                                        style={"flex": 1},
                                    ),
                                ]),
                                dmc.Group([
                                    dmc.Switch(
                                        id="switch-alternate",
                                        label="Alternate Sweep",
                                        checked=False,
                                        size="xs",
                                    ),
                                    dmc.Switch(
                                        id="switch-sort-last",
                                        label="Sort Last Point",
                                        checked=False,
                                        size="xs",
                                    ),
                                ]),
                                dmc.Group([
                                    dmc.Switch(
                                        id="switch-interpolate",
                                        label="Interpolate Reorder",
                                        checked=False,
                                        size="xs",
                                    ),
                                    dmc.Switch(
                                        id="switch-redundant",
                                        label="Redundant Swaps",
                                        checked=True,
                                        size="xs",
                                    ),
                                ]),
                                dmc.Space(h=4),
                                dmc.Group([
                                    dmc.Button(
                                        "Forward Sweep",
                                        id="btn-sweep-forward",
                                        color="blue",
                                        style={"flex": 1},
                                    ),
                                    dmc.Button(
                                        "Backward Sweep",
                                        id="btn-sweep-backward",
                                        color="indigo",
                                        style={"flex": 1},
                                    ),
                                ]),
                                dmc.Button(
                                    "Alternate Sweep",
                                    id="btn-sweep-alternate",
                                    color="cyan",
                                    fullWidth=True,
                                    variant="light",
                                ),
                                dmc.Button(
                                    "Abort Calculation",
                                    id="btn-abort-sweep",
                                    color="red",
                                    fullWidth=True,
                                    variant="subtle",
                                ),
                            ]),
                        ]),
                    ], value="reorder"),

                    # 4. Range Filters & Stencil Pan
                    dmc.AccordionItem([
                        dmc.AccordionControl(
                            dmc.Text("Constraints & Filter Windows", fw=600, size="sm")
                        ),
                        dmc.AccordionPanel([
                            dmc.Stack(gap="xs", children=[
                                dmc.NumberInput(
                                    id="input-max-pan",
                                    label="Max Pan of Stencil",
                                    value=1000,
                                    min=1,
                                ),
                                dmc.NumberInput(
                                    id="input-energy-width",
                                    label="Energy Width Window",
                                    value=10.0,
                                    min=0.0001,
                                    decimalScale=4,
                                ),
                            ]),
                        ]),
                    ], value="constraints"),
                ],
            ),
        ],
    )
