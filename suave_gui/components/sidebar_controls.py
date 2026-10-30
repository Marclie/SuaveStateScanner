"""
Sidebar controls component.
Houses modular accordion docks for display selection, manual interventions,
automated continuity sweeps, and history recovery, enhanced with rich, in-depth tooltips
referencing README.md specifications.
"""

from typing import List, Dict, Any, Optional
import dash_mantine_components as dmc
from dash import html, dcc
from suave_gui.components.tooltip_helper import create_tooltip


def create_sidebar_controls(num_states: int = 9, num_props: int = 8) -> dmc.Box:
    """
    Constructs the sidebar controls drawer.
    Dynamically populates all available properties and the complete list of states,
    fully annotated with descriptive tooltips for every control element.
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

    # Property indices for continuity metric propList (0-indexed properties: Property 0..num_props-1)
    prop_filter_data = [
        {"label": f"Property {i}", "value": str(i)}
        for i in range(max(1, num_props))
    ]
    prop_filter_default = [str(i) for i in range(max(1, num_props))]

    # Derivative order options for multi-select
    order_options = [
        {"label": "1st Derivative (Slope)", "value": "1"},
        {"label": "2nd Derivative (Curvature)", "value": "2"},
        {"label": "3rd Derivative", "value": "3"},
        {"label": "4th Derivative", "value": "4"},
        {"label": "5th Derivative", "value": "5"},
    ]

    return dmc.Box(
        style={
            "width": "380px",
            "minWidth": "350px",
            "padding": "12px",
            "overflowY": "auto",
            "backgroundColor": "#18181B",
            "borderLeft": "1px solid #27272A",
            "height": "100%",
        },
        children=[
            dmc.Accordion(
                multiple=True,
                value=["display", "manual", "reorder", "constraints"],
                children=[
                    # 1. Active Display & Targets
                    dmc.AccordionItem([
                        dmc.AccordionControl(
                            dmc.Text("Active Display & Targets", fw=600, size="sm")
                        ),
                        dmc.AccordionPanel([
                            dmc.Stack(gap="xs", children=[
                                create_tooltip(
                                    dmc.Select(
                                        id="selector-property",
                                        label="Target Property to Plot",
                                        description="Select and plot Energy or any property curve",
                                        value="0",
                                        data=prop_data,
                                        style={"width": "100%"},
                                    ),
                                    title="Target Property to Plot",
                                    description=(
                                        "Selects whether to display State Energies (index 0) or any of the calculated "
                                        "physical properties (Property 1..N, such as transition dipole moments, "
                                        "oscillator strengths, or operator norms). Inspecting property curves helps verify "
                                        "that energy crossings correspond to continuous physical character along the reaction coordinate."
                                    ),
                                    config_name="printVar",
                                    default_val="0 (Energy)",
                                    style={"width": "100%"},
                                    position="left",
                                ),
                                create_tooltip(
                                    dmc.Select(
                                        id="selector-highlight-state",
                                        label="Available States / Highlight",
                                        description="Full list of states or view all simultaneously",
                                        value="all",
                                        data=state_data,
                                        style={"width": "100%"},
                                    ),
                                    title="State Display & Highlighting",
                                    description=(
                                        "Choose to display all electronic states simultaneously, or highlight a single specific "
                                        "state curve with full opacity while dimming all other curves. Crucial for tracing "
                                        "individual adiabatic/diabatic states through crowded avoided crossings."
                                    ),
                                    default_val="All States",
                                    style={"width": "100%"},
                                    position="left",
                                ),
                            ]),
                        ]),
                    ], value="display"),

                    # 2. Manual State Swapping & Interventions (Side-by-Side Layout)
                    dmc.AccordionItem([
                        dmc.AccordionControl(
                            dmc.Text("Manual State Interventions", fw=600, size="sm")
                        ),
                        dmc.AccordionPanel([
                            dmc.Stack(gap="xs", children=[
                                # State 1 and State 2 side by side
                                dmc.SimpleGrid(
                                    cols=2,
                                    spacing="xs",
                                    style={"width": "100%"},
                                    children=[
                                        create_tooltip(
                                            dmc.NumberInput(
                                                id="input-swap-state-1",
                                                label="State 1",
                                                value=0,
                                                min=0,
                                                max=max(0, num_states - 1),
                                                step=1,
                                                size="sm",
                                                style={"width": "100%"},
                                            ),
                                            title="First State Index",
                                            description="0-indexed position of the first electronic state to interchange.",
                                            default_val="0",
                                            style={"width": "100%"},
                                            position="top",
                                        ),
                                        create_tooltip(
                                            dmc.NumberInput(
                                                id="input-swap-state-2",
                                                label="State 2",
                                                value=min(1, max(0, num_states - 1)),
                                                min=0,
                                                max=max(0, num_states - 1),
                                                step=1,
                                                size="sm",
                                                style={"width": "100%"},
                                            ),
                                            title="Second State Index",
                                            description="0-indexed position of the second electronic state to interchange with State 1.",
                                            default_val="1",
                                            style={"width": "100%"},
                                            position="top",
                                        ),
                                    ],
                                ),
                                create_tooltip(
                                    dmc.Button(
                                        "Swap States",
                                        id="btn-apply-swap",
                                        color="grape",
                                        fullWidth=True,
                                        variant="filled",
                                        size="sm",
                                    ),
                                    title="Execute State Swap",
                                    description=(
                                        "Directly interchanges the energies and associated physical properties of State 1 and State 2 "
                                        "across the active Reaction Coordinate Window (Point Range slider). Essential for manual curation "
                                        "and guiding the optimizer through near-degenerate conical intersections."
                                    ),
                                    style={"width": "100%"},
                                    position="left",
                                ),
                                # Shuffle and Reset side by side
                                dmc.SimpleGrid(
                                    cols=2,
                                    spacing="xs",
                                    style={"width": "100%"},
                                    children=[
                                        create_tooltip(
                                            dmc.Button(
                                                "Shuffle in Window",
                                                id="btn-shuffle-values",
                                                color="yellow",
                                                variant="light",
                                                size="xs",
                                                fullWidth=True,
                                            ),
                                            title="Shuffle Points in Window",
                                            description=(
                                                "Randomly shuffles the state ordering for points within the selected Reaction Coordinate Window. "
                                                "As described in the README troubleshooting guide, shuffling can dislodge states trapped in local "
                                                "minima or false crossings where initial configuration order prevents proper state swaps."
                                            ),
                                            config_name="doShuffle",
                                            default_val="False",
                                            style={"width": "100%"},
                                            position="left",
                                        ),
                                        create_tooltip(
                                            dmc.Button(
                                                "Reset Initial",
                                                id="btn-reset-initial",
                                                color="red",
                                                variant="light",
                                                size="xs",
                                                fullWidth=True,
                                            ),
                                            title="Reset Dataset to Initial",
                                            description=(
                                                "Discards all manual swaps, shuffles, and automated continuity sweeps, restoring all states "
                                                "and properties back to the original unmodified configuration from file load."
                                            ),
                                            style={"width": "100%"},
                                            position="left",
                                        ),
                                    ],
                                ),
                                dmc.Divider(my="xs"),
                                # Remove State and Button side by side
                                dmc.SimpleGrid(
                                    cols=2,
                                    spacing="xs",
                                    style={"width": "100%", "alignItems": "flex-end"},
                                    children=[
                                        create_tooltip(
                                            dmc.NumberInput(
                                                id="input-remove-state",
                                                label="Remove State",
                                                value=0,
                                                min=0,
                                                max=max(0, num_states - 1),
                                                step=1,
                                                size="xs",
                                                style={"width": "100%"},
                                            ),
                                            title="State Index to Remove",
                                            description="Index of the electronic state whose points should be cleared over the active window.",
                                            style={"width": "100%"},
                                            position="top",
                                        ),
                                        create_tooltip(
                                            dmc.Button(
                                                "Remove State Data",
                                                id="btn-remove-state",
                                                color="red",
                                                variant="outline",
                                                size="xs",
                                                fullWidth=True,
                                            ),
                                            title="Remove State Data",
                                            description=(
                                                "Sets energy and property values of the chosen state to NaN across the active reaction "
                                                "coordinate window. Useful for removing unphysical intruder states or diverged calculation points."
                                            ),
                                            style={"width": "100%"},
                                            position="left",
                                        ),
                                    ],
                                ),
                            ]),
                        ]),
                    ], value="manual"),

                    # 3. Automated Reordering Sweeps (Clean 2-Column Switch Alignment + Advanced Stencil Disclosure)
                    dmc.AccordionItem([
                        dmc.AccordionControl(
                            dmc.Text("Automated Reordering Sweeps", fw=600, size="sm")
                        ),
                        dmc.AccordionPanel([
                            dmc.Stack(gap="xs", children=[
                                # Primary Sweep Count Input
                                create_tooltip(
                                    dmc.NumberInput(
                                        id="input-num-sweeps",
                                        label="Optimization Sweeps Count",
                                        description="Maximum iterative passes through points and states",
                                        value=1,
                                        min=1,
                                        max=100,
                                        size="sm",
                                        style={"width": "100%"},
                                    ),
                                    title="Optimization Sweeps Count",
                                    description=(
                                        "Maximum number of iterative sweep passes to execute across all coordinate points and states. "
                                        "Sweeps terminate early if the maximum continuity metric change (Δmax) drops below tolerance (1e-12)."
                                    ),
                                    config_name="maxIter",
                                    default_val="1 (up to 1000 in config)",
                                    style={"width": "100%"},
                                    position="top",
                                ),

                                # Perfectly Aligned 2-Column Grid of Switches
                                dmc.SimpleGrid(
                                    cols=2,
                                    spacing="sm",
                                    verticalSpacing="xs",
                                    style={"width": "100%", "marginTop": "4px", "marginBottom": "6px"},
                                    children=[
                                        create_tooltip(
                                            dmc.Switch(
                                                id="switch-alternate",
                                                label="Alternate Sweep",
                                                checked=False,
                                                size="xs",
                                            ),
                                            title="Alternate Sweep Direction",
                                            description=(
                                                "Alternates sweep direction between forward (left-to-right) and backward (right-to-left). "
                                                "Backward sweeps propagate continuity from the right boundary, helping resolve ambiguous crossings "
                                                "and eliminating directional sorting bias."
                                            ),
                                            config_name="sweepBack",
                                            default_val="False",
                                            position="top",
                                        ),
                                        create_tooltip(
                                            dmc.Switch(
                                                id="switch-sort-last",
                                                label="Sort Last Point",
                                                checked=False,
                                                size="xs",
                                            ),
                                            title="Sort Last Point by Energy",
                                            description=(
                                                "Sorts the final coordinate endpoint by energy after a sweep pass, establishing a clean, ordered "
                                                "boundary condition before initiating a reverse sweep."
                                            ),
                                            config_name="sortLast",
                                            default_val="False",
                                            position="top",
                                        ),
                                        create_tooltip(
                                            dmc.Switch(
                                                id="switch-interpolate",
                                                label="Interpolate Reorder",
                                                checked=False,
                                                size="xs",
                                            ),
                                            title="Interpolate Missing Points",
                                            description=(
                                                "Linearly interpolates over NaN or Inf values when calculating finite differences during optimization scans. "
                                                "If disabled (False), missing points are explicitly ignored in finite-difference stencils. "
                                                "Useful for handling failed electronic structure calculations."
                                            ),
                                            config_name="interpolate",
                                            default_val="False",
                                            position="top",
                                        ),
                                        create_tooltip(
                                            dmc.Switch(
                                                id="switch-redundant",
                                                label="Redundant Swaps",
                                                checked=True,
                                                size="xs",
                                            ),
                                            title="Bidirectional Redundant Swaps",
                                            description=(
                                                "Whether to allow redundant swaps. If set to False, a state will only be considered for swapping "
                                                "with higher-lying states, improving performance since fewer states enter the derivatives. "
                                                "If set to True, states can also swap with lower-lying states, preventing states from being trapped "
                                                "in sub-optimal orderings when complex curve crossings occur."
                                            ),
                                            config_name="redundantSwaps",
                                            default_val="True",
                                            position="top",
                                        ),
                                        create_tooltip(
                                            dmc.Switch(
                                                id="switch-make-pos",
                                                label="Make Properties Pos",
                                                checked=False,
                                                size="xs",
                                            ),
                                            title="Make Properties Positive",
                                            description=(
                                                "Takes the absolute value of all property values during continuity sorting. Eliminates dependence "
                                                "on arbitrary sign flips (e.g., transition dipole phase conventions between adjacent geometry steps) "
                                                "common in quantum chemistry."
                                            ),
                                            config_name="makePos",
                                            default_val="False",
                                            position="top",
                                        ),
                                        create_tooltip(
                                            dmc.Switch(
                                                id="switch-normalize",
                                                label="Normalize Properties",
                                                checked=False,
                                                size="xs",
                                            ),
                                            title="Normalize Extracted Properties",
                                            description=(
                                                "Normalizes all property vectors across states during continuity calculations. Ensures properties "
                                                "with disparate numerical scales (such as energies in Hartrees vs. large dipole moments) contribute "
                                                "with balanced weighting to the continuity objective function."
                                            ),
                                            config_name="normalize",
                                            default_val="False",
                                            position="top",
                                        ),
                                    ],
                                ),

                                # Action Buttons
                                dmc.SimpleGrid(
                                    cols=2,
                                    spacing="xs",
                                    style={"width": "100%"},
                                    children=[
                                        create_tooltip(
                                            dmc.Button(
                                                "Forward Sweep",
                                                id="btn-sweep-forward",
                                                color="blue",
                                                size="sm",
                                                fullWidth=True,
                                            ),
                                            title="Run Forward Sweep",
                                            description="Executes automated continuity sweeps from left to right along the reaction coordinate in background threads.",
                                            style={"width": "100%"},
                                            position="left",
                                        ),
                                        create_tooltip(
                                            dmc.Button(
                                                "Backward Sweep",
                                                id="btn-sweep-backward",
                                                color="indigo",
                                                size="sm",
                                                fullWidth=True,
                                            ),
                                            title="Run Backward Sweep",
                                            description="Executes automated continuity sweeps in reverse (right to left) along the reaction coordinate.",
                                            style={"width": "100%"},
                                            position="left",
                                        ),
                                    ],
                                ),
                                create_tooltip(
                                    dmc.Button(
                                        "Alternate Sweep",
                                        id="btn-sweep-alternate",
                                        color="cyan",
                                        fullWidth=True,
                                        variant="light",
                                        size="xs",
                                    ),
                                    title="Run Alternating Sweeps",
                                    description="Alternates between forward and backward passes iteratively to eliminate directional hysteresis and ensure global continuity.",
                                    style={"width": "100%"},
                                    position="left",
                                ),
                                create_tooltip(
                                    dmc.Button(
                                        "Abort Calculation",
                                        id="btn-abort-sweep",
                                        color="red",
                                        fullWidth=True,
                                        variant="subtle",
                                        size="xs",
                                    ),
                                    title="Abort Running Calculation",
                                    description="Immediately halts background sweep execution and preserves the current intermediate state without freezing the UI.",
                                    style={"width": "100%"},
                                    position="left",
                                ),

                                # Advanced Stencil Settings Collapsible (Compact & Non-Intrusive)
                                dmc.Accordion(
                                    chevronPosition="right",
                                    variant="separated",
                                    radius="sm",
                                    style={"marginTop": "6px"},
                                    children=[
                                        dmc.AccordionItem(
                                            [
                                                dmc.AccordionControl(
                                                    dmc.Text("⚙️ Advanced Stencil Settings", size="xs", fw=600, c="#93C5FD")
                                                ),
                                                dmc.AccordionPanel(
                                                    dmc.Stack(gap="xs", children=[
                                                        create_tooltip(
                                                            dmc.MultiSelect(
                                                                id="input-deriv-order",
                                                                label="Derivative Orders",
                                                                description="Select orders for finite differences (e.g. 1 and 3)",
                                                                data=order_options,
                                                                value=["1"],
                                                                size="xs",
                                                                clearable=False,
                                                                style={"width": "100%"},
                                                            ),
                                                            title="Derivative Orders (orders)",
                                                            description=(
                                                                "Select one or multiple derivative orders simultaneously (e.g. 1 and 3) "
                                                                "used to evaluate curve continuity. As noted in the README, order [1] gives "
                                                                "correct behavior for almost all cases."
                                                            ),
                                                            config_name="orders",
                                                            default_val="[1]",
                                                            style={"width": "100%"},
                                                            position="left",
                                                        ),
                                                        dmc.SimpleGrid(
                                                            cols=2,
                                                            spacing="xs",
                                                            style={"width": "100%"},
                                                            children=[
                                                                create_tooltip(
                                                                    dmc.NumberInput(
                                                                        id="input-stencil-width",
                                                                        label="Stencil Width",
                                                                        value=5,
                                                                        min=3,
                                                                        max=50,
                                                                        size="xs",
                                                                        style={"width": "100%"},
                                                                    ),
                                                                    title="Finite Difference Stencil Width",
                                                                    description=(
                                                                        "Grid points included in the finite-difference stencil. "
                                                                        "Must be strictly greater than max derivative order."
                                                                    ),
                                                                    config_name="width",
                                                                    default_val="3 or max(orders)+3",
                                                                    style={"width": "100%"},
                                                                    position="top",
                                                                ),
                                                                create_tooltip(
                                                                    dmc.NumberInput(
                                                                        id="input-future-pnts",
                                                                        label="Future Points",
                                                                        value=0,
                                                                        min=0,
                                                                        max=10,
                                                                        size="xs",
                                                                        style={"width": "100%"},
                                                                    ),
                                                                    title="Future Points in Stencil",
                                                                    description=(
                                                                        "Number of points to the right of the center point included in the stencil. "
                                                                        "Since points to the right are unsorted, 0 is strongly recommended."
                                                                    ),
                                                                    config_name="futurePnts",
                                                                    default_val="0",
                                                                    style={"width": "100%"},
                                                                    position="top",
                                                                ),
                                                            ],
                                                        ),
                                                    ]),
                                                ),
                                            ],
                                            value="advanced-stencil",
                                        ),
                                    ],
                                ),
                            ]),
                        ]),
                    ], value="reorder"),

                    # 4. Constraints & Filter Windows
                    dmc.AccordionItem([
                        dmc.AccordionControl(
                            dmc.Text("Constraints & Filter Windows", fw=600, size="sm")
                        ),
                        dmc.AccordionPanel([
                            dmc.Stack(gap="xs", children=[
                                dmc.SimpleGrid(
                                    cols=2,
                                    spacing="xs",
                                    style={"width": "100%"},
                                    children=[
                                        create_tooltip(
                                            dmc.NumberInput(
                                                id="input-max-pan",
                                                label="Max Pan",
                                                value=1000,
                                                min=1,
                                                size="xs",
                                                style={"width": "100%"},
                                            ),
                                            title="Maximum Stencil Pan",
                                            description=(
                                                "Limits how far the stencil can pivot around the center point. "
                                                "Restricting maxPan reduces tested combinations, speeding up computation."
                                            ),
                                            config_name="maxPan",
                                            default_val="None (unlimited)",
                                            style={"width": "100%"},
                                            position="top",
                                        ),
                                        create_tooltip(
                                            dmc.NumberInput(
                                                id="input-energy-width",
                                                label="Energy Width Window",
                                                value=10.0,
                                                min=0.0001,
                                                decimalScale=4,
                                                size="xs",
                                                style={"width": "100%"},
                                            ),
                                            title="Energy Width Cutoff",
                                            description=(
                                                "Energy difference threshold (ΔE). Only states within this energy delta "
                                                "are evaluated for swapping at any point."
                                            ),
                                            config_name="eWidth",
                                            default_val="None (unlimited)",
                                            style={"width": "100%"},
                                            position="top",
                                        ),
                                    ],
                                ),
                                create_tooltip(
                                    dmc.NumberInput(
                                        id="input-max-state-repeat",
                                        label="Max State Repeat",
                                        description="Consecutive passes without changes before skipping",
                                        value=-1,
                                        min=-1,
                                        step=1,
                                        size="xs",
                                        style={"width": "100%"},
                                    ),
                                    title="Max State Consecutive Repeats",
                                    description=(
                                        "Maximum number of consecutive sweep passes a state can remain unchanged without being skipped in "
                                        "subsequent passes. Default is -1 (unlimited). Setting a threshold accelerates scans once low-lying states stabilize."
                                    ),
                                    config_name="maxStateRepeat",
                                    default_val="-1 (unlimited)",
                                    style={"width": "100%"},
                                    position="left",
                                ),
                                create_tooltip(
                                    dmc.MultiSelect(
                                        id="select-prop-list",
                                        label="Active Properties for Continuity",
                                        description="Filter properties used in finite-difference metric",
                                        data=prop_filter_data,
                                        value=prop_filter_default,
                                        clearable=True,
                                        size="xs",
                                        style={"width": "100%"},
                                    ),
                                    title="Active Continuity Properties List",
                                    description=(
                                        "Specifies which property indices are used to compute the finite-difference metric in getMetric. "
                                        "Other properties remain associated with each state but are excluded from reordering calculations. "
                                        "If left empty, all properties are used."
                                    ),
                                    config_name="propList",
                                    default_val="None (all properties)",
                                    style={"width": "100%"},
                                    position="left",
                                ),
                            ]),
                        ]),
                    ], value="constraints"),
                ],
            ),
        ],
    )
