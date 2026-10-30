"""
Data loading dialog supporting drag-and-drop uploads, local file path picking,
format inspection, and state count auto-configuration, enhanced with comprehensive tooltips.
"""

import dash_mantine_components as dmc
from dash import dcc, html
from suave_gui.components.tooltip_helper import create_tooltip


def create_file_modal() -> dmc.Modal:
    """Creates the data loader modal component with rich tooltips."""
    return dmc.Modal(
        id="file-load-modal",
        title=dmc.Text("Load State Dataset", fw=700, size="lg"),
        opened=False,
        size="lg",
        children=[
            dmc.Stack([
                dmc.Text(
                    "Upload a CSV matrix or provide a local path on the server. "
                    "The file should contain reaction coordinates, energies, and properties.",
                    size="sm",
                    c="dimmed",
                ),

                # Drag and drop upload zone
                create_tooltip(
                    dcc.Upload(
                        id="modal-file-upload",
                        children=dmc.Paper(
                            p="xl",
                            withBorder=True,
                            style={
                                "borderStyle": "dashed",
                                "textAlign": "center",
                                "cursor": "pointer",
                                "backgroundColor": "rgba(39, 39, 42, 0.4)",
                            },
                            children=[
                                dmc.Text("Drag and Drop CSV File Here or Click to Browse", fw=500),
                                dmc.Text("Supports .csv, .txt, .dat matrices", size="xs", c="dimmed"),
                            ],
                        ),
                        multiple=False,
                    ),
                    title="Upload Dataset Matrix",
                    description=(
                        "Drag and drop or browse for a matrix file containing reaction coordinates, state energies, "
                        "and properties. Column 1 must be the reaction coordinate, column 2 is the target energy, "
                        "and subsequent columns are physical properties."
                    ),
                    position="top",
                ),

                dmc.Divider(label="OR USE LOCAL FILE PATH", labelPosition="center"),

                create_tooltip(
                    dmc.TextInput(
                        id="modal-input-filepath",
                        label="Local File Path",
                        placeholder="input.csv",
                        value="input.csv",
                    ),
                    title="Server File Path",
                    description=(
                        "Path to an existing dataset file on the local machine or server filesystem. "
                        "Allows loading large datasets without re-uploading over the web browser."
                    ),
                    config_name="infile",
                    default_val="input.csv",
                    position="top",
                ),

                dmc.Group([
                    create_tooltip(
                        dmc.NumberInput(
                            id="modal-num-states",
                            label="Number of States",
                            description="States per coordinate point",
                            value=9,
                            min=1,
                            max=100,
                            step=1,
                            style={"width": "100%"},
                        ),
                        title="Number of Electronic States",
                        description=(
                            "The number of electronic states present at each reaction coordinate point. "
                            "The total number of rows in the input file must equal (numPoints × numStates)."
                        ),
                        config_name="numStates",
                        default_val="9",
                        style={"flex": 1},
                        position="top",
                    ),
                    create_tooltip(
                        dmc.TextInput(
                            id="modal-config-path",
                            label="Optional Config (JSON)",
                            placeholder="suave.json",
                            value="suave.json",
                            style={"width": "100%"},
                        ),
                        title="Configuration JSON Path",
                        description=(
                            "Optional path to a JSON configuration file specifying custom algorithmic parameters "
                            "(e.g., stencil orders, width, futurePnts, redundantSwaps, energy windows)."
                        ),
                        config_name="configPath",
                        default_val="suave.json",
                        style={"flex": 1},
                        position="top",
                    ),
                ]),

                # Inspection feedback / alert
                html.Div(id="modal-inspection-alert"),

                dmc.Group(
                    justify="flex-end",
                    children=[
                        create_tooltip(
                            dmc.Button("Cancel", id="btn-modal-cancel", variant="subtle", color="gray"),
                            title="Cancel",
                            description="Close this dialog without loading any new dataset.",
                            position="top",
                        ),
                        create_tooltip(
                            dmc.Button("Load Dataset", id="btn-modal-confirm", color="blue"),
                            title="Confirm & Load Dataset",
                            description="Parses the specified file, initializes backend data structures, and updates the plot.",
                            position="top",
                        ),
                    ],
                ),
            ])
        ],
    )
