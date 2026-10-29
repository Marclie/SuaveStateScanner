"""
Data loading dialog supporting drag-and-drop uploads, local file path picking,
format inspection, and state count auto-configuration.
"""

import dash_mantine_components as dmc
from dash import dcc, html


def create_file_modal() -> dmc.Modal:
    """Creates the data loader modal component."""
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

                dmc.Divider(label="OR USE LOCAL FILE PATH", labelPosition="center"),

                dmc.TextInput(
                    id="modal-input-filepath",
                    label="Local File Path",
                    placeholder="input.csv",
                    value="input.csv",
                ),

                dmc.Group([
                    dmc.NumberInput(
                        id="modal-num-states",
                        label="Number of States",
                        description="States per point",
                        value=9,
                        min=1,
                        max=100,
                        step=1,
                        style={"flex": 1},
                    ),
                    dmc.TextInput(
                        id="modal-config-path",
                        label="Optional Config (JSON)",
                        placeholder="suave.json",
                        value="suave.json",
                        style={"flex": 1},
                    ),
                ]),

                # Inspection feedback / alert
                html.Div(id="modal-inspection-alert"),

                dmc.Group(
                    justify="flex-end",
                    children=[
                        dmc.Button("Cancel", id="btn-modal-cancel", variant="subtle", color="gray"),
                        dmc.Button("Load Dataset", id="btn-modal-confirm", color="blue"),
                    ],
                ),
            ])
        ],
    )
