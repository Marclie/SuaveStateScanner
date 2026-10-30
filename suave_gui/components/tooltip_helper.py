"""
Tooltip helper utility for SuaveStateScanner UI components.
Constructs rich, descriptive Dash Mantine Components (dmc) Tooltips with
titles, detailed descriptions, configuration parameter mappings, and defaults
referenced directly from README.md.
"""

from typing import Any, Dict, List, Optional, Union
import dash_mantine_components as dmc
from dash import html


def create_tooltip(
    component: Any,
    title: str,
    description: str,
    config_name: Optional[str] = None,
    default_val: Optional[str] = None,
    width: int = 300,
    position: str = "top",
    style: Optional[Dict[str, Any]] = None,
    open_delay: int = 250,
) -> dmc.Tooltip:
    """
    Wraps a Dash Mantine component with an in-depth, styled tooltip.
    
    :param component: The inner UI element (button, switch, input, etc.)
    :param title: Bold headline title for the tooltip
    :param description: Comprehensive paragraph explaining what the element does
    :param config_name: Associated parameter in suave.json / README.md (e.g. 'redundantSwaps')
    :param default_val: Default value of the parameter (e.g. 'True')
    :param width: Maximum width of the tooltip popover in pixels
    :param position: Mantine tooltip placement ('top', 'bottom', 'left', 'right')
    :param style: Optional CSS dictionary passed to the Tooltip wrapper
    :param open_delay: Hover latency in milliseconds before opening
    :return: dmc.Tooltip component wrapping the child
    """
    label_children: List[Any] = [
        dmc.Text(title, fw=700, size="xs", c="white"),
        dmc.Text(
            description,
            size="xs",
            c="#D4D4D8",
            style={"lineHeight": 1.45, "marginTop": "3px"},
        ),
    ]

    meta_parts: List[str] = []
    if config_name:
        meta_parts.append(f"Config: {config_name}")
    if default_val is not None:
        meta_parts.append(f"Default: {default_val}")

    if meta_parts:
        label_children.append(
            dmc.Text(
                " • ".join(meta_parts),
                size="xs",
                c="#38BDF8",
                fs="italic",
                style={"marginTop": "5px", "fontWeight": 500},
            )
        )

    tooltip_style = style or {}
    box_props = None
    if tooltip_style.get("width") == "100%" or tooltip_style.get("flex") == 1:
        box_props = {"style": {"width": "100%"}}

    return dmc.Tooltip(
        component,
        label=html.Div(label_children),
        multiline=True,
        w=width,
        withArrow=True,
        arrowSize=6,
        arrowRadius=2,
        position=position,
        openDelay=open_delay,
        style=tooltip_style,
        boxWrapperProps=box_props,
        color="#27272A",
    )
