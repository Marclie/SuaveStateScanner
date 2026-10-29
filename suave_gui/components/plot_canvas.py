"""
Reactive Plotly figure canvas builder for multi-state curves.
Utilizes WebGL (Scattergl) acceleration, theme-aware styling, and persistent zoom state.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from suave_gui.core.dataset import StateDataset


class PlotCanvasBuilder:
    """Constructs optimized Plotly figures representing potential energy curves."""

    DARK_THEME = dict(
        paper_bgcolor="#18181B",
        plot_bgcolor="#202024",
        font=dict(color="#F4F4F5", family="Inter, system-ui, -apple-system, sans-serif", size=12),
        hovermode="closest",
        uirevision="constant",  # Keeps current user zoom/pan across data updates
        margin=dict(l=65, r=30, t=45, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(24, 24, 27, 0.75)",
            bordercolor="rgba(82, 82, 91, 0.3)",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            title=dict(text="Reaction Coordinate", font=dict(size=13, color="#E4E4E7")),
            gridcolor="rgba(82, 82, 91, 0.35)",
            zerolinecolor="rgba(82, 82, 91, 0.6)",
            showline=True,
            linecolor="rgba(82, 82, 91, 0.8)",
            tickfont=dict(color="#A1A1AA"),
        ),
        yaxis=dict(
            title=dict(text="Energy (a.u.)", font=dict(size=13, color="#E4E4E7")),
            gridcolor="rgba(82, 82, 91, 0.35)",
            zerolinecolor="rgba(82, 82, 91, 0.6)",
            showline=True,
            linecolor="rgba(82, 82, 91, 0.8)",
            tickfont=dict(color="#A1A1AA"),
        ),
    )

    # Distinct, accessible color cycle for electronic states
    COLOR_CYCLE = [
        "#38BDF8",  # Sky blue
        "#F87171",  # Soft Red
        "#34D399",  # Emerald green
        "#FBBF24",  # Amber
        "#A78BFA",  # Purple
        "#FB923C",  # Orange
        "#2DD4BF",  # Teal
        "#F472B6",  # Pink
        "#818CF8",  # Indigo
        "#A3E635",  # Lime
        "#E879F9",  # Fuchsia
        "#60A5FA",  # Blue
    ]

    @classmethod
    def build_empty_figure(cls, message: str = "No dataset loaded. Click 'Load File' to begin.") -> go.Figure:
        fig = go.Figure(layout=go.Layout(**cls.DARK_THEME))
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#A1A1AA"),
        )
        return fig

    @classmethod
    def build_figure(
        cls,
        dataset: Optional[StateDataset],
        property_idx: int = 0,
        highlight_state: Optional[int] = None,
        point_bounds: Optional[Tuple[int, int]] = None,
        state_bounds: Optional[Tuple[int, int]] = None,
        energy_bounds: Optional[Tuple[float, float]] = None,
        scan_point: Optional[int] = None,
    ) -> go.Figure:
        """
        Build Plotly figure from StateDataset.
        :param dataset: StateDataset instance
        :param property_idx: 0 for Energy, 1..N for properties
        :param highlight_state: state index to highlight (others dimmed)
        :param point_bounds: (min_pnt_idx, max_pnt_idx)
        :param state_bounds: (min_state_idx, max_state_idx)
        :param energy_bounds: (min_energy, max_energy)
        :param scan_point: current reaction coordinate point being scanned (draws red scan line)
        """
        if dataset is None:
            return cls.build_empty_figure()

        fig = go.Figure(layout=go.Layout(**cls.DARK_THEME))

        # Determine y-axis title and values
        if property_idx == 0:
            y_data = dataset.energies
            y_title = "Energy (a.u.)"
        else:
            p_idx = property_idx - 1
            if 0 <= p_idx < dataset.num_properties:
                y_data = dataset.properties[:, :, p_idx]
                y_title = f"Property {property_idx}"
            else:
                y_data = dataset.energies
                y_title = "Energy (a.u.)"

        fig.update_layout(yaxis_title=dict(text=y_title, font=dict(size=13, color="#E4E4E7")))

        p_start = point_bounds[0] if point_bounds else 0
        p_end = point_bounds[1] if point_bounds else dataset.num_points
        p_start = max(0, min(p_start, dataset.num_points - 1))
        p_end = max(p_start + 1, min(p_end, dataset.num_points))

        s_start = state_bounds[0] if state_bounds else 0
        s_end = state_bounds[1] if state_bounds else dataset.num_states
        s_start = max(0, min(s_start, dataset.num_states - 1))
        s_end = max(s_start + 1, min(s_end, dataset.num_states))

        x_coords = dataset.reaction_coordinates[p_start:p_end]

        for state in range(s_start, s_end):
            is_active = (highlight_state is None) or (highlight_state == state)
            opacity = 1.0 if is_active else 0.22
            width = 3.2 if (highlight_state == state) else (2.0 if highlight_state is None else 1.4)
            marker_size = 7 if (highlight_state == state) else (5 if highlight_state is None else 3.5)
            color = cls.COLOR_CYCLE[state % len(cls.COLOR_CYCLE)]

            y_slice = y_data[state, p_start:p_end]

            fig.add_trace(
                go.Scattergl(
                    x=x_coords,
                    y=y_slice,
                    mode="lines+markers",
                    name=f"State {state}",
                    line=dict(color=color, width=width),
                    marker=dict(size=marker_size, color=color),
                    opacity=opacity,
                    hovertemplate=(
                        f"<b>State {state}</b><br>"
                        "RC: %{x:.4f}<br>"
                        f"{y_title}: %{{y:.6f}}<extra></extra>"
                    ),
                )
            )

        # Dynamic red vertical scan indicator showing current scanning position
        if scan_point is not None and 0 <= scan_point < dataset.num_points:
            scan_x = float(dataset.reaction_coordinates[scan_point])
            fig.add_vline(
                x=scan_x,
                line_width=2.5,
                line_color="#EF4444",
                line_dash="solid",
                annotation_text=f"Scan: {scan_x:.4f}",
                annotation_position="top right",
                annotation_font=dict(color="#EF4444", size=11, family="Inter, sans-serif"),
            )

        return fig
