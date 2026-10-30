"""
Integration tests for single-node GUI-backend direct synchronization and requirements.
"""

import numpy as np
import pytest
from suave_state_scanner import SuaveStateScanner
from suave_gui.core.dataset import StateDataset
from suave_gui.components.sidebar_controls import create_sidebar_controls
from suave_gui.components.plot_canvas import PlotCanvasBuilder
from suave_gui.app import create_app


def test_direct_data_structure_sharing():
    """Verify GUI directly manipulates the backend data structures without copies."""
    suave = SuaveStateScanner("input.csv", "output_sync_test.csv", 9, "suave.json")
    dataset = StateDataset(
        reaction_coordinates=suave.allPnts,
        energies=suave.E,
        properties=suave.P,
        suave_instance=suave,
    )

    # 1. Identity assertion: arrays must be the exact same in-memory objects
    assert dataset.energies is suave.E
    assert dataset.properties is suave.P
    assert dataset.reaction_coordinates is suave.allPnts

    # 2. In-place manipulation via GUI: swap states 0 and 1 over points 10 to 30
    initial_e0_at_15 = float(suave.E[0, 15])
    initial_e1_at_15 = float(suave.E[1, 15])

    dataset.apply_swap(0, 1, point_start=10, point_end=30)

    # Changes made via GUI dataset MUST be reflected in backend suave object
    assert suave.E[0, 15] == initial_e1_at_15
    assert suave.E[1, 15] == initial_e0_at_15


def test_shuffle_points_sliding_window():
    """Verify shuffle points only affects the sliding-window region and leaves other points unchanged."""
    suave = SuaveStateScanner("input.csv", "output_sync_test.csv", 9, "suave.json")
    dataset = StateDataset(
        reaction_coordinates=suave.allPnts,
        energies=suave.E,
        properties=suave.P,
        suave_instance=suave,
    )

    e_before = suave.E.copy()
    p_before = suave.P.copy()

    # Shuffle only window [25, 75)
    p_start, p_end = 25, 75
    dataset.shuffle_points(point_start=p_start, point_end=p_end)

    # Points BEFORE window [0, 25) must remain 100% UNCHANGED
    assert np.array_equal(suave.E[:, :p_start], e_before[:, :p_start])
    assert np.array_equal(suave.P[:, :p_start, :], p_before[:, :p_start, :])

    # Points AFTER window [75, 500) must remain 100% UNCHANGED
    assert np.array_equal(suave.E[:, p_end:], e_before[:, p_end:])
    assert np.array_equal(suave.P[:, p_end:, :], p_before[:, p_end:, :])

    # Points WITHIN window [25, 75) should be permuted across states
    assert not np.array_equal(suave.E[:, p_start:p_end], e_before[:, p_start:p_end])


def test_property_dropdown_options():
    """Verify dropdown lists all available properties: Energy, Property 1, Property 2, Property 3, ..."""
    controls_3props = create_sidebar_controls(num_states=4, num_props=3)
    # Find selector-property component
    accordion = controls_3props.children[0]
    display_item = accordion.children[0]
    stack = display_item.children[1].children[0]
    prop_node = stack.children[0]
    prop_select = getattr(prop_node, "children", prop_node)

    assert prop_select.id == "selector-property"
    labels = [opt["label"] for opt in prop_select.data]
    values = [opt["value"] for opt in prop_select.data]

    assert labels == ["Energy", "Property 1", "Property 2", "Property 3"]
    assert values == ["0", "1", "2", "3"]

    # Verify for 8 properties
    controls_8props = create_sidebar_controls(num_states=9, num_props=8)
    accordion8 = controls_8props.children[0]
    prop_node8 = accordion8.children[0].children[1].children[0].children[0]
    prop_select8 = getattr(prop_node8, "children", prop_node8)
    labels8 = [opt["label"] for opt in prop_select8.data]
    assert len(labels8) == 9
    assert labels8[0] == "Energy"
    assert labels8[1] == "Property 1"
    assert labels8[8] == "Property 8"


def test_available_states_dropdown_options():
    """Verify state dropdown shows the full list of states instead of only 'all states'."""
    controls = create_sidebar_controls(num_states=9, num_props=8)
    accordion = controls.children[0]
    display_item = accordion.children[0]
    stack = display_item.children[1].children[0]
    state_node = stack.children[1]
    state_select = getattr(state_node, "children", state_node)

    assert state_select.id == "selector-highlight-state"
    labels = [opt["label"] for opt in state_select.data]
    values = [opt["value"] for opt in state_select.data]

    expected_labels = ["All States"] + [f"State {i}" for i in range(9)]
    expected_values = ["all"] + [str(i) for i in range(9)]

    assert labels == expected_labels
    assert values == expected_values


def test_plot_canvas_property_selection():
    """Verify plot canvas renders the selected property or energy."""
    suave = SuaveStateScanner("input.csv", "output_sync_test.csv", 9, "suave.json")
    dataset = StateDataset(suave.allPnts, suave.E, suave.P, suave_instance=suave)

    # Plot Energy (0)
    fig_energy = PlotCanvasBuilder.build_figure(dataset, property_idx=0)
    assert fig_energy.layout.yaxis.title.text == "Energy (a.u.)"
    assert np.allclose(fig_energy.data[0].y, suave.E[0, :])

    # Plot Property 1 (1)
    fig_prop1 = PlotCanvasBuilder.build_figure(dataset, property_idx=1)
    assert fig_prop1.layout.yaxis.title.text == "Property 1"
    assert np.allclose(fig_prop1.data[0].y, suave.P[0, :, 0])

    # Plot Property 3 (3)
    fig_prop3 = PlotCanvasBuilder.build_figure(dataset, property_idx=3)
    assert fig_prop3.layout.yaxis.title.text == "Property 3"
    assert np.allclose(fig_prop3.data[0].y, suave.P[0, :, 2])


def test_sweep_execution_updates_dataset_and_plot():
    """Verify backend sweep mutations (including prepareSweep array re-binding) update the GUI dataset and plot."""
    suave = SuaveStateScanner("input.csv", "output_sync_test.csv", 9, "suave.json")
    dataset = StateDataset(suave.allPnts, suave.E, suave.P, suave_instance=suave)

    initial_figure = PlotCanvasBuilder.build_figure(dataset, property_idx=0)
    initial_e_trace_0 = np.array(initial_figure.data[0].y)

    # 1. prepareSweep() re-binds suave.E and suave.P via copy.deepcopy
    initial_e_id = id(suave.E)
    lastEvals, lastPvals = suave.prepareSweep()
    new_e_id = id(suave.E)

    # Even though prepareSweep re-bound suave.E to a new numpy array object:
    assert dataset.energies is suave.E
    assert dataset.properties is suave.P
    assert dataset.reaction_coordinates is suave.allPnts

    # 2. Mutate state via backend sweepState or direct point modification
    # Swap state 0 and state 2 at point 50 in suave.E
    val_s0 = suave.E[0, 50]
    val_s2 = suave.E[2, 50]
    suave.E[[0, 2], 50] = suave.E[[2, 0], 50]

    # Verify dataset immediately reflects the change without any manual sync call
    assert dataset.energies[0, 50] == val_s2
    assert dataset.energies[2, 50] == val_s0

    # 3. Verify PlotCanvasBuilder extracts the modified data, NOT the stale initial data
    updated_figure = PlotCanvasBuilder.build_figure(dataset, property_idx=0)
    assert updated_figure.data[0].y[50] == val_s2
    assert updated_figure.data[2].y[50] == val_s0
    assert not np.array_equal(updated_figure.data[0].y, initial_e_trace_0)

    # 4. analyzeSweep re-binds suave.E again
    suave.analyzeSweep(lastEvals, lastPvals)
    assert dataset.energies is suave.E
    assert dataset.properties is suave.P

    # 5. Snapshot restore should propagate in-place to suave.E
    snap = dataset.create_snapshot("Test snapshot")
    # Mutate suave.E
    suave.E[0, 0] += 999.0
    assert dataset.energies[0, 0] == suave.E[0, 0]
    # Restore snapshot
    dataset.restore_snapshot(snap)
    assert suave.E[0, 0] == snap.energies[0, 0]
    assert dataset.energies[0, 0] == snap.energies[0, 0]


def test_plot_canvas_scan_line_and_active_state_highlight():
    """Verify red vertical scan indicator line and active state highlighting on the plot."""
    suave = SuaveStateScanner("input.csv", "output_sync_test.csv", 9, "suave.json")
    dataset = StateDataset(suave.allPnts, suave.E, suave.P, suave_instance=suave)

    # 1. Figure with scan point at index 25
    fig_scanning = PlotCanvasBuilder.build_figure(dataset, highlight_state=2, scan_point=25)
    expected_rc = float(dataset.reaction_coordinates[25])

    # Check shapes for vertical red line
    shapes = fig_scanning.layout.shapes
    assert len(shapes) >= 1
    scan_line = shapes[0]
    assert scan_line.line.color == "#EF4444"
    assert scan_line.x0 == expected_rc
    assert scan_line.x1 == expected_rc

    # Check that active state (state 2) is highlighted prominently while others are dimmed
    trace_active = fig_scanning.data[2]
    trace_inactive = fig_scanning.data[0]
    assert trace_active.opacity == 1.0
    assert trace_active.line.width == 3.2
    assert trace_inactive.opacity == 0.22
    assert trace_inactive.line.width == 1.4

    # 2. Figure without scan point (idle/completed) should not have vertical line shapes
    fig_idle = PlotCanvasBuilder.build_figure(dataset, highlight_state=None, scan_point=None)
    assert fig_idle.layout.shapes is None or len(fig_idle.layout.shapes) == 0
    # All states should have full opacity when no specific state is highlighted
    for trace in fig_idle.data:
        assert trace.opacity == 1.0
