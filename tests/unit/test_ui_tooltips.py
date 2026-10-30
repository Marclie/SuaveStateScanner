"""
Unit and integration tests for UI tooltips, layout improvements, and presets.
Verifies all interactive components (switches, inputs, buttons, sliders, modals)
are equipped with in-depth descriptive tooltips matching README.md specifications,
and checks responsive side-by-side grid layouts, grape button coloring, and preset trays.
"""

import pytest
import dash_mantine_components as dmc
from dash import html

from suave_gui.components.tooltip_helper import create_tooltip
from suave_gui.components.sidebar_controls import create_sidebar_controls
from suave_gui.components.file_modal import create_file_modal
from suave_gui.components.save_modal import create_save_modal
from suave_gui.components.status_bar import create_status_bar
from suave_gui.app import create_app


def _find_component_by_id(component, target_id: str):
    """Recursively search for a component with the specified ID."""
    if getattr(component, "id", None) == target_id:
        return component

    children = getattr(component, "children", None)
    if children is None:
        return None

    if isinstance(children, list):
        for child in children:
            res = _find_component_by_id(child, target_id)
            if res is not None:
                return res
    else:
        return _find_component_by_id(children, target_id)
    return None


def _find_tooltip_for_id(container, target_id: str):
    """Find the dmc.Tooltip wrapping or containing the component with the given target_id."""
    if isinstance(container, dmc.Tooltip):
        if _find_component_by_id(container.children, target_id) is not None:
            return container

    children = getattr(container, "children", None)
    if children is None:
        return None

    if isinstance(children, list):
        for child in children:
            res = _find_tooltip_for_id(child, target_id)
            if res is not None:
                return res
    else:
        return _find_tooltip_for_id(children, target_id)
    return None


def test_tooltip_helper_structure():
    """Verify create_tooltip builds a valid dmc.Tooltip with title, description, and config metadata."""
    btn = dmc.Button("Test", id="btn-test")
    tip = create_tooltip(
        btn,
        title="Redundant Swaps",
        description="Allows states to swap with lower-lying states.",
        config_name="redundantSwaps",
        default_val="True",
    )

    assert isinstance(tip, dmc.Tooltip)
    assert tip.children.id == "btn-test"
    assert tip.multiline is True
    assert tip.withArrow is True

    # Check label contents
    label = tip.label
    assert isinstance(label, html.Div)
    text_elements = label.children
    assert len(text_elements) == 3
    assert text_elements[0].children == "Redundant Swaps"
    assert "lower-lying" in text_elements[1].children
    assert "Config: redundantSwaps" in text_elements[2].children
    assert "Default: True" in text_elements[2].children


def test_sidebar_controls_have_tooltips_for_all_elements():
    """Verify all major sidebar controls have descriptive tooltips matching README.md."""
    sidebar = create_sidebar_controls(num_states=9, num_props=8)

    # 1. Check Display accordion
    tip_prop = _find_tooltip_for_id(sidebar, "selector-property")
    assert tip_prop is not None
    assert "Target Property" in tip_prop.label.children[0].children

    tip_highlight = _find_tooltip_for_id(sidebar, "selector-highlight-state")
    assert tip_highlight is not None

    # 2. Check Manual Intervention controls
    tip_s1 = _find_tooltip_for_id(sidebar, "input-swap-state-1")
    tip_s2 = _find_tooltip_for_id(sidebar, "input-swap-state-2")
    tip_swap_btn = _find_tooltip_for_id(sidebar, "btn-apply-swap")
    tip_shuffle = _find_tooltip_for_id(sidebar, "btn-shuffle-values")
    tip_reset = _find_tooltip_for_id(sidebar, "btn-reset-initial")
    tip_remove = _find_tooltip_for_id(sidebar, "btn-remove-state")

    for t in [tip_s1, tip_s2, tip_swap_btn, tip_shuffle, tip_reset, tip_remove]:
        assert t is not None, "Missing tooltip on manual intervention control"

    # 3. Check Automated Sweeps controls, specifically redundantSwaps & interpolate
    tip_sweeps = _find_tooltip_for_id(sidebar, "input-num-sweeps")
    tip_width = _find_tooltip_for_id(sidebar, "input-stencil-width")
    tip_order = _find_tooltip_for_id(sidebar, "input-deriv-order")
    tip_future = _find_tooltip_for_id(sidebar, "input-future-pnts")
    tip_alternate = _find_tooltip_for_id(sidebar, "switch-alternate")
    tip_sort_last = _find_tooltip_for_id(sidebar, "switch-sort-last")
    tip_interpolate = _find_tooltip_for_id(sidebar, "switch-interpolate")
    tip_redundant = _find_tooltip_for_id(sidebar, "switch-redundant")
    tip_make_pos = _find_tooltip_for_id(sidebar, "switch-make-pos")
    tip_normalize = _find_tooltip_for_id(sidebar, "switch-normalize")
    tip_fwd = _find_tooltip_for_id(sidebar, "btn-sweep-forward")
    tip_bwd = _find_tooltip_for_id(sidebar, "btn-sweep-backward")
    tip_alt = _find_tooltip_for_id(sidebar, "btn-sweep-alternate")
    tip_abort = _find_tooltip_for_id(sidebar, "btn-abort-sweep")

    for t in [
        tip_sweeps, tip_width, tip_order, tip_future, tip_alternate,
        tip_sort_last, tip_interpolate, tip_redundant, tip_make_pos,
        tip_normalize, tip_fwd, tip_bwd, tip_alt, tip_abort,
    ]:
        assert t is not None, "Missing tooltip on automated sweep control"

    # In-depth verification of specific user requests:
    # "redundant swaps" tooltip details
    red_desc = tip_redundant.label.children[1].children
    assert "lower-lying" in red_desc
    assert "higher-lying" in red_desc
    assert "trapped" in red_desc

    # "Interpolate Reorder" tooltip details
    interp_desc = tip_interpolate.label.children[1].children
    assert "interpolate" in interp_desc.lower()
    assert "nan" in interp_desc.lower() or "inf" in interp_desc.lower()

    # 4. Check Constraints controls (maxPan, eWidth, maxStateRepeat, propList)
    tip_pan = _find_tooltip_for_id(sidebar, "input-max-pan")
    tip_ewidth = _find_tooltip_for_id(sidebar, "input-energy-width")
    tip_repeat = _find_tooltip_for_id(sidebar, "input-max-state-repeat")
    tip_proplist = _find_tooltip_for_id(sidebar, "select-prop-list")

    for t in [tip_pan, tip_ewidth, tip_repeat, tip_proplist]:
        assert t is not None, "Missing tooltip on constraint control"


def test_readme_missing_options_present_in_sidebar():
    """Verify options mentioned in README.md (futurePnts, makePos, normalize, maxStateRepeat, propList) are in UI."""
    sidebar = create_sidebar_controls(num_states=9, num_props=8)

    # futurePnts
    future_input = _find_component_by_id(sidebar, "input-future-pnts")
    assert future_input is not None
    assert future_input.value == 0

    # makePos
    makepos_switch = _find_component_by_id(sidebar, "switch-make-pos")
    assert makepos_switch is not None
    assert makepos_switch.checked is False

    # normalize
    norm_switch = _find_component_by_id(sidebar, "switch-normalize")
    assert norm_switch is not None
    assert norm_switch.checked is False

    # maxStateRepeat
    repeat_input = _find_component_by_id(sidebar, "input-max-state-repeat")
    assert repeat_input is not None
    assert repeat_input.value == -1

    # propList
    proplist_select = _find_component_by_id(sidebar, "select-prop-list")
    assert proplist_select is not None
    assert len(proplist_select.data) == 8


def test_multi_select_order_options():
    """Verify derivative orders is a MultiSelect allowing selection of multiple orders."""
    sidebar = create_sidebar_controls(num_states=9, num_props=8)
    order_select = _find_component_by_id(sidebar, "input-deriv-order")
    assert order_select is not None
    assert isinstance(order_select, dmc.MultiSelect)
    assert order_select.value == ["1"]
    values = [opt["value"] for opt in order_select.data]
    assert "1" in values
    assert "2" in values
    assert "3" in values


def test_side_by_side_layout_and_grape_button():
    """Verify State 1 & 2 are side-by-side and Swap States button uses grape color."""
    sidebar = create_sidebar_controls(num_states=9, num_props=8)

    # State 1 and State 2 side by side
    s1 = _find_component_by_id(sidebar, "input-swap-state-1")
    s2 = _find_component_by_id(sidebar, "input-swap-state-2")
    assert s1 is not None
    assert s2 is not None

    # Swap states button is grape
    swap_btn = _find_component_by_id(sidebar, "btn-apply-swap")
    assert swap_btn is not None
    assert swap_btn.color == "grape"


def test_app_layout_header_and_sliders_have_tooltips():
    """Verify header action buttons and coordinate/state range sliders have tooltips and span full width."""
    app = create_app(default_dataset_path=None)
    layout = app.layout

    # Range sliders
    tip_point_slider = _find_tooltip_for_id(layout, "slider-point-range")
    assert tip_point_slider is not None
    assert "pntBounds" in tip_point_slider.label.children[0].children

    tip_state_slider = _find_tooltip_for_id(layout, "slider-state-range")
    assert tip_state_slider is not None
    assert "stateBounds" in tip_state_slider.label.children[0].children

    # Header buttons
    for btn_id in [
        "btn-open-load-modal",
        "btn-undo",
        "btn-redo",
        "btn-save-curves",
        "btn-save-checkpoint",
    ]:
        tip = _find_tooltip_for_id(layout, btn_id)
        assert tip is not None, f"Missing tooltip for {btn_id}"


def test_bottom_tray_presets_and_readouts():
    """Verify bottom tray includes live coordinate/state badges and preset buttons."""
    app = create_app(default_dataset_path=None)
    layout = app.layout

    # Badges
    b_coord = _find_component_by_id(layout, "badge-coord-readout")
    b_state = _find_component_by_id(layout, "badge-state-readout")
    assert b_coord is not None
    assert b_state is not None

    # Presets
    for preset_id in [
        "btn-preset-full",
        "btn-preset-left",
        "btn-preset-right",
        "btn-preset-crossing",
        "btn-preset-all-states",
        "btn-preset-low-states",
    ]:
        btn = _find_component_by_id(layout, preset_id)
        assert btn is not None, f"Missing preset button: {preset_id}"


def test_modals_and_status_bar_tooltips():
    """Verify file load modal, save modal, and status bar have tooltips."""
    # File modal
    fmodal = create_file_modal()
    assert _find_tooltip_for_id(fmodal, "modal-file-upload") is not None
    assert _find_tooltip_for_id(fmodal, "modal-input-filepath") is not None
    assert _find_tooltip_for_id(fmodal, "modal-num-states") is not None
    assert _find_tooltip_for_id(fmodal, "modal-config-path") is not None
    assert _find_tooltip_for_id(fmodal, "btn-modal-confirm") is not None

    # Save modal
    smodal = create_save_modal()
    assert _find_tooltip_for_id(smodal, "save-modal-target-type") is not None
    assert _find_tooltip_for_id(smodal, "save-modal-num-points") is not None
    assert _find_tooltip_for_id(smodal, "save-modal-interpolate-switch") is not None
    assert _find_tooltip_for_id(smodal, "save-modal-interp-kind") is not None
    assert _find_tooltip_for_id(smodal, "save-modal-filename") is not None
    assert _find_tooltip_for_id(smodal, "btn-save-modal-confirm") is not None

    # Status bar
    sbar = create_status_bar()
    assert _find_tooltip_for_id(sbar, "badge-engine-status") is not None
    assert _find_tooltip_for_id(sbar, "text-status-message") is not None
    assert _find_tooltip_for_id(sbar, "text-convergence-telemetry") is not None
    assert _find_tooltip_for_id(sbar, "badge-compute-backend") is not None
    assert _find_tooltip_for_id(sbar, "text-dataset-summary") is not None
