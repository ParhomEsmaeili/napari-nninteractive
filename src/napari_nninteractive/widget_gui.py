from pathlib import Path
from typing import Optional

from interaction_log import NullInteractionLog
from napari.layers import Image, Labels
from napari.viewer import Viewer
from napari_toolkit.containers import setup_vcollapsiblegroupbox, setup_vgroupbox, setup_vscrollarea
from napari_toolkit.widgets import (
    setup_acknowledgements,
    setup_checkbox,
    setup_combobox,
    setup_hswitch,
    setup_iconbutton,
    setup_label,
    setup_layerselect,
    setup_lineedit,
    setup_pushbutton,
    setup_spinbox,
    setup_vswitch,
)
from napari_toolkit.widgets.buttons.icon_button import setup_icon
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QShortcut,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Each custom layer-controls widget (point_controls.py etc.) forces its layer into an
# interactive "add"-style mode the instant it becomes the active layer — a napari
# layer-level property, entirely independent of any Qt button here. Disabling the
# buttons alone doesn't stop a click from visually registering on the layer; the gate
# has to touch this too.
_ADD_MODE_BY_INTERACTION_INDEX = {0: 'add', 1: 'add_rectangle', 2: 'paint', 3: 'add_polygon_lasso'}
_NEUTRAL_MODE = 'pan_zoom'

# Dev convenience — a checkpoint copy gitignored into the repo itself (checkpoints/,
# see .gitignore) so Model Selection has a working default instead of an empty field on
# every fresh launch. Repo-relative, not an absolute machine path, so it degrades to no
# default (falls back to the HF-download combobox path) on a clone without it, rather
# than pointing at a path that doesn't exist there.
_BUNDLED_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "nnInteractive_v1.0"


class BaseGUI(QWidget):
    """
    A base GUI class for building the Base GUI and connect the components with the correct functions.

    Args:
        viewer (Viewer): The Napari viewer instance to connect with the GUI.
        parent (Optional[QWidget], optional): The parent widget. Defaults to None.
    """

    def __init__(self, viewer: Viewer, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._width = 300
        self.setMinimumWidth(self._width)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self._viewer = viewer
        self.session_cfg = None

        # Default — no preset selected yet. Set here (not in the nnInteractiveWidget
        # subclass) because _gate_idle() needs it to exist the moment _unlock_session()
        # first runs, below, before the subclass's own __init__ continues.
        self.interaction_log = NullInteractionLog()

        _main_layout = QVBoxLayout()
        self.setLayout(_main_layout)

        _scroll_widget, _scroll_layout = setup_vscrollarea(_main_layout)

        _scroll_layout.addWidget(self._init_model_selection())  # Model Selection
        _scroll_layout.addWidget(self._init_config_selection())  # Config Selection
        _scroll_layout.addWidget(self._init_preset_selection())  # Preset Selection
        _scroll_layout.addWidget(self._init_case_selection())  # Case Selection
        _scroll_layout.addWidget(self._init_image_selection())  # Image Selection
        _scroll_layout.addWidget(self._init_control_buttons())  # Init and Reset Button
        _scroll_layout.addWidget(self._init_init_buttons())  # Init and Reset Button
        _scroll_layout.addWidget(self._init_prompt_selection())  # Prompt Selection
        _scroll_layout.addWidget(self._init_interaction_selection())  # Interaction Selection
        _scroll_layout.addWidget(self._init_run_button())  # Run Button
        _scroll_layout.addWidget(self._init_export_button())  # Run Button

        _ = setup_acknowledgements(_scroll_layout, width=self._width)  # Acknowledgements

        self._unlock_session()
        self._viewer.bind_key("Ctrl+Q", self._close, overwrite=True)

    # Base Behaviour
    def _close(self):
        """Closes the viewer and quits the application."""
        self._viewer.close()
        quit()

    def _unlock_session(self):
        """Unlocks the session, enabling model and image selection, and initializing controls."""
        self.resume_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.instance_aggregation_ckbx.setEnabled(False)
        self.prompt_button.setEnabled(False)
        self.interaction_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.run_ckbx.setEnabled(False)
        self.export_button.setEnabled(False)
        self.reset_interaction_button.setEnabled(False)
        self.reset_pending_button.setEnabled(False)
        self.propagate_ckbx.setEnabled(False)
        self.label_for_init.setEnabled(False)
        self.class_for_init.setEnabled(False)
        self.auto_refine.setEnabled(False)
        # self.empty_mask_btn.setEnabled(False)
        self.load_mask_btn.setEnabled(False)
        self.add_button.setEnabled(False)
        self.add_ckbx.setEnabled(False)

        # No active session at all — definitely no open timing window either.
        self._gate_idle()
        # _gate_idle() above assumes an in-progress case exists that needs Complete/Abandon
        # before switching — true once a case is actually loaded, but not at true bootstrap
        # (nothing loaded yet, or a prior session just got invalidated by re-browsing
        # config). Override back to the correct bootstrap values: nothing to complete/
        # abandon yet, and case selection stays locked until a preset is chosen — picking a
        # case before a preset is active would leave that case's prompts untimed and
        # undiscarded once the preset does lock in. on_preset_selected() is what opens it up.
        self._set_case_controls_enabled(False)
        self.complete_button.setEnabled(False)
        self.abandon_button.setEnabled(False)

    def _lock_session(self):
        """Locks the session, disabling model and image selection, and enabling control buttons."""
        self.reset_button.setEnabled(True)
        self.instance_aggregation_ckbx.setEnabled(True)
        self.prompt_button.setEnabled(True)
        self.interaction_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.run_ckbx.setEnabled(True)
        self.export_button.setEnabled(True)
        self.reset_interaction_button.setEnabled(True)
        self.reset_pending_button.setEnabled(True)
        self.propagate_ckbx.setEnabled(True)
        self.label_for_init.setEnabled(True)
        self.class_for_init.setEnabled(True)
        self.auto_refine.setEnabled(True)
        # self.empty_mask_btn.setEnabled(True)
        self.load_mask_btn.setEnabled(True)
        self.add_button.setEnabled(True)
        self.add_ckbx.setEnabled(True)

        # A freshly-locked session (new image/case just loaded) hasn't started a timing
        # window yet, and hasn't declared an outcome for it either — reset to idle.
        self._gate_idle()

    # Timing gate — three states, strict cycle: idle -> window_open -> (repeat, or)
    # case_done -> idle (for the next object/case). Each state enables exactly the
    # controls valid to press from it; everything else is unreachable by construction.

    def _gate_idle(self):
        """Fresh — no window open, no outcome declared yet for the current object/case.
        Can Resume (start a window). Abandon is reachable directly too — skipping a case
        with zero interactions is a real, diagnosable outcome. Complete is not reachable
        until at least one Resume->Run window has actually completed (see
        InteractionLog.has_recorded_window) — an empty "completed" object doesn't mean
        anything, unlike an empty abandon. Can't move on yet either way — Next
        Object/case switching require an outcome first.

        With no preset selected (NullInteractionLog), everything below still locks down
        the same way — the only difference is Resume/Complete/Abandon themselves also
        stay disabled, since there's no timing window to open or outcome to declare
        against. Nothing that unlocks the prompt tools should be reachable without an
        active preset (matches the Case Selection hard-gate — see
        nninteractive-implementation-handoff.md); this used to be a no-op here, which let
        Open Case (which auto-chains into on_init(), same as CLoPA's Load Image) unlock
        the prompt tools with no preset active.

        Browse Config is also gated here (see nninteractive-implementation-handoff.md's
        "Re-browsing Config" fix): safe to browse a new config exactly when there's no
        live, undeclared object in progress — no preset active, or no case ever opened
        yet (current_case_id is None). Not safe once a case is open with no outcome
        declared — that's the unsafe half of idle, same reasoning _gate_window_open()
        and _gate_case_done() apply unconditionally below."""
        self.run_button.setEnabled(False)
        self.interaction_button.setEnabled(False)
        self.prompt_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.add_ckbx.setEnabled(False)
        self.reset_button.setEnabled(False)
        self._set_case_controls_enabled(False)
        self._set_active_prompt_layer_mode(_NEUTRAL_MODE)
        if isinstance(self.interaction_log, NullInteractionLog):
            self.resume_button.setEnabled(False)
            self.complete_button.setEnabled(False)
            self.abandon_button.setEnabled(False)
            self.browse_config_button.setEnabled(True)
            return
        self.resume_button.setEnabled(True)
        self.complete_button.setEnabled(self.interaction_log.has_recorded_window)
        self.abandon_button.setEnabled(True)
        self.browse_config_button.setEnabled(getattr(self, 'current_case_id', None) is None)

    def _gate_window_open(self):
        """Timing window open (Resume clicked, not yet closed via Run). Only Run and the
        prompt tools are usable — can't leave a window dangling open."""
        self.resume_button.setEnabled(False)
        self.run_button.setEnabled(True)
        self.interaction_button.setEnabled(True)
        self.prompt_button.setEnabled(True)
        self.add_button.setEnabled(True)
        self.add_ckbx.setEnabled(True)
        self.complete_button.setEnabled(False)
        self.abandon_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self._set_case_controls_enabled(False)
        self.browse_config_button.setEnabled(False)
        self._set_active_prompt_layer_mode(
            _ADD_MODE_BY_INTERACTION_INDEX.get(self.interaction_button.index)
        )

    def _gate_case_done(self):
        """Outcome just declared (Complete/Abandon pressed). Only Next Object/case
        switching are usable — forces an explicit move-on rather than silently
        continuing to place prompts or resuming again on a case just marked done."""
        self.resume_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.interaction_button.setEnabled(False)
        self.prompt_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.add_ckbx.setEnabled(False)
        self.complete_button.setEnabled(False)
        self.abandon_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.browse_config_button.setEnabled(True)
        self._set_case_controls_enabled(True)
        self._set_active_prompt_layer_mode(_NEUTRAL_MODE)

    def _set_active_prompt_layer_mode(self, mode):
        """Sets the currently-selected prompt layer's own napari mode — separate from
        (and untouched by) any of the Qt buttons above. mode=None is a no-op, so an
        unrecognized interaction index safely does nothing rather than raising.

        layer_dict doesn't exist yet the moment _gate_idle() now runs unconditionally
        during true bootstrap (BaseGUI.__init__ calls _unlock_session() before the
        LayerControls subclass's __init__ has set it up) — nothing to set a mode on
        yet either way, so no-op rather than raising."""
        if mode is None:
            return
        layer_dict = getattr(self, 'layer_dict', None)
        if layer_dict is None:
            return
        layer_name = layer_dict.get(self.interaction_button.index)
        if layer_name is not None and layer_name in self._viewer.layers:
            layer = self._viewer.layers[layer_name]
            layer.mode = mode
            # Setting .mode alone doesn't make this layer clickable: napari only routes
            # canvas clicks to whichever layer is the active/selected one, and that
            # selection is stolen by every layer add_label_layer() adds (Open Case's
            # fresh working layer, on_next()'s finished-object layer) — reclaim it here
            # so Resume reliably makes the right layer clickable without a manual
            # reselect in the layers panel.
            self._viewer.layers.selection.active = layer

    def _clear_layers(self):
        """Abstract function to clear all needed layers"""

    def _init_model_selection(self) -> QGroupBox:
        """Initializes the model selection as a combo box."""
        _group_box, _layout = setup_vgroupbox(text="Model Selection:")

        model_options = ["nnInteractive_v1.0"]

        self.model_selection = setup_combobox(
            _layout, options=model_options, function=self.on_model_selected
        )

        _boxlayout = QHBoxLayout()
        _layout.addLayout(_boxlayout)
        self.model_selection_local = setup_lineedit(
            _boxlayout,
            text=str(_BUNDLED_CHECKPOINT_DIR) if _BUNDLED_CHECKPOINT_DIR.is_dir() else None,
            placeholder="Use Local Checkpoint...",
            function=self.on_model_selected,
        )

        def _reset_local_ckpt_lineedit():
            self.model_selection_local.setText("")
            self.on_model_selected()

        btn = setup_iconbutton(
            _boxlayout, "", "delete_shape", self._viewer.theme, function=_reset_local_ckpt_lineedit
        )
        btn.setFixedWidth(30)

        _group_box.setLayout(_layout)
        return _group_box

    def _init_config_selection(self) -> QGroupBox:
        """Config JSON browse — reads export_napari_config.json's full_image_cache for a
        case list (consumed by _init_case_selection, below). Ignores checkpoint_path/
        episode entirely (no adaptation concept in nnInteractive)."""
        _group_box, _layout = setup_vgroupbox(text="Config (timing/experiment):")

        _boxlayout = QHBoxLayout()
        _layout.addLayout(_boxlayout)
        self.config_path_display = setup_lineedit(_boxlayout, placeholder="No config loaded...")
        self.config_path_display.setReadOnly(True)
        self.browse_config_button = setup_iconbutton(
            _boxlayout, "", "path", self._viewer.theme, function=self.on_browse_config
        )

        _group_box.setLayout(_layout)
        return _group_box

    def _set_case_controls_enabled(self, enabled: bool):
        """case_selection/open_case_button/prev_case_button/next_case_button are always
        toggled together — one place to keep them in sync rather than four call sites."""
        self.case_selection.setEnabled(enabled)
        self.open_case_button.setEnabled(enabled)
        self.prev_case_button.setEnabled(enabled)
        self.next_case_button.setEnabled(enabled)

    def _init_preset_selection(self) -> QGroupBox:
        """Initializes the preset selection group box: presets.json picker + preset
        dropdown. Disabled until a config is loaded — resolve_session() needs config_dir/
        config_basename, which only exist once on_browse_config() has run."""
        _group_box, _layout = setup_vgroupbox(text="Preset (timing/logging):")

        _path_layout = QHBoxLayout()
        self.presets_path_lineedit = setup_lineedit(
            _path_layout, placeholder="No presets.json loaded", readonly=True, stretch=3
        )
        self.browse_presets_button = setup_pushbutton(
            _path_layout, "Browse", function=self.on_browse_presets, stretch=1
        )
        _layout.addLayout(_path_layout)

        self.play_around_ckbx = setup_checkbox(
            _layout,
            "Play-around (exclude from timing)",
            False,
            tooltips="Familiarization session, not real timing data — writes to a "
            "separate _play_around.jsonl instead of the normal file, so timing "
            "analysis excludes it by construction. Set before picking a preset — "
            "locked once one's selected, same as the preset dropdown itself.",
        )

        _preset_row = QHBoxLayout()
        self.preset_selection = setup_combobox(
            _preset_row, options=[], placeholder="No preset selected", function=self.on_preset_selected, stretch=3
        )
        self.preset_description_button = setup_iconbutton(
            _preset_row,
            "",
            "info",
            self._viewer.theme,
            self.on_show_preset_description,
            tooltips="Show this preset's description",
            stretch=1,
        )
        _layout.addLayout(_preset_row)

        self.change_preset_button = setup_pushbutton(
            _layout, "Change Preset", function=self.on_change_preset,
        )

        self.browse_presets_button.setEnabled(False)
        self.preset_selection.setEnabled(False)
        self.preset_description_button.setEnabled(False)
        self.change_preset_button.setEnabled(False)
        self.play_around_ckbx.setEnabled(False)

        _group_box.setLayout(_layout)
        return _group_box

    def _init_case_selection(self) -> QGroupBox:
        """Case dropdown + Open Case/Prev/Next — separate group from config-browse
        (above) so Preset can sit between them, matching the required click order
        (Config -> Preset -> Case, not Config+Case together above Preset)."""
        _group_box, _layout = setup_vgroupbox(text="Case Selection:")

        self.case_selection = setup_combobox(_layout, options=[], function=None)

        _case_nav_row = QHBoxLayout()
        self.prev_case_button = setup_iconbutton(
            _case_nav_row, "Prev", "step_left", self._viewer.theme, self.on_prev_case,
            tooltips="Open the previous case in the list", stretch=1,
        )
        self.open_case_button = setup_iconbutton(
            _case_nav_row, "Open Case", "new_points", self._viewer.theme, function=self.on_open_case, stretch=2
        )
        self.next_case_button = setup_iconbutton(
            _case_nav_row, "Next", "step_right", self._viewer.theme, self.on_next_case,
            tooltips="Open the next case in the list", stretch=1,
        )
        _layout.addLayout(_case_nav_row)

        self.task_label = setup_label(_layout, "Target: —")

        _group_box.setLayout(_layout)
        return _group_box

    def _init_image_selection(self) -> QGroupBox:
        """Initializes the image selection combo box in a group box."""
        _group_box, _layout = setup_vgroupbox(text="Image Selection:")

        self.image_selection = setup_layerselect(
            _layout, viewer=self._viewer, layer_type=Image, function=self.on_image_selected
        )

        _group_box.setLayout(_layout)
        return _group_box

    def _init_control_buttons(self) -> QGroupBox:
        """Initializes the control buttons (Initialize and Reset)."""
        _group_box, _layout = setup_vgroupbox(text="")

        # No standalone Initialize button — Open Case is the only path in (auto-chains
        # into on_init(), same as CLoPA's Load Image). A manually-triggered Initialize
        # was only ever reachable at true bootstrap anyway (_lock_session() disables it
        # for the rest of the session the moment any real case is opened), and its only
        # distinct behavior — initializing on an arbitrary dropped-in image outside the
        # case catalog — is a backdoor around the same controlled-catalog design every
        # other gate in this plugin enforces. See nninteractive-implementation-handoff.md.

        self.reset_interaction_button = setup_iconbutton(
            _layout,
            "Reset Object",
            "delete",
            self._viewer.theme,
            self.on_reset_interactions,
            tooltips="Clear the current object entirely — interactions AND its predicted mask — and start it fresh on the next prediction. Model and Image Pair stay loaded. Use Reset Pending instead to just undo un-run interactions - press R",
            shortcut="R",
        )
        self.reset_pending_button = setup_iconbutton(
            _layout,
            "Reset Pending",
            "erase",
            self._viewer.theme,
            self.on_reset_pending_interactions,
            tooltips="Clear interactions placed since the last prediction — keeps the object's prediction history",
        )

        self.reset_button = setup_iconbutton(
            _layout,
            "Next Object",
            "step_right",
            self._viewer.theme,
            self.on_next,
            tooltips="Keep current segmentation and go to the next object - press M",
            shortcut="M",
        )

        self.instance_aggregation_ckbx = setup_checkbox(
            _layout,
            "Instance Aggregation",
            False,
            tooltips="If checked: Add all objects to a single layer. In the case of overlap newer objects overwrite older objects.\n"
            "Otherwise: Create a separate layer for each object. ",
        )

        # Timing/logging controls — explicit actions, not a passive toggle, so an outcome
        # can never be silently left stale from a previous case. Pressing either sets
        # InteractionLog's outcome and locks everything except Next Object/Case selector,
        # forcing an explicit move-on (see _gate_case_done()). Placement not final.
        self.complete_button = setup_iconbutton(
            _layout,
            "Complete",
            "check",
            self._viewer.theme,
            self.on_complete,
            tooltips="Mark this case/object as completed, then use Next Object or Open Case to move on",
        )
        self.abandon_button = setup_iconbutton(
            _layout,
            "Abandon",
            "warning",
            self._viewer.theme,
            self.on_abandon,
            tooltips="Mark this case/object as abandoned, then use Next Object or Open Case to move on",
        )

        _group_box.setLayout(_layout)
        return _group_box

    def _init_init_buttons(self):
        """Initializes the control buttons (Initialize and Reset)."""
        _group_box, _layout = setup_vcollapsiblegroupbox(
            text="Initialize with Segmentation:", collapsed=True
        )

        h_layout = QHBoxLayout()

        self.label_for_init = setup_layerselect(
            h_layout, viewer=self._viewer, layer_type=Labels, stretch=4
        )

        _text = setup_label(h_layout, "Class ID:", stretch=2)
        _text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        _text.setFixedWidth(70)
        self.class_for_init = setup_spinbox(h_layout, default=1, stretch=1)
        self.class_for_init.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

        _layout.addLayout(h_layout)

        self.load_mask_btn = setup_iconbutton(
            _layout,
            "Initialize with Mask",
            "logo_silhouette",
            self._viewer.theme,
            self.on_load_mask,
        )

        self.auto_refine = setup_checkbox(
            _layout, "Auto refine", False, tooltips="Auto Refine the Initial Mask"
        )

        _txt = setup_label(
            _layout, "<b>Warning:</b> This will reset all interactions<br>for the current object"
        )
        _group_box.setLayout(_layout)

        _group_box.setLayout(_layout)
        return _group_box

    def _init_prompt_selection(self) -> QGroupBox:
        """Initializes the prompt selection as switch with options and shortcuts."""
        _group_box, _layout = setup_vgroupbox(text="Prompt Type:")

        self.prompt_button = setup_hswitch(
            _layout,
            options=["positive", "negative"],
            function=self.on_prompt_selected,
            default=0,
            fixed_color="rgb(0,100, 167)",
            shortcut="T",
            tooltips="Press T to switch",
        )

        _group_box.setLayout(_layout)
        return _group_box

    def _init_interaction_selection(self) -> QGroupBox:
        """Initializes the interaction selection as switch with options and shortcuts."""
        _group_box, _layout = setup_vgroupbox(text="Interaction Tools:")

        self.interaction_button = setup_vswitch(
            _layout,
            options=["Point", "BBox", "Scribble", "Lasso"],
            function=self.on_interaction_selected,
            fixed_color="rgb(0,100, 167)",
        )

        setup_icon(self.interaction_button.buttons[0], "new_points", theme=self._viewer.theme)
        setup_icon(self.interaction_button.buttons[1], "rectangle", theme=self._viewer.theme)
        setup_icon(self.interaction_button.buttons[2], "paint", theme=self._viewer.theme)
        setup_icon(self.interaction_button.buttons[3], "polygon_lasso", theme=self._viewer.theme)

        self.propagate_ckbx = setup_checkbox(
            _layout,
            "Auto-zoom",
            True,
            function=self.on_propagate_ckbx,
        )

        for i, shortcut in enumerate(["P", "B", "S", "L"]):
            key = QShortcut(QKeySequence(shortcut), self.interaction_button.buttons[i])
            key.activated.connect(lambda idx=i: self.interaction_button._on_button_pressed(idx))
            self.interaction_button.buttons[i].setToolTip(f"press {shortcut}")

        _group_box.setLayout(_layout)
        return _group_box

    def _init_run_button(self) -> QGroupBox:
        """Initializes the run button and auto-run checkbox"""
        _group_box, _layout = setup_vcollapsiblegroupbox(text="Manual Control:", collapsed=True)

        # Timing/logging control — placement not final, just needs to exist for now.
        self.resume_button = setup_iconbutton(
            _layout,
            "Resume",
            "right_arrow",
            self._viewer.theme,
            self.on_resume,
            tooltips="Start a new timed window",
        )

        h_layout = QHBoxLayout()
        _layout.addLayout(h_layout)

        self.add_button = setup_iconbutton(
            h_layout,
            "Add Interaction",
            "add",
            self._viewer.theme,
            self.add_interaction,
            tooltips="add the current interaction",
        )
        self.run_button = setup_iconbutton(
            h_layout,
            "Run",
            "right_arrow",
            self._viewer.theme,
            self.on_run,
            tooltips="Run the predict step",
        )

        self.run_ckbx = setup_checkbox(
            _layout,
            "Auto Run Prediction",
            False,
            tooltips="Run automatically after each interaction",
        )

        self.add_ckbx = setup_checkbox(
            _layout,
            "Auto Add Interaction",
            True,
            tooltips="Add interaction automatically to session",
        )

        _group_box.setLayout(_layout)
        return _group_box

    def _init_export_button(self) -> QGroupBox:
        """Initializes the export button"""
        _group_box, _layout = setup_vgroupbox(text="")

        self.export_button = setup_iconbutton(
            _layout, "Export", "pop_out", self._viewer.theme, self._export
        )
        _group_box.setLayout(_layout)
        return _group_box

    # Event Handlers
    def on_init(self, *args, **kwargs) -> None:
        """Initializes the session configuration based on the selected model and image."""

    def on_image_selected(self):
        """When a new image is selected reset layers and session (cfg + gui)"""
        self._clear_layers()
        self._unlock_session()

    def on_model_selected(self):
        """When a new model is selected reset layers and session (cfg + gui)"""
        self._clear_layers()
        self._unlock_session()

    def on_reset_interactions(self):
        """Reset only the current interaction"""
        self._clear_layers()

    def on_next(self) -> None:
        """Resets the interactions."""
        print("_reset_interactions")

    def on_prompt_selected(self, *args, **kwargs) -> None:
        """Placeholder method for when a prompt type is selected"""
        print("on_prompt_selected", self.prompt_button.index, self.prompt_button.value)

    def on_interaction_selected(self, *args, **kwargs) -> None:
        """Placeholder method for when an interaction type is selected."""
        print(
            "on_interaction_selected", self.interaction_button.index, self.interaction_button.value
        )

    def on_run(self, *args, **kwargs) -> None:
        """Placeholder method for run operation"""
        print("on_run")

    def on_propagate_ckbx(self, *args, **kwargs):
        print("on_propagate_ckbx", *args, **kwargs)

    def on_load_mask(self):
        pass

    def add_mask_init_layer(self):
        pass

    def _export(self) -> None:
        """Placeholder method for exporting all generated label layers"""
