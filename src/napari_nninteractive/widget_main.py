import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any, Optional

import nnInteractive
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from napari.layers import Image
from napari.utils.notifications import show_warning
from napari.viewer import Viewer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from qtpy.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from napari_nninteractive.widget_controls import LayerControls
import presets
from interaction_log import InteractionLog, NullInteractionLog
from presets import load_presets, resolve_session

# presets.json always lives alongside the front-end-timing package itself (unlike
# configs, which vary per run) — derived from the installed package's own location so
# the browse dialog starts somewhere useful instead of the launch cwd.
PRESETS_DEFAULT_DIR = os.path.dirname(presets.__file__)

# Preset-application/control-locking. locked_controls is fully derived
# (presets.py's _derive_locked_controls()) from algorithm_config + prompt_type — see
# timing-package-plan.md Step 7c. Every entry maps to exactly one GUI control. Shared by
# on_preset_selected() (locks) and on_change_preset() (unlocks the same entries).
_PROMPT_TYPE_TO_BUTTON_INDEX = {'points': 0, 'bbox': 1, 'scribble': 2, 'lasso': 3}


def _get_code_version() -> str | None:
    """This plugin's own git commit hash — front-end-timing can't compute this itself
    (shared across plugins, each in a separate repo), so each widget supplies its own.
    See timing-package-plan.md Step 7e. None if git/the repo isn't available (not fatal —
    records just won't have this field populated)."""
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


class nnInteractiveWidget_(LayerControls):
    """Just a Debug Dummy without all the machine learning stuff"""


class nnInteractiveWidget(LayerControls):
    """
    A widget for the nnInteractive plugin in Napari that manages model inference sessions
    and allows interactive layer-based actions.
    """

    def __init__(self, viewer: Viewer, parent: Optional[QWidget] = None):
        """
        Initialize the nnInteractiveWidget.
        """
        super().__init__(viewer, parent)
        self.session = None
        self._viewer.dims.events.order.connect(self.on_axis_change)
        # Safety net for closing the window (X button / Ctrl+Q) without Finish & Close —
        # see _on_app_quit().
        QApplication.instance().aboutToQuit.connect(self._on_app_quit)
        # self.interaction_log defaults to NullInteractionLog — see BaseGUI.__init__.
        # Replaced with a real InteractionLog once a preset is selected (on_preset_selected()).

    # Timing/logging controls
    def on_resume(self):
        self.interaction_log.resume()
        self._gate_window_open()

    def on_complete(self):
        self.interaction_log.set_outcome("completed")
        self._gate_case_done()

    def on_abandon(self):
        self.interaction_log.set_outcome("abandoned")
        self._gate_case_done()

    def _finalize_current_object(self, case_id: str):
        """Every finalize_case() call site should go through here, not call it directly
        — stamps whatever's currently in the notes field (optional; '' if left blank)
        into the record, then clears the field so it doesn't leak into the next
        object/case."""
        self.interaction_log.set_notes(self.notes_lineedit.text())
        self.interaction_log.finalize_case(case_id)
        self.notes_lineedit.clear()

    def _flush_declared_outcome(self):
        """Writes the current object's record if an outcome was declared but not yet
        flushed (same id logic as leaving via Open Case). No-op otherwise, so calling it
        twice — e.g. Finish & Close, then the aboutToQuit safety net — writes once."""
        if getattr(self, "current_case_id", None) is None or not self.interaction_log.has_outcome:
            return
        if self._object_index_logged:
            self.object_index += 1
        self._finalize_current_object(f"{self.current_case_id}_obj{self.object_index}")
        self._object_index_logged = True

    def _on_app_quit(self):
        try:
            self._flush_declared_outcome()
        except Exception as e:
            warnings.warn(f"Could not write the pending timing record on exit: {e}")

    def on_finish_and_close(self):
        """Ends the session cleanly: flushes the declared outcome, then closes the viewer
        and quits. Refuses if the object has work (a recorded or open window) but no
        Complete/Abandon — the app never writes a null-outcome record."""
        log = self.interaction_log
        if not log.has_outcome and (log.has_recorded_window or log.has_open_window):
            show_warning(
                "Click Complete or Abandon first — this object has work with no outcome yet."
            )
            return
        self._flush_declared_outcome()
        self._viewer.close()
        QApplication.instance().quit()

    def on_browse_presets(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select presets.json", PRESETS_DEFAULT_DIR, "JSON files (*.json)"
        )
        if path == "":
            return
        self.presets_path = path
        self.presets_path_lineedit.setText(path)
        available_presets = load_presets(path)
        # presets.json is shared across every model this package supports, on purpose
        # (see presets.py's module docstring — the same condition compared across
        # models by design), so it isn't pre-filtered by model — selecting a preset
        # meant for a different plugin resolves fine (same dataset, wrong model) and
        # silently writes into that other model's JSONL file instead. nnInteractive
        # is always the one fixed model_registry key (no per-checkpoint versioning
        # like CLoPA's), so this is a plain exact-match filter.
        available_presets = {
            preset_id: preset for preset_id, preset in available_presets.items()
            if preset.get('model') == 'nninteractive'
        }
        self.preset_selection.clear()
        self.preset_selection.addItems(available_presets.keys())
        self.preset_selection.setEnabled(True)

    def on_preset_selected(self):
        preset_id = self.preset_selection.currentText()
        if preset_id == "":
            return

        # resolve_session()'s config_dir/config_basename — derived from the loaded
        # config's path, matching <config_dir>/timing/<config_basename>/<fe_experiment>/...
        config_run_dir = os.path.dirname(self.config_path)
        config_dir = os.path.dirname(config_run_dir)
        config_basename = os.path.basename(config_run_dir)
        config_dataset_name = self.dataset_level_schema['data_schema']['dataset_name']

        try:
            session_context = resolve_session(
                preset_id,
                self.presets_path,
                config_dir,
                config_basename,
                config_dataset_name=config_dataset_name,
                # Preset selection happens before Initialize in the normal flow (config ->
                # preset -> Initialize -> Open Case), so self.checkpoint_path (set by
                # widget_controls.py's on_init) may not exist yet. None until it does —
                # matches presets.py's own "None for plugins with nothing to log yet" case.
                checkpoint_path=getattr(self, "checkpoint_path", None),
                play_around=self.play_around_ckbx.isChecked(),
            )
        except ValueError as e:
            show_warning(str(e))
            self.preset_selection.setCurrentIndex(-1)
            return

        session_context.tags['code_version'] = _get_code_version()
        self.interaction_log = InteractionLog(
            save_path=session_context.save_path, tags=session_context.tags
        )

        for control in session_context.locked_controls:
            if control == 'autozoom':
                locked_value = session_context.tags['algorithm_config'].get('autozoom')
                self.propagate_ckbx.setChecked(locked_value == 'on')
                self.propagate_ckbx.setEnabled(False)
            elif control in _PROMPT_TYPE_TO_BUTTON_INDEX:
                self.interaction_button.buttons[_PROMPT_TYPE_TO_BUTTON_INDEX[control]].setEnabled(False)

        # Default to whichever interaction tool the preset actually permits, in case the
        # previously-selected one just got locked out above.
        allowed_index = _PROMPT_TYPE_TO_BUTTON_INDEX.get(session_context.tags['prompt_type'])
        if allowed_index is not None:
            self.interaction_button._uncheck()
            self.interaction_button._check(allowed_index)

        # Locked while active — on_change_preset() is the escape hatch, not direct
        # re-selection, so a switch always goes through its discard_object() safety step.
        self.preset_selection.setEnabled(False)
        self.browse_presets_button.setEnabled(False)
        self.play_around_ckbx.setEnabled(False)
        self.preset_description_button.setEnabled(True)
        self.change_preset_button.setEnabled(True)

        # Same bootstrap gap as _unlock_session(): _gate_idle() assumes a case is already
        # in progress (needs an outcome before switching), which isn't true yet here —
        # no case has been opened this session. Override back to bootstrap values.
        self._gate_idle()
        self._set_case_controls_enabled(True)
        self.complete_button.setEnabled(False)
        self.abandon_button.setEnabled(False)

    def on_show_preset_description(self):
        QMessageBox.information(
            self, "Preset Description", self.interaction_log.tags.get('description') or "(no description)"
        )

    def on_change_preset(self):
        # discard_object() wipes only genuinely in-flight, undeclared work — everything
        # already finalized to disk under the old preset's save_path is untouched, same
        # data-safety guarantee Reset Object already relies on. But Complete/Abandon may
        # already have declared an outcome for the current object without it having been
        # flushed yet (Next Object/Open Case are what normally flush it) — change_preset_button
        # stays enabled through case_done, so this is reachable. Discarding that would
        # silently drop the whole record instead of just losing in-flight edits, so finalize
        # it here instead, same as leaving via Open Case would.
        if self.interaction_log.has_outcome:
            if self._object_index_logged:
                self.object_index += 1
            self._finalize_current_object(f"{self.current_case_id}_obj{self.object_index}")
            self._object_index_logged = True
        else:
            self.interaction_log.discard_object()
            self.notes_lineedit.clear()

        # Same sweep as on_open_case() — a preset switch abandons whatever case was open
        # just as much as switching cases does.
        self._clear_layers()
        self._clear_finished_object_layers()

        # Reverse on_preset_selected()'s locking loop for whatever this preset locked.
        for control in self.interaction_log.tags.get('locked_controls', []):
            if control == 'autozoom':
                self.propagate_ckbx.setEnabled(True)
            elif control in _PROMPT_TYPE_TO_BUTTON_INDEX:
                self.interaction_button.buttons[_PROMPT_TYPE_TO_BUTTON_INDEX[control]].setEnabled(True)

        self.interaction_log = NullInteractionLog()
        # Avoids on_open_case() finalizing a stale outgoing case against a log that no
        # longer corresponds to it (a fresh Null, or a different preset picked next).
        self.current_case_id = None

        self.preset_selection.setCurrentIndex(-1)
        self.preset_selection.setEnabled(True)
        self.browse_presets_button.setEnabled(True)
        self.play_around_ckbx.setChecked(False)
        self.play_around_ckbx.setEnabled(True)
        self.preset_description_button.setEnabled(False)
        self.change_preset_button.setEnabled(False)

        # Back to the exact bootstrap state — _gate_idle()'s Null-guard branch only
        # touches resume/complete/abandon, so case controls need the same override
        # _unlock_session() uses: locked again until a preset is picked, same as bootstrap
        # (Case Selection hard-gates on an active preset — see
        # nninteractive-implementation-handoff.md).
        self._gate_idle()
        self._set_case_controls_enabled(False)

    # Event Handlers
    def on_init(self, *args, **kwargs):
        """
        Initialize the inference session and setup layers for interaction.

        This method sets up the nnInteractiveInferenceSession, loading from a
        pre-trained model folder and initializing properties based on the viewer layer.
        """
        super().on_init(*args, **kwargs)
        # checkpoint_path is only resolved by super().on_init() just above (model
        # selection happens at Initialize time, after preset selection in the normal
        # flow) — on_preset_selected() logged it as None since it didn't exist yet.
        # Patch it into the already-created log's tags now that it's known, before any
        # finalize_case() call happens.
        if not isinstance(self.interaction_log, NullInteractionLog):
            self.interaction_log.tags['checkpoint_path'] = self.checkpoint_path
        if self.session is None:
            # Get inference class from Checkpoint
            if Path(self.checkpoint_path).joinpath("inference_session_class.json").is_file():
                inference_class = load_json(
                    Path(self.checkpoint_path).joinpath("inference_session_class.json")
                )
                if isinstance(inference_class, dict):
                    inference_class = inference_class["inference_class"]
            else:
                inference_class = "nnInteractiveInferenceSession"

            inference_class = recursive_find_python_class(
                join(nnInteractive.__path__[0], "inference"),
                inference_class,
                "nnInteractive.inference",
            )

            # CPU Fallback if noc Cuda is available
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                show_warning(
                    "Cuda is not available. Using CPU instead. This will result in longer runtimes and additionally auto-zoom will be disabled for runtime reasons"
                )

                device = torch.device("cpu")
                self.propagate_ckbx.setChecked(False)

            # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

            # Initialize the Session
            self.session = inference_class(
                device=device,  # can also be cpu or mps. CPU not recommended
                use_torch_compile=False,
                torch_n_threads=os.cpu_count(),
                verbose=False,
                do_autozoom=self.propagate_ckbx.isChecked(),
            )

            self.session.initialize_from_trained_model_folder(
                self.checkpoint_path,
                0,
                "checkpoint_final.pth",
            )

        _data = np.array(self._viewer.layers[self.session_cfg["name"]].data)
        _data = _data[np.newaxis, ...]

        if self.source_cfg["ndim"] == 2:
            _data = _data[np.newaxis, ...]

        self.session.set_image(_data, {"spacing": self.session_cfg["spacing"]})
        self._snapshot_committed_interactions()

        self.session.set_target_buffer(self._data_result)

        if self._viewer.dims.not_displayed != ():
            self._scribble_brush_size = self.session.preferred_scribble_thickness[
                self._viewer.dims.not_displayed[0]
            ]
        else:
            self._scribble_brush_size = self.session.preferred_scribble_thickness[
                self._viewer.dims.order[0]
            ]
        # Set the prompt type to positive
        self.prompt_button._uncheck()
        self.prompt_button._check(0)

    def on_model_selected(self):
        """Reset the current session completely"""
        super().on_model_selected()  # _clear_layers() + _unlock_session()
        self.session = None
        # _unlock_session() above unconditionally re-locks Case Selection (bootstrap
        # default: no preset active yet) — wrong here whenever a preset already is
        # active, e.g. clearing the local-checkpoint field to switch to the HF-download
        # path mid-session. The case list comes entirely from the loaded config
        # (full_image_cache), structurally independent of which model/checkpoint is
        # selected, so switching checkpoints has no reason to touch it at all. Same
        # bootstrap-gap pattern as on_preset_selected()'s own override, mirrored here.
        if not isinstance(self.interaction_log, NullInteractionLog):
            self._set_case_controls_enabled(True)
            self.complete_button.setEnabled(False)
            self.abandon_button.setEnabled(False)

    def on_image_selected(self):
        """Deliberately does nothing. This fires on every change of which layer the
        Image Selection dropdown shows as current — not only genuine intent to switch
        the working image, e.g. an unrelated Image layer someone dropped into the
        viewer directly (this plugin never opens one itself for anything other than a
        case's own image, via on_open_case()). The old body (super().on_image_selected()
        -> _clear_layers() + _unlock_session(), plus session.reset_interactions() +
        _snapshot_committed_interactions() here) used to fire on that alone, silently
        wiping in-progress prompts/predictions and, once Case Selection hard-gates on an
        active preset, wrongly re-locking it too. Open Case (which auto-chains into
        on_init()) stays the one deliberate trigger for actually switching the working
        image and resetting everything around it."""

    def on_reset_interactions(self):
        """Reset only the current interaction"""
        _ind = self.interaction_button.index
        super().on_reset_interactions()
        if self.session is not None:
            self.session.reset_interactions()
            self._snapshot_committed_interactions()

        self._viewer.layers[self.label_layer_name].refresh()

        self.interaction_button._check(_ind)
        self.on_interaction_selected()
        # self.prompt_button._uncheck()
        self.prompt_button._on_button_pressed(0)
        # Hard reset for this object — deliberately ungated (unlike everything else), so
        # a mistaken Complete/Abandon can always be walked back. Closes any open window too.
        self.interaction_log.discard_object()
        self.notes_lineedit.clear()
        self._gate_idle()

    def on_reset_pending_interactions(self):
        # Only clears interactions placed since the last prediction — unlike
        # on_reset_interactions(), never touches self.session's committed history (no
        # forced re-init on next flush). Restores from the snapshot taken at the last
        # commit point rather than forcing has_positive_bbox False — see
        # [[concept/nninteractive-reset-pending]] for the desync edge case that guards
        # against (a committed positive bbox predicted at zoom 1 without refinement).
        #
        # The visual layers get the same commit-point restore, not a wipe: on_run() never
        # clears them (committed prompts stay visible across Run cycles, by the base
        # plugin's own design — see _snapshot_committed_interactions()'s docstring), so
        # wiping them here would remove already-committed prompts Reset Pending was never
        # supposed to touch. Trimmed via each layer's own remove_last() (not a raw `.data`
        # reassignment — see _snapshot_committed_interactions()'s docstring for why). A
        # layer with no snapshot entry didn't exist at commit time — removed entirely, same
        # as the old unconditional _clear_layers() would have done for it. Scribble is
        # excluded from counting (see _snapshot_committed_interactions()) and still gets the
        # old wipe-the-whole-layer treatment.
        _ind = self.interaction_button.index
        _committed_layer_counts = getattr(self, "_committed_layer_counts", {})
        for layer_name in self.layer_dict.values():
            if layer_name not in self._viewer.layers:
                continue
            if layer_name == self.scribble_layer_name:
                self._viewer.layers.remove(layer_name)
                continue
            target_count = _committed_layer_counts.get(layer_name)
            if target_count is None:
                self._viewer.layers.remove(layer_name)
                continue
            layer = self._viewer.layers[layer_name]
            while len(layer.data) > target_count:
                layer.remove_last()
        if self.session is not None and getattr(self, "_committed_interactions", None) is not None:
            self.session.interactions = self._committed_interactions.clone()
            self.session.has_positive_bbox = self._committed_has_positive_bbox
            self.session.new_interaction_centers = []
            self.session.new_interaction_zoom_out_factors = []
        self.interaction_button._check(_ind)
        self.on_interaction_selected()
        # Discards the whole current window (not just the pending prompts) — see
        # discard_pending()'s docstring — so this also needs a fresh Resume, same as
        # on_reset_interactions().
        self.interaction_log.discard_pending()
        # Reset Pending is deliberately reachable in every gate state (no gate method
        # touches reset_pending_button — see nninteractive-implementation-handoff.md's
        # Revisit checklist). But case_done means an outcome was already declared and is
        # waiting on Next Object/Open Case — unconditionally re-gating to idle here would
        # wrongly re-enable Resume/Complete/Abandon and let the same object be reopened
        # after it's already been marked done. No-op the gate in that state; idle/
        # window_open are the only states this should actually reset to idle from.
        if not self.interaction_log.has_outcome:
            self._gate_idle()

    def on_next(self):
        """Reset the Interactions of current session"""
        _ind = self.interaction_button.index
        super().on_next()
        if self.session is not None:
            self.session.reset_interactions()
            self._snapshot_committed_interactions()

        # if (
        #     self.use_init_ckbx.isChecked()
        #     and self.label_for_init.currentText() in self._viewer.layers
        # ):
        #     self.init_with_mask()

        self._viewer.layers[self.label_layer_name].refresh()

        self.interaction_button._check(_ind)
        self.on_interaction_selected()
        self.prompt_button._check(0)

        # Outcome was already set by on_complete()/on_abandon() — the gate (_gate_case_done())
        # guarantees Next Object is unreachable until one of those ran, so it's always set here.
        # object_index suffix disambiguates multiple objects within the same case — super()
        # already incremented it above, so this is "the object number just committed".
        self._finalize_current_object(f"{self.case_selection.currentText()}_obj{self.object_index}")
        # Marks this object_index value as spent — if the user leaves without another
        # Next Object, on_open_case()/on_change_preset() must bump past it rather than
        # reusing it for a different object (see _object_index_logged's own docstring).
        self._object_index_logged = True
        self._gate_idle()

    def _snapshot_committed_interactions(self):
        """Reset Pending's commit-point snapshot — plugin-side only, no change to the
        separate nnInteractive inference library (see [[concept/nninteractive-reset-pending]]).
        Called after every point at which self.session's interaction tensor becomes the
        new "committed" baseline: set_image (on_init), reset_interactions
        (on_image_selected/on_reset_interactions/on_next), and _predict (on_run in
        widget_controls.py, and the autorun branch of add_interaction() below, which
        flushes via _predict() internally without ever calling on_run()).

        set_image() offloads image preprocessing AND interaction-tensor initialization to
        a background thread and returns immediately — self.session.interactions is still
        None right after it returns. _finish_preprocessing_and_initialize_interactions()
        is the same wait every add_X_interaction() call does internally before touching
        the tensor; calling it here is a no-op once the tensor already exists (every
        other call site), so this is cheap everywhere except right after set_image().

        Also snapshots each point/bbox/lasso layer's item *count* at this same commit point
        (`_committed_layer_counts`) — Reset Pending trims back down to that count via the
        layer's own `remove_last()` (not a raw `.data` reassignment, which would desync each
        layer's parallel color/metadata bookkeeping — e.g. SinglePointLayer.point_colors —
        from the actual point count). This keeps already-committed prompts visible (matching
        on_run()'s own behavior of never touching the layers) while only pending ones vanish.
        A layer not present at commit time (not in this dict) means Reset Pending should
        remove it entirely — it didn't exist before the pending prompts created it.

        Deliberately excludes the scribble layer: ScribbleLayer.remove_last() is a Labels-
        layer undo() off its own history stack, not a count of discrete items — a commit-
        point item count doesn't apply to it the same way, so it's left to the old
        clear-the-whole-layer behavior (see on_reset_pending_interactions()) rather than risk
        a wrong partial-undo."""
        if self.session is None:
            return
        self.session._finish_preprocessing_and_initialize_interactions()
        if self.session.interactions is not None:
            self._committed_interactions = self.session.interactions.clone()
            self._committed_has_positive_bbox = self.session.has_positive_bbox
        self._committed_layer_counts = {
            layer_name: len(self._viewer.layers[layer_name].data)
            for layer_name in self.layer_dict.values()
            if layer_name in self._viewer.layers and layer_name != self.scribble_layer_name
        }

    def on_propagate_ckbx(self, *args, **kwargs):
        if self.session is not None:
            self.session.set_do_autozoom(self.propagate_ckbx.isChecked())

    def on_axis_change(self, event: Any):
        """Change the brush size of the scribble layer when the axis changes"""
        if self.session is not None:

            if self._viewer.dims.not_displayed != ():
                self._scribble_brush_size = self.session.preferred_scribble_thickness[
                    self._viewer.dims.not_displayed[0]
                ]
            else:
                self._scribble_brush_size = self.session.preferred_scribble_thickness[
                    self._viewer.dims.order[0]
                ]

            if self.scribble_layer_name in self._viewer.layers:
                self._viewer.layers[self.scribble_layer_name].brush_size = self._scribble_brush_size

    # Config / case loading — timing/experiment infrastructure, not part of upstream

    def on_browse_config(self):
        """Reads export_napari_config.json's full_image_cache for a case list. Ignores
        checkpoint_path/default_episode_number entirely — no adaptation/episode concept
        in nnInteractive. semantic_id_dict is read for display only, never passed to
        set_image()."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Config JSON", os.getcwd(), "JSON files (*.json)")
        if path == "":
            return

        # Dump the outgoing case, if any — same pattern as on_open_case()'s outgoing
        # finalize. Safe unconditionally: browse_config_button is only enabled (see
        # _gate_idle()/_gate_window_open()/_gate_case_done()) when there's no live,
        # undeclared object in progress — case_done (outcome already set) or no case
        # ever opened at all — so this can never write a None-outcome record.
        _outgoing_case_id = getattr(self, "current_case_id", None)
        if _outgoing_case_id is not None:
            self._finalize_current_object(f"{_outgoing_case_id}_obj{self.object_index}")

        # Full reset — browsing a new config invalidates everything tied to the old
        # one: the preset (locked_controls/save_path are specific to the old config's
        # directory), the case (may not exist in the new config's full_image_cache),
        # and the timing log itself. Mirrors on_change_preset()'s reset, since this is
        # a strictly bigger invalidation (a new config implies a new preset too).
        # Reverse on_preset_selected()'s locking loop for whatever the old preset
        # locked — has to read .tags before resetting to Null below (NullInteractionLog
        # has no .tags at all), and only if a real preset was actually active.
        if not isinstance(self.interaction_log, NullInteractionLog):
            for control in self.interaction_log.tags.get('locked_controls', []):
                if control == 'autozoom':
                    self.propagate_ckbx.setEnabled(True)
                elif control in _PROMPT_TYPE_TO_BUTTON_INDEX:
                    self.interaction_button.buttons[_PROMPT_TYPE_TO_BUTTON_INDEX[control]].setEnabled(True)

        self.interaction_log = NullInteractionLog()
        self.current_case_id = None
        self.preset_selection.setCurrentIndex(-1)
        self.preset_selection.setEnabled(True)
        self.play_around_ckbx.setChecked(False)
        self.play_around_ckbx.setEnabled(True)
        self.preset_description_button.setEnabled(False)
        self.change_preset_button.setEnabled(False)

        self.config_path = path
        with open(path, 'r') as f:
            config = json.load(f)
        self.dataset_level_schema = config['dataset_level_schema']
        self.full_image_cache = self.dataset_level_schema['full_image_cache']
        self.semantic_id_dict = self.dataset_level_schema['segmentation_task_schema']['semantic_id_dict']
        self.config_path_display.setText(path)
        self.case_selection.clear()
        self.case_selection.addItems(self.full_image_cache.keys())

        # Presets need config_dir/config_basename (see resolve_session()), which only
        # exist once config_path is set.
        self.browse_presets_button.setEnabled(True)
        self.play_around_ckbx.setEnabled(True)

    def on_prev_case(self):
        self._step_case(-1)

    def on_next_case(self):
        self._step_case(1)

    def _step_case(self, delta: int):
        _new_idx = self.case_selection.currentIndex() + delta
        if not (0 <= _new_idx < self.case_selection.count()):
            show_warning("No more cases in that direction.")
            return
        self.case_selection.setCurrentIndex(_new_idx)  # currentText() reflects this immediately
        self.on_open_case()

    def on_open_case(self):
        """Convenience layer on top of the existing flow — opens the resolved image path(s)
        into napari and pre-selects it in Image Selection. No hard-refuse gate, deliberately
        unlike CLoPA: manually opening arbitrary images outside the case list must keep
        working, nnInteractive has no concept to validate a case against anyway."""
        # Finalize the outgoing case, if any — read from current_case_id (stashed below on
        # the *previous* call), not case_selection.currentText(): the combobox already
        # reflects the newly-picked case the user selected before clicking Open Case, not
        # the one they're leaving. The gate (_gate_case_done()) guarantees an outcome was
        # already declared before Open Case became reachable, so this is always set here.
        _outgoing_case_id = getattr(self, "current_case_id", None)
        if _outgoing_case_id is not None:
            # If Next Object already spent this object_index on a different object,
            # this one needs a fresh id — otherwise it'd silently collide with that
            # earlier record (same case_obj id, only distinguishable by timestamp).
            if self._object_index_logged:
                self.object_index += 1
            self._finalize_current_object(f"{_outgoing_case_id}_obj{self.object_index}")
            self._object_index_logged = True

        case_id = self.case_selection.currentText()
        if case_id == "":
            return
        self.current_case_id = case_id

        # Sweep whatever the outgoing case left behind — stray interaction (point/bbox/
        # scribble/lasso) layers, and every finished-object layer on_next() ever created
        # this session — neither was cleared by anything else on a case switch. See
        # nninteractive-implementation-handoff.md.
        self._clear_layers()
        self._clear_finished_object_layers()

        case = self.full_image_cache[case_id]
        images = case['images']
        path = images.get('merged') or next(iter(images.values()))

        # Remove the previously-opened case's image layer, if any — Open Case means
        # switching to a case, not accumulating every case ever opened in the viewer.
        # Same fix as napari-clopa's on_open_case() (widget_main.py:241-249).
        old_layer_name = getattr(self, "current_case_layer_name", None)
        if (
            old_layer_name is not None
            and old_layer_name in self._viewer.layers
            and os.path.abspath(self._viewer.layers[old_layer_name].source.path or "") != os.path.abspath(path)
        ):
            self._viewer.layers.remove(old_layer_name)

        # Reuse an already-open layer for this exact file instead of opening a duplicate.
        # Restricted to Image layers — the Labels layer add_label_layer() creates below
        # deliberately copies the image's own source onto itself
        # (label_layer._source = self.session_cfg["source"], in widget_controls.py), so
        # an unfiltered search here wrongly matches the label layer instead of the real
        # image on a same-case reopen: the old image layer gets removed above, the label
        # layer's copied source.path makes this method think the image is "already
        # open" and returns the label layer's name instead of reopening the real image
        # — the image never reappears, and the "No Image Layer selected" check right at
        # the top of on_init() (widget_controls.py:219-220) then crashes.
        existing = next(
            (
                layer for layer in self._viewer.layers
                if isinstance(layer, Image)
                and os.path.abspath(layer.source.path or "") == os.path.abspath(path)
            ),
            None,
        )
        if existing is not None:
            layer_name = existing.name
        else:
            # Explicit plugin, matching napari-clopa's own on_open_case() style
            # (plugin='napari-clopa') rather than relying on auto-discovery. napari's own
            # open() default is the literal string 'napari', which restricts to builtin
            # readers only and silently skips napari-nifti for .nii.gz — nnInteractive has
            # no reader of its own, so this must name napari-nifti explicitly.
            layers = self._viewer.open(path, plugin='napari-nifti')
            layer_name = layers[0].name

        idx = self.image_selection.findText(layer_name)
        if idx >= 0:
            self.image_selection.setCurrentIndex(idx)
        self.current_case_layer_name = layer_name
        self.task_label.setText(
            f"Target: {list(self.semantic_id_dict.keys())} | Channels: {case.get('task_channels', 'unknown')}"
        )

        # Auto-chain into Initialize — after Open Case there's only one sensible next
        # step, same reasoning as CLoPA's on_open_case() auto-chaining into
        # on_load_image(). Cheap on every case after the first: on_init() only
        # reconstructs the model/session when self.session is still None, otherwise it
        # just re-points the existing session at this case's image via set_image().
        self.on_init()

    # Inference Behaviour

    def add_interaction(self):
        _index = self.interaction_button.index
        _layer_name = self.layer_dict.get(_index)
        if (
            _layer_name is not None
            and _layer_name in self._viewer.layers
            and not self._viewer.layers[_layer_name].is_free()
        ):
            data = self._viewer.layers[_layer_name].get_last()

            self._viewer.layers[_layer_name].run()
            # self.inference(_data, _index)

            if data is not None:
                _prompt = self.prompt_button.index == 0
                _auto_run = self.run_ckbx.isChecked()
                self.interaction_log.record_prompt()

                if _index == 0:
                    self._viewer.layers[self.point_layer_name].refresh(force=True)
                    self.session.add_point_interaction(data, _prompt, _auto_run)
                elif _index == 1:
                    # add_bbox_interaction expects [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
                    _min = np.min(data, axis=0)
                    _max = np.max(data, axis=0)
                    bbox = [[_min[0], _max[0]], [_min[1], _max[1]], [_min[2], _max[2]]]
                    self.session.add_bbox_interaction(bbox, _prompt, _auto_run)
                elif _index == 2:
                    self.session.add_scribble_interaction(data, _prompt, _auto_run)
                elif _index == 3:
                    self.session.add_lasso_interaction(data, _prompt, _auto_run)

                if _auto_run:
                    # Autorun flushes inside session.add_X_interaction() above, bypassing
                    # on_run() entirely — so this is the only place that sees an autorun
                    # flush complete and can commit the Reset Pending snapshot and clear
                    # the just-consumed on-screen prompts.
                    self._snapshot_committed_interactions()
                    self._clear_layers()
                    self.interaction_button._check(_index)
                    self.on_interaction_selected()
                    # Same reason: on_run()'s stop_window()/_gate_idle() never run for
                    # an autorun flush, since it never goes through on_run() at all.
                    self.interaction_log.stop_window()
                    self._gate_idle()
                self._viewer.layers[self.label_layer_name].refresh()

    def on_load_mask(self):

        _layer_data = self._viewer.layers[self.label_for_init.currentText()].data

        assert (
            _layer_data.shape == self.session_cfg["shape"]
        )  # Labels and Image should have same shape

        data = _layer_data == self.class_for_init.value()

        if np.any(data):
            if self.session is not None:
                self.session.add_initial_seg_interaction(
                    data.astype(np.uint8), run_prediction=self.auto_refine.isChecked()
                )
                self._viewer.layers[self.label_layer_name].refresh()
        else:
            warnings.warn("Mask is not valid - probably its empty", UserWarning, stacklevel=1)
