"""Tk controller tests with isolated backend substitutes, no end-to-end inference."""

from copy import deepcopy
from pathlib import Path
import sys
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
from model_converter.schema import ConversionError
from model_studio.gui import ModelStudio
from model_studio.presentation import HELP, reduction, validation_view


def session_fixture():
    return SimpleNamespace(source=Path("fixture.pth"), state={"weight": torch.tensor([[1.0, -2.0]])},
                           changes=[], warnings=[], spec=SimpleNamespace(architecture="cnn"))


def report_fixture(session=None, **options):
    factor = 4 if options.get("precision") == "fp32" else 1
    return {"schema": "fixture", "supported": True, "deployment": {"precision": options.get("precision", "int8")},
            "summary": {"total_parameters": 2, "macs": 2, "deployed_parameter_bytes": 2 * factor,
                        "workspace_bytes_lower_bound": 4 * factor}, "warnings": ["fixture only"],
            "layers": [{"name": "layer", "op": "Linear", "input_shape": [1, 2], "output_shape": [1, 1],
                        "parameters": 2, "macs": 2, "weight_bytes": 2 * factor, "activation_bytes": 4}],
            "tensors": [{"name": "weight", "shape": [1, 2], "dtype": "float32", "elements": 2,
                         "bytes": 8, "zero_fraction": 0, "min": -2.0, "max": 1.0}]}


class GuiControllerTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.errors = []
        self.error_patch = patch("model_studio.gui.messagebox.showerror", side_effect=lambda *a, **k: self.errors.append(a))
        self.error_patch.start()
        self.app = ModelStudio(self.root)

    def tearDown(self):
        if hasattr(self, "app"):
            self.wait_idle()
            self.app.close()
        if hasattr(self, "error_patch"):
            self.error_patch.stop()

    def wait_idle(self):
        deadline = time.monotonic() + 10
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.root.update()
        self.assertFalse(self.app.busy, "Worker failed to return to idle")

    def load_fixture(self):
        self.app.model_path.set("fixture.pth")
        with patch("model_studio.gui.load_session", return_value=session_fixture()), \
             patch("model_studio.gui.analyze_session", side_effect=report_fixture):
            self.app.import_model()
            self.wait_idle()

    def test_async_load_renders_layers_and_tensor_selection(self):
        self.load_fixture()
        self.assertEqual(len(self.app.layers.get_children()), 1)
        self.assertEqual(len(self.app.tensors.get_children()), 1)
        self.assertEqual(self.app.card_values["macs"].get(), "2")
        self.app.tensors.selection_set("0")
        self.app.select_tensor()
        self.assertIn("-2.0", self.app.tensor_details.get("1.0", "end"))
        self.app.show_layer(0)
        self.assertIn("Linear", self.app.layer_details.get("1.0", "end"))

    def test_failed_import_clears_stale_graph_and_recovers(self):
        self.load_fixture()
        self.app.model_path.set("bad.pth")
        with patch("model_studio.gui.load_session", side_effect=ConversionError("bad checkpoint")):
            self.app.import_model()
            self.wait_idle()
        self.assertIsNone(self.app.current)
        self.assertFalse(self.app.layers.get_children())
        self.assertTrue(self.errors)
        self.assertEqual(self.app.card_values["macs"].get(), "—")
        self.load_fixture()
        self.assertIsNotNone(self.app.current)

    def test_import_configuration_changes_require_reload(self):
        self.load_fixture()
        self.app.input_length.set("64")
        with self.assertRaises(ConversionError):
            self.app._require_model()
        self.app.prune()
        self.assertFalse(self.app.busy)
        self.assertTrue(self.errors)

    def test_invalid_import_dimensions_clear_model_before_validation(self):
        self.load_fixture()
        self.app.validation = {"samples": 2}
        self.app.input_length.set("bad")
        self.app.import_model()
        self.assertIsNone(self.app.current)
        self.assertIsNone(self.app.analysis)
        self.assertIsNone(self.app.validation)
        self.assertFalse(self.app.layers.get_children())

    def test_render_labels_actual_report_precision_and_clears_details(self):
        self.load_fixture()
        self.app.precision.set("fp32")
        self.app.show_layer(0)
        self.app._render()
        self.assertIn("报告 int8", self.app.subtitle.cget("text"))
        self.assertIn("请更新估算", self.app.subtitle.cget("text"))
        self.assertEqual(self.app.layer_details.get("1.0", "end").strip(), "")

    def test_reanalysis_recomputes_baseline_with_selected_precision(self):
        self.load_fixture()
        self.app.precision.set("fp32")
        with patch("model_studio.gui.analyze_session", side_effect=report_fixture):
            self.app.reanalyze()
            self.wait_idle()
        self.assertEqual(self.app.analysis["summary"]["deployed_parameter_bytes"], 8)
        self.assertEqual(self.app.baseline["summary"]["deployed_parameter_bytes"], 8)

    def test_repeated_imports_do_not_accumulate_stale_rows_or_graph_items(self):
        self.load_fixture()
        graph_items = len(self.app.graph.find_all())
        for _ in range(8):
            self.load_fixture()
            self.assertEqual(len(self.app.layers.get_children()), 1)
            self.assertEqual(len(self.app.tensors.get_children()), 1)
            self.assertEqual(len(self.app.graph.find_all()), graph_items)
            self.assertTrue(self.app.events.empty())

    def test_pruning_keeps_original_and_invalidates_previous_validation(self):
        self.load_fixture()
        original = self.app.original
        candidate = deepcopy(original)
        candidate.changes = [{"method": "structured"}]
        self.app.validation = {"samples": 1}
        with patch("model_studio.gui.prune_session", return_value=(candidate, {"method": "structured"})), \
             patch("model_studio.gui.analyze_session", side_effect=report_fixture):
            self.app.prune()
            self.wait_idle()
        self.assertIs(self.app.original, original)
        self.assertIs(self.app.current, candidate)
        self.assertIsNone(self.app.validation)
        self.app.validation = {"samples": 1}
        self.app.scale.set("0.1")
        self.assertIsNone(self.app.validation)

    def test_compare_shows_table_and_no_raw_json(self):
        self.load_fixture()
        self.assertEqual(len(self.app.compare_table.get_children()), 4)
        values = self.app.compare_table.item("total_parameters", "values")
        self.assertEqual(values, ("可训练参数", "2", "2", "减少 0.00%"))
        self.assertNotIn("{", self.app.compare.get("1.0", "end"))
        self.assertFalse(self.app.diagnostics.winfo_manager())
        self.assertIn("尚未验证", self.app.verdict.get())

    def test_validation_summary_labels_loss_and_no_labels(self):
        result = {"samples": 10, "reference_accuracy": .9, "candidate_accuracy": .7,
                  "prediction_agreement": .8, "rmse": .001, "max_abs_error": .02, "elapsed_seconds": 1.2}
        self.app.validation = result
        self.app._render_validation()
        self.assertEqual(self.app.validation_values["candidate"].get(), "70.00%")
        self.assertEqual(self.app.validation_values["delta"].get(), "-20.00 个百分点")
        self.assertIn("下降", self.app.verdict.get())
        result["reference_accuracy"] = result["candidate_accuracy"] = None
        self.app._render_validation()
        self.assertEqual(self.app.validation_values["candidate"].get(), "未提供标签")
        self.assertIn("未知", self.app.verdict.get())

    def test_diagnostics_are_separate_and_history_bounded(self):
        for index in range(25):
            self.app.append_log(f"操作 {index}", detail={"raw_detail": index})
        self.assertEqual(len(self.app.log_events), 8)
        self.assertEqual(len(self.app.diagnostic_events), 20)
        self.assertNotIn("raw_detail", self.app.log.get("1.0", "end"))
        self.assertIn("raw_detail", self.app.diagnostics.get("1.0", "end"))
        self.app.show_diagnostics.set(True)
        self.app._toggle_diagnostics()
        self.assertEqual(self.app.diagnostics.winfo_manager(), "pack")

    def test_hover_and_keyboard_help_uses_current_quantization(self):
        tip = next(item for item in self.app.help_tips if callable(item.text))
        tip.show()
        self.assertIsNotNone(tip.window)
        self.assertIn("maxabs", tip.window.winfo_children()[0].cget("text"))
        self.app.scaling.set("mse")
        tip.show()
        self.assertIn("不保证分类准确率", tip.window.winfo_children()[0].cget("text"))
        tip.schedule()
        tip.hide()
        self.assertIsNone(tip.timer)
        self.assertIsNone(tip.window)

    def test_dataset_edit_invalidates_old_quality(self):
        self.load_fixture()
        self.app.validation = {"samples": 10}
        self.app.dataset.set("other.npz")
        self.assertIsNone(self.app.validation)
        self.assertIn("尚未验证", self.app.verdict.get())

    def test_taylor_requires_separate_calibration_before_worker(self):
        self.load_fixture()
        self.app.pruning.set("Taylor + 重建（需校准）")
        self.app.prune()
        self.assertFalse(self.app.busy)
        self.assertIn("校准", str(self.errors[-1]))
        self.app.calibration.set("same.npz")
        self.app.dataset.set("same.npz")
        self.app.prune()
        self.assertFalse(self.app.busy)
        self.assertIn("同一文件", str(self.errors[-1]))

    def test_metric_changes_handle_increases_and_zero(self):
        self.assertEqual(reduction(10, 12), "增加 20.00%")
        self.assertEqual(reduction(0, 0), "无变化")
        self.assertEqual(reduction(None, 1), "未知")
        self.assertEqual(validation_view({"samples": 1})["tone"], "pending")

    def test_imported_history_keeps_baseline_and_bad_paths_do_not_crash(self):
        self.load_fixture()
        self.app.current.metadata = {"studio_history": [{"method": "taylor", "calibration": {"path": 123}}]}
        self.app._render()
        self.assertIn("另含 1 条历史", self.app.compare.get("1.0", "end"))
        self.assertEqual(self.app.current.changes, [])
        self.assertFalse(self.app._same_dataset("test.npz", 123))
        self.assertTrue(self.app._same_dataset("test.npz", "test.npz"))

    def test_sidebar_children_share_wheel_binding(self):
        entry = next(widget for widget, state in self.app.controls
                     if widget.winfo_class() == "TEntry")
        self.assertIn("StudioSidebarWheel", entry.bindtags())
        with patch.object(self.app.side_canvas, "yview_scroll") as scroll:
            self.assertEqual(self.app._scroll_sidebar(SimpleNamespace(delta=-120)), "break")
            scroll.assert_called_once_with(1, "units")

    def test_sidebar_has_one_step_and_fixed_actions(self):
        self.assertEqual(self.app.current_step, "import")
        self.assertEqual([key for key, frame in self.app.side_pages.items() if frame.winfo_manager()], ["import"])
        action_parent = self.app.side_primary.master
        self.assertIs(action_parent.master, self.app.side_host)
        self.app._select_step("optimize")
        self.assertEqual([key for key, frame in self.app.side_pages.items() if frame.winfo_manager()], ["optimize"])
        self.assertEqual(self.app.side_primary.cget("text"), "生成优化候选")
        self.assertIs(self.app.side_primary.master, action_parent)
        self.app._select_step("export")
        self.assertIn("导出 C", self.app.side_primary.cget("text"))
        self.assertIn(".pth", self.app.side_secondary.cget("text"))

    def test_advanced_sections_collapse_and_preserve_values(self):
        toggle, content = self.app.advanced_sections["import"]
        self.assertFalse(content.winfo_manager())
        toggle.invoke()
        self.assertEqual(content.winfo_manager(), "pack")
        self.app.input_length.set("256")
        toggle.invoke()
        self.assertFalse(content.winfo_manager())
        self.app._select_step("export")
        self.app._select_step("import")
        self.assertEqual(self.app.input_length.get(), "256")

    def test_translated_choices_keep_backend_values_in_sync(self):
        self.app._select_step("export")
        scaling = next(widget for widget, state in self.app.controls
                       if hasattr(widget, "display_variable") and "最大绝对值" in widget.get())
        self.app.scaling.set("mse")
        self.assertIn("最小重建误差", scaling.get())
        scaling.display_variable.set("最大绝对值 · MaxAbs")
        scaling.event_generate("<<ComboboxSelected>>")
        self.root.update()
        self.assertEqual(self.app.scaling.get(), "maxabs")

    def test_data_driven_methods_reveal_calibration_only_when_needed(self):
        self.assertFalse(self.app.calibration_panel.winfo_manager())
        self.app.pruning.set("重建引导通道剪枝（CNN，需校准）")
        self.assertEqual(self.app.calibration_panel.winfo_manager(), "pack")
        self.assertIn("仅支持", self.app.method_hint.get())
        self.app.pruning.set("FPGM 冗余通道")
        self.assertFalse(self.app.calibration_panel.winfo_manager())

    def test_reconstruction_rejects_wrong_graph_and_calibration_limits(self):
        self.load_fixture()
        self.app.pruning.set("重建引导通道剪枝（CNN，需校准）")
        self.app.current.spec.architecture = "lstm"
        self.app.prune()
        self.assertFalse(self.app.busy)
        self.assertIn("仅支持", str(self.errors[-1]))
        self.app.current.spec.architecture = "cnn"
        self.app.calibration.set("calibration.npz")
        self.app.calibration_samples.set("129")
        self.app.prune()
        self.assertFalse(self.app.busy)
        self.assertIn("1～128", str(self.errors[-1]))

    def test_cache_selection_marks_resources_pending_and_recomputes_both(self):
        self.load_fixture()
        self.app.cache_policy.set("packed")
        self.assertIn("请更新估算", self.app.subtitle.cget("text"))
        def packed_report(session, **options):
            report = report_fixture(session, **options)
            report["deployment"].update(cache_policy=options["cache_policy"],
                                        weight_cache={"bytes": 512, "layers": []})
            report["summary"]["workspace_bytes_lower_bound"] += 512
            return report
        with patch("model_studio.gui.analyze_session", side_effect=packed_report) as analyze:
            self.app.reanalyze()
            self.wait_idle()
        self.assertEqual([call.kwargs["cache_policy"] for call in analyze.call_args_list], ["packed", "packed"])
        self.assertNotIn("请更新估算", self.app.subtitle.cget("text"))
        self.assertIn("512 B 已含在工作区", self.app.deployment_note.get())
        self.assertEqual(self.app.compare_table.item("workspace_bytes_lower_bound", "values")[1:3], ("516 B", "516 B"))

    def test_export_uses_current_candidate_and_cache_policy(self):
        self.load_fixture()
        candidate = deepcopy(self.app.current)
        candidate.changes = [{"method": "structured"}]
        self.app.current = candidate
        self.app.cache_policy.set("packed")
        result = {"directory": "export-fixture", "analysis": report_fixture(),
                  "manifest": {"quality_status": "FP32 only"}}
        with patch("model_studio.gui.filedialog.askdirectory", return_value="export-fixture"), \
             patch("model_studio.gui.export_session", return_value=result) as export, \
             patch("model_studio.gui.analyze_session", side_effect=report_fixture), \
             patch("model_studio.gui.messagebox.showinfo"):
            self.app.export_c()
            self.wait_idle()
        self.assertIs(export.call_args.args[0], candidate)
        self.assertEqual(export.call_args.kwargs["cache_policy"], "packed")

    def test_logs_start_collapsed_and_diagnostics_open_explicitly(self):
        self.assertFalse(self.app.activity_panel.winfo_manager())
        self.app.append_log("导入成功", detail={"source": "fixture.pth"})
        self.assertFalse(self.app.activity_panel.winfo_manager())
        self.assertIn("1 条", self.app.activity_title.get())
        self.app.show_diagnostics.set(True)
        self.app._toggle_diagnostics()
        self.assertEqual(self.app.activity_panel.winfo_manager(), "pack")
        self.assertTrue(self.app.show_activity.get())
        self.app.show_activity.set(False)
        self.app._toggle_activity()
        self.assertFalse(self.app.activity_panel.winfo_manager())
        self.assertFalse(self.app.show_diagnostics.get())

    def test_overview_is_brief_and_does_not_show_raw_warnings(self):
        self.load_fixture()
        self.app.current.warnings = ["raw-private-warning"]
        self.app._render()
        overview = self.app.details.get("1.0", "end")
        self.assertIn("理论计算量最大的层", overview)
        self.assertLess(len(overview), 350)
        self.assertNotIn("raw-private-warning", overview)
        self.assertEqual(self.app.benefit_values["macs"].get(), "减少 0.00%")


if __name__ == "__main__":
    unittest.main()
