"""Headless UI tests for the dockable NEAT assistant."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from NEAT.ui.assistant_panel import AssistantDockWidget, collect_neat_context
from tools.assistant_providers import AssistantProvider, AssistantSettings


class _TextWidget:
    def __init__(self, value: str) -> None:
        self.value = value

    def text(self) -> str:
        return self.value


class _ComboWidget:
    def __init__(self, value: str) -> None:
        self.value = value

    def currentText(self) -> str:
        return self.value


class _Tabs:
    def __init__(self, current_widget) -> None:
        self._current_widget = current_widget

    def currentIndex(self) -> int:
        return 1

    def tabText(self, index: int) -> str:
        return "Bragg Edge Fitting"

    def currentWidget(self):
        return self._current_widget


class _Signal:
    def __init__(self) -> None:
        self.callback = None

    def connect(self, callback) -> None:
        self.callback = callback

    def emit(self, value=None) -> None:
        if self.callback is None:
            return
        if value is None:
            self.callback()
        else:
            self.callback(value)


class _FakeWorker:
    last_context = None
    last_question = None

    def __init__(self, question, context, history, parent=None) -> None:
        self.question = question
        self.context = context
        self.history = history
        self.answer_ready = _Signal()
        self.request_failed = _Signal()
        self.finished = _Signal()
        self._running = False
        _FakeWorker.last_context = context
        _FakeWorker.last_question = question

    def start(self) -> None:
        self._running = True
        citation = SimpleNamespace(number=1, heading_path="FAQ > Macro-pixels")
        result = SimpleNamespace(
            answer="Use the documented macro-pixel guidance.",
            citations=[citation],
            route="parameter_explanation",
            requires_human_review=False,
        )
        self.answer_ready.emit(result)
        self._running = False
        self.finished.emit()

    def isRunning(self) -> bool:
        return self._running

    def deleteLater(self) -> None:
        pass


class AssistantPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_context_is_allow_listed_and_excludes_raw_data_and_paths(self) -> None:
        fitting_tab = object()
        window = SimpleNamespace(
            app_version="4.7.4",
            FittingTab=fitting_tab,
            tabs=_Tabs(fitting_tab),
            phase_dropdown=_ComboWidget("Iron_BCC"),
            min_wavelength_input=_TextWidget("2.1"),
            max_wavelength_input=_TextWidget("3.4"),
            box_width_input=_TextWidget("20"),
            box_height_input=_TextWidget("20"),
            current_d=2.026,
            fitting_data_source="images",
            fitting_mode="individual",
            images=["raw-data"],
            current_fitting_input="C:/private/experiment/file.fits",
            fix_s_enabled=lambda: False,
            fix_t_enabled=lambda: True,
            fix_eta_enabled=lambda: True,
        )
        context = collect_neat_context(window)
        self.assertEqual(context["current_module"], "Bragg Edge Fitting")
        self.assertEqual(context["phase"], "Iron_BCC")
        self.assertTrue(context["eta_fixed"])
        self.assertEqual(context["fitting_data_source"], "images")
        self.assertEqual(context["fitting_mode"], "individual")
        self.assertTrue(context["images_loaded"])
        self.assertNotIn("images", context)
        self.assertNotIn("current_fitting_input", context)
        self.assertFalse(any("private" in str(value) for value in context.values()))

    def test_panel_submits_in_background_contract_and_displays_sources(self) -> None:
        panel = AssistantDockWidget(
            context_provider=lambda: {"current_module": "Fitting"},
            worker_class=_FakeWorker,
            runtime_preparer=lambda: None,
        )
        panel.question_input.setPlainText("What is a macro-pixel?")
        panel.submit_question()
        rendered = panel.transcript.toPlainText()
        self.assertIn("What is a macro-pixel?", rendered)
        self.assertIn("Use the documented macro-pixel guidance.", rendered)
        self.assertIn("Verified NEAT sources", rendered)
        self.assertEqual(_FakeWorker.last_context, {"current_module": "Fitting"})
        self.assertEqual(len(panel.history), 2)
        panel.close()

    def test_semantic_startup_failure_still_uses_release_fallback(self) -> None:
        def unavailable_semantic_runtime():
            raise RuntimeError("offline")

        panel = AssistantDockWidget(
            context_provider=lambda: {"current_module": "Fitting"},
            worker_class=_FakeWorker,
            runtime_preparer=unavailable_semantic_runtime,
        )
        panel.question_input.setPlainText("What is a macro-pixel?")
        panel.submit_question()
        self.assertIn(
            "Use the documented macro-pixel guidance.",
            panel.transcript.toPlainText(),
        )
        panel.close()

    def test_local_provider_is_ready_without_api_key(self) -> None:
        repository = SimpleNamespace(
            load=lambda: AssistantSettings(
                provider=AssistantProvider.LOCAL,
                model="qwen3:8b",
                base_url="http://localhost:11434/v1",
            )
        )
        with patch(
            "tools.assistant_settings_store.AssistantSettingsRepository",
            return_value=repository,
        ):
            panel = AssistantDockWidget(
                context_provider=lambda: {},
                worker_class=_FakeWorker,
                runtime_preparer=lambda: None,
            )
        self.assertIn("Local model", panel.status_label.text())
        self.assertIn("qwen3:8b", panel.status_label.text())
        self.assertNotIn("No ", panel.status_label.text())
        panel.close()

    def test_explain_current_screen_submits_active_module_and_preserves_draft(self) -> None:
        panel = AssistantDockWidget(
            context_provider=lambda: {
                "neat_version": "4.7.4",
                "current_module": "Bragg Edge Fitting",
                "phase": "Iron_BCC",
            },
            worker_class=_FakeWorker,
            runtime_preparer=lambda: None,
        )
        panel.question_input.setPlainText("My unfinished draft")

        panel.explain_screen_button.click()

        self.assertIn("Bragg Edge Fitting", _FakeWorker.last_question)
        self.assertIn("main areas and controls", _FakeWorker.last_question)
        self.assertEqual(
            _FakeWorker.last_context["current_module"],
            "Bragg Edge Fitting",
        )
        self.assertEqual(
            panel.question_input.toPlainText(),
            "My unfinished draft",
        )
        self.assertIn("Explain the current NEAT screen", panel.transcript.toPlainText())
        panel.close()

    def test_feedback_buttons_log_only_allow_listed_answer_metadata(self) -> None:
        captured = []
        panel = AssistantDockWidget(
            context_provider=lambda: {
                "neat_version": "4.7.4",
                "current_module": "Fitting",
                "private_path": "C:/private/data.fits",
            },
            worker_class=_FakeWorker,
            runtime_preparer=lambda: None,
            feedback_logger=lambda **fields: captured.append(fields),
        )
        panel.question_input.setPlainText("What is a macro-pixel?")
        panel.submit_question()
        self.assertFalse(panel.feedback_widget.isHidden())

        panel.not_helpful_button.click()

        self.assertEqual(len(captured), 1)
        record = captured[0]
        self.assertEqual(record["rating"], "not_helpful")
        self.assertEqual(record["question"], "What is a macro-pixel?")
        self.assertEqual(record["route"], "parameter_explanation")
        self.assertEqual(record["neat_version"], "4.7.4")
        self.assertEqual(
            record["citations"],
            [
                {
                    "source_id": "",
                    "heading_path": "FAQ > Macro-pixels",
                }
            ],
        )
        self.assertNotIn("answer", record)
        self.assertNotIn("context", record)
        self.assertNotIn("private_path", str(record))
        self.assertFalse(panel.not_helpful_button.isEnabled())
        panel.close()

    def test_feedback_confirmation_displays_the_saved_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            saved_path = Path(directory) / "assistant_feedback.jsonl"
            panel = AssistantDockWidget(
                context_provider=lambda: {"neat_version": "4.7.4"},
                worker_class=_FakeWorker,
                runtime_preparer=lambda: None,
                feedback_logger=lambda **fields: saved_path,
            )
            panel.question_input.setPlainText("What is a macro-pixel?")
            panel.submit_question()
            panel.helpful_button.click()

            self.assertIn(str(saved_path), panel.status_label.text())
            self.assertEqual(panel.status_label.toolTip(), str(saved_path))
            panel.close()


if __name__ == "__main__":
    unittest.main()
