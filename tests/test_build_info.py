"""Source-checkout commit display and packaged-version regressions."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from NEAT import build_info
from NEAT.ui import main_window as main_window_module
from NEAT.ui.main_window import FitsViewer

from PyQt5.QtWidgets import QApplication, QMainWindow


class DevelopmentCommitTests(unittest.TestCase):
    def setUp(self):
        build_info.development_commit.cache_clear()
        self.addCleanup(build_info.development_commit.cache_clear)

    def test_git_checkout_revision_is_reported_and_cached(self):
        root = Path(build_info.__file__).resolve().parent.parent
        with patch.object(build_info.subprocess, "run") as run:
            run.side_effect = [
                subprocess.CompletedProcess([], 0, stdout=str(root) + "\n"),
                subprocess.CompletedProcess([], 0, stdout="a1b2c3d4\n"),
            ]
            self.assertEqual(build_info.development_commit(), "a1b2c3d4")
            self.assertEqual(build_info.development_commit(), "a1b2c3d4")
            self.assertEqual(run.call_count, 2)

    def test_unrelated_parent_checkout_is_not_misattributed(self):
        root = Path(build_info.__file__).resolve().parent.parent
        with patch.object(build_info.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess(
                [], 0, stdout=str(root.parent) + "\n"
            )
            self.assertIsNone(build_info.development_commit())
            run.assert_called_once()

    def test_missing_git_or_invalid_revision_does_not_prevent_launch(self):
        with patch.object(build_info.subprocess, "run", side_effect=FileNotFoundError):
            self.assertIsNone(build_info.development_commit())
        build_info.development_commit.cache_clear()
        root = Path(build_info.__file__).resolve().parent.parent
        with patch.object(build_info.subprocess, "run") as run:
            run.side_effect = [
                subprocess.CompletedProcess([], 0, stdout=str(root) + "\n"),
                subprocess.CompletedProcess([], 0, stdout="<invalid>\n"),
            ]
            self.assertIsNone(build_info.development_commit())

    def test_packaged_application_never_looks_up_git(self):
        with patch.object(build_info.sys, "frozen", True, create=True):
            with patch.object(build_info.subprocess, "run") as run:
                self.assertIsNone(build_info.development_commit())
                run.assert_not_called()


class AboutBuildInformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_about_dialog_displays_source_commit(self):
        host = QMainWindow()
        host.app_version = "4.8.2"
        host.development_commit = "a1b2c3d4"
        with patch.object(main_window_module.QMessageBox, "about") as about:
            FitsViewer.show_about_dialog(host)
        text = about.call_args.args[2]
        self.assertIn("v4.8.2", text)
        self.assertIn("Development commit:", text)
        self.assertIn("a1b2c3d4", text)
        host.close()

    def test_about_dialog_omits_commit_for_packaged_install(self):
        host = QMainWindow()
        host.app_version = "4.8.2"
        host.development_commit = None
        with patch.object(main_window_module.QMessageBox, "about") as about:
            FitsViewer.show_about_dialog(host)
        text = about.call_args.args[2]
        self.assertIn("v4.8.2", text)
        self.assertNotIn("Development commit:", text)
        host.close()


if __name__ == "__main__":
    unittest.main()
