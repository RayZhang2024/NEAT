"""Headless regression tests for the top-level AI Assistant menu."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QDockWidget, QMainWindow

from NEAT.ui.main_window import FitsViewer


class _MenuHost(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.assistant_dock = QDockWidget("NEAT AI Assistant", self)
        self.assistant_dock.toggleViewAction().setText("AI Assistant")
        self.assistant_dock.open_settings_dialog = lambda: None
        self.check_updates_on_startup = False
        self.show_about_dialog = lambda: None
        self.on_update_check_startup_toggled = lambda checked: None
        self.check_for_updates_now = lambda: None
        self.open_user_manual = lambda: None
        self.open_github_repository = lambda: None
        self.open_video_tutorials = lambda: None


class MainWindowAssistantMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_assistant_controls_have_their_own_top_level_menu(self) -> None:
        host = _MenuHost()
        FitsViewer._add_assistant_menu(host, host.menuBar())
        FitsViewer._add_about_menu(host, host.menuBar())

        menus = {
            action.text(): action.menu()
            for action in host.menuBar().actions()
        }
        self.assertIn("AI Assistant", menus)
        self.assertIn("About", menus)
        assistant_actions = [
            action.text() for action in menus["AI Assistant"].actions()
        ]
        about_actions = [action.text() for action in menus["About"].actions()]
        self.assertEqual(
            assistant_actions,
            ["AI Assistant", "AI Assistant Settings..."],
        )
        self.assertNotIn("AI Assistant", about_actions)
        self.assertNotIn("AI Assistant Settings...", about_actions)
        host.close()


if __name__ == "__main__":
    unittest.main()
