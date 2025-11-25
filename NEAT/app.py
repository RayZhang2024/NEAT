"""Application entry point for NEAT."""

import sys

from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtWidgets import QApplication

from NEAT.ui import FitsViewer


def create_app():
    """Configure and create the QApplication instance."""
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyleSheet("""
    QLineEdit {
        max-height: 30px;
    }
    """)
    return app


def main():
    """Launch the NEAT GUI."""
    app = create_app()
    viewer = FitsViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
