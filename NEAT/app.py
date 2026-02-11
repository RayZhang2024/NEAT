"""Application entry point for NEAT."""

import sys
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen


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


def _load_launch_splash_pixmap():
    """Load splash image from bundled assets, if available."""
    candidates = []

    # PyInstaller one-folder/one-file runtime location.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "NEAT" / "assets" / "launch_splash.png")

    # Source/package layout.
    candidates.append(Path(__file__).resolve().parent / "assets" / "launch_splash.png")

    for candidate in candidates:
        if not candidate.exists():
            continue
        pixmap = QPixmap(str(candidate))
        if not pixmap.isNull():
            return pixmap
    return None


def create_launch_splash():
    """Create and show the startup splash screen."""
    pixmap = _load_launch_splash_pixmap()
    if pixmap is not None:
        max_width = 1100
        max_height = 720
        if pixmap.width() > max_width or pixmap.height() > max_height:
            pixmap = pixmap.scaled(
                max_width,
                max_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
    else:
        pixmap = QPixmap(520, 260)
        pixmap.fill(QColor("#1f2933"))

        painter = QPainter(pixmap)
        painter.setPen(QColor("#e5eef6"))
        painter.setFont(QFont("Arial", 22, QFont.Bold))
        painter.drawText(30, 90, "NEAT")
        painter.setFont(QFont("Arial", 11))
        painter.drawText(30, 125, "Neutron Bragg Edge Analysis Toolkit")
        painter.setPen(QColor("#b6c6d7"))
        painter.drawText(30, 155, "Preparing interface...")
        painter.end()

    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash.showMessage(
        "Starting NEAT...",
        Qt.AlignHCenter | Qt.AlignBottom,
        QColor("#111111"),
    )
    splash.show()
    QApplication.processEvents()
    return splash


def update_splash_message(splash, message):
    """Update splash progress text and force repaint."""
    if splash is None:
        return
    splash.showMessage(message, Qt.AlignHCenter | Qt.AlignBottom, QColor("#111111"))
    QApplication.processEvents()


def main():
    """Launch the NEAT GUI."""
    app = create_app()
    splash = create_launch_splash()
    update_splash_message(splash, "Loading modules...")

    from NEAT.ui import FitsViewer

    update_splash_message(splash, "Building main window...")
    viewer = FitsViewer()
    viewer.show()
    splash.finish(viewer)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
