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


def _screen_device_pixel_ratio() -> float:
    """Return the primary screen device pixel ratio."""
    app = QApplication.instance()
    if app is None:
        return 1.0
    screen = app.primaryScreen()
    if screen is None:
        return 1.0
    # Prefer floating-point DPR where available.
    dpr_fn = getattr(screen, "devicePixelRatio", None)
    if callable(dpr_fn):
        try:
            dpr = float(dpr_fn())
            if dpr > 0:
                return dpr
        except (TypeError, ValueError):
            pass
    return 1.0


def _prepare_splash_pixmap_for_display(source: QPixmap, max_width: int, max_height: int) -> QPixmap:
    """
    Prepare splash pixmap for crisp display on HiDPI screens while keeping
    approximately the same logical on-screen size as legacy behavior.
    """
    if source.isNull():
        return source

    src_w = source.width()
    src_h = source.height()

    # Legacy logical size cap.
    logical_pixmap = source
    if src_w > max_width or src_h > max_height:
        logical_pixmap = source.scaled(
            max_width,
            max_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

    logical_w = max(1, logical_pixmap.width())
    logical_h = max(1, logical_pixmap.height())

    # Determine how much HiDPI density the source can support at this logical size.
    screen_dpr = _screen_device_pixel_ratio()
    source_supported_dpr = min(src_w / logical_w, src_h / logical_h)
    effective_dpr = min(screen_dpr, source_supported_dpr)

    if effective_dpr <= 1.0:
        return logical_pixmap

    target_w = max(1, int(round(logical_w * effective_dpr)))
    target_h = max(1, int(round(logical_h * effective_dpr)))
    hidpi_pixmap = source.scaled(
        target_w,
        target_h,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )
    hidpi_pixmap.setDevicePixelRatio(effective_dpr)
    return hidpi_pixmap


def create_launch_splash():
    """Create and show the startup splash screen."""
    pixmap = _load_launch_splash_pixmap()
    if pixmap is not None:
        max_width = 1100
        max_height = 720
        pixmap = _prepare_splash_pixmap_for_display(pixmap, max_width, max_height)
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
    try:
        exit_code = app.exec_()
    except KeyboardInterrupt:
        # Allow Ctrl+C in console-launched sessions without a traceback.
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
