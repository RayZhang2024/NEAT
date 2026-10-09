"""Application entry point for NEAT."""

import os
import sys
import traceback
from pathlib import Path

# Load ONNX Runtime before Qt on Windows. Loading Qt first can introduce a
# conflicting native DLL and make the assistant's semantic retriever fail in a
# packaged executable. This remains optional for core-only source installs.
try:  # pragma: no cover - depends on optional native runtime and operating system
    import onnxruntime as _onnxruntime  # noqa: F401
except (ImportError, OSError):
    _onnxruntime = None

from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen

from NEAT.package_resources import assistant_knowledge_root, launch_splash_resource


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

    # Source/package resource fallback.
    for candidate in candidates:
        if not candidate.is_file():
            continue
        pixmap = QPixmap(str(candidate))
        if not pixmap.isNull():
            return pixmap
    try:
        resource = launch_splash_resource()
        if resource.is_file():
            pixmap = QPixmap()
            if pixmap.loadFromData(resource.read_bytes()):
                return pixmap
    except (FileNotFoundError, OSError):
        pass
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


def _write_release_smoke_result(message: str) -> None:
    """Persist packaged smoke-test progress when a result path is configured."""

    result_path = os.environ.get("NEAT_RELEASE_SMOKE_RESULT", "").strip()
    if not result_path:
        return
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message.rstrip() + "\n")


def _run_release_smoke_test() -> int:
    """Validate packaged assistant imports and approved knowledge without an API call."""

    _write_release_smoke_result("START")
    from importlib.metadata import version as distribution_version

    from NEAT import __version__

    installed_version = distribution_version("NEAT")
    expected_version = os.environ.get("NEAT_EXPECTED_RELEASE_VERSION", "").strip()
    if installed_version != __version__:
        raise RuntimeError(
            "Installed distribution version does not match NEAT.__version__: "
            f"{installed_version} != {__version__}"
        )
    if expected_version and installed_version != expected_version:
        raise RuntimeError(
            "Packaged NEAT version does not match the expected release version: "
            f"{installed_version} != {expected_version}"
        )
    _write_release_smoke_result(f"OK package version={installed_version}")

    import anthropic  # noqa: F401
    _write_release_smoke_result("OK anthropic")
    import chromadb  # noqa: F401
    _write_release_smoke_result("OK chromadb")
    import keyring  # noqa: F401
    _write_release_smoke_result("OK keyring")
    try:
        import onnxruntime  # noqa: F401
    except (ImportError, OSError) as exc:
        _write_release_smoke_result(
            f"OPTIONAL onnxruntime unavailable; BM25 fallback required: {exc}"
        )
    else:
        _write_release_smoke_result("OK onnxruntime")
    import openai  # noqa: F401
    _write_release_smoke_result("OK openai")
    from tools import assistant_anthropic  # noqa: F401
    from tools import assistant_google  # noqa: F401
    from tools import assistant_openai  # noqa: F401
    from tools import assistant_openai_compatible  # noqa: F401
    from tools.assistant_retrieval import BM25Retriever, load_knowledge_base
    from tools.assistant_shared_client import is_shared_service_configured
    _write_release_smoke_result("OK provider adapters")

    sections = load_knowledge_base(assistant_knowledge_root())
    if not sections:
        raise RuntimeError("The packaged assistant knowledge base is empty.")
    _write_release_smoke_result(f"OK knowledge sections={len(sections)}")
    fallback_matches = BM25Retriever(sections).search("How do I use NEAT?", limit=3)
    if not fallback_matches:
        raise RuntimeError("The packaged BM25 retrieval fallback returned no results.")
    _write_release_smoke_result("OK BM25 fallback")
    if getattr(sys, "_MEIPASS", None):
        if not is_shared_service_configured():
            raise RuntimeError(
                "The packaged automatic shared-access configuration is missing."
            )
        _write_release_smoke_result("OK automatic shared access configured")
    return len(sections)


def main():
    """Launch the NEAT GUI."""

    release_smoke_requested = (
        os.environ.get("NEAT_RELEASE_SMOKE_TEST", "").strip() == "1"
        or "--release-smoke-test" in sys.argv
    )
    if release_smoke_requested:
        try:
            _run_release_smoke_test()
            _write_release_smoke_result("PASS")
            exit_code = 0
        except BaseException:
            _write_release_smoke_result("FAIL")
            _write_release_smoke_result(traceback.format_exc())
            exit_code = 1
        # Some optional libraries start background cleanup that can keep a
        # windowed PyInstaller process alive after the probe has completed.
        # This mode performs no user work, so an immediate, deterministic exit
        # is appropriate after the result file has been flushed.
        os._exit(exit_code)
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
