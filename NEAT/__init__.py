"""GUI for Bragg edge imaging data analysis."""

from importlib.metadata import version

__author__ = " Ruiyao Zhang "
# Package metadata is the only maintained release-version value.  Setuptools
# writes it from ``pyproject.toml`` for editable installs and wheels; NEAT.spec
# bundles that metadata for the standalone application.
__version__ = version("NEAT")

__all__ = ["FitsViewer", "__version__"]


def __getattr__(name):
    """Load the GUI lazily so numerical services can be imported headlessly."""
    if name != "FitsViewer":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Preserve ONNX-before-Qt ordering for application users while keeping
    # headless package imports independent of the optional native runtime.
    try:  # pragma: no cover - depends on optional runtime and operating system
        import onnxruntime as _onnxruntime  # noqa: F401
    except (ImportError, OSError):
        _onnxruntime = None
    globals()["_onnxruntime"] = _onnxruntime

    from .ui import FitsViewer

    return FitsViewer
