""" GUI for Bragg edge imging data analysis. """
__author__ = " Ruiyao Zhang "
__version__ = "4.8.1"

# ONNX Runtime must load before PyQt on Windows. PyQt can otherwise load a
# conflicting DLL first, causing the assistant's local semantic search to fail
# even though onnxruntime is installed. Keep this optional for core-only installs.
try:  # pragma: no cover - behavior depends on optional runtime and OS DLL loading
    import onnxruntime as _onnxruntime  # noqa: F401
except (ImportError, OSError):
    _onnxruntime = None

from .ui import FitsViewer

__all__ = ["FitsViewer"]
