"""GUI for Bragg edge imaging data analysis."""

from importlib.metadata import version

__author__ = " Ruiyao Zhang "
# Package metadata is the only maintained release-version value.  Setuptools
# writes it from ``pyproject.toml`` for editable installs and wheels; NEAT.spec
# bundles that metadata for the standalone application.
__version__ = version("NEAT")

# ONNX Runtime must load before PyQt on Windows. PyQt can otherwise load a
# conflicting DLL first, causing the assistant's local semantic search to fail
# even though onnxruntime is installed. Keep this optional for core-only installs.
try:  # pragma: no cover - behavior depends on optional runtime and OS DLL loading
    import onnxruntime as _onnxruntime  # noqa: F401
except (ImportError, OSError):
    _onnxruntime = None

from .ui import FitsViewer

__all__ = ["FitsViewer", "__version__"]
