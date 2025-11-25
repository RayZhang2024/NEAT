"""Compatibility shim exposing the NEAT application entry point."""

from .app import main
from .ui import FitsViewer

__all__ = ["FitsViewer", "main"]

if __name__ == "__main__":
    main()
