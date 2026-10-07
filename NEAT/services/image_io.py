"""Headless generic image loading and FITS-writing helpers."""

from __future__ import annotations

import os
from importlib import import_module
from types import ModuleType

import numpy as np
from astropy.io import fits

imageio: ModuleType | None
try:
    imageio = import_module("imageio.v2")
except ImportError:  # pragma: no cover - exercised by fallback tests
    imageio = None

Image: ModuleType | None
try:
    Image = import_module("PIL.Image")
except ImportError:  # pragma: no cover - exercised by unavailable-reader tests
    Image = None


def _flip_vertical_image_axis(data):
    """Flip the image row axis while preserving leading stack dimensions."""
    arr = np.asarray(data)
    if arr.ndim < 2:
        return arr
    return np.flip(arr, axis=-2)


def load_image_file(file_path):
    """Read one FITS/TIFF image with NEAT's established display orientation."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".fits", ".fit", ".fts"):
        return _flip_vertical_image_axis(fits.getdata(file_path, memmap=False))
    if imageio is not None:
        arr = imageio.imread(file_path)
        if ext in (".tiff", ".tif"):
            arr = _flip_vertical_image_axis(arr)
        return arr
    if Image is not None:
        with Image.open(file_path) as img:
            arr = np.array(img)
        if ext in (".tiff", ".tif"):
            arr = _flip_vertical_image_axis(arr)
        return arr
    raise ImportError("Neither imageio nor Pillow is available to read TIFF files.")


def write_fits_image_file(file_path, data, header=None, overwrite=True):
    """Write a NEAT display-oriented image so ImageJ renders it the same way."""
    fits.writeto(
        file_path,
        _flip_vertical_image_axis(data),
        header=header,
        overwrite=overwrite,
    )
