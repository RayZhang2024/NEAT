"""Small headless image-writing helpers shared by NEAT services and workers."""

from __future__ import annotations

import numpy as np
from astropy.io import fits


def _flip_vertical_image_axis(data):
    """Flip the image row axis while preserving leading stack dimensions."""
    arr = np.asarray(data)
    if arr.ndim < 2:
        return arr
    return np.flip(arr, axis=-2)


def write_fits_image_file(file_path, data, header=None, overwrite=True):
    """Write a NEAT display-oriented image so ImageJ renders it the same way."""
    fits.writeto(
        file_path,
        _flip_vertical_image_axis(data),
        header=header,
        overwrite=overwrite,
    )
