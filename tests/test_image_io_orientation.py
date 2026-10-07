import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from astropy.io import fits

from NEAT.services import image_io
from NEAT.services.image_io import load_image_file as service_load_image_file
from NEAT.workers.batch import (
    load_image_file,
    write_fits_image_file,
)


class TestImageIoOrientation(unittest.TestCase):
    def test_fits_load_returns_imagej_display_orientation(self):
        stored = np.array([[1, 2], [3, 4]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "frame_00001.fits")
            fits.writeto(path, stored, overwrite=True)

            loaded = load_image_file(path)

        np.testing.assert_array_equal(loaded, np.flipud(stored))

    def test_fits_write_round_trips_display_orientation(self):
        displayed = np.array([[10, 11], [12, 13]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "processed_00001.fits")
            write_fits_image_file(path, displayed, overwrite=True)

            stored = fits.getdata(path, memmap=False)
            loaded = load_image_file(path)

        np.testing.assert_array_equal(stored, np.flipud(displayed))
        np.testing.assert_array_equal(loaded, displayed)

    def test_generic_reader_supports_all_historical_fits_extensions(self):
        stored = np.array([[1, 2], [3, 4]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            for extension in (".fits", ".fit", ".fts"):
                path = os.path.join(folder, f"frame{extension}")
                fits.writeto(path, stored, overwrite=True)
                np.testing.assert_array_equal(
                    service_load_image_file(path), np.flipud(stored)
                )

    def test_generic_reader_prefers_imageio_for_tiff_and_flips_orientation(self):
        stored = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        imageio = SimpleNamespace(imread=Mock(return_value=stored))
        with patch.object(image_io, "imageio", imageio):
            loaded = service_load_image_file("synthetic.tif")
        imageio.imread.assert_called_once_with("synthetic.tif")
        np.testing.assert_array_equal(loaded, np.flipud(stored))

    def test_generic_reader_uses_pillow_when_imageio_is_unavailable(self):
        from PIL import Image

        stored = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "frame.tiff")
            Image.fromarray(stored).save(path)
            with patch.object(image_io, "imageio", None):
                loaded = service_load_image_file(path)
        np.testing.assert_array_equal(loaded, np.flipud(stored))

    def test_generic_reader_keeps_missing_tiff_reader_error(self):
        with (
            patch.object(image_io, "imageio", None),
            patch.object(image_io, "Image", None),
            self.assertRaisesRegex(
                ImportError,
                "Neither imageio nor Pillow is available to read TIFF files.",
            ),
        ):
            service_load_image_file("unreadable.tif")

    def test_worker_batch_reader_is_compatibility_import(self):
        self.assertIs(load_image_file, service_load_image_file)


if __name__ == "__main__":
    unittest.main()
