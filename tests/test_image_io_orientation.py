import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from NEAT.workers.batch import load_image_file, write_fits_image_file


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


if __name__ == "__main__":
    unittest.main()
