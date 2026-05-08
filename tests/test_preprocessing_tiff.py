import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from NEAT.workers.preprocessing import FullProcessWorker


@unittest.skipIf(Image is None, "Pillow is required to write TIFF fixtures")
class TestFullProcessWorkerTiffLoading(unittest.TestCase):
    def test_load_run_dict_accepts_tif_and_tiff_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            tif_data = np.array([[1, 2], [3, 4]], dtype=np.uint16)
            tiff_data = np.array([[5, 6], [7, 8]], dtype=np.uint16)

            Image.fromarray(tif_data).save(folder / "sample_00001.tif")
            Image.fromarray(tiff_data).save(folder / "sample_00002.tiff")
            Image.fromarray(tiff_data).save(folder / "sample_no_suffix.tiff")

            worker = FullProcessWorker(str(folder), str(folder), str(folder), "base", 1, 0)
            run = worker.load_run_dict(str(folder))

        self.assertEqual(sorted(run["images"]), ["00001", "00002"])
        np.testing.assert_array_equal(run["images"]["00001"], np.flipud(tif_data).astype(np.float32))
        np.testing.assert_array_equal(run["images"]["00002"], np.flipud(tiff_data).astype(np.float32))


if __name__ == "__main__":
    unittest.main()
