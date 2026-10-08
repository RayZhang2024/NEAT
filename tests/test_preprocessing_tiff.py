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

        self.assertEqual(sorted(run["images"]), ["00001", "00002", "suffix"])
        np.testing.assert_array_equal(run["images"]["00001"], np.flipud(tif_data).astype(np.float32))
        np.testing.assert_array_equal(run["images"]["00002"], np.flipud(tiff_data).astype(np.float32))
        np.testing.assert_array_equal(run["images"]["suffix"], np.flipud(tiff_data).astype(np.float32))
        self.assertEqual(run["load_errors"], [])

    def test_overlap_stage_fails_when_required_sidecars_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_folder = root / "input"
            output_folder = root / "output"
            input_folder.mkdir()
            output_folder.mkdir()
            Image.fromarray(np.ones((2, 2), dtype=np.uint16)).save(
                input_folder / "sample_frame.tif"
            )
            worker = FullProcessWorker(
                str(input_folder),
                str(input_folder),
                str(output_folder),
                "base",
                10,
                1,
            )

            with self.assertRaisesRegex(
                RuntimeError, "2_correction_Sample failed"
            ):
                worker.do_overlap_correction(str(input_folder), "Sample")

    def test_stop_is_forwarded_to_active_child(self):
        class Child:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        with tempfile.TemporaryDirectory() as tmp:
            worker = FullProcessWorker(tmp, tmp, tmp, "base", 10, 1)
            child = Child()
            worker._active_child = child

            worker.stop()

            self.assertTrue(child.stopped)
            self.assertFalse(worker._is_running)


if __name__ == "__main__":
    unittest.main()
