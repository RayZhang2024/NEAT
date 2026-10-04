"""Subprocess checks for the fitting service's GUI import boundary."""

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestFittingImports(unittest.TestCase):
    def run_fresh_python(self, script):
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_engine_import_does_not_load_pyqt_in_fresh_process(self):
        self.run_fresh_python(
            """
            import sys
            from NEAT.services.fitting_engine import FittingEngine

            assert FittingEngine is not None
            assert not any(
                name == "PyQt5" or name.startswith("PyQt5.")
                for name in sys.modules
            )
            """
        )

    def test_top_level_fits_viewer_import_resolves_in_fresh_process(self):
        self.run_fresh_python(
            """
            from NEAT import FitsViewer

            assert FitsViewer is not None
            """
        )


if __name__ == "__main__":
    unittest.main()
