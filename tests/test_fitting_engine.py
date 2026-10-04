"""Headless regression coverage for the full-pattern fitting service."""

import unittest

import numpy as np

from NEAT.core import fitting_function_3
from NEAT.services.fitting_engine import FittingEngine


class TestFittingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = FittingEngine()
        self.wavelengths = np.linspace(1.0, 2.2, 1600)
        self.hkl = (1, 1, 0)
        self.lattice_params = {"a": 1.2}
        self.shape = {"s": 0.006, "t": 0.05, "eta": 0.35}
        self.regions = [
            {"min_wavelength": 1.4, "max_wavelength": 1.55},
            {"min_wavelength": 1.2, "max_wavelength": 1.35},
            {"min_wavelength": 1.55, "max_wavelength": 1.95},
        ]
        self.intensities = fitting_function_3(
            self.wavelengths,
            0.3,
            0.2,
            0.15,
            0.07,
            self.shape["s"],
            self.shape["t"],
            self.shape["eta"],
            [self.hkl],
            1.55,
            1.95,
            "bcc",
            self.lattice_params,
        )
        self.fit_config = {
            "structure_type": "bcc",
            "lattice_params": self.lattice_params,
            "bragg_rows": [
                {
                    "row": 0,
                    "hkl": self.hkl,
                    "regions": self.regions,
                    **self.shape,
                    "valid": True,
                }
            ],
        }

    def fit(self, *, fix_s=False, fix_t=False, fix_eta=False, config=None):
        return self.engine.fit_full_pattern(
            self.wavelengths,
            self.intensities,
            config or self.fit_config,
            fix_s=fix_s,
            fix_t=fix_t,
            fix_eta=fix_eta,
            max_nfev=300,
            curve_fit_maxfev=5000,
        )

    def test_direct_engine_free_fixed_and_mixed_parameter_regression(self):
        original_lattice = dict(self.fit_config["lattice_params"])
        free_result, error = self.fit()
        self.assertIsNone(error)
        self.assertEqual(self.fit_config["lattice_params"], original_lattice)
        self.assertTrue(free_result["success"])
        self.assertAlmostEqual(free_result["lattice_params"]["a"], 1.2, places=2)
        self.assertIn(self.hkl, free_result["ab_fits"])
        self.assertTrue(np.isfinite(free_result["edge_heights"][self.hkl]))
        self.assertTrue(np.isfinite(free_result["edge_widths"][self.hkl]))
        self.assertIn("residuals", free_result)
        self.assertEqual(free_result["x_data"].shape, free_result["y_data"].shape)

        fixed_result, error = self.fit(fix_s=True, fix_t=True, fix_eta=True)
        self.assertIsNone(error)
        self.assertTrue(np.isnan(fixed_result["s_uncertainties"][self.hkl]))
        self.assertTrue(np.isnan(fixed_result["t_uncertainties"][self.hkl]))
        self.assertTrue(np.isnan(fixed_result["eta_uncertainties"][self.hkl]))
        self.assertEqual(fixed_result["fitted_s"][self.hkl], self.shape["s"])

        mixed_result, error = self.fit(fix_s=True, fix_t=False, fix_eta=True)
        self.assertIsNone(error)
        self.assertTrue(np.isnan(mixed_result["s_uncertainties"][self.hkl]))
        self.assertTrue(np.isfinite(mixed_result["t_uncertainties"][self.hkl]))
        self.assertTrue(np.isnan(mixed_result["eta_uncertainties"][self.hkl]))

    def test_custom_shape_bounds_are_applied_by_engine(self):
        config = dict(self.fit_config)
        config["fitting_parameter_bounds"] = {
            "s": (0.004, 0.0045),
            "t": (0.035, 0.045),
            "eta": (0.25, 0.32),
        }
        result, error = self.fit(config=config)
        self.assertIsNone(error)
        self.assertTrue(0.004 <= result["fitted_s"][self.hkl] <= 0.0045)
        self.assertTrue(0.035 <= result["fitted_t"][self.hkl] <= 0.045)
        self.assertTrue(0.25 <= result["fitted_eta"][self.hkl] <= 0.32)

    def test_invalid_structure_and_missing_context_return_errors(self):
        config = dict(self.fit_config, structure_type="unsupported")
        result, error = self.fit(config=config)
        self.assertIsNone(result)
        self.assertEqual(error, "Unsupported structure type: unsupported")

        config = dict(self.fit_config, lattice_params={})
        result, error = self.fit(config=config)
        self.assertIsNone(result)
        self.assertEqual(error, "Lattice parameters not initialized")


if __name__ == "__main__":
    unittest.main()
