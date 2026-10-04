"""Headless regression coverage for the full-pattern fitting service."""

import unittest

import numpy as np

from NEAT.core import fitting_function_3
from NEAT.services.fitting_engine import FittingEngine


class TestFittingEngine(unittest.TestCase):
    # Captured by running the pre-extraction FittingMixin implementation from
    # commit a79e4b5fa79e714b1f04eea207de4154c4c73430 in a detached worktree.
    # That run used this class's fixture, all s/t/eta free, max_nfev=300,
    # curve_fit_maxfev=5000, and apply_lattice_update=False.
    PRE_REFACTOR_GOLDEN = {
        "lattice_a": 1.1999999999285176,
        "fitted_s": 0.006000000002919007,
        "fitted_t": 0.04999999918906637,
        "fitted_eta": 0.349999997513927,
        "edge_height": 0.09704296293608283,
        "edge_width": 0.028378945196835703,
        "residual_l2": 6.815095383653195e-10,
        "rss": 4.64455250882911e-19,
    }
    # Per-value (rtol, atol): tight for fitted parameters and shape metrics,
    # but tolerant of small platform/SciPy floating-point differences. The
    # residual norm is near zero for this noiseless fixture, so it uses an
    # absolute tolerance scaled to the observed optimization residual.
    PRE_REFACTOR_TOLERANCES = {
        "lattice_a": (1e-7, 1e-9),
        "fitted_s": (1e-5, 1e-9),
        "fitted_t": (1e-5, 1e-8),
        "fitted_eta": (1e-5, 1e-7),
        "edge_height": (1e-6, 1e-8),
        "edge_width": (1e-5, 1e-7),
        "residual_l2": (1e-3, 1e-9),
        "rss": (1e-2, 1e-18),
    }

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

    def test_matches_pre_refactor_commit_numerical_golden(self):
        result, error = self.fit()
        self.assertIsNone(error)
        self.assertTrue(result["success"])
        residual_l2 = float(np.linalg.norm(result["residuals"]))
        actual = {
            "lattice_a": float(result["lattice_params"]["a"]),
            "fitted_s": float(result["fitted_s"][self.hkl]),
            "fitted_t": float(result["fitted_t"][self.hkl]),
            "fitted_eta": float(result["fitted_eta"][self.hkl]),
            "edge_height": float(result["edge_heights"][self.hkl]),
            "edge_width": float(result["edge_widths"][self.hkl]),
            "residual_l2": residual_l2,
            "rss": residual_l2**2,
        }
        for name, expected in self.PRE_REFACTOR_GOLDEN.items():
            rtol, atol = self.PRE_REFACTOR_TOLERANCES[name]
            with self.subTest(quantity=name):
                np.testing.assert_allclose(actual[name], expected, rtol=rtol, atol=atol)

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
