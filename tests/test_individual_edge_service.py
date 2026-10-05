"""Pre-extraction goldens from main 4a1f57228bc40c37a92e55268828810d2c022c86.

The fixtures call the original FittingMixin.fit_region with explicit arrays and
row_data, before its numerical code was moved. SciPy defaults and Region-3
maxfev=300 are identical on both sides of the comparison.
"""

import os
import subprocess
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from NEAT.core import fitting_function_3
from NEAT.domain import FittingParameterBounds, IndividualEdgeFitConfig, WavelengthRegion
from NEAT.services.fitting_engine import FittingEngine


class TestIndividualEdgeService(unittest.TestCase):
    GOLDEN = {
        "known": {
            "r1": (0.5631301493066353, 0.1854390532885369),
            "r2": (-0.09874525449252843, 0.0742043658175223),
            "params": (
                1.6970562747465978, 0.006000000002918513,
                0.04999999918907101, 0.34999999751333233,
                1.143600449665017e-11, 7.708356281416253e-12,
                4.104833955043006e-11, 4.5926984973321436e-10,
            ),
            "rms": 2.951884759343016e-11,
            "height": 0.09704296293680348,
            "width": 0.028378945205101758,
            "curve": (0.4202919665457192, 0.49141272920106766,
                      0.5006473858128486, 248.52160139958198),
        },
        "unknown": {
            "r1": (0.3160010106084053, 0.42928065967974627),
            "r2": (-0.012844148988621407, 0.015475891718901654),
            "params": (
                0.9999999998497303, 0.004999999934935156,
                0.059999996724268014, 0.40000000110866907,
                1.1000703241081304e-11, 9.080462873459044e-12,
                1.4780507234364932e-10, 4.050966907299933e-10,
            ),
            "rms": 4.731917121997985e-11,
            "height": 0.09067736102544804,
            "width": 0.018679669908122642,
            "curve": (0.495214202225566, 0.5595883602854255,
                      0.5676020342939307, 282.17999097169525),
        },
    }

    def fixture(self, known=True):
        if known:
            wavelengths = np.linspace(1.0, 2.2, 1600)
            hkl, a, baseline, shape = (1, 1, 0), 1.2, (0.3, 0.2, 0.15, 0.07), (0.006, 0.05, 0.35)
            windows = ((1.4, 1.55), (1.2, 1.35), (1.55, 1.95))
        else:
            wavelengths = np.linspace(0.6, 1.4, 1400)
            hkl, a, baseline, shape = None, 0.5, (0.2, 0.3, 0.1, 0.15), (0.005, 0.06, 0.4)
            windows = ((0.7, 0.82), (0.8, 0.95), (0.9, 1.2))
        intensities = fitting_function_3(
            wavelengths, *baseline, *shape, [hkl] if hkl else [], *windows[2],
            "bcc" if known else "cubic", {"a": a},
        )
        row = {
            "hkl": hkl, "d": None if known else 1.0,
            "regions": [
                {"min_wavelength": lower, "max_wavelength": upper}
                for lower, upper in windows
            ],
            "s": shape[0], "t": shape[1], "eta": shape[2],
        }
        config = IndividualEdgeFitConfig.from_legacy_row(
            row, source_row=4, is_known_phase=known,
            structure_type="bcc" if known else "cubic",
            lattice_params={"a": a},
            fitting_parameter_bounds=FittingParameterBounds(),
        )
        return wavelengths, intensities, row, config

    def assert_golden(self, known):
        wavelengths, intensities, _, config = self.fixture(known)
        attempt = FittingEngine().fit_individual_edge(wavelengths, intensities, config)
        self.assertIsNone(attempt.error_message)
        result = attempt.result
        self.assertIsNotNone(result)
        expected = self.GOLDEN["known" if known else "unknown"]
        # Stage fits and fitted values: rtol 1e-5, atol 1e-7 for platform SciPy variation.
        np.testing.assert_allclose(result.region1.params, expected["r1"], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(result.region2.params, expected["r2"], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(result.baseline_params["a0"], 0.3 if known else 0.2, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(result.fit_params[:4], expected["params"][:4], rtol=1e-5, atol=1e-7)
        # Near-zero covariance errors and RMS use absolute tolerances.
        np.testing.assert_allclose(result.fit_params[4:], expected["params"][4:], rtol=1e-3, atol=1e-8)
        np.testing.assert_allclose(result.rms, expected["rms"], rtol=1e-3, atol=1e-8)
        np.testing.assert_allclose(result.edge_height, expected["height"], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(result.edge_width, expected["width"], rtol=1e-5, atol=1e-6)
        curve = result.region3.fit
        summary = (curve[0], curve[len(curve)//2], curve[-1], curve.sum())
        np.testing.assert_allclose(summary, expected["curve"], rtol=1e-5, atol=1e-6)
        self.assertEqual(result.source_row, 4)
        self.assertEqual(result.hkl, (1, 1, 0) if known else None)
        self.assertEqual(len(result.region1.x), len(result.region1.y))
        self.assertEqual(len(result.region2.x), len(result.region2.y))
        self.assertEqual(len(result.region3.x), len(result.region3.y))
        return result

    def test_known_phase_golden(self):
        self.assert_golden(True)

    def test_unknown_phase_golden_and_source_row_label(self):
        result = self.assert_golden(False)
        self.assertEqual(result.to_legacy_dict(skip_ui_updates=True)["hkl"], "edge5")

    def test_exact_legacy_key_sets(self):
        result = self.assert_golden(True)
        self.assertEqual(set(result.to_legacy_dict(skip_ui_updates=True)), {
            "hkl", "x", "fit", "d_fit", "d_unc", "s_fit", "s_unc", "t_fit", "t_unc",
            "eta_fit", "eta_unc", "fit_params", "edge_height", "edge_width", "rms", "baseline_params",
        })
        self.assertEqual(set(result.to_legacy_dict(skip_ui_updates=False)), {
            "hkl", "x", "y", "fit", "fit_params", "edge_height", "edge_width", "rms", "baseline_params",
        })

    def test_fixed_and_mixed_shape_parameters(self):
        wavelengths, intensities, _, config = self.fixture(True)
        engine = FittingEngine()
        fixed = engine.fit_individual_edge(wavelengths, intensities, config, fix_s=True, fix_t=True, fix_eta=True).result
        self.assertIsNotNone(fixed)
        np.testing.assert_allclose(fixed.fit_params[:4],
            (1.6970562748477112, 0.006, 0.05, 0.35), rtol=1e-5, atol=1e-7)
        self.assertTrue(all(np.isnan(value) for value in (fixed.s_unc, fixed.t_unc, fixed.eta_unc)))
        mixed = engine.fit_individual_edge(wavelengths, intensities, config, fix_s=True, fix_eta=True).result
        self.assertIsNotNone(mixed)
        np.testing.assert_allclose(mixed.t_fit, 0.0499999999148089, rtol=1e-5, atol=1e-7)
        self.assertTrue(np.isfinite(mixed.t_unc))
        self.assertTrue(np.isnan(mixed.s_unc))
        self.assertTrue(np.isnan(mixed.eta_unc))

        unknown_x, unknown_y, _, unknown_config = self.fixture(False)
        unknown_fixed = engine.fit_individual_edge(
            unknown_x, unknown_y, unknown_config,
            fix_s=True, fix_t=True, fix_eta=True,
        ).result
        self.assertIsNotNone(unknown_fixed)
        np.testing.assert_allclose(
            unknown_fixed.fit_params[:4], (1.0000000001731293, 0.005, 0.06, 0.4),
            rtol=1e-5, atol=1e-7,
        )
        self.assertTrue(all(np.isnan(v) for v in
            (unknown_fixed.s_unc, unknown_fixed.t_unc, unknown_fixed.eta_unc)))
        unknown_mixed = engine.fit_individual_edge(
            unknown_x, unknown_y, unknown_config, fix_s=True, fix_eta=True,
        ).result
        self.assertIsNotNone(unknown_mixed)
        np.testing.assert_allclose(unknown_mixed.t_fit, 0.05999999997510184, rtol=1e-5, atol=1e-7)
        self.assertTrue(np.isfinite(unknown_mixed.t_unc))
        self.assertTrue(np.isnan(unknown_mixed.s_unc))
        self.assertTrue(np.isnan(unknown_mixed.eta_unc))

    def test_explicit_and_batch_rows_keep_source_identity_and_ownership(self):
        _, _, row, config = self.fixture(False)
        self.assertEqual(config.source_row, 4)
        self.assertIsNone(config.hkl)
        self.assertEqual(config.d, 1.0)
        self.assertIsInstance(config.regions[0], WavelengthRegion)
        row["regions"][0]["min_wavelength"] = 99
        self.assertEqual(config.window(0), (0.7, 0.82))
        source_lattice = {"a": 1.2}
        owned = replace(config, lattice_params=source_lattice)
        source_lattice["a"] = 9.9
        self.assertEqual(owned.lattice_params["a"], 1.2)
        batch = dict(row, row=9, valid=True)
        first = IndividualEdgeFitConfig.from_legacy_row(batch, source_row=batch["row"],
            is_known_phase=False, structure_type="cubic", lattice_params={},
            fitting_parameter_bounds=FittingParameterBounds())
        second = IndividualEdgeFitConfig.from_legacy_row(batch, source_row=10,
            is_known_phase=False, structure_type="cubic", lattice_params={},
            fitting_parameter_bounds=FittingParameterBounds())
        self.assertEqual((first.source_row, second.source_row), (9, 10))
        self.assertEqual(len((first, second)), 2)
        with self.assertRaises(ValueError):
            FittingParameterBounds(s=(0.02, 0.01))

    def test_missing_lattice_and_unknown_d_are_service_failures(self):
        for known in (True, False):
            wavelengths, intensities, _, config = self.fixture(known)
            config = replace(config, lattice_params={}) if known else replace(config, d=None)
            attempt = FittingEngine().fit_individual_edge(wavelengths, intensities, config)
            self.assertIsNone(attempt.result)
            self.assertEqual(attempt.error_stage, "setup")
            self.assertIn("Missing cubic lattice parameter 'a'" if known else "Missing 'd' value", attempt.error_message)

    def test_legacy_inverted_region_keeps_stage_failure(self):
        wavelengths, intensities, row, config = self.fixture(True)
        row["regions"][1] = {"min_wavelength": 1.35, "max_wavelength": 1.2}
        config = IndividualEdgeFitConfig.from_legacy_row(row, source_row=4,
            is_known_phase=True, structure_type="bcc", lattice_params={"a": 1.2},
            fitting_parameter_bounds=FittingParameterBounds())
        attempt = FittingEngine().fit_individual_edge(wavelengths, intensities, config)
        self.assertEqual((attempt.error_stage, attempt.error_message), ("Region 1", "has no data in range"))
        with self.assertRaises(ValueError):
            replace(config, legacy_window_order=False)

    def test_region3_failure_retains_prior_stages(self):
        wavelengths, intensities, _, config = self.fixture(True)
        from NEAT.services import individual_edge as service
        real_fit = service.curve_fit
        calls = []
        def fail_third(*args, **kwargs):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("forced Region 3 failure")
            return real_fit(*args, **kwargs)
        with patch.object(service, "curve_fit", side_effect=fail_third):
            attempt = FittingEngine().fit_individual_edge(wavelengths, intensities, config)
        self.assertIsNone(attempt.result)
        self.assertEqual(attempt.error_stage, "Region 3 fit")
        self.assertIsNotNone(attempt.region1.fit)
        self.assertIsNotNone(attempt.region2.fit)
        self.assertIsNotNone(attempt.region3)

    def test_region1_and_region2_failures_retain_completed_stages(self):
        wavelengths, intensities, row, config = self.fixture(True)
        from NEAT.services import individual_edge as service
        real_fit = service.curve_fit
        for failing_stage in (1, 2):
            calls = []
            def fail_selected(*args, **kwargs):
                calls.append(1)
                if len(calls) == failing_stage:
                    raise RuntimeError("forced")
                return real_fit(*args, **kwargs)
            with self.subTest(stage=failing_stage), patch.object(service, "curve_fit", side_effect=fail_selected):
                attempt = FittingEngine().fit_individual_edge(wavelengths, intensities, config)
                self.assertEqual(attempt.error_stage, f"Region {failing_stage} fit")
                self.assertIsNone(attempt.result)
                if failing_stage == 2:
                    self.assertIsNotNone(attempt.region1.fit)
                    self.assertIsNotNone(attempt.region2)
                    self.assertIsNone(attempt.region2.fit)
        row["regions"][0] = {"min_wavelength": 1.55, "max_wavelength": 1.4}
        empty_r2 = IndividualEdgeFitConfig.from_legacy_row(
            row, source_row=4, is_known_phase=True, structure_type="bcc",
            lattice_params={"a": 1.2}, fitting_parameter_bounds=FittingParameterBounds(),
        )
        attempt = FittingEngine().fit_individual_edge(wavelengths, intensities, empty_r2)
        self.assertEqual((attempt.error_stage, attempt.error_message),
                         ("Region 2", "has no data in range"))
        self.assertIsNotNone(attempt.region1.fit)

    def test_optimizer_limits_and_baseline_bounds_match_legacy(self):
        wavelengths, intensities, _, config = self.fixture(True)
        from NEAT.services import individual_edge as service
        real_fit = service.curve_fit
        calls = []
        def capture(*args, **kwargs):
            calls.append(kwargs)
            return real_fit(*args, **kwargs)
        with patch.object(service, "curve_fit", side_effect=capture):
            attempt = FittingEngine().fit_individual_edge(wavelengths, intensities, config)
        self.assertIsNotNone(attempt.result)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["p0"], [0, 0])
        self.assertEqual(calls[0]["bounds"], ([-10, -10], [10, 10]))
        self.assertNotIn("maxfev", calls[0])
        self.assertEqual(calls[1]["p0"], [0, 0])
        self.assertNotIn("bounds", calls[1])
        self.assertNotIn("maxfev", calls[1])
        self.assertEqual(calls[2]["maxfev"], 300)
        r1_a0, r1_b0 = self.GOLDEN["known"]["r1"]
        lower, upper = calls[2]["bounds"]
        self.assertAlmostEqual(lower[0], r1_a0 - max(abs(r1_a0), 1), places=7)
        self.assertAlmostEqual(upper[1], r1_b0 + max(abs(r1_b0), 1), places=7)
        self.assertAlmostEqual(lower[4], 1.2 * 0.95)
        self.assertAlmostEqual(upper[4], 1.2 * 1.05)

    def test_known_metric_uses_supplied_phase_state(self):
        wavelengths, intensities, _, config = self.fixture(True)
        config = replace(config, lattice_params={"a": 1.22})
        from NEAT.services import individual_edge as service
        real_calculate = service.calculate_x_hkl_general
        with patch.object(service, "calculate_x_hkl_general", wraps=real_calculate) as calculate:
            attempt = FittingEngine().fit_individual_edge(wavelengths, intensities, config)
        self.assertIsNotNone(attempt.result)
        calculate.assert_called_once_with("bcc", config.lattice_params, [(1, 1, 0)])

    def test_non_cubic_selected_phase_not_rejected(self):
        wavelengths, intensities, _, config = self.fixture(True)
        config = replace(config, structure_type="hexagonal", lattice_params={"a": 1.2, "c": 1.5})
        attempt = FittingEngine().fit_individual_edge(wavelengths, intensities, config)
        self.assertNotIn("Unsupported", attempt.error_message or "")
        self.assertIsNotNone(attempt.result)

    def test_qt_free_import_in_fresh_process(self):
        env = os.environ.copy()
        process = subprocess.run(
            [sys.executable, "-c", "from NEAT.services.fitting_engine import FittingEngine; "
             "from NEAT.domain import IndividualEdgeFitConfig; "
             "import sys; assert 'PyQt5' not in sys.modules"],
            env=env, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
