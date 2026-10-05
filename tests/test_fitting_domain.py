"""Typed full-pattern contract and legacy-boundary regressions."""

import unittest

import numpy as np

from NEAT.core.fitting import DEFAULT_FITTING_PARAMETER_BOUNDS
from NEAT.domain import (
    FittingParameterBounds,
    FullPatternEdgeFit,
    FullPatternFitConfig,
    FullPatternFitResult,
    PatternFitRow,
    WavelengthRegion,
)


class TestFittingDomain(unittest.TestCase):
    def test_wavelength_region_validates_true_bounds(self):
        self.assertEqual(WavelengthRegion(1.2, 1.4).min_wavelength, 1.2)
        for lower, upper in ((1.4, 1.2), (1.2, 1.2), (float("nan"), 1.4)):
            with self.subTest(bounds=(lower, upper)), self.assertRaises(ValueError):
                WavelengthRegion(lower, upper)

    def test_typed_bounds_validate_and_legacy_bounds_fall_back(self):
        bounds = FittingParameterBounds(s=(0.002, 0.006), t=(0.03, 0.08), eta=(0.2, 0.8))
        self.assertEqual(bounds.s, (0.002, 0.006))
        self.assertEqual(FittingParameterBounds().s, DEFAULT_FITTING_PARAMETER_BOUNDS["s"])
        for invalid in ((0.02, 0.01), (float("inf"), 0.03)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                FittingParameterBounds(s=invalid)
        self.assertEqual(
            FittingParameterBounds.from_legacy_dict({"s": (0.02, 0.01)}),
            FittingParameterBounds(),
        )

    def test_config_owns_nested_values_and_excludes_output_metadata(self):
        rows = [{
            "hkl": [1, 1, 0],
            "regions": [
                {"min_wavelength": 1.4, "max_wavelength": 1.55},
                {"min_wavelength": 1.2, "max_wavelength": 1.35},
                {"min_wavelength": 1.55, "max_wavelength": 1.95},
            ],
            "s": 0.006, "t": 0.05, "eta": 0.35, "valid": True,
        }]
        legacy = {
            "structure_type": "bcc", "lattice_params": {"a": 1.2},
            "fitting_parameter_bounds": {"s": [0.002, 0.008]},
            "bragg_rows": rows, "flight_path": 16.0,
            "bragg_rows_text": ["table provenance"],
        }
        config = FullPatternFitConfig.from_legacy_dict(legacy)
        self.assertIsInstance(config.bragg_rows[0], PatternFitRow)
        self.assertIsInstance(config.bragg_rows[0].regions[0], WavelengthRegion)
        self.assertEqual(config.bragg_rows[0].hkl, (1, 1, 0))
        legacy["lattice_params"]["a"] = 2.0
        legacy["bragg_rows"][0]["regions"][0]["min_wavelength"] = 9.0
        legacy["bragg_rows"][0]["hkl"][0] = 9
        legacy["fitting_parameter_bounds"]["s"][0] = 0.004
        legacy["bragg_rows"].append({"valid": False})
        self.assertEqual(config.lattice_params["a"], 1.2)
        self.assertEqual(config.bragg_rows[0].regions[0].min_wavelength, 1.4)
        self.assertEqual(config.bragg_rows[0].hkl, (1, 1, 0))
        self.assertEqual(config.fitting_parameter_bounds.s, (0.002, 0.008))
        self.assertEqual(len(config.bragg_rows), 1)
        self.assertNotIn("flight_path", vars(config))
        self.assertNotIn("bragg_rows_text", vars(config))
        with self.assertRaises(TypeError):
            config.lattice_params["a"] = 3.0

    def test_invalid_row_is_retained_in_typed_config(self):
        config = FullPatternFitConfig.from_legacy_dict({
            "bragg_rows": [{"valid": False, "regions": None, "hkl": None}]
        })
        self.assertEqual(len(config.bragg_rows), 1)
        self.assertFalse(config.bragg_rows[0].valid)
        self.assertIsNone(config.bragg_rows[0].regions)

    def test_direct_config_detaches_lattice_and_row_collections(self):
        source_regions = [WavelengthRegion(1.2, 1.3)] * 3
        row = PatternFitRow(
            source_row=0, hkl=(1, 1, 0), regions=source_regions,
            s=0.006, t=0.05, eta=0.35, valid=True,
        )
        source_regions[0] = WavelengthRegion(1.4, 1.5)
        source_rows = [row]
        source_lattice = {"a": 1.2}
        config = FullPatternFitConfig(
            structure_type="bcc", lattice_params=source_lattice,
            bragg_rows=source_rows,
        )
        source_lattice["a"] = 2.0
        source_rows.clear()
        self.assertEqual(row.regions[0].min_wavelength, 1.2)
        self.assertEqual(config.lattice_params["a"], 1.2)
        self.assertEqual(len(config.bragg_rows), 1)
        with self.assertRaises(ValueError):
            PatternFitRow(0, (1, 1), None, None, None, None, False)

    def test_typed_ordered_edges_recreate_complete_legacy_payload(self):
        region = WavelengthRegion(1.2, 1.4)
        spectrum = np.array([1.0, 2.0])
        edges = tuple(
            FullPatternEdgeFit(
                hkl=hkl, a0=0.1, b0=0.2, a_hkl=0.3, b_hkl=0.4,
                s=0.006, t=0.05, eta=0.35, x_r3=spectrum, y_r3=spectrum,
                regions=(region, region, region),
            )
            for hkl in ((1, 1, 0), (2, 0, 0))
        )
        result = FullPatternFitResult(
            ab_fits={}, bragg_edges=edges, structure_type="bcc",
            lattice_params={"a": 1.2}, lattice_uncertainties={"a": 0.01},
            fitted_s={}, fitted_t={}, fitted_eta={},
            s_uncertainties={}, t_uncertainties={}, eta_uncertainties={},
            x_data=spectrum, y_data=spectrum, x_exp_sorted=spectrum,
            y_exp_sorted=spectrum, residuals=spectrum, success=True,
            message="ok", edge_heights={}, edge_widths={},
        )
        legacy = result.to_legacy_dict()
        self.assertEqual(set(legacy), {
            "ab_fits", "bragg_edges", "structure_type", "lattice_params",
            "lattice_uncertainties", "fitted_s", "fitted_t", "fitted_eta",
            "s_uncertainties", "t_uncertainties", "eta_uncertainties",
            "x_data", "y_data", "x_exp_sorted", "y_exp_sorted", "residuals",
            "success", "message", "edge_heights", "edge_widths",
        })
        self.assertEqual([edge["hkl"] for edge in legacy["bragg_edges"]],
                         [(1, 1, 0), (2, 0, 0)])
        self.assertEqual(set(legacy["bragg_edges"][0]), {
            "hkl", "a0", "b0", "a_hkl", "b_hkl", "s", "t", "eta",
            "x_r3", "y_r3", "regions",
        })
        self.assertEqual(legacy["bragg_edges"][0]["regions"][0],
                         {"min_wavelength": 1.2, "max_wavelength": 1.4})
        self.assertEqual(legacy["bragg_edges"][0]["a_hkl"], 0.3)
        self.assertEqual(legacy["bragg_edges"][0]["eta"], 0.35)
        self.assertIs(legacy["bragg_edges"][0]["x_r3"], spectrum)


if __name__ == "__main__":
    unittest.main()
