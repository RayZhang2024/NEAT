import math
import unittest

import numpy as np

from NEAT.core import (
    PHASE_DATA,
    calculate_d_spacing_general,
    calculate_theoretical_bragg_edges,
    calculate_uncertainty_estimator_constant,
    calculate_x_hkl_general,
    estimate_uncertainty_parameter,
    fitting_function_3,
)


class TestCoreBraggEdges(unittest.TestCase):
    def test_copper_builtin_uses_fcc_copper_lattice_parameter(self):
        copper = PHASE_DATA["Cu_fcc"]

        self.assertEqual(copper["structure"], "fcc")
        self.assertEqual(copper["lattice_params"], {"a": 3.615})

    def test_beta_titanium_builtin_uses_bcc_structure(self):
        beta_titanium = PHASE_DATA["Ti_Beta"]

        self.assertEqual(beta_titanium["structure"], "bcc")
        self.assertEqual(beta_titanium["lattice_params"], {"a": 3.32})
        self.assertTrue(
            all(sum(hkl) % 2 == 0 for hkl in beta_titanium["hkl_list"])
        )

    def test_edge_model_eta_endpoints_are_finite_and_distinct(self):
        x = np.linspace(1.8, 2.2, 101)
        common = dict(
            a0=0.1,
            b0=0.02,
            a_hkl=0.3,
            b_hkl=0.01,
            s=0.01,
            t=0.05,
            hkl_list=[(1, 0, 0)],
            min_wavelength=1.8,
            max_wavelength=2.2,
            structure_type="cubic",
            lattice_params={"a": 1.0},
        )

        gaussian = fitting_function_3(x, eta=0.0, **common)
        lorentzian = fitting_function_3(x, eta=1.0, **common)

        self.assertTrue(np.all(np.isfinite(gaussian)))
        self.assertTrue(np.all(np.isfinite(lorentzian)))
        self.assertFalse(np.allclose(gaussian, lorentzian))

    def test_edge_model_ignores_edge_outside_fit_window(self):
        x = np.linspace(1.0, 1.5, 21)
        result = fitting_function_3(
            x,
            0.1,
            0.02,
            0.3,
            0.01,
            0.01,
            0.05,
            0.5,
            [(1, 0, 0)],
            1.0,
            1.5,
            "cubic",
            {"a": 1.0},
        )

        np.testing.assert_array_equal(result, np.zeros_like(x))

    def test_calculate_d_spacing_general_cubic(self):
        d = calculate_d_spacing_general("cubic", {"a": 4.0}, (1, 1, 1))
        self.assertAlmostEqual(d, 4.0 / math.sqrt(3.0), places=10)

    def test_calculate_theoretical_bragg_edges_marks_invalid(self):
        edges = calculate_theoretical_bragg_edges(
            "orthorhombic",
            {"a": 3.0, "b": 4.0},  # missing c -> invalid
            [(1, 0, 0)],
        )
        self.assertEqual(len(edges), 1)
        self.assertTrue(np.isnan(edges[0][1]))

    def test_calculate_x_hkl_general_matches_edge_helper(self):
        hkls = [(1, 1, 0), (2, 0, 0)]
        x_vals = calculate_x_hkl_general("bcc", {"a": 2.86}, hkls)
        edges = calculate_theoretical_bragg_edges("bcc", {"a": 2.86}, hkls)
        self.assertEqual(len(x_vals), len(edges))
        for x, (_, edge) in zip(x_vals, edges):
            self.assertAlmostEqual(x, edge, places=12)

    def test_uncertainty_estimator_solves_each_parameter(self):
        constant = calculate_uncertainty_estimator_constant(30.0, 4.0, 0.2)
        self.assertAlmostEqual(constant, 4.8, places=12)

        self.assertAlmostEqual(
            estimate_uncertainty_parameter(
                constant,
                "macro_pixel_size",
                uamp=4.0,
                fitting_uncertainty=0.2,
            ),
            30.0,
            places=12,
        )
        self.assertAlmostEqual(
            estimate_uncertainty_parameter(
                constant,
                "uamp",
                macro_pixel_size=30.0,
                fitting_uncertainty=0.2,
            ),
            4.0,
            places=12,
        )
        self.assertAlmostEqual(
            estimate_uncertainty_parameter(
                constant,
                "fitting_uncertainty",
                macro_pixel_size=30.0,
                uamp=4.0,
            ),
            0.2,
            places=12,
        )

    def test_uncertainty_estimator_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            calculate_uncertainty_estimator_constant(0.0, 4.0, 0.2)
        with self.assertRaises(ValueError):
            estimate_uncertainty_parameter(
                4.8,
                "fitting_uncertainty",
                macro_pixel_size=-30.0,
                uamp=4.0,
            )


if __name__ == "__main__":
    unittest.main()
