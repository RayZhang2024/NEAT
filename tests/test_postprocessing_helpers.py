import unittest
from types import SimpleNamespace

import numpy as np

from NEAT.ui.dialogs import ParameterPlotDialog


class TestPostProcessingHelpers(unittest.TestCase):
    def test_coordinate_edges_are_midpoints_with_extrapolated_ends(self):
        centers = np.array([0.0, 1.0, 3.0])

        edges = ParameterPlotDialog.calculate_edges(None, centers)

        np.testing.assert_allclose(edges, [-0.5, 0.5, 2.0, 4.0])

    def test_parameter_map_resize_uses_requested_shape(self):
        source = np.array([[1.0, 2.0], [3.0, 4.0]])

        resized = ParameterPlotDialog.resize_parameter_map(
            None, source, (4, 6)
        )

        self.assertEqual(resized.shape, (4, 6))
        self.assertAlmostEqual(resized[0, 0], 1.0)
        self.assertAlmostEqual(resized[-1, -1], 4.0)

    def test_parameter_map_export_preserves_source_shape_and_values(self):
        source = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

        exported = ParameterPlotDialog.prepare_parameter_map_for_export(source)

        self.assertEqual(exported.shape, source.shape)
        self.assertEqual(exported.dtype, np.float32)
        np.testing.assert_allclose(exported, source, equal_nan=True)

    def test_binary_mask_validation(self):
        self.assertTrue(
            ParameterPlotDialog.is_binary_mask(np.array([[0, 1], [1, 0]]))
        )
        self.assertFalse(
            ParameterPlotDialog.is_binary_mask(np.array([[0.0, 0.5], [1.0, 0.0]]))
        )
        self.assertFalse(
            ParameterPlotDialog.is_binary_mask(np.array([[0.0, np.nan]]))
        )

    def test_strain_source_is_limited_to_d_spacing_metrics(self):
        self.assertTrue(ParameterPlotDialog.is_d_spacing_parameter("d_110"))
        self.assertTrue(
            ParameterPlotDialog.is_d_spacing_parameter("Filtered d_110")
        )
        self.assertFalse(ParameterPlotDialog.is_d_spacing_parameter("d_unc_110"))
        self.assertFalse(ParameterPlotDialog.is_d_spacing_parameter("s_110"))
        self.assertFalse(ParameterPlotDialog.is_d_spacing_parameter("Strain"))

    def test_line_profile_interpolates_in_y_x_grid_order(self):
        dialog_state = SimpleNamespace(
            X_unique=np.array([0.0, 1.0]),
            Y_unique=np.array([0.0, 1.0]),
            Z=np.array([[0.0, 1.0], [2.0, 3.0]]),
        )

        values = ParameterPlotDialog.interpolate_z_values(
            dialog_state,
            np.array([0.5]),
            np.array([0.5]),
        )

        np.testing.assert_allclose(values, [1.5])


if __name__ == "__main__":
    unittest.main()
