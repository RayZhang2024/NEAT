import math
import unittest

import numpy as np

from NEAT.core import (
    calculate_d_spacing_general,
    calculate_theoretical_bragg_edges,
    calculate_x_hkl_general,
)


class TestCoreBraggEdges(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

