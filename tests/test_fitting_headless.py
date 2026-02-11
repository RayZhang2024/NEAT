import unittest

import numpy as np

from NEAT.core import fitting_function_3
from NEAT.ui.mixins.fitting import FittingMixin


class _DummyItem:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _DummyTable:
    def __init__(self, rows):
        self._rows = rows

    def rowCount(self):
        return len(self._rows)

    def columnCount(self):
        return max((len(row) for row in self._rows), default=0)

    def item(self, row, col):
        if row < 0 or row >= len(self._rows):
            return None
        current_row = self._rows[row]
        if col < 0 or col >= len(current_row):
            return None
        value = current_row[col]
        if value is None:
            return None
        return _DummyItem(str(value))


class _DummyTextInput:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _DummyDropdown:
    def __init__(self, value):
        self._value = value

    def currentText(self):
        return self._value


class _HeadlessFitting(FittingMixin):
    pass


class TestFittingHeadless(unittest.TestCase):
    def test_build_batch_fit_context_parses_valid_row(self):
        obj = _HeadlessFitting()
        obj.bragg_table = _DummyTable(
            [
                [
                    "(1, 1, 0)",
                    "1.697",
                    "1.40",
                    "1.55",
                    "1.20",
                    "1.35",
                    "1.55",
                    "1.95",
                    "0.006",
                    "0.05",
                    "0.35",
                ]
            ]
        )
        obj.phase_dropdown = _DummyDropdown("Fe_bcc")
        obj.min_wavelength_input = _DummyTextInput("1.0")
        obj.max_wavelength_input = _DummyTextInput("2.0")
        obj.structure_type = "bcc"
        obj.lattice_params = {"a": 1.2}
        obj.flight_path = 16.0

        ctx = obj._build_batch_fit_context()
        self.assertEqual(ctx["structure_type"], "bcc")
        self.assertEqual(ctx["selected_phase"], "Fe_bcc")
        self.assertEqual(ctx["bragg_rows"][0]["hkl"], (1, 1, 0))
        self.assertTrue(ctx["bragg_rows"][0]["valid"])
        self.assertIn("(1; 1; 0)", ctx["bragg_rows_text"][0])

    def test_build_batch_fit_context_marks_invalid_bounds(self):
        obj = _HeadlessFitting()
        obj.bragg_table = _DummyTable(
            [
                [
                    "(1, 1, 0)",
                    "1.697",
                    "1.40",
                    "1.55",
                    "1.35",
                    "1.20",  # invalid: min >= max
                    "1.55",
                    "1.95",
                    "0.006",
                    "0.05",
                    "0.35",
                ]
            ]
        )
        obj.phase_dropdown = _DummyDropdown("Fe_bcc")
        obj.min_wavelength_input = _DummyTextInput("1.0")
        obj.max_wavelength_input = _DummyTextInput("2.0")
        obj.structure_type = "bcc"
        obj.lattice_params = {"a": 1.2}

        ctx = obj._build_batch_fit_context()
        self.assertFalse(ctx["bragg_rows"][0]["valid"])
        self.assertIsNone(ctx["bragg_rows"][0]["regions"])

    def test_fit_region_headless_known_phase(self):
        obj = _HeadlessFitting()

        wavelengths = np.linspace(1.0, 2.2, 1600)
        hkl = (1, 1, 0)
        a_lattice = 1.2
        a0, b0 = 0.3, 0.2
        a_hkl, b_hkl = 0.15, 0.07
        s_val, t_val, eta_val = 0.006, 0.05, 0.35
        r1 = (1.2, 1.35)
        r2 = (1.4, 1.55)
        r3 = (1.55, 1.95)
        intensities = fitting_function_3(
            wavelengths,
            a0,
            b0,
            a_hkl,
            b_hkl,
            s_val,
            t_val,
            eta_val,
            [hkl],
            r3[0],
            r3[1],
            "bcc",
            {"a": a_lattice},
        )
        row_data = {
            "hkl": hkl,
            "d": None,
            "regions": [
                {"min_wavelength": r2[0], "max_wavelength": r2[1]},
                {"min_wavelength": r1[0], "max_wavelength": r1[1]},
                {"min_wavelength": r3[0], "max_wavelength": r3[1]},
            ],
            "s": s_val,
            "t": t_val,
            "eta": eta_val,
        }

        result = obj.fit_region(
            0,
            skip_ui_updates=True,
            row_data=row_data,
            wavelengths=wavelengths,
            intensities=intensities,
            fit_flags=(False, False, False),
            selected_phase="Fe_bcc",
            structure_type="bcc",
            lattice_params={"a": a_lattice},
        )
        self.assertIsInstance(result, dict)
        self.assertIn("fit_params", result)
        self.assertTrue(np.isfinite(result["d_fit"]))

    def test_fit_region_headless_unknown_phase_has_finite_metrics(self):
        obj = _HeadlessFitting()

        wavelengths = np.linspace(0.6, 1.4, 1400)
        # Unknown-phase model in fitting_function_3 uses x_hkl = 2*a when hkl_list is empty.
        a_unknown = 0.5  # edge position around lambda=1.0
        a0, b0 = 0.2, 0.3
        a_hkl, b_hkl = 0.1, 0.15
        s_val, t_val, eta_val = 0.005, 0.06, 0.4
        r1 = (0.8, 0.95)
        r2 = (0.7, 0.82)
        r3 = (0.9, 1.2)
        intensities = fitting_function_3(
            wavelengths,
            a0,
            b0,
            a_hkl,
            b_hkl,
            s_val,
            t_val,
            eta_val,
            [],
            r3[0],
            r3[1],
            "cubic",
            {"a": a_unknown},
        )
        row_data = {
            "hkl": None,
            "d": 1.0,
            "regions": [
                {"min_wavelength": r2[0], "max_wavelength": r2[1]},
                {"min_wavelength": r1[0], "max_wavelength": r1[1]},
                {"min_wavelength": r3[0], "max_wavelength": r3[1]},
            ],
            "s": s_val,
            "t": t_val,
            "eta": eta_val,
        }

        result = obj.fit_region(
            0,
            skip_ui_updates=True,
            row_data=row_data,
            wavelengths=wavelengths,
            intensities=intensities,
            fit_flags=(False, False, False),
            selected_phase="Unknown_Phase",
            structure_type="cubic",
            lattice_params={"a": 1.0},  # intentionally different from fitted edge location
        )

        self.assertIsInstance(result, dict)
        self.assertTrue(np.isfinite(result["edge_height"]))
        self.assertTrue(np.isfinite(result["edge_width"]))
        self.assertGreater(result["edge_height"], 0.0)
        self.assertGreater(result["edge_width"], 0.0)

    def test_fit_full_pattern_core_with_context(self):
        obj = _HeadlessFitting()

        wavelengths = np.linspace(1.0, 2.2, 1600)
        hkl = (1, 1, 0)
        a_lattice = 1.2
        a0, b0 = 0.3, 0.2
        a_hkl, b_hkl = 0.15, 0.07
        s_val, t_val, eta_val = 0.006, 0.05, 0.35
        r1 = (1.2, 1.35)
        r2 = (1.4, 1.55)
        r3 = (1.55, 1.95)
        intensities = fitting_function_3(
            wavelengths,
            a0,
            b0,
            a_hkl,
            b_hkl,
            s_val,
            t_val,
            eta_val,
            [hkl],
            r3[0],
            r3[1],
            "bcc",
            {"a": a_lattice},
        )
        fit_context = {
            "structure_type": "bcc",
            "lattice_params": {"a": a_lattice},
            "bragg_rows": [
                {
                    "row": 0,
                    "hkl": hkl,
                    "d": None,
                    "regions": [
                        {"min_wavelength": r2[0], "max_wavelength": r2[1]},
                        {"min_wavelength": r1[0], "max_wavelength": r1[1]},
                        {"min_wavelength": r3[0], "max_wavelength": r3[1]},
                    ],
                    "s": s_val,
                    "t": t_val,
                    "eta": eta_val,
                    "valid": True,
                }
            ],
        }

        result, error = obj.fit_full_pattern_core(
            fix_s=False,
            fix_t=False,
            fix_eta=False,
            max_nfev=200,
            curve_fit_maxfev=5000,
            fit_context=fit_context,
            wavelengths=wavelengths,
            intensities=intensities,
            apply_lattice_update=False,
        )
        self.assertIsNone(error)
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(float(result["lattice_params"]["a"]), a_lattice, places=2)
        self.assertIn(hkl, result["edge_heights"])
        self.assertIn(hkl, result["edge_widths"])
        self.assertTrue(np.isfinite(result["edge_heights"][hkl]))
        self.assertTrue(np.isfinite(result["edge_widths"][hkl]))


if __name__ == "__main__":
    unittest.main()
