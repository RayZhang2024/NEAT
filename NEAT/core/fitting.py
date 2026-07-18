"""Core fitting utilities for NEAT."""

import numpy as np
from scipy import special


UNCERTAINTY_ESTIMATOR_TARGETS = (
    "macro_pixel_size",
    "uamp",
    "fitting_uncertainty",
)


def _validate_positive_parameter(name, value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc

    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric_value


def calculate_uncertainty_estimator_constant(
    macro_pixel_size,
    uamp,
    fitting_uncertainty,
):
    """Return K = macro pixel size * Uamp * fitting uncertainty squared."""
    macro_pixel_size = _validate_positive_parameter("Macro pixel size", macro_pixel_size)
    uamp = _validate_positive_parameter("Uamp", uamp)
    fitting_uncertainty = _validate_positive_parameter(
        "Fitting uncertainty", fitting_uncertainty
    )
    return macro_pixel_size * uamp * fitting_uncertainty**2


def estimate_uncertainty_parameter(
    constant,
    target,
    *,
    macro_pixel_size=None,
    uamp=None,
    fitting_uncertainty=None,
):
    """Estimate one parameter from K = macro pixel size * Uamp * uncertainty^2."""
    constant = _validate_positive_parameter("Estimator constant", constant)
    if target not in UNCERTAINTY_ESTIMATOR_TARGETS:
        raise ValueError(f"Unknown uncertainty estimator target: {target}")

    if target == "macro_pixel_size":
        uamp = _validate_positive_parameter("Uamp", uamp)
        fitting_uncertainty = _validate_positive_parameter(
            "Fitting uncertainty", fitting_uncertainty
        )
        return constant / (uamp * fitting_uncertainty**2)

    if target == "uamp":
        macro_pixel_size = _validate_positive_parameter(
            "Macro pixel size", macro_pixel_size
        )
        fitting_uncertainty = _validate_positive_parameter(
            "Fitting uncertainty", fitting_uncertainty
        )
        return constant / (macro_pixel_size * fitting_uncertainty**2)

    macro_pixel_size = _validate_positive_parameter("Macro pixel size", macro_pixel_size)
    uamp = _validate_positive_parameter("Uamp", uamp)
    return np.sqrt(constant / (macro_pixel_size * uamp))


def calculate_d_spacing_general(structure_type, lattice_params, hkl):
    """Return d_hkl for the given structure/lattice; np.nan if invalid."""
    try:
        h, k, l = hkl
        structure = (structure_type or "").lower()

        if structure in ("cubic", "fcc", "bcc"):
            a = lattice_params["a"]
            return a / np.sqrt(h**2 + k**2 + l**2)

        if structure == "tetragonal":
            a, c = lattice_params["a"], lattice_params["c"]
            return 1.0 / np.sqrt((h**2 + k**2) / (a**2) + (l**2) / (c**2))

        if structure == "hexagonal":
            a, c = lattice_params["a"], lattice_params["c"]
            return 1.0 / np.sqrt((4.0 / 3.0) * ((h**2 + h * k + k**2) / (a**2)) + (l**2) / (c**2))

        if structure == "orthorhombic":
            a, b, c = lattice_params["a"], lattice_params["b"], lattice_params["c"]
            return 1.0 / np.sqrt((h**2) / (a**2) + (k**2) / (b**2) + (l**2) / (c**2))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return np.nan

    return np.nan


def calculate_theoretical_bragg_edges(structure_type, lattice_params, hkl_list):
    """Return list of ((h, k, l), lambda_hkl) pairs where lambda_hkl = 2*d_hkl."""
    edges = []
    for hkl in hkl_list:
        d_hkl = calculate_d_spacing_general(structure_type, lattice_params, hkl)
        if np.isnan(d_hkl) or d_hkl <= 0:
            edges.append((hkl, np.nan))
        else:
            edges.append((hkl, 2.0 * d_hkl))
    return edges


def fitting_function_1(x, a0, b0):
    return np.exp(-(a0 + b0 * x))


def fitting_function_2(x, a_hkl, b_hkl, a0, b0):
    return np.exp(-(a0 + (b0 * x))) * np.exp(-(a_hkl + (b_hkl * x)))


def fitting_function_3(
    x,
    a0,
    b0,
    a_hkl,
    b_hkl,
    s,
    t,
    eta,
    hkl_list,
    min_wavelength,
    max_wavelength,
    structure_type,
    lattice_params,
):
    """Region-3 model with general crystal-structure Bragg-edge support."""
    if hkl_list:
        x_hkl_list = calculate_x_hkl_general(structure_type, lattice_params, hkl_list)
    else:
        x_hkl_list = [2 * lattice_params.get("a", 0)]

    total_intensity = np.zeros_like(x, dtype=float)

    for x_hkl in x_hkl_list:
        if np.isnan(x_hkl):
            continue
        if (x_hkl < min_wavelength) or (x_hkl > max_wavelength):
            continue

        try:
            def _l_int_exp_tail(x_local, x_edge, s_local, t_local):
                """Integrated Lorentzian convolved with a one-sided exponential tail."""
                u = (x_local - x_edge) / (s_local * np.sqrt(2))
                l_plain = 0.5 + np.arctan(u) / np.pi
                shift = np.exp(-(x_local - x_edge) / t_local)
                u_shift = (x_local - x_edge - s_local**2 / t_local) / (s_local * np.sqrt(2))
                l_shift = 0.5 + np.arctan(u_shift) / np.pi
                return l_plain - shift * (l_plain - l_shift)

            g_term = 0.5 * (
                special.erfc(-(x - x_hkl) / (np.sqrt(2) * s))
                - np.exp(-(x - x_hkl) / t + s**2 / (2 * t**2))
                * special.erfc(-(x - x_hkl) / (np.sqrt(2) * s) + s / t)
            )

            l_term = _l_int_exp_tail(x, x_hkl, s, t)
            step = (1.0 - eta) * g_term + eta * l_term

            pre = np.exp(-(a_hkl + b_hkl * x))
            edge = np.exp(-(a0 + b0 * x)) * (pre + (1.0 - pre) * step)
            total_intensity += edge

        except (FloatingPointError, ValueError, OverflowError):
            continue

    return total_intensity


def calculate_x_hkl_general(structure_type, lattice_params, hkl_list):
    """Return theoretical Bragg-edge positions (lambda_hkl = 2*d_hkl)."""
    x_hkl_list = []
    for hkl in hkl_list:
        d_hkl = calculate_d_spacing_general(structure_type, lattice_params, hkl)
        if d_hkl > 0:
            x_hkl_list.append(2.0 * d_hkl)
        else:
            x_hkl_list.append(np.nan)
    return x_hkl_list


_FCC_HKL_DEFAULT = [
    (1, 1, 1),
    (2, 0, 0),
    (2, 2, 0),
    (3, 1, 1),
    (2, 2, 2),
    (4, 0, 0),
    (3, 3, 1),
    (4, 2, 0),
]

_BCC_HKL_DEFAULT = [
    (1, 1, 0),
    (2, 0, 0),
    (2, 1, 1),
    (2, 2, 0),
    (3, 1, 0),
    (2, 2, 2),
    (3, 2, 1),
    (3, 3, 0),
]

PHASE_DATA = {
    "Unknown_Phase": {
        "structure": "unknown",
        "lattice_params": {},
        "hkl_list": [],
    },
    "Cu_fcc": {
        "structure": "fcc",
        "lattice_params": {"a": 3.615},
        "hkl_list": list(_FCC_HKL_DEFAULT),
    },
    "Fe_bcc": {
        "structure": "bcc",
        "lattice_params": {"a": 2.86},
        "hkl_list": list(_BCC_HKL_DEFAULT),
    },
    "Fe_fcc": {
        "structure": "fcc",
        "lattice_params": {"a": 3.61},
        "hkl_list": list(_FCC_HKL_DEFAULT),
    },
    "Al": {
        "structure": "fcc",
        "lattice_params": {"a": 4.05},
        "hkl_list": list(_FCC_HKL_DEFAULT),
    },
    "Ni_gamma": {
        "structure": "fcc",
        "lattice_params": {"a": 3.60},
        "hkl_list": list(_FCC_HKL_DEFAULT),
    },
    "CeO2": {
        "structure": "fcc",
        "lattice_params": {"a": 5.41},
        "hkl_list": list(_FCC_HKL_DEFAULT),
    },
    "Ti_Beta": {
        "structure": "bcc",
        "lattice_params": {"a": 3.32},
        "hkl_list": [(1, 1, 0), (2, 0, 0), (2, 1, 1), (2, 2, 0), (3, 1, 0), (2, 2, 2)],
    },
    "Ti_alpha_hex": {
        "structure": "hexagonal",
        "lattice_params": {"a": 2.95, "c": 4.68},
        "hkl_list": [(1, 0, 0), (0, 0, 2), (1, 0, 1), (1, 1, 0), (2, 0, 0), (1, 1, 2), (2, 1, 1)],
    },
}


__all__ = [
    "UNCERTAINTY_ESTIMATOR_TARGETS",
    "calculate_uncertainty_estimator_constant",
    "estimate_uncertainty_parameter",
    "fitting_function_1",
    "fitting_function_2",
    "fitting_function_3",
    "calculate_d_spacing_general",
    "calculate_theoretical_bragg_edges",
    "calculate_x_hkl_general",
    "PHASE_DATA",
]
