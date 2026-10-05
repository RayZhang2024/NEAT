"""Staged individual-edge numerics; deliberately preserves the legacy algorithm."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from ..core import calculate_x_hkl_general, fitting_function_1, fitting_function_2, fitting_function_3
from ..core.fitting import initial_value_within_bounds
from ..domain import (
    IndividualEdgeFitAttempt,
    IndividualEdgeFitConfig,
    IndividualEdgeFitResult,
    RegionFitCurve,
)


def fit_individual_edge(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    fit_config: IndividualEdgeFitConfig,
    *,
    fix_s: bool = False,
    fix_t: bool = False,
    fix_eta: bool = False,
) -> IndividualEdgeFitAttempt:
    """Return completed stages even when a later fit fails."""
    if not isinstance(fit_config, IndividualEdgeFitConfig):
        raise TypeError("fit_config must be IndividualEdgeFitConfig")
    try:
        wavelengths_data = np.asarray(wavelengths)
        intensities_data = np.asarray(intensities)
        if fit_config.is_known_phase:
            if "a" not in fit_config.lattice_params:
                raise ValueError("Missing cubic lattice parameter 'a'")
            a_guess = fit_config.lattice_params["a"]
        else:
            if fit_config.d is None:
                raise ValueError("Missing 'd' value for unknown phase")
            a_guess = float(fit_config.d) / 2.0
        r2_min, r2_max = fit_config.window(0)
        r1_min, r1_max = fit_config.window(1)
        r3_min, r3_max = fit_config.window(2)
    except Exception as exc:
        return IndividualEdgeFitAttempt(error_stage="setup", error_message=str(exc))

    region1 = None
    region2 = None
    region3 = None
    try:
        mask_r1 = (wavelengths_data >= r1_min) & (wavelengths_data <= r1_max)
        x_r1, y_r1 = wavelengths_data[mask_r1], intensities_data[mask_r1]
        if x_r1.size == 0:
            return IndividualEdgeFitAttempt(error_stage="Region 1", error_message="has no data in range")
        region1 = RegionFitCurve(x_r1, y_r1)
        popt_r1, _ = curve_fit(
            fitting_function_1, x_r1, y_r1, p0=[0, 0], bounds=([-10, -10], [10, 10])
        )
        a0, b0 = popt_r1
        region1 = RegionFitCurve(
            x_r1, y_r1, fitting_function_1(x_r1, *popt_r1), tuple(popt_r1)
        )
    except Exception as exc:
        return IndividualEdgeFitAttempt(region1=region1, error_stage="Region 1 fit", error_message=str(exc))

    try:
        mask_r2 = (wavelengths_data >= r2_min) & (wavelengths_data <= r2_max)
        x_r2, y_r2 = wavelengths_data[mask_r2], intensities_data[mask_r2]
        if x_r2.size == 0:
            return IndividualEdgeFitAttempt(
                region1=region1, error_stage="Region 2", error_message="has no data in range"
            )
        region2 = RegionFitCurve(x_r2, y_r2)
        popt_r2, _ = curve_fit(
            lambda xx, a_hkl, b_hkl: fitting_function_2(xx, a_hkl, b_hkl, a0, b0),
            x_r2, y_r2, p0=[0, 0],
        )
        a_hkl, b_hkl = popt_r2
        region2 = RegionFitCurve(
            x_r2, y_r2, fitting_function_2(x_r2, a_hkl, b_hkl, a0, b0), tuple(popt_r2)
        )
    except Exception as exc:
        return IndividualEdgeFitAttempt(
            region1=region1, region2=region2, error_stage="Region 2 fit", error_message=str(exc)
        )

    try:
        def span50(value: float) -> tuple[float, float]:
            distance = max(abs(value), 1)
            return value - distance, value + distance

        a0_hat, b0_hat = popt_r1
        a_hkl_hat, b_hkl_hat = popt_r2
        lb4, ub4 = zip(*(span50(p) for p in (a0_hat, b0_hat, a_hkl_hat, b_hkl_hat)))
        if fit_config.is_known_phase:
            if fit_config.hkl is None:
                raise TypeError("Missing hkl for known phase")
            h, k, l = fit_config.hkl
            hkl = (h, k, l)
        else:
            hkl = None

        mask_r3 = (wavelengths_data >= r3_min) & (wavelengths_data <= r3_max)
        x_r3, y_r3 = wavelengths_data[mask_r3], intensities_data[mask_r3]
        region3 = RegionFitCurve(x_r3, y_r3)

        p0 = [a0_hat, b0_hat, a_hkl_hat, b_hkl_hat, a_guess]
        lb = list(lb4) + [a_guess * 0.95]
        ub = list(ub4) + [a_guess * 1.05]
        for name, fixed, value in (
            ("s", fix_s, fit_config.s),
            ("t", fix_t, fit_config.t),
            ("eta", fix_eta, fit_config.eta),
        ):
            if not fixed:
                p0.append(initial_value_within_bounds(value, fit_config.fitting_parameter_bounds[name]))
                lb.append(fit_config.fitting_parameter_bounds[name][0])
                ub.append(fit_config.fitting_parameter_bounds[name][1])

        def func_r3(x: np.ndarray, *params: float) -> np.ndarray:
            a0_fit, b0_fit, a_hkl_fit, b_hkl_fit = params[:4]
            idx = 4
            a_fit = params[idx]
            idx += 1
            s_fit = fit_config.s if fix_s else params[idx]
            idx += 0 if fix_s else 1
            t_fit = fit_config.t if fix_t else params[idx]
            idx += 0 if fix_t else 1
            eta_fit = fit_config.eta if fix_eta else params[idx]
            return fitting_function_3(
                x, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                s_fit, t_fit, eta_fit,
                [hkl] if fit_config.is_known_phase else [],
                r3_min, r3_max, "cubic", {"a": a_fit},
            )

        popt_3, pcov_3 = curve_fit(func_r3, x_r3, y_r3, p0=p0, bounds=(lb, ub), maxfev=300)
        y3_fit = func_r3(x_r3, *popt_3)
        region3 = RegionFitCurve(x_r3, y_r3, y3_fit)
        rms_3 = np.sqrt(np.mean((y_r3 - y3_fit) ** 2))
        a0_fit, b0_fit, a_hkl_fit, b_hkl_fit = popt_3[:4]
        idx = 4
        a_fit = popt_3[idx] * 2
        a_unc = np.sqrt(pcov_3[idx, idx]) * 2
        idx += 1
        if not fix_s:
            s_fit, s_unc = popt_3[idx], np.sqrt(pcov_3[idx, idx])
            idx += 1
        else:
            s_fit, s_unc = fit_config.s, np.nan
        if not fix_t:
            t_fit, t_unc = popt_3[idx], np.sqrt(pcov_3[idx, idx])
            idx += 1
        else:
            t_fit, t_unc = fit_config.t, np.nan
        if not fix_eta:
            eta_fit, eta_unc = popt_3[idx], np.sqrt(pcov_3[idx, idx])
        else:
            eta_fit, eta_unc = fit_config.eta, np.nan

        if fit_config.is_known_phase:
            assert hkl is not None
            denom = np.sqrt(h ** 2 + k ** 2 + l ** 2)
            d_fit, d_unc = a_fit / denom, a_unc / denom
        else:
            d_fit, d_unc = a_fit, a_unc

        try:
            if fit_config.is_known_phase:
                structure = fit_config.structure_type
                lat = fit_config.lattice_params or {"a": a_fit}
                d_vals = calculate_x_hkl_general(structure, lat, [hkl])
                d_hkl = d_vals[0] / 2.0 if d_vals and not np.isnan(d_vals[0]) else np.nan
            else:
                structure = "cubic"
                d_hkl = a_fit / 2.0
                lat = {"a": d_hkl}
        except (TypeError, ValueError, KeyError):
            d_hkl = np.nan

        if np.isnan(d_hkl) or d_hkl <= 0:
            edge_height = np.nan
            edge_width = np.nan
        else:
            x_edge = 2.0 * d_hkl
            try:
                xx_h = np.linspace(r3_min, r3_max, 4000)
                yy_h = fitting_function_3(
                    xx_h, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                    s_fit, t_fit, eta_fit, [hkl] if fit_config.is_known_phase else [],
                    r3_min, r3_max, structure, lat,
                )
                edge_height = yy_h.max() - yy_h.min() if yy_h.size else np.nan
            except (ValueError, FloatingPointError):
                edge_height = np.nan
            try:
                span = max(0.2, 0.1 * (r3_max - r3_min))
                start = max(r3_min, x_edge - span)
                end = min(r3_max, x_edge + span)
                if end <= start:
                    raise ValueError("Invalid span for width computation")
                xx = np.linspace(start, end, 4000)
                yy = fitting_function_3(
                    xx, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                    s_fit, t_fit, eta_fit, [hkl] if fit_config.is_known_phase else [],
                    start, end, structure, lat,
                )
                if yy.size < 5:
                    raise ValueError("Insufficient points for width computation")
                dy = np.gradient(yy, xx)
                dy_max = dy.max()
                if dy_max <= 0:
                    raise ValueError("Non-positive derivative peak")
                half = dy_max / 2.0
                indices = np.where(dy >= half)[0]
                if indices.size == 0:
                    raise ValueError("No points above half maximum")
                edge_width = xx[indices[-1]] - xx[indices[0]]
            except (ValueError, FloatingPointError):
                edge_width = np.nan

        result = IndividualEdgeFitResult(
            source_row=fit_config.source_row,
            hkl=hkl,
            region1=region1,
            region2=region2,
            region3=region3,
            d_fit=d_fit,
            d_unc=d_unc,
            s_fit=s_fit,
            s_unc=s_unc,
            t_fit=t_fit,
            t_unc=t_unc,
            eta_fit=eta_fit,
            eta_unc=eta_unc,
            edge_height=edge_height,
            edge_width=edge_width,
            rms=rms_3,
            baseline_params={"a0": a0_fit, "b0": b0_fit, "a_hkl": a_hkl_fit, "b_hkl": b_hkl_fit},
        )
        return IndividualEdgeFitAttempt(region1=region1, region2=region2, region3=region3, result=result)
    except Exception as exc:
        return IndividualEdgeFitAttempt(
            region1=region1, region2=region2, region3=region3,
            error_stage="Region 3 fit", error_message=str(exc),
        )
