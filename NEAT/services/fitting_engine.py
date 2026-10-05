"""GUI-independent numerical engine for full-pattern fitting."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.optimize import curve_fit, least_squares

from ..core import fitting_function_1, fitting_function_2, fitting_function_3
from ..core.fitting import (
    calculate_x_hkl_general,
    initial_value_within_bounds,
)
from ..domain import (
    FullPatternEdgeFit, FullPatternFitConfig, FullPatternFitResult,
    IndividualEdgeFitAttempt, IndividualEdgeFitConfig,
)
from .individual_edge import fit_individual_edge as _fit_individual_edge


class FittingEngine:
    """Run full-pattern fitting from explicit arrays and typed scientific config."""

    _STRUCTURE_CONFIG: ClassVar[dict[str, list[str]]] = {
        "cubic": ["a"],
        "fcc": ["a"],
        "bcc": ["a"],
        "tetragonal": ["a", "c"],
        "hexagonal": ["a", "c"],
        "orthorhombic": ["a", "b", "c"],
    }

    def fit_individual_edge(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        fit_config: IndividualEdgeFitConfig,
        *,
        fix_s: bool = False,
        fix_t: bool = False,
        fix_eta: bool = False,
    ) -> IndividualEdgeFitAttempt:
        return _fit_individual_edge(
            wavelengths, intensities, fit_config,
            fix_s=fix_s, fix_t=fix_t, fix_eta=fix_eta,
        )

    def fit_full_pattern(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        fit_config: FullPatternFitConfig,
        *,
        fix_s: bool = False,
        fix_t: bool = False,
        fix_eta: bool = False,
        max_nfev: int = 300,
        curve_fit_maxfev: int | None = None,
    ) -> tuple[FullPatternFitResult | None, str | None]:
        """Fit a spectrum using the typed full-pattern scientific contract."""
        if not isinstance(fit_config, FullPatternFitConfig):
            raise TypeError("fit_config must be FullPatternFitConfig")
        structure_type = fit_config.structure_type
        if structure_type not in self._STRUCTURE_CONFIG:
            return None, f"Unsupported structure type: {structure_type}"

        required_params = self._STRUCTURE_CONFIG[structure_type]
        lattice_params = dict(fit_config.lattice_params)
        if not lattice_params:
            return None, "Lattice parameters not initialized"
        missing_params = [name for name in required_params if name not in lattice_params]
        if missing_params:
            return None, f"Missing parameters for {structure_type}: {missing_params}"

        parameter_bounds = fit_config.fitting_parameter_bounds

        wavelengths_data = np.asarray(wavelengths)
        intensities_data = np.asarray(intensities)
        if wavelengths_data.shape != intensities_data.shape:
            return None, "Wavelength and intensity arrays must have the same shape"

        source_rows = fit_config.bragg_rows
        if not source_rows:
            return None, "No Bragg edges to fit"

        bragg_edges: list[FullPatternEdgeFit] = []
        for row_data in source_rows:
            if not row_data.valid:
                continue
            hkl = row_data.hkl
            regions = row_data.regions
            if hkl is None or not regions or len(regions) < 3:
                continue
            s_val, t_val, eta_val = row_data.s, row_data.t, row_data.eta
            if s_val is None or t_val is None or eta_val is None:
                continue

            r3_min = regions[2].min_wavelength
            r3_max = regions[2].max_wavelength
            mask_r3 = (wavelengths_data >= r3_min) & (wavelengths_data <= r3_max)
            x_r3 = wavelengths_data[mask_r3]
            y_r3 = intensities_data[mask_r3]
            if len(x_r3) == 0:
                continue

            try:
                mask_r1 = (wavelengths_data >= regions[1].min_wavelength) & (
                    wavelengths_data <= regions[1].max_wavelength
                )
                x_r1, y_r1 = wavelengths_data[mask_r1], intensities_data[mask_r1]
                cf_kwargs = {} if curve_fit_maxfev is None else {"maxfev": curve_fit_maxfev}
                popt_r1, _ = curve_fit(
                    fitting_function_1,
                    x_r1,
                    y_r1,
                    p0=[0, 0],
                    bounds=([-10, -10], [10, 10]),
                    **cf_kwargs,
                )
                a0, b0 = popt_r1

                mask_r2 = (wavelengths_data >= regions[0].min_wavelength) & (
                    wavelengths_data <= regions[0].max_wavelength
                )
                x_r2, y_r2 = wavelengths_data[mask_r2], intensities_data[mask_r2]
                popt_r2, _ = curve_fit(
                    lambda xx, a, b, a0=a0, b0=b0: fitting_function_2(xx, a, b, a0, b0),
                    x_r2,
                    y_r2,
                    p0=[0, 0],
                    **cf_kwargs,
                )
                a_hkl, b_hkl = popt_r2
            except (RuntimeError, ValueError, TypeError, FloatingPointError) as exc:
                return None, f"Fitting error for hkl{hkl}: {exc}"

            bragg_edges.append(FullPatternEdgeFit(
                hkl=hkl, a0=a0, b0=b0, a_hkl=a_hkl, b_hkl=b_hkl,
                s=s_val, t=t_val, eta=eta_val,
                x_r3=x_r3, y_r3=y_r3, regions=regions,
            ))

        if not bragg_edges:
            return None, "No valid edges with Region 3 data"

        lattice_initial = [lattice_params[name] for name in required_params]
        initial_guess = list(lattice_initial)
        lower_bounds = [value * 0.95 for value in lattice_initial]
        upper_bounds = [value * 1.05 for value in lattice_initial]
        for edge in bragg_edges:
            for key in ("a0", "b0", "a_hkl", "b_hkl"):
                value = getattr(edge, key)
                half = max(abs(value), 1)
                initial_guess.append(value)
                lower_bounds.append(value - half)
                upper_bounds.append(value + half)

        s_initial = [edge.s for edge in bragg_edges]
        t_initial = [edge.t for edge in bragg_edges]
        eta_initial = [edge.eta for edge in bragg_edges]
        for fixed, values, name in (
            (fix_s, s_initial, "s"),
            (fix_t, t_initial, "t"),
            (fix_eta, eta_initial, "eta"),
        ):
            if not fixed:
                lower, upper = parameter_bounds[name]
                for value in values:
                    initial_guess.append(initial_value_within_bounds(value, (lower, upper)))
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)

        concatenated_x = np.concatenate([edge.x_r3 for edge in bragg_edges])
        concatenated_y = np.concatenate([edge.y_r3 for edge in bragg_edges])
        edge_indices = np.concatenate(
            [np.full_like(edge.x_r3, i, dtype=int) for i, edge in enumerate(bragg_edges)]
        )
        n_lat = len(required_params)
        n_edge = len(bragg_edges)
        n_ab = 4 * n_edge

        def residuals(params):
            idx = n_lat
            lattice_dict = dict(zip(required_params, params[:n_lat]))
            ab_block = params[idx : idx + n_ab].reshape(n_edge, 4)
            idx += n_ab
            if not fix_s:
                s_params = params[idx : idx + n_edge]
                idx += n_edge
            else:
                s_params = s_initial
            if not fix_t:
                t_params = params[idx : idx + n_edge]
                idx += n_edge
            else:
                t_params = t_initial
            eta_params = params[idx : idx + n_edge] if not fix_eta else eta_initial

            model = np.zeros_like(concatenated_y)
            for edge_idx, edge in enumerate(bragg_edges):
                mask = edge_indices == edge_idx
                a0_fit, b0_fit, a_hkl_fit, b_hkl_fit = ab_block[edge_idx]
                r3_min = edge.regions[2].min_wavelength
                r3_max = edge.regions[2].max_wavelength
                model[mask] = fitting_function_3(
                    concatenated_x[mask],
                    a0_fit,
                    b0_fit,
                    a_hkl_fit,
                    b_hkl_fit,
                    s_params[edge_idx],
                    t_params[edge_idx],
                    eta_params[edge_idx],
                    [edge.hkl],
                    r3_min,
                    r3_max,
                    structure_type,
                    lattice_dict,
                )
            return concatenated_y - model

        try:
            result = least_squares(
                residuals,
                initial_guess,
                bounds=(lower_bounds, upper_bounds),
                max_nfev=max_nfev,
                verbose=0,
            )
        except Exception as exc:  # noqa: BLE001 - preserve the compatibility error contract
            return None, f"Optimization failed: {exc}"
        if not result.success:
            return None, f"Fit did not converge: {result.message}"

        final_params = result.x
        lattice_fit = dict(zip(required_params, final_params[:n_lat]))
        ab_fit_block = final_params[n_lat : n_lat + n_ab].reshape(n_edge, 4)
        final_res = residuals(final_params)
        n_observations, n_parameters = len(concatenated_y), len(final_params)
        variance = np.sum(final_res**2) / max(1, n_observations - n_parameters)
        try:
            covariance = np.linalg.inv(result.jac.T @ result.jac) * variance
            param_stderr = np.sqrt(np.diag(covariance))
        except np.linalg.LinAlgError:
            param_stderr = np.full_like(final_params, np.inf)

        idx = n_lat + n_ab
        fitted_s_vals: np.ndarray | list[float]
        s_uncertainties: np.ndarray | list[float]
        if not fix_s:
            fitted_s_vals = final_params[idx : idx + n_edge]
            s_uncertainties = param_stderr[idx : idx + n_edge]
            idx += n_edge
        else:
            fitted_s_vals, s_uncertainties = s_initial, [np.nan] * n_edge
        fitted_t_vals: np.ndarray | list[float]
        t_uncertainties: np.ndarray | list[float]
        if not fix_t:
            fitted_t_vals = final_params[idx : idx + n_edge]
            t_uncertainties = param_stderr[idx : idx + n_edge]
            idx += n_edge
        else:
            fitted_t_vals, t_uncertainties = t_initial, [np.nan] * n_edge
        fitted_eta_vals: np.ndarray | list[float]
        eta_uncertainties: np.ndarray | list[float]
        if not fix_eta:
            fitted_eta_vals = final_params[idx : idx + n_edge]
            eta_uncertainties = param_stderr[idx : idx + n_edge]
        else:
            fitted_eta_vals, eta_uncertainties = eta_initial, [np.nan] * n_edge

        lattice_uncertainties = {
            name: param_stderr[i] for i, name in enumerate(required_params)
        }
        s_unc_dict, t_unc_dict, eta_unc_dict = {}, {}, {}
        for edge, s_value, s_unc, t_value, t_unc, eta_value, eta_unc in zip(
            bragg_edges,
            fitted_s_vals,
            s_uncertainties,
            fitted_t_vals,
            t_uncertainties,
            fitted_eta_vals,
            eta_uncertainties,
        ):
            s_unc_dict[edge.hkl] = s_unc
            t_unc_dict[edge.hkl] = t_unc
            eta_unc_dict[edge.hkl] = eta_unc

        model_vals = concatenated_y - final_res
        order = np.argsort(concatenated_x)
        edge_heights, edge_widths = {}, {}
        for i, edge in enumerate(bragg_edges):
            hkl = edge.hkl
            x_vals = calculate_x_hkl_general(structure_type, lattice_fit, [hkl])
            d_hkl = x_vals[0] / 2.0 if x_vals and not np.isnan(x_vals[0]) else np.nan
            if np.isnan(d_hkl) or d_hkl <= 0:
                edge_heights[hkl] = edge_widths[hkl] = np.nan
                continue
            x_edge = 2.0 * d_hkl
            a0_fit, b0_fit, a_hkl_fit, b_hkl_fit = ab_fit_block[i]
            s_value, t_value, eta_value = (
                fitted_s_vals[i],
                fitted_t_vals[i],
                fitted_eta_vals[i],
            )
            r3_min = edge.regions[2].min_wavelength
            r3_max = edge.regions[2].max_wavelength
            try:
                xx_h = np.linspace(r3_min, r3_max, 4000)
                yy_h = fitting_function_3(
                    xx_h, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                    s_value, t_value, eta_value, [hkl], r3_min, r3_max,
                    structure_type, lattice_fit
                )
                edge_height = yy_h.max() - yy_h.min() if yy_h.size else np.nan
            except (ValueError, FloatingPointError, TypeError):
                edge_height = np.nan
            try:
                span = max(0.2, 0.1 * (r3_max - r3_min))
                start, end = max(r3_min, x_edge - span), min(r3_max, x_edge + span)
                if end <= start:
                    raise ValueError("Invalid span for width computation")
                xx = np.linspace(start, end, 4000)
                yy = fitting_function_3(
                    xx, a0_fit, b0_fit, a_hkl_fit, b_hkl_fit,
                    s_value, t_value, eta_value, [hkl], start, end,
                    structure_type, lattice_fit
                )
                if yy.size < 5:
                    raise ValueError("Insufficient points for width computation")
                derivative = np.gradient(yy, xx)
                derivative_max = derivative.max()
                if derivative_max <= 0:
                    raise ValueError("Non-positive derivative peak")
                indices = np.where(derivative >= derivative_max / 2.0)[0]
                if indices.size == 0:
                    raise ValueError("No points above half maximum")
                edge_width = xx[indices[-1]] - xx[indices[0]]
            except (ValueError, FloatingPointError, TypeError):
                edge_width = np.nan
            edge_heights[hkl] = edge_height
            edge_widths[hkl] = edge_width

        return FullPatternFitResult(
            ab_fits={edge.hkl: tuple(ab_fit_block[i]) for i, edge in enumerate(bragg_edges)},
            bragg_edges=tuple(bragg_edges),
            structure_type=structure_type,
            lattice_params=lattice_fit,
            lattice_uncertainties=lattice_uncertainties,
            fitted_s={edge.hkl: value for edge, value in zip(bragg_edges, fitted_s_vals)},
            fitted_t={edge.hkl: value for edge, value in zip(bragg_edges, fitted_t_vals)},
            fitted_eta={edge.hkl: value for edge, value in zip(bragg_edges, fitted_eta_vals)},
            s_uncertainties=s_unc_dict,
            t_uncertainties=t_unc_dict,
            eta_uncertainties=eta_unc_dict,
            x_data=concatenated_x[order],
            y_data=model_vals[order],
            x_exp_sorted=concatenated_x[order],
            y_exp_sorted=concatenated_y[order],
            residuals=final_res,
            success=result.success,
            message=result.message,
            edge_heights=edge_heights,
            edge_widths=edge_widths,
        ), None
