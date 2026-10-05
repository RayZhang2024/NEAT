"""Typed scientific inputs and outputs for full-pattern fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping, cast

import numpy as np

from ..core.fitting import (
    DEFAULT_FITTING_PARAMETER_BOUNDS,
    normalize_fitting_parameter_bounds,
)

HKL = tuple[int, int, int]


@dataclass(frozen=True)
class WavelengthRegion:
    min_wavelength: float
    max_wavelength: float

    def __post_init__(self) -> None:
        lower = float(self.min_wavelength)
        upper = float(self.max_wavelength)
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError("Region bounds must be finite and increasing")
        object.__setattr__(self, "min_wavelength", lower)
        object.__setattr__(self, "max_wavelength", upper)

    def to_legacy_dict(self) -> dict[str, float]:
        return {"min_wavelength": self.min_wavelength, "max_wavelength": self.max_wavelength}


@dataclass(frozen=True)
class PatternFitRow:
    source_row: int
    hkl: HKL | None
    regions: tuple[WavelengthRegion, ...] | None
    s: float | None
    t: float | None
    eta: float | None
    valid: bool

    def __post_init__(self) -> None:
        if self.hkl is not None:
            hkl = tuple(self.hkl)
            if len(hkl) != 3 or not all(isinstance(value, Integral) for value in hkl):
                raise ValueError("HKL must have three integer components")
            object.__setattr__(self, "hkl", tuple(int(value) for value in hkl))
        if self.regions is not None:
            regions = tuple(self.regions)
            if not all(isinstance(region, WavelengthRegion) for region in regions):
                raise TypeError("Regions must contain WavelengthRegion values")
            object.__setattr__(self, "regions", regions)

    @classmethod
    def from_legacy_dict(cls, row: Mapping[str, Any], source_row: int) -> PatternFitRow:
        """Retain unusable rows so the engine can distinguish them from no rows."""
        raw_hkl = row.get("hkl")
        hkl: HKL | None = None
        if raw_hkl is not None:
            try:
                values = tuple(raw_hkl)
                if len(values) == 3 and all(isinstance(value, Integral) for value in values):
                    hkl = cast(HKL, tuple(int(value) for value in values))
            except TypeError:
                pass

        regions: tuple[WavelengthRegion, ...] | None = None
        raw_regions = row.get("regions")
        if raw_regions:
            try:
                regions = tuple(
                    WavelengthRegion(region["min_wavelength"], region["max_wavelength"])
                    for region in raw_regions
                )
            except (KeyError, TypeError, ValueError):
                pass
        return cls(
            source_row=source_row,
            hkl=hkl,
            regions=regions,
            s=row.get("s", 0.001),
            t=row.get("t", 0.01),
            eta=row.get("eta", 0.5),
            valid=bool(row.get("valid")),
        )


@dataclass(frozen=True)
class FittingParameterBounds:
    s: tuple[float, float] = field(default_factory=lambda: DEFAULT_FITTING_PARAMETER_BOUNDS["s"])
    t: tuple[float, float] = field(default_factory=lambda: DEFAULT_FITTING_PARAMETER_BOUNDS["t"])
    eta: tuple[float, float] = field(default_factory=lambda: DEFAULT_FITTING_PARAMETER_BOUNDS["eta"])

    def __post_init__(self) -> None:
        normalized = normalize_fitting_parameter_bounds(
            {"s": self.s, "t": self.t, "eta": self.eta}
        )
        for name, pair in normalized.items():
            object.__setattr__(self, name, pair)

    @classmethod
    def from_legacy_dict(cls, value: Any) -> FittingParameterBounds:
        try:
            normalized = normalize_fitting_parameter_bounds(value)
        except ValueError:
            normalized = normalize_fitting_parameter_bounds()
        return cls(**normalized)

    def __getitem__(self, name: str) -> tuple[float, float]:
        return {"s": self.s, "t": self.t, "eta": self.eta}[name]


@dataclass(frozen=True)
class FullPatternFitConfig:
    structure_type: str = "cubic"
    lattice_params: Mapping[str, float] = field(default_factory=dict)
    fitting_parameter_bounds: FittingParameterBounds = field(default_factory=FittingParameterBounds)
    bragg_rows: tuple[PatternFitRow, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.fitting_parameter_bounds, FittingParameterBounds):
            raise TypeError("Fitting bounds must be FittingParameterBounds")
        object.__setattr__(self, "lattice_params", MappingProxyType(dict(self.lattice_params)))
        rows = tuple(self.bragg_rows)
        if not all(isinstance(row, PatternFitRow) for row in rows):
            raise TypeError("Pattern rows must be PatternFitRow values")
        object.__setattr__(self, "bragg_rows", rows)

    @classmethod
    def from_legacy_dict(cls, context: Mapping[str, Any]) -> FullPatternFitConfig:
        """Convert only scientific fields from the shared legacy table snapshot."""
        return cls(
            structure_type=context.get("structure_type", "cubic"),
            lattice_params=context.get("lattice_params") or {},
            fitting_parameter_bounds=FittingParameterBounds.from_legacy_dict(
                context.get("fitting_parameter_bounds", DEFAULT_FITTING_PARAMETER_BOUNDS)
            ),
            bragg_rows=tuple(
                PatternFitRow.from_legacy_dict(row, index)
                for index, row in enumerate(context.get("bragg_rows") or [])
            ),
        )


@dataclass
class FullPatternEdgeFit:
    hkl: HKL
    a0: float
    b0: float
    a_hkl: float
    b_hkl: float
    s: float
    t: float
    eta: float
    x_r3: np.ndarray
    y_r3: np.ndarray
    regions: tuple[WavelengthRegion, ...]

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "hkl": self.hkl,
            "a0": self.a0,
            "b0": self.b0,
            "a_hkl": self.a_hkl,
            "b_hkl": self.b_hkl,
            "s": self.s,
            "t": self.t,
            "eta": self.eta,
            "x_r3": self.x_r3,
            "y_r3": self.y_r3,
            "regions": [region.to_legacy_dict() for region in self.regions],
        }


@dataclass
class FullPatternFitResult:
    ab_fits: dict[HKL, tuple[float, float, float, float]]
    bragg_edges: tuple[FullPatternEdgeFit, ...]
    structure_type: str
    lattice_params: dict[str, float]
    lattice_uncertainties: dict[str, float]
    fitted_s: dict[HKL, float]
    fitted_t: dict[HKL, float]
    fitted_eta: dict[HKL, float]
    s_uncertainties: dict[HKL, float]
    t_uncertainties: dict[HKL, float]
    eta_uncertainties: dict[HKL, float]
    x_data: np.ndarray
    y_data: np.ndarray
    x_exp_sorted: np.ndarray
    y_exp_sorted: np.ndarray
    residuals: np.ndarray
    success: bool
    message: str
    edge_heights: dict[HKL, float]
    edge_widths: dict[HKL, float]

    def to_legacy_dict(self) -> dict[str, Any]:
        """Reconstruct the established GUI result keys and ordered edge payload."""
        return {
            "ab_fits": self.ab_fits,
            "bragg_edges": [edge.to_legacy_dict() for edge in self.bragg_edges],
            "structure_type": self.structure_type,
            "lattice_params": self.lattice_params,
            "lattice_uncertainties": self.lattice_uncertainties,
            "fitted_s": self.fitted_s,
            "fitted_t": self.fitted_t,
            "fitted_eta": self.fitted_eta,
            "s_uncertainties": self.s_uncertainties,
            "t_uncertainties": self.t_uncertainties,
            "eta_uncertainties": self.eta_uncertainties,
            "x_data": self.x_data,
            "y_data": self.y_data,
            "x_exp_sorted": self.x_exp_sorted,
            "y_exp_sorted": self.y_exp_sorted,
            "residuals": self.residuals,
            "success": self.success,
            "message": self.message,
            "edge_heights": self.edge_heights,
            "edge_widths": self.edge_widths,
        }
