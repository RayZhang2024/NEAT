"""GUI-independent contract for the staged individual-edge fit."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping, cast

import numpy as np

from .fitting import FittingParameterBounds, HKL, WavelengthRegion

EdgeWindow = WavelengthRegion | tuple[float, float]


@dataclass(frozen=True)
class IndividualEdgeFitConfig:
    source_row: int
    hkl: HKL | None
    d: float | None
    regions: tuple[EdgeWindow, EdgeWindow, EdgeWindow]
    s: float
    t: float
    eta: float
    is_known_phase: bool
    structure_type: str
    lattice_params: Mapping[str, float]
    fitting_parameter_bounds: FittingParameterBounds = field(default_factory=FittingParameterBounds)
    # Legacy reversed/empty windows must reach their original fitting stage.
    legacy_window_order: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.regions) != 3:
            raise ValueError("Invalid region bounds for selected edge")
        regions: list[EdgeWindow] = []
        for region in self.regions:
            if isinstance(region, WavelengthRegion):
                regions.append(region)
            else:
                lower, upper = (float(value) for value in region)
                if self.legacy_window_order and (
                    not isfinite(lower) or not isfinite(upper) or lower >= upper
                ):
                    regions.append((lower, upper))
                else:
                    regions.append(WavelengthRegion(lower, upper))
        object.__setattr__(self, "regions", cast(tuple[EdgeWindow, EdgeWindow, EdgeWindow], tuple(regions)))
        if self.hkl is not None:
            hkl = tuple(self.hkl)
            if len(hkl) != 3 or not all(isinstance(value, Integral) for value in hkl):
                raise ValueError("HKL must have three integer components")
            object.__setattr__(self, "hkl", cast(HKL, tuple(int(value) for value in hkl)))
        object.__setattr__(self, "lattice_params", MappingProxyType(dict(self.lattice_params)))
        if not isinstance(self.fitting_parameter_bounds, FittingParameterBounds):
            raise TypeError("Fitting bounds must be FittingParameterBounds")
        for name in ("s", "t", "eta"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.d is not None:
            object.__setattr__(self, "d", float(self.d))

    def window(self, index: int) -> tuple[float, float]:
        region = self.regions[index]
        if isinstance(region, WavelengthRegion):
            return region.min_wavelength, region.max_wavelength
        return region

    @classmethod
    def from_legacy_row(
        cls,
        row: Mapping[str, Any],
        *,
        source_row: int,
        is_known_phase: bool,
        structure_type: str,
        lattice_params: Mapping[str, float],
        fitting_parameter_bounds: FittingParameterBounds,
    ) -> IndividualEdgeFitConfig:
        raw_regions = row.get("regions") or []
        if len(raw_regions) != 3:
            raise ValueError("Invalid region bounds for selected edge")
        regions = tuple(
            (region["min_wavelength"], region["max_wavelength"])
            for region in raw_regions
        )
        raw_hkl = row.get("hkl")
        hkl: HKL | None = None
        if raw_hkl is not None:
            try:
                values = tuple(raw_hkl)
                if len(values) == 3 and all(isinstance(value, Integral) for value in values):
                    hkl = cast(HKL, tuple(int(value) for value in values))
            except TypeError:
                pass
        return cls(
            source_row=source_row,
            hkl=hkl,
            d=row.get("d"),
            regions=cast(tuple[EdgeWindow, EdgeWindow, EdgeWindow], regions),
            s=row["s"],
            t=row["t"],
            eta=row["eta"],
            is_known_phase=is_known_phase,
            structure_type=structure_type,
            lattice_params=lattice_params,
            fitting_parameter_bounds=fitting_parameter_bounds,
            legacy_window_order=True,
        )


@dataclass(frozen=True)
class RegionFitCurve:
    x: np.ndarray
    y: np.ndarray
    fit: np.ndarray | None = None
    params: tuple[float, ...] | None = None


@dataclass(frozen=True)
class IndividualEdgeFitResult:
    source_row: int
    hkl: HKL | None
    region1: RegionFitCurve
    region2: RegionFitCurve
    region3: RegionFitCurve
    d_fit: float
    d_unc: float
    s_fit: float
    s_unc: float
    t_fit: float
    t_unc: float
    eta_fit: float
    eta_unc: float
    edge_height: float
    edge_width: float
    rms: float
    baseline_params: Mapping[str, float]

    @property
    def fit_params(self) -> tuple[float, float, float, float, float, float, float, float]:
        return (self.d_fit, self.s_fit, self.t_fit, self.eta_fit,
                self.d_unc, self.s_unc, self.t_unc, self.eta_unc)

    def to_legacy_dict(self, *, skip_ui_updates: bool) -> dict[str, Any]:
        identity: HKL | str = self.hkl if self.hkl is not None else f"edge{self.source_row + 1}"
        common = {
            "hkl": identity,
            "x": self.region3.x,
            "fit": self.region3.fit,
            "fit_params": self.fit_params,
            "edge_height": self.edge_height,
            "edge_width": self.edge_width,
            "rms": self.rms,
            "baseline_params": dict(self.baseline_params),
        }
        if skip_ui_updates:
            common.update({
                "d_fit": self.d_fit, "d_unc": self.d_unc,
                "s_fit": self.s_fit, "s_unc": self.s_unc,
                "t_fit": self.t_fit, "t_unc": self.t_unc,
                "eta_fit": self.eta_fit, "eta_unc": self.eta_unc,
            })
        else:
            common["y"] = self.region3.y
        return common


@dataclass(frozen=True)
class IndividualEdgeFitAttempt:
    region1: RegionFitCurve | None = None
    region2: RegionFitCurve | None = None
    region3: RegionFitCurve | None = None
    result: IndividualEdgeFitResult | None = None
    error_stage: str | None = None
    error_message: str | None = None
