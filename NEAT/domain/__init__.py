"""GUI-independent scientific contracts."""

from .fitting import (
    FittingParameterBounds,
    FullPatternEdgeFit,
    FullPatternFitConfig,
    FullPatternFitResult,
    PatternFitRow,
    WavelengthRegion,
)
from .individual_edge import (
    IndividualEdgeFitAttempt,
    IndividualEdgeFitConfig,
    IndividualEdgeFitResult,
    RegionFitCurve,
)

__all__ = [
    "FittingParameterBounds",
    "FullPatternEdgeFit",
    "FullPatternFitConfig",
    "FullPatternFitResult",
    "PatternFitRow",
    "WavelengthRegion",
    "IndividualEdgeFitAttempt",
    "IndividualEdgeFitConfig",
    "IndividualEdgeFitResult",
    "RegionFitCurve",
]
