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
from .preprocessing import (
    PreprocessingOperationResult,
    PreprocessingStatus,
    ProducedOutput,
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
    "PreprocessingOperationResult",
    "PreprocessingStatus",
    "ProducedOutput",
]
