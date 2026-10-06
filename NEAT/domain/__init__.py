"""GUI-independent scientific contracts."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_EXPORT_MODULES = {
    "FittingParameterBounds": "fitting",
    "FullPatternEdgeFit": "fitting",
    "FullPatternFitConfig": "fitting",
    "FullPatternFitResult": "fitting",
    "PatternFitRow": "fitting",
    "WavelengthRegion": "fitting",
    "IndividualEdgeFitAttempt": "individual_edge",
    "IndividualEdgeFitConfig": "individual_edge",
    "IndividualEdgeFitResult": "individual_edge",
    "RegionFitCurve": "individual_edge",
    "PreprocessingOperationResult": "preprocessing",
    "PreprocessingStatus": "preprocessing",
    "ProducedOutput": "preprocessing",
}

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


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
