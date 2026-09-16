"""
OpenJudge: Open-Source System-One Decision Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Non-autoregressive, calibrated, type-safe decision framework for software automation.
"""

from openjudge.types import (
    Decision,
    DecisionResult,
    FieldSpec,
    ConfidenceScore,
    DecisionField,
)
from openjudge.engine import OpenJudgeEngine
from openjudge.calibration import (
    TemperatureScaler,
    PlattScaler,
    ExpectedCalibrationError,
)

__version__ = "0.1.0"
__all__ = [
    "Decision",
    "DecisionResult",
    "FieldSpec",
    "ConfidenceScore",
    "DecisionField",
    "OpenJudgeEngine",
    "TemperatureScaler",
    "PlattScaler",
    "ExpectedCalibrationError",
]
