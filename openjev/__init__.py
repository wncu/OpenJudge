"""
OpenJev: Open-Source System-One Decision Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Non-autoregressive, calibrated, type-safe decision framework for software automation.
"""

from openjev.types import (
    Decision,
    DecisionResult,
    FieldSpec,
    ConfidenceScore,
    DecisionField,
)
from openjev.engine import OpenJevEngine
from openjev.calibration import (
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
    "OpenJevEngine",
    "TemperatureScaler",
    "PlattScaler",
    "ExpectedCalibrationError",
]
