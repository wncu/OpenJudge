from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin
from pydantic import BaseModel, Field, create_model
import numpy as np


class ConfidenceScore(BaseModel):
    selected_value: Any
    confidence: float
    probabilities: Dict[str, float]
    entropy: float
    is_confident: bool = True


class FieldSpec(BaseModel):
    name: str
    description: Optional[str] = None
    options: List[str]
    is_boolean: bool = False
    temperature: float = 1.0


class DecisionResult(BaseModel):
    values: Dict[str, Any]
    confidence_scores: Dict[str, ConfidenceScore]
    latency_ms: float
    total_tokens_evaluated: int
    is_fully_calibrated: bool = True

    def get_typed_instance(self, model_cls: Type[BaseModel]) -> BaseModel:
        """Instantiate target Pydantic model with guaranteed type safety."""
        return model_cls.model_validate(self.values)


class Decision:
    """Helper methods to inspect Pydantic models and extract candidate decision fields."""

    @staticmethod
    def extract_specs(model_cls: Type[BaseModel]) -> List[FieldSpec]:
        specs = []
        for name, field_info in model_cls.model_fields.items():
            annotation = field_info.annotation
            desc = field_info.description or name

            # Handle Enums
            if isinstance(annotation, type) and issubclass(annotation, Enum):
                options = [e.value if isinstance(e.value, str) else e.name for e in annotation]
                specs.append(FieldSpec(name=name, description=desc, options=options, is_boolean=False))
            # Handle Booleans
            elif annotation is bool:
                specs.append(FieldSpec(name=name, description=desc, options=["True", "False"], is_boolean=True))
            # Handle Literal types
            elif get_origin(annotation) is type(Union) or str(get_origin(annotation)) == "typing.Literal":
                args = get_args(annotation)
                options = [str(arg) for arg in args]
                specs.append(FieldSpec(name=name, description=desc, options=options, is_boolean=False))
            else:
                # Default generic categorical field fallback
                specs.append(FieldSpec(name=name, description=desc, options=["True", "False"], is_boolean=True))
        return specs


def DecisionField(description: str, **kwargs):
    """Pydantic field helper for OpenJev decision models."""
    return Field(description=description, **kwargs)
