from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field

class ChurnRequest(BaseModel):
    data: dict[str, Any] = Field(..., description="Customer features as key/value pairs")

class Factor(BaseModel):
    feature: str
    value: Optional[Any] = None
    contribution: float
    abs_contribution: float

class ChurnResponse(BaseModel):
    churn_probability: float
    churn_label: int
    threshold: float
    model_version: str
    top_factors: list[Factor]