from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prices": [
                    285.12,
                    286.70,
                    284.95,
                    287.31,
                ],
                "horizon": 1,
            }
        }
    )

    prices: list[float] = Field(
        ...,
        min_length=2,
        description=(
            "Fechamentos do mais antigo para o mais recente."
        ),
    )
    horizon: int = Field(
        default=1,
        ge=1,
        le=30,
        description=(
            "Quantidade de dias previstos recursivamente."
        ),
    )

    @field_validator("prices")
    @classmethod
    def validate_prices(
        cls,
        values: list[float],
    ) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "Todos os preços devem ser números finitos."
            )
        if any(value <= 0 for value in values):
            raise ValueError(
                "Todos os preços devem ser positivos."
            )
        return values


class PredictResponse(BaseModel):
    symbol: str | None
    last_price: float
    horizon: int
    predictions: list[float]


class FeedbackRequest(BaseModel):
    predicted_price: float = Field(..., gt=0)
    actual_price: float = Field(..., gt=0)


class FeedbackResponse(BaseModel):
    absolute_error: float
    absolute_percentage_error: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_ready: bool
    symbol: str | None
    window_size: int | None
    min_prices: int | None
    resources: dict[str, float]
    startup_error: str | None = None
