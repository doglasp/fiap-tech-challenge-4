from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    # O exemplo traz os 61 fechamentos que o modelo atual exige
    # (janela de 60 + 1), para que o "Try it out" do Swagger funcione
    # sem edição. São dados reais de AAPL, de 2026-03-24 a 2026-06-18.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prices": [
                    251.41, 252.39, 252.66, 248.57, 246.4, 253.56,
                    255.39, 255.68, 258.62, 253.27, 258.66, 260.25,
                    260.24, 258.96, 258.59, 266.18, 263.16, 269.98,
                    272.8, 265.93, 272.92, 273.18, 270.81, 267.36,
                    270.46, 269.92, 271.1, 279.88, 276.58, 283.92,
                    287.25, 287.18, 293.05, 292.68, 294.8, 298.87,
                    298.21, 300.23, 297.84, 298.97, 302.25, 304.99,
                    308.82, 308.33, 310.85, 312.51, 312.06, 306.31,
                    315.2, 310.26, 311.23, 307.34, 301.54, 290.55,
                    291.58, 295.63, 291.13, 296.42, 299.24, 295.95,
                    298.01,
                ],
                "horizon": 1,
            }
        }
    )

    prices: list[float] = Field(
        ...,
        min_length=2,
        description=(
            "Fechamentos do mais antigo para o mais recente. "
            "A quantidade mínima depende da janela do modelo "
            "carregado e é informada por /ready em 'min_prices'."
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
