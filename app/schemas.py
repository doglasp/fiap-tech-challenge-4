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
                "symbol": "AAPL",
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
    symbol: str | None = Field(
        default=None,
        max_length=20,
        description=(
            "Ação a que os preços se referem. É opcional e serve "
            "apenas para rotular a resposta: o modelo carregado é "
            "utilizado de qualquer forma. Consulte "
            "'model_trained_on' na resposta para saber em qual ação "
            "esse modelo foi treinado."
        ),
    )

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        return normalized or None

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
    # 'symbol' descreve a ação que a requisição diz representar e
    # 'model_trained_on' descreve o modelo que produziu a previsão.
    # Quando divergem, o modelo está sendo aplicado a uma ação
    # diferente da que viu no treino — e isso precisa ficar visível
    # em vez de ser mascarado por um único campo.
    symbol: str | None = Field(
        default=None,
        description=(
            "Ação informada na requisição. Nulo quando não informada."
        ),
    )
    model_trained_on: str | None = Field(
        default=None,
        description=(
            "Ação em que o modelo carregado foi treinado."
        ),
    )
    last_price: float
    horizon: int
    predictions: list[float]


class FeedbackRequest(BaseModel):
    # Sem exemplo explícito o Swagger monta um a partir do gt=0 e
    # exibe 1 nos dois campos, o que produz erro zero. O exemplo usa
    # o D+1 devolvido pelo exemplo de PredictRequest (297.71) e um
    # preço observado plausível, como no README.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_price": 297.71,
                "actual_price": 298.01,
            }
        }
    )

    predicted_price: float = Field(
        ...,
        gt=0,
        description=(
            "Preço devolvido anteriormente por /predict."
        ),
    )
    actual_price: float = Field(
        ...,
        gt=0,
        description=(
            "Preço de fechamento realmente observado no dia previsto."
        ),
    )


class FeedbackResponse(BaseModel):
    absolute_error: float
    absolute_percentage_error: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_ready: bool
    model_trained_on: str | None
    window_size: int | None
    min_prices: int | None
    resources: dict[str, float]
    startup_error: str | None = None
