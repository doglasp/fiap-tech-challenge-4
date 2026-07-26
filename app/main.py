from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import make_asgi_app

from app.config import Settings
from app.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_IN_PROGRESS,
    HTTP_REQUESTS_TOTAL,
    MODEL_ABSOLUTE_ERROR,
    MODEL_ABSOLUTE_PERCENTAGE_ERROR,
    MODEL_FEEDBACK_TOTAL,
    MODEL_FORECAST_POINTS_TOTAL,
    MODEL_INFERENCE_DURATION,
    MODEL_LAST_ABSOLUTE_ERROR,
    MODEL_LAST_ABSOLUTE_PERCENTAGE_ERROR,
    MODEL_LAST_PREDICTED_PRICE,
    MODEL_LAST_RELATIVE_CHANGE_PERCENT,
    MODEL_LOAD_FAILURES_TOTAL,
    MODEL_PREDICTION_REQUESTS_TOTAL,
    MODEL_READY,
    REGISTRY,
    collect_resource_metrics,
)
from app.predictor import Predictor
from app.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

LOGGER = logging.getLogger("lstm_api")
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

KNOWN_PATHS = {
    "/",
    "/health",
    "/ready",
    "/predict",
    "/feedback",
    "/openapi.json",
    "/docs",
    "/redoc",
}


def _metric_path(path: str) -> str:
    """Evita labels de alta cardinalidade."""

    return path if path in KNOWN_PATHS else "/other"


def create_app(
    settings: Settings | None = None,
    predictor_instance: Any | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.predictor = predictor_instance
        application.state.startup_error = None
        sampler_task: asyncio.Task | None = None

        if application.state.predictor is None:
            try:
                LOGGER.info("Carregando modelo e artefatos...")
                loaded_predictor = Predictor(
                    model_path=settings.model_path,
                    scaler_path=settings.scaler_path,
                    metadata_path=settings.metadata_path,
                )
                loaded_predictor.warm_up()
                application.state.predictor = loaded_predictor
                MODEL_READY.set(1)
                LOGGER.info(
                    "Modelo pronto | symbol=%s | window=%s",
                    loaded_predictor.symbol,
                    loaded_predictor.window,
                )
            except Exception as exc:
                application.state.startup_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                MODEL_READY.set(0)
                MODEL_LOAD_FAILURES_TOTAL.inc()
                LOGGER.exception(
                    "Falha ao carregar o modelo."
                )
        else:
            warm_up = getattr(
                application.state.predictor,
                "warm_up",
                None,
            )
            if callable(warm_up):
                warm_up()
            MODEL_READY.set(1)

        async def sample_resources() -> None:
            while True:
                collect_resource_metrics()
                await asyncio.sleep(
                    settings.resource_sample_seconds
                )

        sampler_task = asyncio.create_task(
            sample_resources()
        )

        try:
            yield
        finally:
            if sampler_task is not None:
                sampler_task.cancel()
                with suppress(asyncio.CancelledError):
                    await sampler_task

    application = FastAPI(
        title=(
            "Tech Challenge Fase 4 — "
            "API de Previsão LSTM"
        ),
        description=(
            "Prevê preços futuros a partir de fechamentos "
            "históricos e expõe métricas Prometheus."
        ),
        version=settings.app_version,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def prometheus_middleware(
        request: Request,
        call_next,
    ):
        raw_path = request.url.path

        if raw_path.startswith("/metrics"):
            return await call_next(request)

        path = _metric_path(raw_path)
        method = request.method
        status_code = 500
        started_at = perf_counter()

        HTTP_REQUESTS_IN_PROGRESS.labels(
            method=method,
            path=path,
        ).inc()

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = perf_counter() - started_at

            HTTP_REQUESTS_IN_PROGRESS.labels(
                method=method,
                path=path,
            ).dec()

            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                path=path,
                status=str(status_code),
            ).inc()

            HTTP_REQUEST_DURATION.labels(
                method=method,
                path=path,
            ).observe(elapsed)

    def get_predictor(request: Request):
        current_predictor = getattr(
            request.app.state,
            "predictor",
            None,
        )
        if current_predictor is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Modelo indisponível. Consulte /ready "
                    "e os logs de inicialização."
                ),
            )
        return current_predictor

    @application.get("/")
    def root():
        return {
            "name": application.title,
            "version": application.version,
            "docs": "/docs",
            "health": "/health",
            "readiness": "/ready",
            "metrics": "/metrics/",
        }

    @application.get(
        "/health",
        response_model=HealthResponse,
    )
    def health(request: Request):
        current_predictor = getattr(
            request.app.state,
            "predictor",
            None,
        )
        ready = current_predictor is not None

        return HealthResponse(
            status="ok" if ready else "degraded",
            model_ready=ready,
            model_trained_on=getattr(
                current_predictor,
                "symbol",
                None,
            ),
            window_size=getattr(
                current_predictor,
                "window",
                None,
            ),
            min_prices=getattr(
                current_predictor,
                "min_prices",
                None,
            ),
            resources=collect_resource_metrics(),
            startup_error=getattr(
                request.app.state,
                "startup_error",
                None,
            ),
        )

    @application.get("/ready")
    def ready(request: Request):
        current_predictor = getattr(
            request.app.state,
            "predictor",
            None,
        )
        if current_predictor is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "error": getattr(
                        request.app.state,
                        "startup_error",
                        None,
                    ),
                },
            )

        return {
            "status": "ready",
            "model_loaded": True,
            "scaler_loaded": (
                getattr(
                    current_predictor,
                    "scaler",
                    True,
                )
                is not None
            ),
            "model_trained_on": current_predictor.symbol,
            "window_size": current_predictor.window,
            "min_prices": current_predictor.min_prices,
        }

    @application.post(
        "/predict",
        response_model=PredictResponse,
    )
    def predict_prices(
        request_body: PredictRequest,
        request: Request,
    ):
        current_predictor = get_predictor(request)
        started_at = perf_counter()

        try:
            predictions = current_predictor.predict(
                request_body.prices,
                horizon=request_body.horizon,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        finally:
            MODEL_INFERENCE_DURATION.observe(
                perf_counter() - started_at
            )

        last_price = float(request_body.prices[-1])
        first_prediction = float(predictions[0])
        relative_change = (
            (first_prediction - last_price)
            / last_price
            * 100
        )

        MODEL_PREDICTION_REQUESTS_TOTAL.labels(
            horizon=str(request_body.horizon)
        ).inc()
        MODEL_FORECAST_POINTS_TOTAL.inc(
            len(predictions)
        )
        MODEL_LAST_PREDICTED_PRICE.set(
            first_prediction
        )
        MODEL_LAST_RELATIVE_CHANGE_PERCENT.set(
            relative_change
        )

        return PredictResponse(
            symbol=request_body.symbol,
            model_trained_on=current_predictor.symbol,
            last_price=last_price,
            horizon=request_body.horizon,
            predictions=[
                float(value)
                for value in predictions
            ],
        )

    @application.post(
        "/feedback",
        response_model=FeedbackResponse,
    )
    def feedback(request_body: FeedbackRequest):
        absolute_error = abs(
            request_body.actual_price
            - request_body.predicted_price
        )
        absolute_percentage_error = (
            absolute_error
            / request_body.actual_price
            * 100
        )

        MODEL_FEEDBACK_TOTAL.inc()
        MODEL_ABSOLUTE_ERROR.observe(
            absolute_error
        )
        MODEL_ABSOLUTE_PERCENTAGE_ERROR.observe(
            absolute_percentage_error
        )
        MODEL_LAST_ABSOLUTE_ERROR.set(
            absolute_error
        )
        MODEL_LAST_ABSOLUTE_PERCENTAGE_ERROR.set(
            absolute_percentage_error
        )

        return FeedbackResponse(
            absolute_error=float(absolute_error),
            absolute_percentage_error=float(
                absolute_percentage_error
            ),
        )

    application.mount(
        "/metrics",
        make_asgi_app(registry=REGISTRY),
    )

    return application


app = create_app()
