from __future__ import annotations

import os

import psutil
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)


REGISTRY = CollectorRegistry()

HTTP_REQUESTS_TOTAL = Counter(
    "lstm_api_http_requests_total",
    "Quantidade total de requisições HTTP.",
    ["method", "path", "status"],
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION = Histogram(
    "lstm_api_http_request_duration_seconds",
    "Duração total das requisições HTTP em segundos.",
    ["method", "path"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        5,
        10,
    ),
    registry=REGISTRY,
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "lstm_api_http_requests_in_progress",
    "Quantidade de requisições em processamento.",
    ["method", "path"],
    registry=REGISTRY,
)

MODEL_READY = Gauge(
    "lstm_model_ready",
    "Indica se o modelo foi carregado e está pronto.",
    registry=REGISTRY,
)

MODEL_LOAD_FAILURES_TOTAL = Counter(
    "lstm_model_load_failures_total",
    "Quantidade de falhas ao carregar os artefatos.",
    registry=REGISTRY,
)

MODEL_INFERENCE_DURATION = Histogram(
    "lstm_model_inference_duration_seconds",
    "Tempo gasto somente na inferência do modelo.",
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        5,
        10,
    ),
    registry=REGISTRY,
)

MODEL_PREDICTION_REQUESTS_TOTAL = Counter(
    "lstm_model_prediction_requests_total",
    "Quantidade de requisições de previsão.",
    ["horizon"],
    registry=REGISTRY,
)

MODEL_FORECAST_POINTS_TOTAL = Counter(
    "lstm_model_forecast_points_total",
    "Quantidade total de pontos futuros produzidos.",
    registry=REGISTRY,
)

MODEL_LAST_PREDICTED_PRICE = Gauge(
    "lstm_model_last_predicted_price",
    "Último preço previsto pelo modelo.",
    registry=REGISTRY,
)

MODEL_LAST_RELATIVE_CHANGE_PERCENT = Gauge(
    "lstm_model_last_relative_change_percent",
    (
        "Variação percentual da previsão D+1 em relação "
        "ao último preço informado."
    ),
    registry=REGISTRY,
)

MODEL_FEEDBACK_TOTAL = Counter(
    "lstm_model_feedback_total",
    "Quantidade de observações reais recebidas.",
    registry=REGISTRY,
)

MODEL_ABSOLUTE_ERROR = Histogram(
    "lstm_model_absolute_error",
    "Erro absoluto entre o preço previsto e o observado.",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 50, 100),
    registry=REGISTRY,
)

MODEL_ABSOLUTE_PERCENTAGE_ERROR = Histogram(
    "lstm_model_absolute_percentage_error",
    "Erro percentual absoluto da previsão.",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50),
    registry=REGISTRY,
)

MODEL_LAST_ABSOLUTE_ERROR = Gauge(
    "lstm_model_last_absolute_error",
    "Último erro absoluto registrado.",
    registry=REGISTRY,
)

MODEL_LAST_ABSOLUTE_PERCENTAGE_ERROR = Gauge(
    "lstm_model_last_absolute_percentage_error",
    "Último erro percentual absoluto registrado.",
    registry=REGISTRY,
)

SYSTEM_CPU_PERCENT = Gauge(
    "lstm_api_system_cpu_percent",
    "Utilização total de CPU do sistema em percentual.",
    registry=REGISTRY,
)

SYSTEM_MEMORY_PERCENT = Gauge(
    "lstm_api_system_memory_percent",
    "Utilização total de memória do sistema em percentual.",
    registry=REGISTRY,
)

PROCESS_CPU_PERCENT = Gauge(
    "lstm_api_process_cpu_percent",
    (
        "CPU do processo em percentual. Pode ultrapassar 100 "
        "quando utiliza mais de um núcleo."
    ),
    registry=REGISTRY,
)

PROCESS_CPU_NORMALIZED_PERCENT = Gauge(
    "lstm_api_process_cpu_normalized_percent",
    "CPU do processo normalizada pela quantidade de CPUs lógicas.",
    registry=REGISTRY,
)

PROCESS_RSS_BYTES = Gauge(
    "lstm_api_process_resident_memory_bytes",
    "Memória residente do processo em bytes.",
    registry=REGISTRY,
)

PROCESS = psutil.Process(os.getpid())
CPU_COUNT = psutil.cpu_count(logical=True) or 1

psutil.cpu_percent(interval=None)
PROCESS.cpu_percent(interval=None)


def collect_resource_metrics() -> dict[str, float]:
    """Atualiza os gauges e devolve uma fotografia dos recursos."""

    system_cpu = float(psutil.cpu_percent(interval=None))
    system_memory = float(psutil.virtual_memory().percent)
    process_cpu = float(PROCESS.cpu_percent(interval=None))
    process_cpu_normalized = min(
        process_cpu / CPU_COUNT,
        100.0,
    )
    process_rss = float(PROCESS.memory_info().rss)

    SYSTEM_CPU_PERCENT.set(system_cpu)
    SYSTEM_MEMORY_PERCENT.set(system_memory)
    PROCESS_CPU_PERCENT.set(process_cpu)
    PROCESS_CPU_NORMALIZED_PERCENT.set(
        process_cpu_normalized
    )
    PROCESS_RSS_BYTES.set(process_rss)

    return {
        "system_cpu_percent": system_cpu,
        "system_memory_percent": system_memory,
        "process_cpu_percent": process_cpu,
        "process_cpu_normalized_percent": (
            process_cpu_normalized
        ),
        "process_rss_mb": process_rss / (1024**2),
    }
