from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
from pathlib import Path
from time import perf_counter

import httpx


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return (
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight
    )


def read_close_prices(
    csv_path: Path,
    minimum: int,
) -> list[float]:
    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("CSV sem cabeçalho.")

        close_column = next(
            (
                column
                for column in reader.fieldnames
                if column.strip().lower() == "close"
            ),
            None,
        )
        if close_column is None:
            raise ValueError(
                "A coluna 'Close' não foi encontrada no CSV."
            )

        prices = [
            float(row[close_column])
            for row in reader
            if row.get(close_column)
        ]

    if len(prices) < minimum:
        raise ValueError(
            f"O CSV precisa ter ao menos {minimum} preços."
        )
    return prices[-minimum:]


async def run(args) -> dict:
    base_url = args.url.rstrip("/")

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=args.timeout,
    ) as client:
        health = await client.get("/health")
        health.raise_for_status()
        health_data = health.json()

        minimum = int(health_data["min_prices"])
        prices = read_close_prices(
            Path(args.prices_file),
            minimum,
        )
        payload = {
            "prices": prices,
            "horizon": args.horizon,
        }
        semaphore = asyncio.Semaphore(
            args.concurrency
        )

        async def request_once(index: int) -> dict:
            async with semaphore:
                started_at = perf_counter()
                response = await client.post(
                    "/predict",
                    json=payload,
                )
                latency_ms = (
                    perf_counter() - started_at
                ) * 1000
                return {
                    "index": index,
                    "status": response.status_code,
                    "latency_ms": latency_ms,
                }

        started_at = perf_counter()
        rows = await asyncio.gather(
            *[
                request_once(index)
                for index in range(args.requests)
            ]
        )
        elapsed = perf_counter() - started_at

    latencies = [
        row["latency_ms"]
        for row in rows
    ]
    errors = [
        row
        for row in rows
        if row["status"] != 200
    ]

    return {
        "url": base_url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "horizon": args.horizon,
        "elapsed_seconds": elapsed,
        "throughput_requests_per_second": (
            args.requests / elapsed
        ),
        "error_rate_percent": (
            len(errors) / args.requests * 100
        ),
        "latency_mean_ms": statistics.fmean(
            latencies
        ),
        "latency_p50_ms": percentile(
            latencies,
            0.50,
        ),
        "latency_p95_ms": percentile(
            latencies,
            0.95,
        ),
        "latency_p99_ms": percentile(
            latencies,
            0.99,
        ),
        "latency_max_ms": max(latencies),
        "status_counts": {
            str(status): sum(
                row["status"] == status
                for row in rows
            )
            for status in sorted(
                {row["status"] for row in rows}
            )
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Teste de carga da API LSTM."
        )
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
    )
    parser.add_argument(
        "--prices-file",
        required=True,
        help="CSV gerado pelo notebook 01.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = asyncio.run(run(arguments))
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )
