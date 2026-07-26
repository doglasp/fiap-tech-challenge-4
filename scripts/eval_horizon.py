"""Mede o erro da previsão recursiva em vários horizontes.

A avaliação do notebook 02 cobre apenas D+1, mas a API aceita horizon
até 30. Este script replica o split do notebook 01 e a recursão do
app/predictor.py para medir quanto o erro cresce a cada passo, sem
retreinar nada.

O horizonte 1 reproduz o MAE, o RMSE e o MAPE gravados em
artifacts/metrics.pkl, o que serve de verificação da replicação.

Uso:

    python scripts/eval_horizon.py

Requer o TensorFlow instalado. Dentro do contêiner da API:

    docker run --rm -i lstm-api:slim python - < scripts/eval_horizon.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np


def read_close_prices(csv_path: Path) -> np.ndarray:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
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

    if not prices:
        raise ValueError("O CSV não contém preços.")
    return np.asarray(prices, dtype=np.float64)


def build_validation_indices(
    total_prices: int,
    split_price_idx: int,
    window: int,
) -> list[int]:
    """Índices k da validação, como o make_windows do notebook 01."""

    return [
        k
        for k in range(split_price_idx, total_prices)
        if k - 1 - window >= 0
    ]


def evaluate(args: argparse.Namespace) -> dict:
    prices = read_close_prices(Path(args.prices_file))
    returns = np.diff(np.log(prices))

    meta = joblib.load(Path(args.metadata_path))
    window = int(meta["window_size"])
    train_ratio = float(meta.get("train_ratio", 0.8))

    split_price_idx = int(len(prices) * train_ratio)

    scaler = joblib.load(Path(args.scaler_path))
    returns_scaled = scaler.transform(
        returns.reshape(-1, 1)
    ).flatten()

    # Import tardio: só é necessário quando o script roda de fato.
    from tensorflow.keras.models import load_model

    model = load_model(Path(args.model_path), compile=False)

    mean_ = float(scaler.mean_[0])
    scale_ = float(scaler.scale_[0])

    ks = build_validation_indices(
        len(prices),
        split_price_idx,
        window,
    )

    resultado: dict[str, dict] = {}

    for horizon in args.horizons:
        # Só os k cujo preço real de D+horizon existe na série.
        ks_h = [k for k in ks if k + horizon - 1 < len(prices)]
        if not ks_h:
            continue

        janelas = np.asarray(
            [returns_scaled[k - 1 - window : k - 1] for k in ks_h],
            dtype=np.float32,
        )
        previstos = np.asarray(
            [prices[k - 1] for k in ks_h],
            dtype=np.float64,
        )
        reais = np.asarray(
            [prices[k + horizon - 1] for k in ks_h],
            dtype=np.float64,
        )
        # Baseline naïve: o preço de referência não muda.
        base = previstos.copy()

        # Mesma recursão do Predictor, porém em lote: um passo por vez
        # para todas as amostras de uma só vez.
        for _ in range(horizon):
            saida = model(
                janelas.reshape(-1, window, 1),
                training=False,
            )
            passo_escalado = np.asarray(saida).reshape(-1)
            passo_retorno = passo_escalado * scale_ + mean_
            previstos = previstos * np.exp(passo_retorno)
            janelas = np.concatenate(
                [janelas[:, 1:], passo_escalado.reshape(-1, 1)],
                axis=1,
            ).astype(np.float32)

        erro = np.abs(previstos - reais)
        erro_naive = np.abs(base - reais)

        resultado[str(horizon)] = {
            "n": len(ks_h),
            "MAE": float(erro.mean()),
            "RMSE": float(
                np.sqrt(((previstos - reais) ** 2).mean())
            ),
            "MAPE": float((erro / reais).mean() * 100),
            "MAE_naive": float(erro_naive.mean()),
            "MAPE_naive": float(
                (erro_naive / reais).mean() * 100
            ),
            # Quanto o modelo projeta de variação contra quanto o
            # preço realmente variou: quantifica o achatamento.
            "variacao_prevista_%": float(
                (np.abs(previstos - base) / base).mean() * 100
            ),
            "variacao_real_%": float(
                (np.abs(reais - base) / base).mean() * 100
            ),
        }

    return resultado


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mede o erro da previsão recursiva por horizonte."
        )
    )
    parser.add_argument(
        "--prices-file",
        default="artifacts/AAPL_clean.csv",
    )
    parser.add_argument(
        "--model-path",
        default="models/lstm_final.keras",
    )
    parser.add_argument(
        "--scaler-path",
        default="artifacts/ret_scaler.pkl",
    )
    parser.add_argument(
        "--metadata-path",
        default="artifacts/inference_meta.pkl",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 10, 20],
    )
    return parser.parse_args()


def main() -> None:
    resultado = evaluate(parse_args())
    print(json.dumps(resultado, indent=2))

    print()
    cabecalho = (
        f"{'h':>3}  {'n':>4}  {'MAE':>8}  {'MAPE %':>8}  "
        f"{'MAPE naive':>11}  {'prev %':>7}  {'real %':>7}"
    )
    print(cabecalho)
    print("-" * len(cabecalho))
    for horizon, linha in resultado.items():
        print(
            f"{horizon:>3}  {linha['n']:>4}  "
            f"{linha['MAE']:>8.3f}  {linha['MAPE']:>8.3f}  "
            f"{linha['MAPE_naive']:>11.3f}  "
            f"{linha['variacao_prevista_%']:>7.3f}  "
            f"{linha['variacao_real_%']:>7.3f}"
        )


if __name__ == "__main__":
    main()
