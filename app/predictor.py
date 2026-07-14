from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from typing import Iterable

import joblib
import numpy as np


def _configure_tensorflow_threads(tf_module) -> None:
    """Limita threads quando as variáveis de ambiente forem informadas."""

    intra = os.getenv("TF_NUM_INTRAOP_THREADS")
    inter = os.getenv("TF_NUM_INTEROP_THREADS")

    if intra:
        tf_module.config.threading.set_intra_op_parallelism_threads(
            int(intra)
        )
    if inter:
        tf_module.config.threading.set_inter_op_parallelism_threads(
            int(inter)
        )


class Predictor:
    """Carrega os artefatos e executa inferência recursiva."""

    def __init__(
        self,
        model_path: str | Path,
        scaler_path: str | Path,
        metadata_path: str | Path,
        serialize_inference: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.metadata_path = Path(metadata_path)

        missing = [
            str(path)
            for path in (
                self.model_path,
                self.scaler_path,
                self.metadata_path,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Artefatos ausentes: " + ", ".join(missing)
            )

        # O import é tardio para que testes com Predictor falso não
        # precisem inicializar o TensorFlow.
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        _configure_tensorflow_threads(tf)

        self.scaler = joblib.load(self.scaler_path)
        self.meta = joblib.load(self.metadata_path)
        self.model = load_model(self.model_path, compile=False)

        self.window = int(self.meta["window_size"])
        self.symbol = self.meta.get("symbol")
        self.target = self.meta.get("target")

        if self.window < 1:
            raise ValueError("window_size deve ser maior que zero.")
        if not hasattr(self.scaler, "transform"):
            raise TypeError("O scaler não possui o método transform.")
        if not hasattr(self.scaler, "inverse_transform"):
            raise TypeError(
                "O scaler não possui o método inverse_transform."
            )

        self._lock = Lock() if serialize_inference else None

    @property
    def min_prices(self) -> int:
        """Quantidade mínima de preços: janela + 1."""

        return self.window + 1

    def warm_up(self) -> None:
        """Inicializa o grafo antes da primeira requisição real."""

        sample = np.zeros((1, self.window, 1), dtype=np.float32)
        context = self._lock if self._lock is not None else nullcontext()

        with context:
            self.model(sample, training=False)

    def predict(
        self,
        prices: Iterable[float],
        horizon: int = 1,
    ) -> list[float]:
        prices_array = np.asarray(list(prices), dtype=float)

        if prices_array.ndim != 1:
            raise ValueError(
                "'prices' deve ser uma lista 1D de fechamentos."
            )
        if len(prices_array) < self.min_prices:
            raise ValueError(
                f"São necessários pelo menos {self.min_prices} preços "
                f"(janela={self.window}); recebidos "
                f"{len(prices_array)}."
            )
        if not np.all(np.isfinite(prices_array)):
            raise ValueError(
                "Todos os preços devem ser números finitos."
            )
        if np.any(prices_array <= 0):
            raise ValueError(
                "Todos os preços devem ser positivos "
                "(o alvo é log-retorno)."
            )
        if not 1 <= int(horizon) <= 30:
            raise ValueError(
                "'horizon' deve estar entre 1 e 30."
            )

        recent = prices_array[-self.min_prices :]
        log_returns = np.diff(np.log(recent))
        window_scaled = self.scaler.transform(
            log_returns.reshape(-1, 1)
        ).flatten()

        predictions: list[float] = []
        last_price = float(recent[-1])
        context = self._lock if self._lock is not None else nullcontext()

        # A serialização reduz riscos de concorrência no runtime do
        # TensorFlow. Para maior vazão, escale réplicas da API.
        with context:
            for _ in range(int(horizon)):
                model_input = window_scaled.reshape(
                    1,
                    self.window,
                    1,
                )
                prediction_tensor = self.model(
                    model_input,
                    training=False,
                )
                scaled_return = float(
                    np.asarray(prediction_tensor).reshape(-1)[0]
                )
                predicted_return = float(
                    self.scaler.inverse_transform(
                        [[scaled_return]]
                    )[0, 0]
                )

                next_price = float(
                    last_price * np.exp(predicted_return)
                )
                predictions.append(next_price)

                window_scaled = np.append(
                    window_scaled[1:],
                    scaled_return,
                )
                last_price = next_price

        return predictions
