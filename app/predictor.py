from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from threading import Lock

import joblib
import numpy as np


def _session_options(ort_module):
    """Aplica os limites de thread informados por variáveis de ambiente."""

    options = ort_module.SessionOptions()

    intra = os.getenv("ORT_NUM_INTRAOP_THREADS")
    inter = os.getenv("ORT_NUM_INTEROP_THREADS")

    if intra:
        options.intra_op_num_threads = int(intra)
    if inter:
        options.inter_op_num_threads = int(inter)

    return options


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
        # precisem inicializar o runtime de inferência.
        import onnxruntime as ort

        self.scaler = joblib.load(self.scaler_path)
        self.meta = joblib.load(self.metadata_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=_session_options(ort),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self.session.get_inputs()[0].name

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

        janela_modelo = self.session.get_inputs()[0].shape[1]
        if isinstance(janela_modelo, int) and janela_modelo != self.window:
            raise ValueError(
                f"O modelo espera janela {janela_modelo}, mas os metadados "
                f"informam {self.window}. Reexporte com "
                "scripts/export_onnx.py."
            )

        self._lock = Lock() if serialize_inference else None

    @property
    def min_prices(self) -> int:
        """Quantidade mínima de preços: janela + 1."""

        return self.window + 1

    def warm_up(self) -> None:
        """Executa a primeira inferência antes da primeira requisição real."""

        sample = np.zeros((1, self.window, 1), dtype=np.float32)
        context = self._lock if self._lock is not None else nullcontext()

        with context:
            self._infer(sample)

    def _infer(self, model_input: np.ndarray) -> float:
        saida = self.session.run(
            None,
            {self._input_name: model_input},
        )
        return float(np.asarray(saida[0]).reshape(-1)[0])

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

        # A serialização protege a sessão de chamadas concorrentes. Para
        # maior vazão, escale réplicas da API.
        with context:
            for _ in range(int(horizon)):
                # O grafo exportado espera float32; os pesos já são
                # float32, então não há perda adicional.
                model_input = window_scaled.reshape(
                    1,
                    self.window,
                    1,
                ).astype(np.float32)
                scaled_return = self._infer(model_input)
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
