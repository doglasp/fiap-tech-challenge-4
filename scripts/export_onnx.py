"""Exporta o modelo Keras treinado para ONNX e verifica a equivalência.

O runtime da API usa apenas o ONNX; o `.keras` continua sendo o artefato
de origem, produzido pelo notebook 02. Este script é a receita que liga
os dois e deve ser reexecutado sempre que o modelo for retreinado.

    python scripts/export_onnx.py

Requer as dependências de desenvolvimento (`requirements-dev.txt`), que
incluem TensorFlow e tf2onnx. A imagem de produção não os instala.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# O Grappler do TensorFlow aborta com "Bad StatusOr access: CUDA Runtime
# error" quando encontra o pacote de GPU sem driver disponível. Como a
# exportação é sempre em CPU, desligamos a GPU antes de importar o TF.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import numpy as np

OPSET = 17
TOLERANCIA = 1e-4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keras",
        type=Path,
        default=Path("models/lstm_final.keras"),
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/lstm_final.onnx"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/inference_meta.pkl"),
    )
    parser.add_argument(
        "--amostras",
        type=int,
        default=512,
        help="janelas aleatórias usadas na verificação de equivalência",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    for caminho in (args.keras, args.metadata):
        if not caminho.is_file():
            print(f"artefato ausente: {caminho}", file=sys.stderr)
            return 1

    window = int(joblib.load(args.metadata)["window_size"])

    from tensorflow.keras.models import load_model

    modelo = load_model(args.keras, compile=False)
    print(f"keras carregado | janela={window} | params={modelo.count_params()}")

    args.onnx.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="savedmodel_") as saved_model_dir:
        modelo.export(saved_model_dir, verbose=False)
        conversao = subprocess.run(
            [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--saved-model",
                saved_model_dir,
                "--output",
                str(args.onnx),
                "--opset",
                str(OPSET),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    if conversao.returncode != 0:
        print(conversao.stdout[-2000:], file=sys.stderr)
        print(conversao.stderr[-2000:], file=sys.stderr)
        print("falha na conversão para ONNX", file=sys.stderr)
        return 1

    import onnxruntime as ort

    sessao = ort.InferenceSession(
        str(args.onnx),
        providers=["CPUExecutionProvider"],
    )
    entrada = sessao.get_inputs()[0].name

    amostras = np.random.default_rng(42).standard_normal(
        (args.amostras, window, 1)
    ).astype(np.float32)

    saida_keras = modelo.predict(amostras, verbose=0).reshape(-1)
    saida_onnx = sessao.run(None, {entrada: amostras})[0].reshape(-1)
    diferenca = float(np.abs(saida_keras - saida_onnx).max())

    print(f"onnx gerado    | {args.onnx} | opset {OPSET}")
    print(f"tamanho        | keras {args.keras.stat().st_size / 1024:.0f} KB"
          f" -> onnx {args.onnx.stat().st_size / 1024:.0f} KB")
    print(f"equivalência   | {args.amostras} janelas | dif. máx {diferenca:.3e}")

    if diferenca > TOLERANCIA:
        print(
            f"divergência acima da tolerância ({TOLERANCIA:.0e}); "
            "o ONNX não substitui o modelo Keras.",
            file=sys.stderr,
        )
        return 1

    print("ok: o ONNX reproduz o modelo Keras dentro da tolerância.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
