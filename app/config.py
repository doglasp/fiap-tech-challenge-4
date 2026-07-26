from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Configurações carregadas por variáveis de ambiente."""

    model_path: Path
    scaler_path: Path
    metadata_path: Path
    resource_sample_seconds: float
    app_version: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_path=Path(
                os.getenv("MODEL_PATH", "models/lstm_final.keras")
            ),
            scaler_path=Path(
                os.getenv("SCALER_PATH", "artifacts/ret_scaler.pkl")
            ),
            metadata_path=Path(
                os.getenv(
                    "META_PATH",
                    "artifacts/inference_meta.pkl",
                )
            ),
            resource_sample_seconds=float(
                os.getenv("RESOURCE_SAMPLE_SECONDS", "5")
            ),
            app_version=os.getenv("APP_VERSION", "2.0.0"),
        )
