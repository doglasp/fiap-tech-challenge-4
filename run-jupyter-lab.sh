#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IPYTHONDIR="$ROOT_DIR/.ipython"
export JUPYTER_CONFIG_DIR="$ROOT_DIR/.jupyter"
export JUPYTER_DATA_DIR="$ROOT_DIR/.jupyter"
export JUPYTER_PATH="$ROOT_DIR/.jupyter/share/jupyter"
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"

mkdir -p "$IPYTHONDIR" "$JUPYTER_CONFIG_DIR" "$JUPYTER_PATH" "$MPLCONFIGDIR"

exec "$ROOT_DIR/.venv/bin/python" -m jupyter lab "$@"
