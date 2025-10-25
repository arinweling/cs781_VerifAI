#!/usr/bin/env bash
# Setup virtual environment and install project dependencies via Poetry
# Usage: bash setup_venv.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"



echo "[1/8] Ensuring python3-venv is installed (sudo required if missing)..."
if ! dpkg -s python3-venv >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3-venv
fi

echo "[2/8] Creating virtual environment (.venv) if absent..."
if [ ! -d "$ROOT_DIR/.venv" ]; then
  python3 -m venv "$ROOT_DIR/.venv"
fi

echo "[3/8] Activating virtual environment..."
# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

echo "[4/8] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

echo "[5/8] Installing Poetry inside this venv..."
pip install --upgrade poetry

echo "[6/8] Installing project dependencies with Poetry (using existing venv)..."
poetry config virtualenvs.create false --local || true
poetry install -E examples --no-root

echo "[7/8] Performing editable install (pip install -e .)..."
cd "$ROOT_DIR"
python -m pip install -e .

source "$ROOT_DIR/.venv/bin/activate"
echo "Done. Activate with: source .venv/bin/activate (from repo root)."