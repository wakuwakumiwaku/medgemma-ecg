#!/usr/bin/env bash
set -euo pipefail

venv_dir="${MEDGEMMA_VENV:-$HOME/.venvs/medgemma-ecg}"
mkdir -p "$(dirname "$venv_dir")"
if [ -e .venv ] && [ ! -L .venv ]; then
  printf '%s\n' ".venv exists and is not a symlink; remove it or set MEDGEMMA_VENV." >&2
  exit 1
fi
python3 -m venv "$venv_dir"
ln -sfn "$venv_dir" .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e ".[train,ecg,dev]"
.venv/bin/python -m medgemma_ecg.doctor
