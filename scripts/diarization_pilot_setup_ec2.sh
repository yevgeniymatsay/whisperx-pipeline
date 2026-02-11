#!/usr/bin/env bash
set -euo pipefail

# Create an isolated pilot venv on EC2 (do not touch the main call-extractor venv).

VENV_DIR="${VENV_DIR:-/home/ubuntu/venvs/diarization-pilot}"
REQ_FILE="${REQ_FILE:-setup/requirements_diarization_pilot.txt}"

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "$REQ_FILE"

python - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
except Exception as e:
    print("torch_import_error", repr(e))
PY

echo "OK: pilot venv ready at $VENV_DIR"
