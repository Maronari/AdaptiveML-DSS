#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python3.13}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter '${PYTHON_BIN}' not found."
  echo "Install Python 3.13.x and rerun: scripts/setup_full_env.sh python3.13"
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install --upgrade "pip<26" "setuptools<81" wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.10.0+cpu"
python -m pip install --index-url "${PYPI_INDEX_URL}" -r requirements.txt
python -m pip install --index-url "${PYPI_INDEX_URL}" --no-deps "LightAutoML==0.4.1"
