#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "Python 3 is required. Install Python and run this file again."
  read -r -p "Press Enter to close..."
  exit 1
fi

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Creating the virtual environment..."
  rm -rf "$PROJECT_DIR/.venv"
  "$PYTHON_BIN" -m venv "$PROJECT_DIR/.venv"
fi

if ! "$VENV_PYTHON" -c "import streamlit, pandas, numpy, sklearn, plotly, matplotlib" >/dev/null 2>&1; then
  echo "Installing project packages..."
  "$VENV_PYTHON" -m pip install --upgrade pip
  "$VENV_PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
fi

echo "Starting Customer Segmentation Dashboard..."
exec "$VENV_PYTHON" -m streamlit run "$PROJECT_DIR/app.py"
