#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { echo "Creating virtual environment..."; python3 -m venv .venv; }
. .venv/bin/activate
echo "Installing / checking packages (first run takes a few minutes)..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
python scripts/check_setup.py || true
streamlit run app.py
