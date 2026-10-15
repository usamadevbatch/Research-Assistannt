#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# onnxruntime (used for embeddings) has no wheels for every Python/OS combo; prefer 3.12/3.11.
if [ ! -d .venv ]; then
    for py in python3.12 python3.11 python3.13 python3; do
        command -v "$py" >/dev/null && break
    done
    "$py" -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env
exec python -m streamlit run app.py
