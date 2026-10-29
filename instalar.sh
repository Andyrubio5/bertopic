#!/usr/bin/env bash
set -euo pipefail
BER_TOPIC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BER_TOPIC_DIR"
python3 -I -m venv .venv
.venv/bin/python -I -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python -I -m pip check
printf '%s\n' 'Entorno instalado. Ejecuta: bash ejecutar.sh'
