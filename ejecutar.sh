#!/usr/bin/env bash
set -euo pipefail
BER_TOPIC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BER_TOPIC_DIR"
if [[ ! -x .venv/bin/python ]]; then
    printf '%s\n' 'Falta el entorno. Primero ejecuta: bash instalar.sh' >&2
    exit 1
fi
# -I impide mezclar los paquetes del proyecto con PYTHONPATH o paquetes externos.
exec .venv/bin/python -I pipeline.py "$@"
