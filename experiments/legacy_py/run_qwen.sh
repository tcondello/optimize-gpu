#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${TINYGRAD_PYTHON:-$HOME/tinygrad/.venv/bin/python3}"

export PATH="$HOME/.local/bin:$PATH"
export DEV="${DEV:-NV}"
export JIT_BATCH_SIZE="${JIT_BATCH_SIZE:-0}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/qwen_engine.py" "$@"
