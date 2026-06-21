#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <python-module> [args...]" >&2
  echo "See README.md for stage commands and examples." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE="$1"
shift

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "python executable not found. Activate the environment or set PYTHON_BIN." >&2
    exit 1
  fi
fi

exec "${PYTHON_BIN}" -m "${MODULE}" "$@"
