#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <python-module> [--nproc-per-node N] [args...]" >&2
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

NPROC="${NPROC_PER_NODE:-2}"
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --nproc-per-node|--nproc_per_node)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      NPROC="$2"
      shift 2
      ;;
    --nproc-per-node=*|--nproc_per_node=*)
      NPROC="${1#*=}"
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

for arg in "${ARGS[@]}"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    exec "${PYTHON_BIN}" -m "${MODULE}" "${ARGS[@]}"
  fi
done

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" --module "${MODULE}" "${ARGS[@]}"
