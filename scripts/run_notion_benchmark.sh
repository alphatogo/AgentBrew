#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${AGENTBREW_ENV_FILE:-${PROJECT_ROOT}/.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing environment file: ${ENV_FILE}" >&2
  echo "Copy .env.example to .env and fill in your credentials." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN:-python}" scripts/run.py \
  agentbrew/configs/runs/notion_benchmark.yaml \
  --env-file "${ENV_FILE}" \
  "$@"
