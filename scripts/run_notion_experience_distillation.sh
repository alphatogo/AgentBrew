#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${AGENTBREW_ENV_FILE:-${PROJECT_ROOT}/.env}"
INPUT_PATH="${1:-${PROJECT_ROOT}/outputs/notion_trajectory_sample/trajectories}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing environment file: ${ENV_FILE}" >&2
  echo "Copy .env.example to .env and fill in your credentials." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

MODEL_PATH="${AGENTBREW_MODEL_PATH:-./Qwen3-32B}"
BASE_URL="${LOCAL_LLM_BASE_URL:-http://localhost:2024/v1}"
API_KEY="${LOCAL_LLM_API_KEY:-}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN:-python}" -m agentbrew.experience_distillation.pipeline \
  --environment notion \
  --input-path "${INPUT_PATH}" \
  --model "${MODEL_PATH}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --max-concurrent-files "${MAX_CONCURRENT_FILES:-8}" \
  --max-concurrent-tasks "${MAX_CONCURRENT_TASKS:-64}" \
  --max-concurrent-chat "${MAX_CONCURRENT_CHAT:-16}" \
  --max-concurrent-logprob "${MAX_CONCURRENT_LOGPROB:-96}" \
  --hindsight-timeout "${HINDSIGHT_TIMEOUT:-120}" \
  --logprob-timeout "${LOGPROB_TIMEOUT:-60}" \
  --retries "${DISTILLATION_RETRIES:-3}"
