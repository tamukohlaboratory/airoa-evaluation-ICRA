#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required}"
HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"

# pi05_hsr_task47_ep50_v2, pi0_hsr_airoa-moma
POLICY_CONFIG_NAME="${POLICY_CONFIG_NAME:-pi0_hsr_airoa-moma}"
POLICY_DEFAULT_PROMPT="${POLICY_DEFAULT_PROMPT:-}"
POLICY_RECORD_DIR="${POLICY_RECORD_DIR:-}"
POLICY_PYTORCH_DEVICE="${POLICY_PYTORCH_DEVICE:-}"

ARGS=(  
  "--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}"
  "--config-name" "${POLICY_CONFIG_NAME}"
  "--host" "${HOST}"
  "--port" "${PORT}"
)

if [[ -n "${POLICY_DEFAULT_PROMPT}" ]]; then
  ARGS+=("--default-prompt" "${POLICY_DEFAULT_PROMPT}")
fi

if [[ -n "${POLICY_RECORD_DIR}" ]]; then
  ARGS+=("--record-dir" "${POLICY_RECORD_DIR}")
fi

if [[ -n "${POLICY_PYTORCH_DEVICE}" ]]; then
  ARGS+=("--pytorch-device" "${POLICY_PYTORCH_DEVICE}")
fi

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
