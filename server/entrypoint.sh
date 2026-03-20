#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required}"
HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"
POLICY_CONFIG_NAME="${POLICY_CONFIG_NAME:-relocate_all_ep300_epoch100_convert_gripper_False_fix_select_episodes}"

ARGS=(
  "--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}"
  "--config-name" "${POLICY_CONFIG_NAME}" 
  # "--config-name" pi0_hsr_airoa-moma
  "--host" "${HOST}"
  "--port" "${PORT}"
)

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
