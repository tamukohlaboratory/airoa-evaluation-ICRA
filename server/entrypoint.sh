#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required}"
HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"
PYTORCH_DEVICE="${POLICY_PYTORCH_DEVICE:-}"
USE_MBR="${POLICY_USE_MBR:-}"
MBR_NUM_CANDIDATES="${POLICY_MBR_NUM_CANDIDATES:-}"
MBR_NUM_REFERENCE_CANDIDATES="${POLICY_MBR_NUM_REFERENCE_CANDIDATES:-}"

# pi05_hsr_task47_ep50_v2, pi0_hsr_airoa-moma
POLICY_CONFIG_NAME="${POLICY_CONFIG_NAME:-0412_pi05_airoa_hsr_fullfinetuning_statediff_horizon8_relocate}"

ARGS=(  
  "--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}"
  "--config-name" "${POLICY_CONFIG_NAME}"
  "--host" "${HOST}"
  "--port" "${PORT}"
)

if [[ -n "${PYTORCH_DEVICE}" ]]; then
  ARGS+=("--pytorch-device" "${PYTORCH_DEVICE}")
fi

if [[ -n "${USE_MBR}" ]]; then
  ARGS+=("--use-mbr" "${USE_MBR}")
fi

if [[ -n "${MBR_NUM_CANDIDATES}" ]]; then
  ARGS+=("--mbr-num-candidates" "${MBR_NUM_CANDIDATES}")
fi

if [[ -n "${MBR_NUM_REFERENCE_CANDIDATES}" ]]; then
  ARGS+=("--mbr-num-reference-candidates" "${MBR_NUM_REFERENCE_CANDIDATES}")
fi

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
