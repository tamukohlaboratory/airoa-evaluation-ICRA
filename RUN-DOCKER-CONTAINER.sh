#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-up}"

is_true() {
  case "${1:-}" in
    1 | true | TRUE | True | yes | YES | on | ON | y | Y)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

ensure_paths() {
  : "${POLICY_CACHE_DIR:=${PWD}/.docker_cache/policy_cache}"
  : "${HF_CACHE_DIR:=${PWD}/.docker_cache/hf}"
  : "${ROSBAG_DIR:=${PWD}/datasets/rosbags}"
  : "${POLICY_CHECKPOINT_PATH:?Set POLICY_CHECKPOINT_PATH to your checkpoint full path}"

  export POLICY_CACHE_DIR
  export HF_CACHE_DIR
  export ROSBAG_DIR
  export POLICY_CHECKPOINT_PATH

  if [[ ! -d "${POLICY_CHECKPOINT_PATH}" ]]; then
    echo "[ERROR] POLICY_CHECKPOINT_PATH does not exist: ${POLICY_CHECKPOINT_PATH}"
    exit 1
  fi

  mkdir -p "${POLICY_CACHE_DIR}" "${HF_CACHE_DIR}" "${ROSBAG_DIR}"
}

ensure_ros2_network_env() {
  : "${ROS_DOMAIN_ID:=0}"

  if [[ -z "${ROS_LOCALHOST_ONLY:-}" ]]; then
    if is_true "${TEST_MODE:-true}"; then
      ROS_LOCALHOST_ONLY=1
    else
      ROS_LOCALHOST_ONLY=0
    fi
  fi

  export ROS_DOMAIN_ID
  export ROS_LOCALHOST_ONLY
}

print_env_summary() {
  echo "[INFO] TEST_MODE=${TEST_MODE:-true}"
  echo "[INFO] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
  echo "[INFO] ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}"
  echo "[INFO] POLICY_CHECKPOINT_PATH=${POLICY_CHECKPOINT_PATH:-}"
  echo "[INFO] POLICY_CACHE_DIR=${POLICY_CACHE_DIR:-}"
  echo "[INFO] HF_CACHE_DIR=${HF_CACHE_DIR:-}"
  echo "[INFO] ROSBAG_DIR=${ROSBAG_DIR:-}"
  echo "[INFO] POLICY_SERVER_HOST=${POLICY_SERVER_HOST:-127.0.0.1}"
  echo "[INFO] POLICY_SERVER_PORT=${POLICY_SERVER_PORT:-8000}"
  echo "[INFO] POLICY_SERVER_API_KEY set=$([[ -n "${POLICY_SERVER_API_KEY:-}" ]] && echo true || echo false)"
}

cmd_up() {
  ensure_paths
  ensure_ros2_network_env
  print_env_summary
  docker compose up --build -d
  echo "[INFO] Containers started."
  echo "[INFO] Next step: ./RUN-DOCKER-CONTAINER.sh shell"
}

cmd_shell() {
  local container_name="${HSR_CLIENT_CONTAINER_NAME:-airoa_hsr_client}"
  if ! docker ps --format '{{.Names}}' | grep -qx "${container_name}"; then
    echo "[ERROR] Container '${container_name}' is not running."
    echo "Start it first with: ./RUN-DOCKER-CONTAINER.sh up"
    exit 1
  fi
  docker exec -it \
    -e POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-127.0.0.1}" \
    -e POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-8000}" \
    -e POLICY_SERVER_API_KEY="${POLICY_SERVER_API_KEY:-}" \
    -e TEST_MODE="${TEST_MODE:-true}" \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    -e ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}" \
    "${container_name}" bash -lc '
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
export PATH="/home/policy/.venv/bin:${PATH}"
export PYTHONPATH="/workspace/packages/policy-client/src:${PYTHONPATH:-}"
echo "[INFO] ROS 2 environment loaded."
echo "[INFO] Run inside this shell:"
echo "ros2 launch hsr_policy_client hsr_policy_client.launch.py test_mode:=${TEST_MODE:-true}"
if [[ "${TEST_MODE:-true}" == "false" ]]; then echo "[INFO] See REAL_ROBOT_CHECKLIST.md before running on hardware."; fi
exec bash
'
}

cmd_launch() {
  echo "[INFO] Opening hsr_client shell. Run ros2 launch inside the container."
  cmd_shell
}

cmd_down() {
  local policy_container="${POLICY_SERVER_CONTAINER_NAME:-airoa_policy_server}"
  local client_container="${HSR_CLIENT_CONTAINER_NAME:-airoa_hsr_client}"
  local found=false

  for container in "${client_container}" "${policy_container}"; do
    if docker ps -a --format '{{.Names}}' | grep -qx "${container}"; then
      found=true
      echo "[INFO] Removing container: ${container}"
      docker rm -f "${container}" >/dev/null
    fi
  done

  if [[ "${found}" == "false" ]]; then
    echo "[INFO] No managed containers found."
  else
    echo "[INFO] Containers stopped and removed."
  fi
}

cmd_logs() {
  local service="${2:-}"
  local policy_container="${POLICY_SERVER_CONTAINER_NAME:-airoa_policy_server}"
  local client_container="${HSR_CLIENT_CONTAINER_NAME:-airoa_hsr_client}"

  if [[ -n "${service}" ]]; then
    local container_name="${service}"
    case "${service}" in
      policy_server)
        container_name="${policy_container}"
        ;;
      hsr_client)
        container_name="${client_container}"
        ;;
    esac
    docker logs -f "${container_name}"
    return
  fi

  echo "[INFO] Showing logs for ${policy_container} and ${client_container} (Ctrl+C to stop)."
  docker logs -f "${policy_container}" &
  local pid1=$!
  docker logs -f "${client_container}" &
  local pid2=$!
  trap 'kill ${pid1} ${pid2} 2>/dev/null || true' INT TERM
  wait
}

case "${MODE}" in
  up)
    cmd_up
    ;;
  shell)
    cmd_shell
    ;;
  launch)
    cmd_launch
    ;;
  down)
    cmd_down
    ;;
  logs)
    cmd_logs "$@"
    ;;
  *)
    echo "Usage: $0 [up|shell|launch|logs [service]|down]"
    exit 1
    ;;
esac
