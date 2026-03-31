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

ensure_client_paths() {
  : "${ROSBAG_DIR:=${PWD}/datasets/rosbags}"

  export ROSBAG_DIR

  mkdir -p "${ROSBAG_DIR}"
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
  echo "[INFO] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-}"
  echo "[INFO] CYCLONEDDS_URI=${CYCLONEDDS_URI:-}"
  echo "[INFO] POLICY_CHECKPOINT_PATH=${POLICY_CHECKPOINT_PATH:-}"
  echo "[INFO] POLICY_CACHE_DIR=${POLICY_CACHE_DIR:-}"
  echo "[INFO] POLICY_CONFIG_NAME=${POLICY_CONFIG_NAME:-pi0_hsr_airoa-moma}"
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

cmd_up_server() {
  ensure_paths
  print_env_summary
  docker compose up --build -d policy_server
  echo "[INFO] policy_server started."
}

cmd_up_client() {
  ensure_client_paths
  ensure_ros2_network_env
  print_env_summary
  docker compose up --build -d --no-deps hsr_client
  echo "[INFO] hsr_client started."
  echo "[INFO] Next step: ./RUN-DOCKER-CONTAINER.sh shell"
}

cmd_shell() {
  local container_name="${HSR_CLIENT_CONTAINER_NAME:-airoa_hsr_client}"
  local cyclonedds_uri_for_container="${CYCLONEDDS_URI:-}"
  local cyclonedds_interface="${CYCLONEDDS_INTERFACE:-}"
  local cyclonedds_interface_address="${CYCLONEDDS_INTERFACE_ADDRESS:-}"
  local cyclonedds_trace_verbosity="${CYCLONEDDS_TRACE_VERBOSITY:-}"
  local cyclonedds_trace_output="${CYCLONEDDS_TRACE_OUTPUT:-stderr}"
  local cyclonedds_peers="${CYCLONEDDS_PEERS:-}"
  if ! docker ps --format '{{.Names}}' | grep -qx "${container_name}"; then
    echo "[ERROR] Container '${container_name}' is not running."
    echo "Start it first with: ./RUN-DOCKER-CONTAINER.sh up"
    exit 1
  fi

  if [[ -n "${cyclonedds_interface}" || -n "${cyclonedds_interface_address}" ]]; then
    local generated_cyclonedds_host_path="/tmp/cyclonedds-${USER:-user}.xml"
    local network_interface_attrs='autodetermine="false"'
    local allow_multicast_value="default"
    local participant_index_value="none"
    local peers_block=""
    if [[ -n "${cyclonedds_interface_address}" ]]; then
      network_interface_attrs="${network_interface_attrs} address=\"${cyclonedds_interface_address}\""
    elif [[ -n "${cyclonedds_interface}" ]]; then
      network_interface_attrs="${network_interface_attrs} name=\"${cyclonedds_interface}\""
    fi
    if [[ -n "${cyclonedds_peers}" ]]; then
      allow_multicast_value="false"
      participant_index_value="auto"
      local peer_entries=""
      local peer
      IFS=',' read -r -a cyclonedds_peer_array <<< "${cyclonedds_peers}"
      for peer in "${cyclonedds_peer_array[@]}"; do
        peer="${peer//[[:space:]]/}"
        [[ -z "${peer}" ]] && continue
        peer_entries="${peer_entries}
        <Peer Address=\"${peer}\" />"
      done
      if [[ -n "${peer_entries}" ]]; then
        peers_block="      <Peers>${peer_entries}
      </Peers>"
      fi
    fi
    cat > "${generated_cyclonedds_host_path}" <<EOF
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface ${network_interface_attrs} priority="default" multicast="default" />
      </Interfaces>
      <AllowMulticast>${allow_multicast_value}</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Discovery>
      <ParticipantIndex>${participant_index_value}</ParticipantIndex>
${peers_block}
    </Discovery>
    <Internal>
      <SocketReceiveBufferSize min="10MB"/>
      <Watermarks>
        <WhcHigh>500kB</WhcHigh>
      </Watermarks>
    </Internal>
$(if [[ -n "${cyclonedds_trace_verbosity}" ]]; then cat <<TRACE
    <Tracing>
      <Verbosity>${cyclonedds_trace_verbosity}</Verbosity>
      <OutputFile>${cyclonedds_trace_output}</OutputFile>
    </Tracing>
TRACE
fi)
  </Domain>
</CycloneDDS>
EOF
    cyclonedds_uri_for_container="file://${generated_cyclonedds_host_path}"
    if [[ -n "${cyclonedds_interface_address}" ]]; then
      echo "[INFO] Generated container CycloneDDS config for address: ${cyclonedds_interface_address}"
    else
      echo "[INFO] Generated container CycloneDDS config for interface: ${cyclonedds_interface}"
    fi
    if [[ -n "${cyclonedds_trace_verbosity}" ]]; then
      echo "[INFO] CycloneDDS tracing enabled: ${cyclonedds_trace_verbosity} -> ${cyclonedds_trace_output}"
    fi
  fi

  if [[ -n "${cyclonedds_uri_for_container}" && "${cyclonedds_uri_for_container}" == file://* ]]; then
    local cyclonedds_host_path="${cyclonedds_uri_for_container#file://}"
    local cyclonedds_container_path="/tmp/cyclonedds.xml"
    if [[ -f "${cyclonedds_host_path}" ]]; then
      docker cp "${cyclonedds_host_path}" "${container_name}:${cyclonedds_container_path}"
      cyclonedds_uri_for_container="file://${cyclonedds_container_path}"
      echo "[INFO] Copied CycloneDDS config into container: ${cyclonedds_host_path} -> ${cyclonedds_container_path}"
    else
      echo "[WARN] CYCLONEDDS_URI points to a missing host file: ${cyclonedds_host_path}"
      echo "[WARN] Unsetting CYCLONEDDS_URI for this container shell."
      cyclonedds_uri_for_container=""
    fi
  fi

  docker exec -it \
    -e POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-127.0.0.1}" \
    -e POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-8000}" \
    -e POLICY_SERVER_API_KEY="${POLICY_SERVER_API_KEY:-}" \
    -e TEST_MODE="${TEST_MODE:-true}" \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    -e ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}" \
    -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-}" \
    -e CYCLONEDDS_URI="${cyclonedds_uri_for_container}" \
    "${container_name}" bash -lc '
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
export PATH="/home/policy/.venv/bin:${PATH}"
export PYTHONPATH="/workspace/packages/policy-client/src:${PYTHONPATH:-}"
echo "[INFO] ROS 2 environment loaded."
echo "[INFO] Run inside this shell:"
echo "ros2 launch hsr_policy_client hsr_policy_client.launch.py policy_server_host:=${POLICY_SERVER_HOST:-127.0.0.1} policy_server_port:=${POLICY_SERVER_PORT:-8000} policy_server_api_key:=${POLICY_SERVER_API_KEY:-} test_mode:=${TEST_MODE:-true}"
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
  local generated_cyclonedds_host_path="/tmp/cyclonedds-${USER:-user}.xml"
  local cyclonedds_container_path="/tmp/cyclonedds.xml"
  local found=false

  for container in "${client_container}" "${policy_container}"; do
    if docker ps -a --format '{{.Names}}' | grep -qx "${container}"; then
      found=true
      if [[ "${container}" == "${client_container}" ]]; then
        docker exec "${container}" rm -f "${cyclonedds_container_path}" >/dev/null 2>&1 || true
      fi
      echo "[INFO] Removing container: ${container}"
      docker rm -f "${container}" >/dev/null
    fi
  done

  if [[ -f "${generated_cyclonedds_host_path}" ]]; then
    rm -f "${generated_cyclonedds_host_path}"
    echo "[INFO] Removed temporary CycloneDDS config: ${generated_cyclonedds_host_path}"
  fi

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
  up-server)
    cmd_up_server
    ;;
  up-client)
    cmd_up_client
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
    echo "Usage: $0 [up|up-server|up-client|shell|launch|logs [service]|down]"
    exit 1
    ;;
esac
