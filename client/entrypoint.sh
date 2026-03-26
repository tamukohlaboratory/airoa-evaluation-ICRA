#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
set -u

export PATH="/home/policy/.venv/bin:${PATH}"
export PYTHONPATH="/workspace/packages/policy-client/src:${PYTHONPATH:-}"

exec "$@"
