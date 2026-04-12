# Deploy Guide (ROS 2 / Real HSR)

ROS 2 deployment guide for the HSR policy client.

For a short Japanese ops memo focused on the split inference-PC / control-PC workflow, see `deploy/REAL_ROBOT_WORKFLOW_JA.md`.

## 1. Host prerequisites

- Linux
- Docker Engine + Docker Compose v2
- NVIDIA driver + NVIDIA Container Toolkit
- Network route to HSR

```bash
docker --version
docker compose version
nvidia-smi
```

Verified environment (2026-02-20):

- OS: Ubuntu 24.04.3 LTS
- GPU: NVIDIA GeForce RTX 5070 Ti
- NVIDIA driver: 580.126.09
- Docker: 29.0.1
- Docker Compose: v2.40.3

## 2. Required environment variables

```bash
export TEST_MODE=false
export POLICY_CHECKPOINT_PATH=/abs/path/to/checkpoint_dir
export POLICY_SERVER_HOST=127.0.0.1
export POLICY_SERVER_PORT=8000
export POLICY_SERVER_API_KEY=
export POLICY_CACHE_DIR=$PWD/.docker_cache/policy_cache
export HF_CACHE_DIR=$PWD/.docker_cache/hf
export ROSBAG_DIR=$PWD/datasets/rosbags
```

## 3. Optional ROS 2 network variables

```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=
```

## 4. Optional policy-specific variables

Set policy-specific variables only when your server implementation requires them.

OpenPI example:

```bash
export POLICY_CONFIG_NAME=<openpi_config_name>
export POLICY_DEFAULT_PROMPT=
export POLICY_RECORD_DIR=
export POLICY_PYTORCH_DEVICE=cuda
```

## 5. Start and verify containers

```bash
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh logs policy_server
./RUN-DOCKER-CONTAINER.sh logs hsr_client
```

For split deployment across two hosts:

- Inference PC: run only `policy_server`

```bash
export POLICY_CHECKPOINT_PATH=/abs/path/to/checkpoint_dir
export POLICY_SERVER_PORT=8000
./RUN-DOCKER-CONTAINER.sh up-server
```

- Control PC: run only `hsr_client`, pointing `POLICY_SERVER_HOST` at the inference PC

```bash
export TEST_MODE=false
export ROS_DOMAIN_ID=<robot_domain_id>
export ROS_LOCALHOST_ONLY=0
export POLICY_SERVER_HOST=<inference_pc_ip>
export POLICY_SERVER_PORT=8000
./RUN-DOCKER-CONTAINER.sh up-client
```

If CycloneDDS on the control PC interferes with DDS discovery when `hsr_client` runs in Docker,
prefer running the client directly on the host instead of in the container:

```bash
export TEST_MODE=false
export ROS_DOMAIN_ID=<robot_domain_id>
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///abs/path/to/cyclonedds.xml
export POLICY_SERVER_HOST=<inference_pc_ip>
export POLICY_SERVER_PORT=8000
./RUN-HOST-CLIENT.sh setup
./RUN-HOST-CLIENT.sh launch
```

## 6. Run deploy launch

1. Enter client shell:

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

2. Launch inside client container:

```bash
ros2 launch hsr_policy_client hsr_policy_client.launch.py test_mode:=false
```

Optional pre-check in the same shell:

```bash
ros2 launch hsr_policy_client hsr_policy_client.launch.py
```

This pre-check runs in `test_mode:=true` by default and uses synthetic random observations.

## 7. Update language instruction at runtime

From a shell with ROS 2 environment loaded:

```bash
ros2 service call /hsr_policy_client/update_instruction hsr_policy_client_interfaces/srv/StringTrigger "{message: 'Pick up the coffee bottle on the right'}"
```


## 8. Interface packages

- `hsr_policy_client_interfaces` contains the custom service definitions used by the client.
- `tmc_control_msgs` is kept as a local shim only for `GripperApplyEffort.action`. If your HSR ROS 2 workspace already provides the upstream `tmc_control_msgs`, prefer building against the upstream package instead of duplicating it in the same workspace.

## 9. Restart rules

Run `down` and `up` after changing:

- `POLICY_CHECKPOINT_PATH`
- policy-specific server environment values

## 10. Troubleshooting

- `Container 'airoa_hsr_client' is not running`: run `./RUN-DOCKER-CONTAINER.sh up`
- `POLICY_CHECKPOINT_PATH is missing`: set `POLICY_CHECKPOINT_PATH`
- client waits for WebSocket server: check `./RUN-DOCKER-CONTAINER.sh logs policy_server`
- real-robot DDS discovery fails: verify `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY`, host-network reachability, and robot-side DDS settings

## 11. Stop

```bash
./RUN-DOCKER-CONTAINER.sh down
```
