# Deploy Guide (Real HSR)

Real HSR deployment guide.

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
export HSR_IP=100.119.167.94
export ROS_MASTER_URI=http://100.119.167.94:11311
export ROS_IP=<YOUR_HOST_IP>
export POLICY_CHECKPOINT_PATH=/abs/path/to/checkpoint_dir
export POLICY_SERVER_HOST=127.0.0.1
export POLICY_SERVER_PORT=8000
export POLICY_SERVER_API_KEY=
export POLICY_CACHE_DIR=$PWD/.docker_cache/policy_cache
export HF_CACHE_DIR=$PWD/.docker_cache/hf
export ROSBAG_DIR=$PWD/datasets/rosbags
```

## 3. Optional policy-specific variables

Set policy-specific variables only when your server implementation requires them.

OpenPI example:

```bash
export POLICY_CONFIG_NAME=<openpi_config_name>
export POLICY_DEFAULT_PROMPT=
export POLICY_RECORD_DIR=
export POLICY_PYTORCH_DEVICE=cuda
```

For checkpoints trained with masked HSR action sampling, the sample mask is enabled automatically when the checkpoint path or `POLICY_CONFIG_NAME` contains `mask`. If the name does not contain `mask`, enable it manually:

```bash
export POLICY_ENABLE_ACTION_SAMPLE_MASK=true
# Optional; defaults to the HSR 32-dim padded action layout.
export POLICY_ACTION_SAMPLE_MASK_VALID_DIMS=0,1,2,3,4,6,11,12,13,14,15
```

## 4. Start and verify containers

```bash
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh logs policy_server
./RUN-DOCKER-CONTAINER.sh logs hsr_client
```

## 5. Run deploy launch

1. Enter client shell:

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

2. Launch inside client container:

```bash
roslaunch hsr_policy_client hsr_policy_client.launch test_mode:=false
```

Optional pre-check in the same shell:

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

This pre-check runs in `test_mode:=true` by default and uses synthetic random observations.

## 6. Update language instruction at runtime

From a shell with ROS environment loaded:

```bash
rosservice call /hsr_policy_client/update_instruction "message: 'Pick up the coffee bottle on the right'"
```

## 7. Restart rules

Run `down` and `up` after changing:

- `POLICY_CHECKPOINT_PATH`
- policy-specific server environment values

## 8. Troubleshooting

- `required variable ROS_MASTER_URI is missing`: set `ROS_MASTER_URI`
- `Container 'airoa_hsr_client' is not running`: run `./RUN-DOCKER-CONTAINER.sh up`
- `POLICY_CHECKPOINT_PATH is missing`: set `POLICY_CHECKPOINT_PATH`
- client waits for WebSocket server: check `./RUN-DOCKER-CONTAINER.sh logs policy_server`
- `unable to contact ROS master`: check network route, `ROS_MASTER_URI`, and `ROS_IP`

## 9. Stop

```bash
./RUN-DOCKER-CONTAINER.sh down
```
