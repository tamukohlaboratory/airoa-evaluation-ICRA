# 実機運用メモ

このメモは，今の構成で魔改造HSRを動かす時に，推論PCと制御PCで何をやるかをざっくりまとめた物です．  

## 1. 推論PCですること

リポジトリへ移動して，checkpoint と port をセットして server を起動します．

```bash
cd ~/usr/watanabe_ws/icra_compe/src/airoa-evaluation-ICRA

export POLICY_CHECKPOINT_PATH=/abs/path/to/checkpoint_dir
export POLICY_CONFIG_NAME=config_name
export POLICY_SERVER_PORT=8000

./RUN-DOCKER-CONTAINER.sh up
```

必要ならログ確認です．

```bash
./RUN-DOCKER-CONTAINER.sh logs policy_server
```

## 2. 制御PCですること

```bash
cd ~/usr/watanabe_ws/icra_compe/src/airoa-evaluation-ICRA

export TEST_MODE=false
export ROS_DOMAIN_ID=55
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/hma/cyclonedds.xml
unset CYCLONEDDS_INTERFACE
export CYCLONEDDS_INTERFACE_ADDRESS=192.168.11.5
export CYCLONEDDS_PEERS=192.168.11.55
# 魔改造HSRのIPアドレス

export POLICY_SERVER_HOST=172.30.21.216
# 推論PCのIPアドレス

export POLICY_SERVER_PORT=8000

# for C055
export TEST_MODE=false
export ROS_DOMAIN_ID=55
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/hma/cyclonedds.xml
unset CYCLONEDDS_INTERFACE
export CYCLONEDDS_INTERFACE_ADDRESS=192.168.11.5
export CYCLONEDDS_PEERS=192.168.11.55
export POLICY_SERVER_HOST=172.30.21.216
export POLICY_SERVER_PORT=8000

# for 魔改造
export TEST_MODE=false
export ROS_DOMAIN_ID=74
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/hma/cyclonedds.xml
unset CYCLONEDDS_INTERFACE
export CYCLONEDDS_INTERFACE_ADDRESS=192.168.10.6
export CYCLONEDDS_PEERS=192.168.10.50
export POLICY_SERVER_HOST=172.30.21.216
export POLICY_SERVER_PORT=8000
```

そのあと client container を起動します．

```bash
./RUN-DOCKER-CONTAINER.sh down
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh shell
```

## 3. 実際に client を起動する

制御PCの shell の中でこれを実行します．

```bash
ros2 launch hsr_policy_client hsr_policy_client.launch.py \
  test_mode:=false \
  instruction:='Pick up an object'

relocat
ros2 launch hsr_policy_client hsr_policy_client.launch.py \
  test_mode:=false \
  instruction:='From a rectangle by relocating the mug that is not at a rectangle corner 3'
```
