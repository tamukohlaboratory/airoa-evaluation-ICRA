FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git git-lfs curl wget ca-certificates gnupg2 lsb-release \
    python3 python3-venv python3-pip \
    qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-desktop \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-default-generators \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/ros/rosdep/sources.list.d && \
    if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then \
      rosdep init; \
    fi && \
    rosdep update --rosdistro humble

WORKDIR /workspace

# Keep Python dependency installation in a stable layer so deploy-side changes
# do not force a full redownload on every image rebuild.
COPY packages/policy-client/pyproject.toml ./packages/policy-client/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.10 --seed /home/policy/.venv && \
    uv pip install --link-mode=copy --python /home/policy/.venv/bin/python \
    "dm-tree>=0.1.8" \
    "msgpack>=1.0.5" \
    "numpy==1.26.4" \
    "opencv-python" \
    "pillow>=9.0.0" \
    "pyyaml" \
    "rospkg>=1.6.1" \
    "catkin_pkg" \
    "tree>=0.2.4" \
    "typing_extensions>=4.0.0" \
    "websockets>=11.0"

COPY deploy ./deploy
COPY packages/policy-client ./packages/policy-client

RUN mkdir -p /root/ros2_ws/src && \
    cp -a /workspace/deploy/hsr_policy_client_interfaces /root/ros2_ws/src/hsr_policy_client_interfaces && \
    cp -a /workspace/deploy/hsr_policy_client /root/ros2_ws/src/hsr_policy_client && \
    cp -a /workspace/deploy/tmc_control_msgs /root/ros2_ws/src/tmc_control_msgs && \
    source /opt/ros/humble/setup.bash && \
    cd /root/ros2_ws && \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro humble && \
    colcon build --merge-install

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --link-mode=copy --python /home/policy/.venv/bin/python --no-deps -e /workspace/packages/policy-client

COPY client/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PATH="/home/policy/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace/packages/policy-client/src"
ENV ROS_DISTRO=humble
ENV ROS_HOME=/root/.ros

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash", "-lc", "sleep infinity"]
