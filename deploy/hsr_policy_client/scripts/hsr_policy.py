#!/usr/bin/env python3
from collections import deque
import threading

import os
import re
import time
from typing import Any, Optional

import cv2
from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from hsr_policy_client_interfaces.srv import StringTrigger
from policy_client.websocket_client_policy import WebsocketClientPolicy
from tmc_control_msgs.action import GripperApplyEffort


def _fmt_log(msg: str, *args: Any) -> str:
    if not args:
        return str(msg)
    try:
        return str(msg) % args
    except Exception:
        return " ".join([str(msg)] + [str(arg) for arg in args])


def _loginfo(logger: Any, msg: str, *args: Any) -> None:
    logger.info(_fmt_log(msg, *args))


def _logwarn(logger: Any, msg: str, *args: Any) -> None:
    logger.warning(_fmt_log(msg, *args))


MODE_CONTINUOUS = "continuous"
MODE_DISCRETE = "discrete"
MODE_HYBRID = "hybrid"
MODES = [MODE_CONTINUOUS, MODE_DISCRETE, MODE_HYBRID]

UPSAMPLE_METHOD_SPLINE = "spline"
UPSAMPLE_METHOD_LINEAR = "linear"
UPSAMPLE_METHODS = [UPSAMPLE_METHOD_SPLINE, UPSAMPLE_METHOD_LINEAR]

ACTION_SMOOTHING_NONE = "none"
ACTION_SMOOTHING_EMA = "ema"
ACTION_SMOOTHING_MA = "moving_average"
ACTION_SMOOTHING_METHODS = [ACTION_SMOOTHING_NONE, ACTION_SMOOTHING_EMA, ACTION_SMOOTHING_MA]

SYNTH_TEST_IMAGE_HEIGHT = 480
SYNTH_TEST_IMAGE_WIDTH = 640
SYNTH_TEST_RANDOM_SEED = 0

CAMERA_TOPICS_BY_HSR_ID: dict[str, dict[str, str]] = {
    "C055": {
        "head_compressed": "/head_rgbd_sensor/rgb/image_raw/compressed",
        "head_raw": "/head_rgbd_sensor/rgb/image_rect_color",
        "hand_compressed": "/hand_camera/image_raw/compressed",
        "hand_raw": "/hand_camera/image_raw",
    },
    "MHSRC": {
        "head_compressed": "/head_rgbd_sensor/color/image_raw/compressed",
        "head_raw": "/head_rgbd_sensor/color/image_raw",
        "hand_compressed": "/hand_camera/color/image_rect_raw/compressed",
        "hand_raw": "/hand_camera/color/image_rect_raw",
    },
}
DEFAULT_HSR_ID = "MHSRC"


def _resolve_camera_topics_for_hsr(hsr_id: Any) -> tuple[str, dict[str, str]]:
    normalized = str(hsr_id or "").strip().upper()
    if normalized in CAMERA_TOPICS_BY_HSR_ID:
        return normalized, CAMERA_TOPICS_BY_HSR_ID[normalized]
    return DEFAULT_HSR_ID, CAMERA_TOPICS_BY_HSR_ID[DEFAULT_HSR_ID]


def _float_tag(x: float, *, ndigits: int = 3) -> str:
    """Filesystem-friendly float tag, e.g. 0.2 -> 0p200, -1.5 -> m1p500."""
    try:
        x = float(x)
    except Exception:
        return "nan"
    sign = "m" if x < 0 else ""
    x = abs(x)
    s = f"{x:.{ndigits}f}".replace(".", "p")
    return f"{sign}{s}"


def _build_trace_group_name(
    *,
    config_name: str,
    adopted_action_chunks: int,
    update_freq: int,
    upsample: bool,
    upsample_hz: int,
    upsample_method: str,
    action_smoothing: str,
    ema_alpha: float,
    ma_window: int,
    smooth_gripper: bool,
    smooth_base: bool,
) -> str:
    parts: list[str] = []
    parts.append(f"ac{int(adopted_action_chunks)}")
    parts.append(f"uf{int(update_freq)}")

    if upsample:
        parts.append(f"up{int(upsample_hz)}")
        parts.append(f"um{str(upsample_method)}")
    else:
        parts.append("original")

    if action_smoothing and action_smoothing != ACTION_SMOOTHING_NONE:
        parts.append(f"sm{str(action_smoothing)}")
        if action_smoothing == ACTION_SMOOTHING_EMA:
            parts.append(f"a{_float_tag(ema_alpha)}")
        elif action_smoothing == ACTION_SMOOTHING_MA:
            parts.append(f"w{int(ma_window)}")
        if smooth_gripper:
            parts.append("sg")
        if smooth_base:
            parts.append("sb")

    return f"{config_name}_" + "_".join(parts)


def _natural_cubic_spline_interpolate(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Natural cubic spline interpolation (2nd derivative = 0 at both ends).

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Strictly increasing knot positions.
    y : np.ndarray, shape (N, D)
        Values at knots.
    xq : np.ndarray, shape (M,)
        Query positions in [x[0], x[-1]].

    Returns
    -------
    np.ndarray, shape (M, D)
        Interpolated values.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    xq = np.asarray(xq, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if y.shape[0] != x.shape[0]:
        raise ValueError("y must have the same length as x.")
    if x.shape[0] < 2:
        return np.repeat(y[:1], xq.shape[0], axis=0)

    n = x.shape[0]
    dim = y.shape[1]
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing.")

    if n == 2:
        # Linear interpolation fallback.
        t = (xq - x[0]) / (x[1] - x[0])
        t = t[:, None]
        return y[0:1] * (1.0 - t) + y[1:2] * t

    # Solve for second-derivative coefficients c (natural boundary: c0=cn-1=0).
    c = np.zeros((n, dim), dtype=np.float32)
    m = n - 2  # number of interior points
    if m > 0:
        lower = h[:-1]  # (m-1,)
        diag = 2.0 * (h[:-1] + h[1:])  # (m,)
        upper = h[1:]  # (m-1,)
        rhs = 3.0 * (
            (y[2:] - y[1:-1]) / h[1:, None] - (y[1:-1] - y[:-2]) / h[:-1, None]
        )  # (m, dim)

        if m == 1:
            c[1:-1] = rhs / diag[:, None]
        else:
            # Thomas algorithm for tridiagonal systems.
            cp = np.empty((m - 1,), dtype=np.float32)
            dp = np.empty((m, dim), dtype=np.float32)

            cp[0] = upper[0] / diag[0]
            dp[0] = rhs[0] / diag[0]
            for i in range(1, m - 1):
                denom = diag[i] - lower[i - 1] * cp[i - 1]
                cp[i] = upper[i] / denom
                dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / denom
            denom = diag[m - 1] - lower[m - 2] * cp[m - 2]
            dp[m - 1] = (rhs[m - 1] - lower[m - 2] * dp[m - 2]) / denom

            c_inner = np.empty((m, dim), dtype=np.float32)
            c_inner[m - 1] = dp[m - 1]
            for i in range(m - 2, -1, -1):
                c_inner[i] = dp[i] - cp[i] * c_inner[i + 1]
            c[1:-1] = c_inner

    # Coefficients for each segment [x_i, x_{i+1})
    b = (y[1:] - y[:-1]) / h[:, None] - (h[:, None] * (2.0 * c[:-1] + c[1:]) / 3.0)
    d = (c[1:] - c[:-1]) / (3.0 * h[:, None])
    a = y[:-1]
    c_seg = c[:-1]

    # Evaluate.
    idx = np.searchsorted(x[1:], xq, side="right")
    idx = np.clip(idx, 0, n - 2)
    dx = (xq - x[idx])[:, None]
    return a[idx] + b[idx] * dx + c_seg[idx] * (dx**2) + d[idx] * (dx**3)


def _linear_interpolate(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Piecewise linear interpolation.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Strictly increasing knot positions.
    y : np.ndarray, shape (N, D)
        Values at knots.
    xq : np.ndarray, shape (M,)
        Query positions in [x[0], x[-1]].

    Returns
    -------
    np.ndarray, shape (M, D)
        Interpolated values.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    xq = np.asarray(xq, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if y.shape[0] != x.shape[0]:
        raise ValueError("y must have the same length as x.")
    if x.shape[0] < 2:
        return np.repeat(y[:1], xq.shape[0], axis=0)

    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing.")

    # Find segment indices so that x[idx] <= xq < x[idx+1].
    idx = np.searchsorted(x[1:], xq, side="right")
    idx = np.clip(idx, 0, x.shape[0] - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    denom = (x1 - x0)[:, None]
    # Avoid division by zero in pathological cases.
    denom = np.where(denom == 0, 1.0, denom)
    w = ((xq - x0)[:, None]) / denom
    return (y0 * (1.0 - w) + y1 * w).astype(np.float32, copy=False)


def _cubic_spline_upsample_actions(
    actions: np.ndarray,
    *,
    in_hz: float,
    out_hz: float,
    out_steps: Optional[int] = None,
) -> np.ndarray:
    """Upsample action sequence with natural cubic spline interpolation."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError("actions must be 2D (T, D).")
    in_steps = int(actions.shape[0])
    if out_steps is None:
        out_steps = int(round(in_steps * float(out_hz) / float(in_hz)))
    out_steps = max(int(out_steps), 1)

    if in_steps == 0:
        return actions
    if in_steps == 1:
        return np.repeat(actions, out_steps, axis=0)
    if out_steps == in_steps:
        return actions

    duration_s = in_steps / float(in_hz)
    x = np.linspace(0.0, duration_s, in_steps + 1, dtype=np.float32)
    y = np.concatenate([actions, actions[-1:, :]], axis=0)
    xq = np.arange(out_steps, dtype=np.float32) / float(out_hz)
    return _natural_cubic_spline_interpolate(x, y, xq).astype(np.float32, copy=False)


def _linear_upsample_actions(
    actions: np.ndarray,
    *,
    in_hz: float,
    out_hz: float,
    out_steps: Optional[int] = None,
) -> np.ndarray:
    """Upsample action sequence with linear interpolation."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError("actions must be 2D (T, D).")
    in_steps = int(actions.shape[0])
    if out_steps is None:
        out_steps = int(round(in_steps * float(out_hz) / float(in_hz)))
    out_steps = max(int(out_steps), 1)

    if in_steps == 0:
        return actions
    if in_steps == 1:
        return np.repeat(actions, out_steps, axis=0)
    if out_steps == in_steps:
        return actions

    duration_s = in_steps / float(in_hz)
    x = np.linspace(0.0, duration_s, in_steps + 1, dtype=np.float32)
    y = np.concatenate([actions, actions[-1:, :]], axis=0)
    xq = np.arange(out_steps, dtype=np.float32) / float(out_hz)
    return _linear_interpolate(x, y, xq).astype(np.float32, copy=False)


class ActionSmoother:
    def __init__(
        self,
        *,
        logger: Any,
        method: str,
        ema_alpha: float,
        ma_window: int,
        dims_mask: np.ndarray,
    ):
        self.logger = logger
        self.method = str(method)
        if self.method not in ACTION_SMOOTHING_METHODS:
            _logwarn(
                self.logger,
                "Unknown action_smoothing '%s'. Falling back to '%s'. Available: %s",
                self.method,
                ACTION_SMOOTHING_NONE,
                ", ".join(ACTION_SMOOTHING_METHODS),
            )
            self.method = ACTION_SMOOTHING_NONE

        self.ema_alpha = float(ema_alpha)
        self.ema_alpha = float(np.clip(self.ema_alpha, 0.0, 1.0))
        self.ma_window = max(int(ma_window), 1)
        self.dims_mask = np.asarray(dims_mask, dtype=bool).reshape(-1)

        self._ema_state: Optional[np.ndarray] = None
        self._ma_buf: deque[np.ndarray] = deque(maxlen=self.ma_window)

    def update(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.method == ACTION_SMOOTHING_NONE:
            return action

        if self.dims_mask.shape[0] != action.shape[0]:
            mask = np.ones_like(action, dtype=bool)
        else:
            mask = self.dims_mask

        out = np.array(action, copy=True)
        if self.method == ACTION_SMOOTHING_EMA:
            if self._ema_state is None or self._ema_state.shape != action.shape:
                self._ema_state = np.array(action, copy=True)
                return out
            a = self.ema_alpha
            self._ema_state[mask] = a * action[mask] + (1.0 - a) * self._ema_state[mask]
            out[mask] = self._ema_state[mask]
            return out

        self._ma_buf.append(action)
        if len(self._ma_buf) == 0:
            return out
        stacked = np.stack(list(self._ma_buf), axis=0)
        out[mask] = stacked[:, mask].mean(axis=0).astype(np.float32, copy=False)
        return out


def _param_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


class HSRPolicyClientNode(Node):
    DEFAULT_PARAMETERS = [
        ("instruction", "Grasp the apple."),
        ("config_name", "remote_policy"),
        ("policy_server_host", "127.0.0.1"),
        ("policy_server_port", 8000),
        ("policy_server_api_key", ""),
        ("adopted_action_chunks", 1),
        ("update_freq", 5),
        ("upsample", False),
        ("upsample_hz", 50),
        ("upsample_method", UPSAMPLE_METHOD_SPLINE),
        ("action_smoothing", ACTION_SMOOTHING_NONE),
        ("ema_alpha", 0.2),
        ("ma_window", 5),
        ("smooth_gripper", False),
        ("smooth_base", False),
        ("gripper_mode", "continuous"),
        ("require_control_mode", False),
        ("expected_control_mode", "auto"),
        ("test_mode", True),
        ("save_exec_trace", False),
    ]

    def __init__(self):
        super().__init__("hsr_policy_client")
        self.declare_parameters(namespace="", parameters=self.DEFAULT_PARAMETERS)

    def param(self, name: str) -> Any:
        clean_name = str(name)
        if clean_name.startswith("~"):
            clean_name = clean_name[1:]
        return self.get_parameter(clean_name).value


class SyntheticReplayEnv:
    """Environment for synthetic test-mode replay without requiring a robot."""

    def __init__(
        self,
        node: HSRPolicyClientNode,
        *,
        update_freq: int = 10,
    ):
        self.node = node
        self.update_freq = max(int(update_freq), 1)
        self.sleep_period_s = 1.0 / float(self.update_freq)
        self.image_height = SYNTH_TEST_IMAGE_HEIGHT
        self.image_width = SYNTH_TEST_IMAGE_WIDTH
        self.random_seed = SYNTH_TEST_RANDOM_SEED
        self._rng = np.random.default_rng(self.random_seed)

        self.instruction = str(self.node.param("instruction"))
        self.gripper_state = 0
        self.control_mode = "auto"
        self.joint_state: Optional[np.ndarray] = None
        self._last_action: Optional[np.ndarray] = None

        self.joint_state_names: list[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]
        self.base_action_names: list[str] = ["base_x", "base_y", "base_theta"]

        self._instruction_service = self.node.create_service(
            StringTrigger,
            "/hsr_policy_client/update_instruction",
            self.update_instruction_srv,
        )
        _loginfo(
            self.node.get_logger(),
            "Test mode enabled. Using synthetic random data image=%dx%d seed=%d (infinite loop)",
            self.image_height,
            self.image_width,
            self.random_seed,
        )

    def update_instruction_srv(self, request: StringTrigger.Request, response: StringTrigger.Response):
        self.instruction = request.message
        response.success = True
        _loginfo(self.node.get_logger(), "Instruction updated: %s", self.instruction)
        return response

    def is_finished(self) -> bool:
        return False

    def reset_observation(self, *, reset_joint_state: bool = True):
        _ = reset_joint_state
        return

    def _make_sample(self) -> dict[str, Any]:
        head_rgb = self._rng.integers(
            low=0, high=256, size=(self.image_height, self.image_width, 3), dtype=np.uint8
        )
        hand_rgb = self._rng.integers(
            low=0, high=256, size=(self.image_height, self.image_width, 3), dtype=np.uint8
        )
        joint_state = self._rng.normal(loc=0.0, scale=0.5, size=(8,)).astype(np.float32)

        return {
            "head_rgb": head_rgb,
            "hand_rgb": hand_rgb,
            "joint_state": joint_state,
            "instruction": self.instruction,
            "gripper_state": 0,
            "control_mode": "auto",
        }

    def get_observations(self):
        sample = self._make_sample()
        self.joint_state = np.asarray(sample["joint_state"], dtype=np.float32)
        return sample

    def execute_actions(self, action: np.ndarray) -> bool:
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        return True

    def sleep(self):
        time.sleep(self.sleep_period_s)


class HSREnv:
    """
    Runtime environment class that receives HSR sensor observations via ROS 2
    and applies actions to the robot.
    """

    GRIPPER_OPEN = 1
    GRIPPER_CLOSE = 0
    GRIPPER_CLOSE_THRESHOLD = 0.5

    def __init__(self, node: HSRPolicyClientNode, update_freq: int = 10):
        self.node = node
        self.update_freq = max(int(update_freq), 1)
        self.sleep_period_s = 1.0 / float(self.update_freq)

        self.head_rgb: Optional[np.ndarray] = None
        self.hand_rgb: Optional[np.ndarray] = None
        self.joint_state: Optional[np.ndarray] = None
        self.gripper_state = 0
        self.control_mode: Optional[str] = None
        self.gripper_mode = str(self.node.param("gripper_mode"))
        self.require_control_mode = _param_to_bool(self.node.param("require_control_mode"))
        self.expected_control_mode = str(self.node.param("expected_control_mode"))
        self.instruction = str(self.node.param("instruction"))
        self._control_mode_received = False
        self._last_missing_obs_log_t = 0.0
        self._last_missing_joint_log_t = 0.0
        self._last_control_mode_log_t = 0.0
        self._last_image_decode_log_t = 0.0
        self._logged_head_ready = False
        self._logged_hand_ready = False
        self._logged_joint_ready = False
        self.hsr_id = str(os.environ.get("HSR_ID", DEFAULT_HSR_ID)).strip()
        self._resolved_hsr_id, camera_topics = _resolve_camera_topics_for_hsr(self.hsr_id)
        self.head_compressed_topic = camera_topics["head_compressed"]
        self.head_raw_topic = camera_topics["head_raw"]
        self.hand_compressed_topic = camera_topics["hand_compressed"]
        self.hand_raw_topic = camera_topics["hand_raw"]
        if self.hsr_id.upper() != self._resolved_hsr_id:
            _logwarn(
                self.node.get_logger(),
                "Unknown HSR_ID='%s'. Falling back to %s camera topics.",
                self.hsr_id,
                self._resolved_hsr_id,
            )
        _loginfo(
            self.node.get_logger(),
            (
                "Using camera topics for HSR_ID=%s: "
                "head(compressed=%s raw=%s) hand(compressed=%s raw=%s)"
            ),
            self._resolved_hsr_id,
            self.head_compressed_topic,
            self.head_raw_topic,
            self.hand_compressed_topic,
            self.hand_raw_topic,
        )

        self.joint_state_names: list[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]

        self.arm_action_names: list[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        self.head_action_names: list[str] = ["head_pan_joint", "head_tilt_joint"]
        self.base_action_names: list[str] = ["base_x", "base_y", "base_theta"]

        reliable_sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        reliable_joint_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.arm_pub = self.node.create_publisher(JointTrajectory, "/arm_trajectory_controller/joint_trajectory", 1)
        self.head_pub = self.node.create_publisher(JointTrajectory, "/head_trajectory_controller/joint_trajectory", 1)
        self.gripper_pub = self.node.create_publisher(JointTrajectory, "/gripper_controller/joint_trajectory", 1)
        self.base_pub = self.node.create_publisher(Twist, "/omni_base_controller/cmd_vel", 1)
        self.gripper_close_client = ActionClient(self.node, GripperApplyEffort, "/gripper_controller/grasp")
        self._latest_gripper_goal_future = None
        self._warned_missing_gripper_server = False

        self._instruction_service = self.node.create_service(
            StringTrigger,
            "/hsr_policy_client/update_instruction",
            self.update_instruction_srv,
        )

        self._head_sub = self.node.create_subscription(
            CompressedImage,
            self.head_compressed_topic,
            self.head_image_callback,
            reliable_sensor_qos,
        )
        self._head_raw_sub = self.node.create_subscription(
            Image,
            self.head_raw_topic,
            self.head_image_raw_callback,
            reliable_sensor_qos,
        )
        self._hand_sub = self.node.create_subscription(
            CompressedImage,
            self.hand_compressed_topic,
            self.hand_image_callback,
            reliable_sensor_qos,
        )
        self._hand_raw_sub = self.node.create_subscription(
            Image,
            self.hand_raw_topic,
            self.hand_image_raw_callback,
            reliable_sensor_qos,
        )
        self._joint_sub = self.node.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            reliable_joint_qos,
        )
        self._whole_body_joint_sub = self.node.create_subscription(
            JointState,
            "/whole_body/joint_states",
            self.whole_body_joint_state_callback,
            reliable_joint_qos,
        )
        self._gripper_open_sub = self.node.create_subscription(
            JointTrajectory,
            "/gripper_controller/joint_trajectory",
            self.gripper_open_callback,
            1,
        )
        self._control_mode_sub = self.node.create_subscription(
            String,
            "/control_mode",
            self.control_mode_callback,
            1,
        )

    def head_image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            self._log_image_decode_failure(self.head_compressed_topic, "cv2.imdecode returned None")
            return
        self.head_rgb = np.array(image[:, :, :])
        if not self._logged_head_ready:
            _loginfo(self.node.get_logger(), "Received head image from compressed topic. shape=%s", str(self.head_rgb.shape))
            self._logged_head_ready = True

    def head_image_raw_callback(self, msg: Image):
        image = self._decode_raw_image(msg)
        if image is not None:
            self.head_rgb = image
            if not self._logged_head_ready:
                _loginfo(self.node.get_logger(), "Received head image from raw topic. shape=%s", str(self.head_rgb.shape))
                self._logged_head_ready = True

    def hand_image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            self._log_image_decode_failure(self.hand_compressed_topic, "cv2.imdecode returned None")
            return
        self.hand_rgb = np.array(image[:, :, :])
        if not self._logged_hand_ready:
            _loginfo(self.node.get_logger(), "Received hand image from compressed topic. shape=%s", str(self.hand_rgb.shape))
            self._logged_hand_ready = True

    def hand_image_raw_callback(self, msg: Image):
        image = self._decode_raw_image(msg)
        if image is not None:
            self.hand_rgb = image
            if not self._logged_hand_ready:
                _loginfo(self.node.get_logger(), "Received hand image from raw topic. shape=%s", str(self.hand_rgb.shape))
                self._logged_hand_ready = True

    def _log_image_decode_failure(self, topic_name: str, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_image_decode_log_t >= 2.0:
            _logwarn(self.node.get_logger(), "Failed to decode image from %s: %s", topic_name, reason)
            self._last_image_decode_log_t = now

    def _decode_raw_image(self, msg: Image) -> Optional[np.ndarray]:
        encoding = str(msg.encoding).lower()
        bytes_per_pixel_map = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
            "yuv422": 2,
            "yuv422_yuy2": 2,
            "yuv422_yuyv": 2,
            "yuyv": 2,
        }
        bytes_per_pixel = bytes_per_pixel_map.get(encoding)
        if bytes_per_pixel is None:
            _logwarn(self.node.get_logger(), "Unsupported image encoding on raw topic: %s", msg.encoding)
            return None

        data = np.frombuffer(msg.data, dtype=np.uint8)
        height = int(msg.height)
        width = int(msg.width)
        step = int(msg.step)
        expected = height * step
        if data.size < expected:
            _logwarn(
                self.node.get_logger(),
                "Raw image payload too small: encoding=%s size=%d expected>=%d",
                msg.encoding,
                int(data.size),
                int(expected),
            )
            return None

        used_row_bytes = width * bytes_per_pixel
        if step < used_row_bytes:
            _logwarn(
                self.node.get_logger(),
                "Raw image step too small: encoding=%s step=%d required>=%d",
                msg.encoding,
                step,
                used_row_bytes,
            )
            return None

        image = data[:expected].reshape(height, step)
        image = image[:, :used_row_bytes]
        if bytes_per_pixel == 1:
            image = image.reshape(height, width)
        else:
            image = image.reshape(height, width, bytes_per_pixel)

        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif encoding == "rgba8":
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif encoding == "bgra8":
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif encoding == "mono8":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif encoding in {"yuv422", "yuv422_yuy2", "yuv422_yuyv", "yuyv"}:
            yuy2_code = getattr(cv2, "COLOR_YUV2BGR_YUY2", None)
            if yuy2_code is None:
                yuy2_code = getattr(cv2, "COLOR_YUV2BGR_YUYV", None)
            if yuy2_code is None:
                _logwarn(self.node.get_logger(), "OpenCV does not support YUY2 conversion on this build.")
                return None
            image = cv2.cvtColor(image, yuy2_code)

        return np.array(image)

    def joint_state_callback(self, msg: JointState):
        self._update_joint_state(msg, "/joint_states")

    def whole_body_joint_state_callback(self, msg: JointState):
        self._update_joint_state(msg, "/whole_body/joint_states")

    def _update_joint_state(self, msg: JointState, topic_name: str):
        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        missing = [name for name in self.joint_state_names if name not in name_to_index]
        if missing:
            now = time.monotonic()
            if now - self._last_missing_joint_log_t >= 2.0:
                _logwarn(
                    self.node.get_logger(),
                    "JointState on %s is missing required joints: %s",
                    topic_name,
                    ", ".join(missing),
                )
                self._last_missing_joint_log_t = now
            return
        joints = [msg.position[name_to_index[name]] for name in self.joint_state_names]
        self.joint_state = np.asarray(joints, dtype=np.float32)
        if not self._logged_joint_ready:
            _loginfo(self.node.get_logger(), "Received joint state from %s.", topic_name)
            self._logged_joint_ready = True

    def gripper_open_callback(self, msg: JointTrajectory):
        _ = msg
        self.gripper_state = self.GRIPPER_OPEN

    def control_mode_callback(self, msg: String):
        self.control_mode = msg.data
        self._control_mode_received = True

    def update_instruction_srv(self, request: StringTrigger.Request, response: StringTrigger.Response):
        self.instruction = request.message
        response.success = True
        _loginfo(self.node.get_logger(), "Instruction updated: %s", self.instruction)
        return response

    def reset_observation(self, *, reset_joint_state: bool = True):
        self.head_rgb = None
        self.hand_rgb = None
        if reset_joint_state:
            self.joint_state = None

    def get_observations(self):
        missing: list[str] = []
        if self.head_rgb is None:
            missing.append("head_rgb")
        if self.hand_rgb is None:
            missing.append("hand_rgb")
        if self.joint_state is None:
            missing.append("joint_state")
        if missing:
            now = time.monotonic()
            if now - self._last_missing_obs_log_t >= 2.0:
                _logwarn(
                    self.node.get_logger(),
                    "Waiting for observations. missing=%s control_mode=%s",
                    ", ".join(missing),
                    str(self.control_mode),
                )
                self._last_missing_obs_log_t = now
            return None
        return {
            "head_rgb": self.head_rgb,
            "hand_rgb": self.hand_rgb,
            "joint_state": self.joint_state,
            "instruction": self.instruction,
            "gripper_state": self.gripper_state,
            "control_mode": self.control_mode,
        }

    def _send_gripper_close_goal(self, effort: float) -> None:
        goal = GripperApplyEffort.Goal()
        goal.effort = float(effort)
        goal.do_control_stop = False
        if not self.gripper_close_client.wait_for_server(timeout_sec=0.0):
            if not self._warned_missing_gripper_server:
                _logwarn(self.node.get_logger(), "Gripper action server is not available: /gripper_controller/grasp")
                self._warned_missing_gripper_server = True
            return
        self._latest_gripper_goal_future = self.gripper_close_client.send_goal_async(goal)
        self._warned_missing_gripper_server = False

    def execute_actions(self, action: np.ndarray) -> bool:
        if self._control_mode_received:
            if self.control_mode != self.expected_control_mode:
                now = time.monotonic()
                if now - self._last_control_mode_log_t >= 2.0:
                    _logwarn(
                        self.node.get_logger(),
                        "Action blocked because control_mode is '%s' (expected '%s').",
                        str(self.control_mode),
                        self.expected_control_mode,
                    )
                    self._last_control_mode_log_t = now
                return False
        elif self.require_control_mode:
            now = time.monotonic()
            if now - self._last_control_mode_log_t >= 2.0:
                _logwarn(
                    self.node.get_logger(),
                    "Action blocked because control_mode is unavailable (expected '%s').",
                    self.expected_control_mode,
                )
                self._last_control_mode_log_t = now
            return False
        else:
            now = time.monotonic()
            if now - self._last_control_mode_log_t >= 5.0:
                _logwarn(
                    self.node.get_logger(),
                    "control_mode topic is unavailable. Allowing actions because require_control_mode=false.",
                )
                self._last_control_mode_log_t = now

        action = np.asarray(action, dtype=np.float32).reshape(-1)

        arm_traj = JointTrajectory()
        arm_traj.joint_names = self.arm_action_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = action[:5].astype(np.float64).tolist()
        arm_point.velocities = []
        arm_point.time_from_start = Duration(seconds=1.0 / float(self.update_freq) / 2.0).to_msg()
        arm_traj.points = [arm_point]

        head_traj = JointTrajectory()
        head_traj.joint_names = self.head_action_names
        head_point = JointTrajectoryPoint()
        head_point.positions = action[6:8].astype(np.float64).tolist()
        head_point.velocities = []
        head_point.time_from_start = Duration(seconds=1.0 / float(self.update_freq) / 2.0).to_msg()
        head_traj.points = [head_point]

        twist = Twist()
        twist.linear.x = float(action[8])
        twist.linear.y = float(action[9])
        twist.angular.z = float(action[10])

        if self.gripper_mode == "continuous":
            gripper_traj = JointTrajectory()
            gripper_traj.joint_names = ["hand_motor_joint"]
            gripper_point = JointTrajectoryPoint()
            gripper_value = float(np.clip(action[5], -0.1, 1.23))
            gripper_point.positions = [gripper_value]
            gripper_point.velocities = []
            gripper_point.time_from_start = Duration(seconds=1.0).to_msg()
            gripper_traj.points = [gripper_point]
            self.gripper_pub.publish(gripper_traj)
        elif self.gripper_mode == "discrete":
            gripper_action = self.GRIPPER_CLOSE if action[5] < self.GRIPPER_CLOSE_THRESHOLD else self.GRIPPER_OPEN
            if self.gripper_state != gripper_action:
                if gripper_action == self.GRIPPER_CLOSE:
                    self._send_gripper_close_goal(effort=-0.018)
                else:
                    gripper_traj = JointTrajectory()
                    gripper_traj.joint_names = ["hand_motor_joint"]
                    gripper_point = JointTrajectoryPoint()
                    gripper_point.positions = [1.239183768915874]
                    gripper_point.velocities = []
                    gripper_point.time_from_start = Duration(seconds=1.0).to_msg()
                    gripper_traj.points = [gripper_point]
                    self.gripper_pub.publish(gripper_traj)
                self.gripper_state = gripper_action
        elif self.gripper_mode == "hybrid":
            gripper_value = float(action[5])
            if gripper_value < self.GRIPPER_CLOSE_THRESHOLD:
                if self.gripper_state != self.GRIPPER_CLOSE:
                    self._send_gripper_close_goal(effort=-0.018)
                    self.gripper_state = self.GRIPPER_CLOSE
            else:
                gripper_traj = JointTrajectory()
                gripper_traj.joint_names = ["hand_motor_joint"]
                gripper_point = JointTrajectoryPoint()
                gripper_point.positions = [float(np.clip(gripper_value, -0.1, 1.23))]
                gripper_point.velocities = []
                gripper_point.time_from_start = Duration(seconds=1.0).to_msg()
                gripper_traj.points = [gripper_point]
                self.gripper_pub.publish(gripper_traj)
                self.gripper_state = self.GRIPPER_OPEN

        self.arm_pub.publish(arm_traj)
        self.head_pub.publish(head_traj)
        self.base_pub.publish(twist)
        return True

    def sleep(self):
        time.sleep(self.sleep_period_s)


class OpenpiPolicy:
    """
    Policy runner that performs inference and returns actions.
    It consumes observations from HSREnv and applies model outputs back to the environment.
    """

    def __init__(
        self,
        logger,
        policy_server_host: str = "127.0.0.1",
        policy_server_port: int | None = 8000,
        policy_server_api_key: Optional[str] = None,
        adopted_action_chunks: int = 15,  # Number of actions consumed from each inferred action chunk.
        action_hz: int = 10,  # Action chunk time resolution (Hz). Usually matches update_freq.
        upsample: bool = False,
        upsample_hz: int = 50,
        upsample_method: str = UPSAMPLE_METHOD_SPLINE,
    ):
        self.logger = logger
        self.policy = WebsocketClientPolicy(
            host=policy_server_host,
            port=policy_server_port,
            api_key=policy_server_api_key,
        )
        try:
            metadata = self.policy.get_server_metadata()
            _loginfo(self.logger, "Connected to policy server. metadata=%s", metadata)
        except Exception as e:
            _logwarn(self.logger, "Failed to read policy server metadata: %s", e)

        self.adopted_action_chunks: int = adopted_action_chunks
        self.action_hz: int = action_hz
        self.upsample: bool = upsample
        self.upsample_hz: int = upsample_hz
        self.upsample_method: str = str(upsample_method)
        if self.upsample_method not in UPSAMPLE_METHODS:
            _logwarn(self.logger, 
                "Unknown upsample_method '%s'. Falling back to '%s'. Available: %s",
                self.upsample_method,
                UPSAMPLE_METHOD_SPLINE,
                ", ".join(UPSAMPLE_METHODS),
            )
            self.upsample_method = UPSAMPLE_METHOD_SPLINE

        self.execution_action_chunks: int = adopted_action_chunks
        if self.upsample:
            self.execution_action_chunks = max(
                int(round(adopted_action_chunks * float(self.upsample_hz) / float(self.action_hz))),
                1,
            )

        self.action_queue: deque = deque(maxlen=self.execution_action_chunks)
        self._last_original_action_chunk: Optional[np.ndarray] = None
        self._infer_latencies_s: list[float] = []

    def _record_infer_timing(self, *, start_s: float, end_s: float) -> None:
        latency = float(end_s - start_s)
        if latency >= 0:
            self._infer_latencies_s.append(latency)

    def log_inference_stats(self) -> None:
        def _summarize(values: list[float]) -> tuple[int, float, float] | None:
            if not values:
                return None
            arr = np.asarray(values, dtype=np.float64)
            return int(arr.size), float(arr.mean()), float(arr.var())

        lat = _summarize(self._infer_latencies_s)

        if lat is None:
            _loginfo(self.logger, "Inference stats: no inference calls recorded.")
            return

        n_lat, mean_lat, var_lat = lat
        _loginfo(self.logger, 
            "Inference latency (infer() only): n=%d mean=%.1fms var=%.3f(ms^2)",
            n_lat,
            mean_lat * 1e3,
            var_lat * 1e6,
        )

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """
        Receive observation data and return a single action.
        obs: Dict[str, Any]
            Observation dictionary
            {
                "head_rgb": <np.ndarray shape (H, W, 3)>,
                "hand_rgb": <np.ndarray shape (H, W, 3)>,
                "joint_state": <np.ndarray shape (8,)>, # ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint","hand_motor_joint(gripper)", "head_pan_joint", "head_tilt_joint"]
                "instruction": <str>,
            }
        return: np.ndarray : shape (11,)
            Action vector
            [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "gripper",
                "head_pan_joint",
                "head_tilt_joint",
                "base_x",
                "base_y",
                "base_t",
            ]
        """

        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            # Convert delta-style arm/head outputs back to absolute values.
            return action + np.concatenate(
                [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
            )  # Gripper/base dimensions are not delta-form, so add zeros there.
        # Build input dictionary for policy inference.
        policy_input = {
            "head_rgb": obs["head_rgb"],
            "hand_rgb": obs["hand_rgb"],
            "state": obs["joint_state"],
            "prompt": obs["instruction"],
        }
        infer_start_s = time.perf_counter()
        raw_action_chunk = np.asarray(self.policy.infer(policy_input)["actions"], dtype=np.float32)
        infer_end_s = time.perf_counter()
        self._record_infer_timing(start_s=infer_start_s, end_s=infer_end_s)
        self._last_original_action_chunk = raw_action_chunk[: self.adopted_action_chunks]

        if self.upsample:
            action_chunk = raw_action_chunk[: self.adopted_action_chunks]
            if self.upsample_method == UPSAMPLE_METHOD_LINEAR:
                action_chunk = _linear_upsample_actions(
                    action_chunk,
                    in_hz=self.action_hz,
                    out_hz=self.upsample_hz,
                    out_steps=self.execution_action_chunks,
                )
            else:
                action_chunk = _cubic_spline_upsample_actions(
                    action_chunk,
                    in_hz=self.action_hz,
                    out_hz=self.upsample_hz,
                    out_steps=self.execution_action_chunks,
                )
        else:
            action_chunk = raw_action_chunk[: self.adopted_action_chunks]

        self.action_queue.extend(action_chunk[1:])
        action = action_chunk[0]  # Return only the first action now; queue the rest.

        # Convert delta-style arm/head outputs back to absolute values.
        return action + np.concatenate(
            [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
        )  # Gripper/base dimensions are not delta-form, so add zeros there.

    def get_last_original_action_chunk(self) -> Optional[np.ndarray]:
        return self._last_original_action_chunk


class ExecTraceRecorder:
    def __init__(
        self,
        logger,
        *,
        enabled: bool,
        config_name: str,
        joint_dim_names: Optional[list[str]] = None,
        base_action_names: Optional[list[str]] = None,
        base_dir: str = "/home/policy/deploy_record",
    ):
        self.enabled = bool(enabled)
        self.config_name = str(config_name)
        self.joint_dim_names = list(joint_dim_names or [])
        self.base_action_names = list(base_action_names or [])
        self.base_dir = str(base_dir)
        self.logger = logger

        self._t: list[float] = []
        self._joint_state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._t_action_original: list[float] = []
        self._action_original_delta: list[np.ndarray] = []
        self._t_chunk_start: list[float] = []

    def add(self, *, stamp_s: float, joint_state: np.ndarray, action: np.ndarray) -> None:
        if not self.enabled:
            return
        self._t.append(float(stamp_s))
        self._joint_state.append(np.asarray(joint_state, dtype=np.float32).reshape(-1))
        self._action.append(np.asarray(action, dtype=np.float32).reshape(-1))

    def add_chunk_start(self, *, stamp_s: float) -> None:
        if not self.enabled:
            return
        self._t_chunk_start.append(float(stamp_s))

    def add_original_action_chunk(self, *, base_stamp_s: float, action_chunk: np.ndarray, action_hz: float) -> None:
        if not self.enabled:
            return
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim != 2:
            return
        hz = float(action_hz)
        if hz <= 0:
            return
        dt = 1.0 / hz
        base = float(base_stamp_s)
        for k in range(int(action_chunk.shape[0])):
            self._t_action_original.append(base + k * dt)
            self._action_original_delta.append(action_chunk[k].reshape(-1))

    def _output_dir(self) -> str:
        safe_name = self.config_name.replace("/", "_").replace(os.sep, "_").strip()
        if safe_name == "":
            safe_name = "unknown_config"
        return os.path.join(self.base_dir, safe_name)

    def _next_run_index(self, out_dir: str) -> int:
        try:
            names = os.listdir(out_dir)
        except FileNotFoundError:
            return 1

        max_idx = 0
        for name in names:
            m = re.match(r"^(\d+)\.(npz|png)$", name)
            if m is None:
                continue
            try:
                idx = int(m.group(1))
            except ValueError:
                continue
            max_idx = max(max_idx, idx)
        return max_idx + 1

    def _dim_name(self, dim_idx: int) -> str:
        if 0 <= dim_idx < len(self.joint_dim_names):
            return self.joint_dim_names[dim_idx]
        base_i = dim_idx - len(self.joint_dim_names)
        if 0 <= base_i < len(self.base_action_names):
            return self.base_action_names[base_i]
        return f"dim[{dim_idx}]"

    def save_and_plot(self) -> None:
        if not self.enabled:
            return
        if len(self._t) == 0:
            _logwarn(self.logger, "ExecTraceRecorder: no samples to save.")
            return

        out_dir = self._output_dir()
        os.makedirs(out_dir, exist_ok=True)
        run_idx = self._next_run_index(out_dir)
        stem = f"{run_idx:04d}"

        npz_path = os.path.join(out_dir, f"{stem}.npz")
        plot_path = os.path.join(out_dir, f"{stem}.png")

        payload: dict[str, np.ndarray] = {
            "t": np.asarray(self._t, dtype=np.float64),
            "joint_state": np.stack(self._joint_state, axis=0),
            "action": np.stack(self._action, axis=0),
            "joint_dim_names": np.asarray(self.joint_dim_names, dtype=str),
            "base_action_names": np.asarray(self.base_action_names, dtype=str),
        }
        if len(self._t_chunk_start) > 0:
            payload["t_chunk_start"] = np.asarray(self._t_chunk_start, dtype=np.float64)
        if len(self._t_action_original) > 0 and len(self._action_original_delta) == len(self._t_action_original):
            payload["t_action_original"] = np.asarray(self._t_action_original, dtype=np.float64)
            payload["action_original_delta"] = np.stack(self._action_original_delta, axis=0)

        np.savez_compressed(npz_path, **payload)
        _loginfo(self.logger, "Saved exec trace: %s", npz_path)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            _logwarn(self.logger, "ExecTraceRecorder: matplotlib unavailable, skipping plot (%s).", e)
            return

        t = np.asarray(self._t, dtype=np.float64)
        joint_state = np.stack(self._joint_state, axis=0)
        action = np.stack(self._action, axis=0)
        t_chunk_start = np.asarray(self._t_chunk_start, dtype=np.float64) if len(self._t_chunk_start) > 0 else None
        t_action_original = (
            np.asarray(self._t_action_original, dtype=np.float64) if len(self._t_action_original) > 0 else None
        )
        action_original_delta = (
            np.stack(self._action_original_delta, axis=0) if len(self._action_original_delta) > 0 else None
        )

        n_joint = int(joint_state.shape[1]) if joint_state.ndim == 2 else 1
        n_action = int(action.shape[1]) if action.ndim == 2 else 1
        n_action_original = (
            int(action_original_delta.shape[1])
            if action_original_delta is not None and action_original_delta.ndim == 2
            else 0
        )
        nrows = max(max(n_joint, n_action), 1)

        fig_h = max(6.0, 1.1 * float(nrows))
        fig, axes = plt.subplots(nrows, 1, figsize=(14, fig_h), sharex=True)
        if nrows == 1:
            axes = [axes]

        for i in range(nrows):
            ax = axes[i]
            has_joint = i < n_joint
            has_action = i < n_action
            has_action_original = action_original_delta is not None and i < n_action_original
            dim_name = self._dim_name(i)

            if has_joint:
                ax.plot(t, joint_state[:, i], label=f"{dim_name} (joint)")
            if has_action:
                ax.plot(t, action[:, i], label=f"{dim_name} (action)")
            if has_action_original and t_action_original is not None:
                # Convert delta->command using joint_state sampled at the closest previous time.
                idx = np.searchsorted(t, t_action_original, side="right") - 1
                idx = np.clip(idx, 0, max(len(t) - 1, 0))
                js = joint_state[idx]
                original_cmd = np.array(action_original_delta[:, i], copy=True)
                if i < 5:
                    original_cmd = original_cmd + js[:, i]
                elif 6 <= i < 8:
                    original_cmd = original_cmd + js[:, i]
                ax.plot(
                    t_action_original,
                    original_cmd,
                    linestyle="None",
                    marker="+",
                    markersize=4.5,
                    alpha=0.9,
                    label="original action",
                )
            ax.set_ylabel(dim_name)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title("joint/action (same index overlaid when available)")
            if has_joint or has_action:
                ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("time [s]")

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, format="png")
        plt.close(fig)
        _loginfo(self.logger, "Saved exec trace plot: %s", plot_path)


def main(args=None):
    print("Start hsr_policy_client")

    rclpy.init(args=args)
    node = HSRPolicyClientNode()
    logger = node.get_logger()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    config_name: str = str(node.param("config_name"))
    policy_server_host: str = str(node.param("policy_server_host"))
    policy_server_port: int = int(node.param("policy_server_port"))
    policy_server_api_key: Optional[str] = str(node.param("policy_server_api_key"))
    if policy_server_api_key.strip() == "":
        policy_server_api_key = None
    adopted_action_chunks = int(node.param("adopted_action_chunks"))
    update_freq: int = int(node.param("update_freq"))
    upsample: bool = _param_to_bool(node.param("upsample"))
    upsample_hz: int = int(node.param("upsample_hz"))
    upsample_method: str = str(node.param("upsample_method"))
    execution_freq: int = upsample_hz if upsample else update_freq

    action_smoothing: str = str(node.param("action_smoothing"))
    ema_alpha: float = float(node.param("ema_alpha"))
    ma_window: int = int(node.param("ma_window"))
    smooth_gripper: bool = _param_to_bool(node.param("smooth_gripper"))
    smooth_base: bool = _param_to_bool(node.param("smooth_base"))
    require_control_mode: bool = _param_to_bool(node.param("require_control_mode"))
    expected_control_mode: str = str(node.param("expected_control_mode"))
    test_mode: bool = _param_to_bool(node.param("test_mode"))
    gripper_mode: str = str(node.param("gripper_mode"))

    save_exec_trace: bool = _param_to_bool(node.param("save_exec_trace"))
    trace_group_name = _build_trace_group_name(
        config_name=config_name,
        adopted_action_chunks=int(adopted_action_chunks),
        update_freq=int(update_freq),
        upsample=bool(upsample),
        upsample_hz=int(upsample_hz),
        upsample_method=str(upsample_method),
        action_smoothing=str(action_smoothing),
        ema_alpha=float(ema_alpha),
        ma_window=int(ma_window),
        smooth_gripper=bool(smooth_gripper),
        smooth_base=bool(smooth_base),
    )

    _loginfo(logger, "config_name: %s", config_name)
    _loginfo(logger, "policy_server_host: %s", policy_server_host)
    _loginfo(logger, "policy_server_port: %s", policy_server_port)
    _loginfo(logger, "policy_server_api_key set: %s", policy_server_api_key is not None)
    _loginfo(logger, "adopted_action_chunks: %s", adopted_action_chunks)
    _loginfo(logger, "update_freq: %s", update_freq)
    _loginfo(logger, "upsample: %s", upsample)
    _loginfo(logger, "upsample_hz: %s", upsample_hz)
    _loginfo(logger, "upsample_method: %s", upsample_method)
    _loginfo(logger, "action_smoothing: %s", action_smoothing)
    _loginfo(logger, "ema_alpha: %s", ema_alpha)
    _loginfo(logger, "ma_window: %s", ma_window)
    _loginfo(logger, "smooth_gripper: %s", smooth_gripper)
    _loginfo(logger, "smooth_base: %s", smooth_base)
    _loginfo(logger, "require_control_mode: %s", require_control_mode)
    _loginfo(logger, "expected_control_mode: %s", expected_control_mode)
    _loginfo(logger, "test_mode: %s", test_mode)
    _loginfo(logger, "execution_freq: %s", execution_freq)
    _loginfo(logger, "gripper_mode: %s", gripper_mode)
    _loginfo(logger, "save_exec_trace: %s", save_exec_trace)
    _loginfo(logger, "exec_trace_group_name: %s", trace_group_name)

    if test_mode:
        env = SyntheticReplayEnv(node, update_freq=execution_freq)
    else:
        env = HSREnv(node, update_freq=execution_freq)

    policy = OpenpiPolicy(
        logger=logger,
        policy_server_host=policy_server_host,
        policy_server_port=policy_server_port,
        policy_server_api_key=policy_server_api_key,
        adopted_action_chunks=adopted_action_chunks,
        action_hz=update_freq,
        upsample=upsample,
        upsample_hz=upsample_hz,
        upsample_method=upsample_method,
    )

    base_mask = np.array([True, True, True, True, True, False, True, True, False, False, False], dtype=bool)
    if smooth_gripper:
        base_mask[5] = True
    if smooth_base:
        base_mask[8:11] = True
    action_smoother = ActionSmoother(
        logger=logger,
        method=action_smoothing,
        ema_alpha=ema_alpha,
        ma_window=ma_window,
        dims_mask=base_mask,
    )
    recorder = ExecTraceRecorder(
        logger=logger,
        enabled=save_exec_trace,
        config_name=trace_group_name,
        joint_dim_names=env.joint_state_names,
        base_action_names=env.base_action_names,
    )

    log_interval = 1
    if upsample and update_freq > 0:
        log_interval = max(int(round(execution_freq / update_freq)), 1)
    tick = 0
    perf0 = time.perf_counter()
    chunk_gaps_s: list[float] = []
    last_chunk_end_t_s: Optional[float] = None

    def log_chunk_gap_stats() -> None:
        if not chunk_gaps_s:
            _loginfo(logger, "Chunk gap stats: no chunk gaps recorded.")
            return
        arr = np.asarray(chunk_gaps_s, dtype=np.float64)
        _loginfo(
            logger,
            "Chunk gap (prev chunk last action -> next chunk first action): n=%d mean=%.1fms var=%.3f(ms^2)",
            int(arr.size),
            float(arr.mean()) * 1e3,
            float(arr.var()) * 1e6,
        )

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        while rclpy.ok():
            will_infer = len(policy.action_queue) == 0
            obs = env.get_observations()
            if obs is None:
                can_continue_chunk = (not will_infer) and (env.joint_state is not None) and (upsample or test_mode)
                if can_continue_chunk:
                    obs = {"joint_state": env.joint_state, "instruction": env.instruction}
                else:
                    _loginfo(logger, "Observations are not ready.")
                    tick += 1
                    env.sleep()
                    continue

            action = policy.act(obs)
            action_t_s = time.perf_counter() - perf0
            action_to_send = action_smoother.update(action)
            is_executed = env.execute_actions(action_to_send)
            sent_t_s = time.perf_counter() - perf0

            if is_executed:
                if will_infer:
                    recorder.add_chunk_start(stamp_s=sent_t_s)
                    if last_chunk_end_t_s is not None:
                        gap = float(sent_t_s - last_chunk_end_t_s)
                        if gap >= 0:
                            chunk_gaps_s.append(gap)

                if (not will_infer) and len(policy.action_queue) == 0:
                    last_chunk_end_t_s = sent_t_s

            if is_executed and "joint_state" in obs:
                recorder.add(
                    stamp_s=sent_t_s,
                    joint_state=obs["joint_state"],
                    action=action_to_send,
                )
                if upsample and will_infer:
                    original_chunk = policy.get_last_original_action_chunk()
                    if original_chunk is not None and original_chunk.ndim == 2:
                        recorder.add_original_action_chunk(
                            base_stamp_s=action_t_s,
                            action_chunk=original_chunk,
                            action_hz=update_freq,
                        )

            if tick % log_interval == 0:
                if is_executed:
                    _loginfo(logger, "Action executed.")
                else:
                    _loginfo(logger, "Action not executed.")
                _loginfo(logger, "Language instruction: %s", obs.get("instruction", ""))
                _loginfo(logger, "Action: %s", action)

            if not test_mode:
                if not upsample:
                    env.reset_observation()
                elif will_infer:
                    env.reset_observation(reset_joint_state=False)
            tick += 1
            env.sleep()
    except KeyboardInterrupt:
        _loginfo(logger, "KeyboardInterrupt received. Saving exec trace and shutting down.")
    finally:
        try:
            recorder.save_and_plot()
        except Exception as e:
            _logwarn(logger, "Failed to save execution trace: %s", e)
        try:
            policy.log_inference_stats()
        except Exception as e:
            _logwarn(logger, "Failed to log inference stats: %s", e)
        try:
            log_chunk_gap_stats()
        except Exception as e:
            _logwarn(logger, "Failed to log chunk gap stats: %s", e)
        executor.shutdown()
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            _logwarn(logger, "Ignoring shutdown error: %s", e)
        executor_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
