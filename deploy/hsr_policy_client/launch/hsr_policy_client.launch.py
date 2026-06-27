from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    declared_arguments = [
        DeclareLaunchArgument("instruction", default_value="Grasp the apple."),
        DeclareLaunchArgument("config_name", default_value="remote_policy"),
        DeclareLaunchArgument("action_mode", default_value="auto"),
        DeclareLaunchArgument(
            "hsr_id",
            default_value=EnvironmentVariable("HSR_ID", default_value="B075"),
        ),
        DeclareLaunchArgument(
            "policy_server_host",
            default_value=EnvironmentVariable("POLICY_SERVER_HOST", default_value="127.0.0.1"),
        ),
        DeclareLaunchArgument(
            "policy_server_port",
            default_value=EnvironmentVariable("POLICY_SERVER_PORT", default_value="8000"),
        ),
        DeclareLaunchArgument(
            "policy_server_api_key",
            default_value=EnvironmentVariable("POLICY_SERVER_API_KEY", default_value=""),
        ),
        DeclareLaunchArgument("update_freq", default_value="10"),
        DeclareLaunchArgument("adopted_action_chunks", default_value="10"),
        DeclareLaunchArgument("upsample", default_value="true"),
        DeclareLaunchArgument("upsample_hz", default_value="30"),
        DeclareLaunchArgument("upsample_method", default_value="spline"),
        DeclareLaunchArgument("action_smoothing", default_value="ema"),
        DeclareLaunchArgument("ema_alpha", default_value="0.2"),
        DeclareLaunchArgument("ma_window", default_value="5"),
        DeclareLaunchArgument("smooth_gripper", default_value="true"),
     
        DeclareLaunchArgument("smooth_base", default_value="true"),
        DeclareLaunchArgument("gripper_mode", default_value="discrete"),
        DeclareLaunchArgument("require_control_mode", default_value="false"),
        DeclareLaunchArgument("expected_control_mode", default_value="auto"),
        DeclareLaunchArgument("save_exec_trace", default_value="false"),
        DeclareLaunchArgument("test_mode", default_value="true"),
    ]

    parameters = {
        "instruction": LaunchConfiguration("instruction"),
        "config_name": LaunchConfiguration("config_name"),
        "action_mode": LaunchConfiguration("action_mode"),
        "hsr_id": LaunchConfiguration("hsr_id"),
        "policy_server_host": LaunchConfiguration("policy_server_host"),
        "policy_server_port": ParameterValue(LaunchConfiguration("policy_server_port"), value_type=int),
        "policy_server_api_key": LaunchConfiguration("policy_server_api_key"),
        "update_freq": ParameterValue(LaunchConfiguration("update_freq"), value_type=int),
        "adopted_action_chunks": ParameterValue(LaunchConfiguration("adopted_action_chunks"), value_type=int),
        "upsample": ParameterValue(LaunchConfiguration("upsample"), value_type=bool),
        "upsample_hz": ParameterValue(LaunchConfiguration("upsample_hz"), value_type=int),
        "upsample_method": LaunchConfiguration("upsample_method"),
        "action_smoothing": LaunchConfiguration("action_smoothing"),
        "ema_alpha": ParameterValue(LaunchConfiguration("ema_alpha"), value_type=float),
        "ma_window": ParameterValue(LaunchConfiguration("ma_window"), value_type=int),
        "smooth_gripper": ParameterValue(LaunchConfiguration("smooth_gripper"), value_type=bool),
        "smooth_base": ParameterValue(LaunchConfiguration("smooth_base"), value_type=bool),
        "gripper_mode": LaunchConfiguration("gripper_mode"),
        "require_control_mode": ParameterValue(LaunchConfiguration("require_control_mode"), value_type=bool),
        "expected_control_mode": LaunchConfiguration("expected_control_mode"),
        "save_exec_trace": ParameterValue(LaunchConfiguration("save_exec_trace"), value_type=bool),
        "test_mode": ParameterValue(LaunchConfiguration("test_mode"), value_type=bool),
    }

    client_node = Node(
        package="hsr_policy_client",
        executable="hsr_policy.py",
        name="hsr_policy_client",
        output="screen",
        parameters=[parameters],
    )

    return LaunchDescription(declared_arguments + [client_node])
