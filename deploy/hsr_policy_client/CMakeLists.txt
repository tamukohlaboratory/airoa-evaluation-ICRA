cmake_minimum_required(VERSION 3.8)
project(hsr_policy_client)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hsr_policy_client_interfaces REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tmc_control_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

install(PROGRAMS
  scripts/hsr_policy.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
