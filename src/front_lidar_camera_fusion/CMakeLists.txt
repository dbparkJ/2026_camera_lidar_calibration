cmake_minimum_required(VERSION 3.0.2)
project(front_lidar_camera_fusion)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_filters
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/camera_checkerboard_calibrator.py
  scripts/extrinsic_override.py
  scripts/lidar_camera_extrinsic_tuner.py
  scripts/lidar_colorize_node.py
  scripts/lidar_global_transform_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
