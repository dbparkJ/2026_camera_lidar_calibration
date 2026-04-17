# front_lidar_camera_fusion

ROS1 Noetic package for a single front lidar and a single front camera.

Default live-topic assumptions in this repository:

- lidar: `/lidar0/velodyne_points`
- front camera image: `/roof_clpe_ros/roof_cam_1/image_raw`
- camera TF frame for projection: `avmFront`
- fused position: `/filter/positionlla`
- fused orientation: `/filter/quaternion`

Published topics:

- `/fusion/front_lidar/colored_points`
- `/fusion/front_lidar/colored_global_points`

Manual camera-lidar alignment:

- launch: `roslaunch front_lidar_camera_fusion lidar_camera_extrinsic_tuner.launch`
- default override YAML: `~/.ros/front_lidar_camera_extrinsic_override.yaml`
- RViz workflow: choose the `Publish Point` tool so RViz publishes to `/clicked_point`, click a lidar feature in the 3D view, then left-click the matching camera pixel in the tuner window
- the tuner auto-adds one 3D-2D correspondence when both clicks are available, solves an initial extrinsic from 4 or more pairs, and still allows small manual `tx/ty/tz/roll/pitch/yaw` nudges before save
- `front_pipeline.launch` automatically loads the saved override on startup

Checkerboard calibration:

- launch: `roslaunch front_lidar_camera_fusion camera_checkerboard_calibration.launch`
- default output YAML: `~/.ros/front_cam_1_checkerboard.yaml`
- board spec in this repo: `14 x 10` squares, `0.03m` square size
- OpenCV uses inner corners, so this board is calibrated as `13 x 9` inner corners

Notes:

- The manual reports camera resolution `1920x1080` and view angle `73V / 120H / 146D`.
- No live `CameraInfo` topic was detected during implementation, so the colorizer now prefers `camera_calibration_file` and falls back to FOV-based pinhole intrinsics only when that file is missing.
- The image header frame `roof_cam_1_link` was not present in `/tf`, so the projection step uses the configurable TF frame `avmFront`.
- The current default launch now uses `config/roof_cam_1_checkerboard.yaml`, populated from your 44-sample ROS `camera_calibration` result.
