#!/usr/bin/env python3

import math
import os
import sys
from typing import List, Tuple

import cv2
import message_filters
import numpy as np
import rospy
import tf2_ros
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from extrinsic_override import apply_extrinsic_override, load_extrinsic_override


POINTFIELD_TO_DTYPE = {
    PointField.INT8: np.int8,
    PointField.UINT8: np.uint8,
    PointField.INT16: np.int16,
    PointField.UINT16: np.uint16,
    PointField.INT32: np.int32,
    PointField.UINT32: np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}

NP_TO_POINTFIELD = {
    np.dtype(np.int8): PointField.INT8,
    np.dtype(np.uint8): PointField.UINT8,
    np.dtype(np.int16): PointField.INT16,
    np.dtype(np.uint16): PointField.UINT16,
    np.dtype(np.int32): PointField.INT32,
    np.dtype(np.uint32): PointField.UINT32,
    np.dtype(np.float32): PointField.FLOAT32,
    np.dtype(np.float64): PointField.FLOAT64,
}


def build_dtype(fields: List[PointField], point_step: int) -> np.dtype:
    dtype_fields: List[Tuple[str, object]] = []
    offset = 0
    for field in sorted(fields, key=lambda item: item.offset):
        if offset < field.offset:
            dtype_fields.append((f"__pad_{offset}", f"V{field.offset - offset}"))
            offset = field.offset
        base_dtype = np.dtype(POINTFIELD_TO_DTYPE[field.datatype])
        if field.count == 1:
            dtype_fields.append((field.name, base_dtype))
        else:
            dtype_fields.append((field.name, base_dtype, (field.count,)))
        offset += base_dtype.itemsize * field.count
    if offset < point_step:
        dtype_fields.append((f"__pad_{offset}", f"V{point_step - offset}"))
    return np.dtype(dtype_fields)


def pointcloud2_to_array(msg: PointCloud2) -> np.ndarray:
    dtype = build_dtype(msg.fields, msg.point_step)
    count = msg.width * msg.height
    return np.frombuffer(msg.data, dtype=dtype, count=count)


def fields_from_dtype(dtype: np.dtype) -> List[PointField]:
    fields = []
    for name in dtype.names:
        if name.startswith("__pad_"):
            continue
        field_dtype, offset = dtype.fields[name][:2]
        base_dtype = field_dtype.base
        count = int(np.prod(field_dtype.shape)) if field_dtype.shape else 1
        fields.append(
            PointField(
                name=name,
                offset=offset,
                datatype=NP_TO_POINTFIELD[np.dtype(base_dtype)],
                count=count,
            )
        )
    return fields


def array_to_pointcloud2(array: np.ndarray, header: Header) -> PointCloud2:
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(array.shape[0])
    msg.fields = fields_from_dtype(array.dtype)
    msg.is_bigendian = False
    msg.point_step = array.dtype.itemsize
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = array.tobytes()
    return msg


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


class LidarColorizeNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cloud_topic = rospy.get_param("~cloud_topic", "/lidar0/velodyne_points")
        self.image_topic = rospy.get_param("~image_topic", "/roof_clpe_ros/roof_cam_1/image_raw")
        self.camera_tf_frame = rospy.get_param("~camera_tf_frame", "avmFront")
        self.output_topic = rospy.get_param("~colored_cloud_topic", "/fusion/front_lidar/colored_points")
        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 10))
        self.sync_slop_sec = float(rospy.get_param("~sync_slop_sec", 0.15))
        self.tf_timeout_sec = float(rospy.get_param("~tf_timeout_sec", 0.2))
        self.min_depth_m = float(rospy.get_param("~min_depth_m", 0.3))
        self.drop_out_of_image_points = bool(rospy.get_param("~drop_out_of_image_points", False))
        self.enable_occlusion_filter = bool(rospy.get_param("~enable_occlusion_filter", True))
        self.occlusion_depth_tolerance_m = float(
            rospy.get_param("~occlusion_depth_tolerance_m", 0.15)
        )
        self.projection_model = rospy.get_param("~projection_model", "vehicle_neg_x_front")
        self.camera_calibration_file = os.path.expanduser(
            rospy.get_param("~camera_calibration_file", "")
        ).strip()
        self.extrinsic_override_file = os.path.expanduser(
            rospy.get_param("~extrinsic_override_file", "")
        ).strip()

        self.image_width = int(rospy.get_param("~image_width", 1920))
        self.image_height = int(rospy.get_param("~image_height", 1080))
        self.hfov_deg = float(rospy.get_param("~hfov_deg", 120.0))
        self.vfov_deg = float(rospy.get_param("~vfov_deg", 73.0))
        self.fx = float(rospy.get_param("~fx", 0.0))
        self.fy = float(rospy.get_param("~fy", 0.0))
        self.cx = float(rospy.get_param("~cx", 0.0))
        self.cy = float(rospy.get_param("~cy", 0.0))

        self.calibration_loaded = False
        self.calibration_image_width = self.image_width
        self.calibration_image_height = self.image_height
        self.camera_matrix = None
        self.dist_coeffs = None
        self.extrinsic_override = load_extrinsic_override(self.extrinsic_override_file)
        self._load_intrinsics()

        if self.extrinsic_override_file:
            if self.extrinsic_override.loaded_from_file:
                rospy.loginfo(
                    "Loaded camera-lidar extrinsic override from %s",
                    self.extrinsic_override.path,
                )
            else:
                rospy.loginfo(
                    "No camera-lidar extrinsic override file found at %s, using TF only.",
                    self.extrinsic_override.path,
                )

        self.output_pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        cloud_sub = message_filters.Subscriber(self.cloud_topic, PointCloud2)
        image_sub = message_filters.Subscriber(self.image_topic, Image)
        sync = message_filters.ApproximateTimeSynchronizer(
            [cloud_sub, image_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
            allow_headerless=False,
        )
        sync.registerCallback(self.callback)

        rospy.loginfo(
            "front_lidar_colorize_node listening: cloud=%s image=%s camera_tf=%s output=%s",
            self.cloud_topic,
            self.image_topic,
            self.camera_tf_frame,
            self.output_topic,
        )

    def _load_intrinsics(self) -> None:
        if self.camera_calibration_file and os.path.exists(self.camera_calibration_file):
            try:
                with open(self.camera_calibration_file, "r", encoding="utf-8") as stream:
                    calibration = yaml.safe_load(stream) or {}
                camera_data = calibration["camera_matrix"]["data"]
                distortion_data = calibration.get("distortion_coefficients", {}).get("data", [])
                self.camera_matrix = np.array(camera_data, dtype=np.float64).reshape(3, 3)
                self.dist_coeffs = np.array(distortion_data, dtype=np.float64).reshape(-1, 1)
                self.calibration_image_width = int(calibration.get("image_width", self.image_width))
                self.calibration_image_height = int(calibration.get("image_height", self.image_height))
                self.fx = float(self.camera_matrix[0, 0])
                self.fy = float(self.camera_matrix[1, 1])
                self.cx = float(self.camera_matrix[0, 2])
                self.cy = float(self.camera_matrix[1, 2])
                self.calibration_loaded = True
                rospy.loginfo("Loaded camera calibration from %s", self.camera_calibration_file)
                return
            except Exception as exc:
                rospy.logwarn(
                    "Failed to load calibration file %s, using fallback FOV intrinsics: %s",
                    self.camera_calibration_file,
                    exc,
                )

        if self.fx <= 0.0:
            self.fx = (self.image_width * 0.5) / math.tan(math.radians(self.hfov_deg * 0.5))
        if self.fy <= 0.0:
            self.fy = (self.image_height * 0.5) / math.tan(math.radians(self.vfov_deg * 0.5))
        if self.cx <= 0.0:
            self.cx = self.image_width * 0.5
        if self.cy <= 0.0:
            self.cy = self.image_height * 0.5
        self.camera_matrix = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self.calibration_image_width = self.image_width
        self.calibration_image_height = self.image_height
        self.calibration_loaded = False

    def _scaled_camera_matrix(self, image_width: int, image_height: int) -> np.ndarray:
        matrix = self.camera_matrix.copy()
        scale_x = float(image_width) / float(self.calibration_image_width)
        scale_y = float(image_height) / float(self.calibration_image_height)
        matrix[0, 0] *= scale_x
        matrix[1, 1] *= scale_y
        matrix[0, 2] *= scale_x
        matrix[1, 2] *= scale_y
        return matrix

    def callback(self, cloud_msg: PointCloud2, image_msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Failed to convert image: %s", exc)
            return

        current_height, current_width = image.shape[:2]
        if current_width != self.image_width or current_height != self.image_height:
            self.image_height = current_height
            self.image_width = current_width
            if not self.calibration_loaded:
                self.fx = 0.0
                self.fy = 0.0
                self.cx = 0.0
                self.cy = 0.0
                self._load_intrinsics()

        try:
            transform = self.tf_buffer.lookup_transform(
                self.camera_tf_frame,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
                rospy.Duration(self.tf_timeout_sec),
            )
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "TF lookup failed for colorization: %s", exc)
            return

        cloud_array = pointcloud2_to_array(cloud_msg)
        if cloud_array.size == 0:
            return

        output_dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("intensity", np.float32),
                ("ring", np.uint16),
                ("time", np.float32),
                ("rgb", np.float32),
            ]
        )
        output_array = np.zeros(cloud_array.shape[0], dtype=output_dtype)
        output_array["x"] = cloud_array["x"].astype(np.float32, copy=False)
        output_array["y"] = cloud_array["y"].astype(np.float32, copy=False)
        output_array["z"] = cloud_array["z"].astype(np.float32, copy=False)
        if "intensity" in cloud_array.dtype.names:
            output_array["intensity"] = cloud_array["intensity"].astype(np.float32, copy=False)
        if "ring" in cloud_array.dtype.names:
            output_array["ring"] = cloud_array["ring"].astype(np.uint16, copy=False)
        if "time" in cloud_array.dtype.names:
            output_array["time"] = cloud_array["time"].astype(np.float32, copy=False)

        xyz = np.stack(
            (
                cloud_array["x"].astype(np.float64, copy=False),
                cloud_array["y"].astype(np.float64, copy=False),
                cloud_array["z"].astype(np.float64, copy=False),
            ),
            axis=1,
        )
        finite_mask = np.isfinite(xyz).all(axis=1)
        if not np.any(finite_mask):
            return

        xyz_valid = xyz[finite_mask]
        rotation = quaternion_to_rotation_matrix(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        )
        translation = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        camera_frame_points = xyz_valid @ rotation.T + translation
        camera_frame_points = apply_extrinsic_override(camera_frame_points, self.extrinsic_override)
        optical_points = self._to_optical_frame(camera_frame_points)

        depth = optical_points[:, 2]
        visible_mask = depth > self.min_depth_m
        if not np.any(visible_mask):
            self.output_pub.publish(array_to_pointcloud2(output_array, cloud_msg.header))
            return

        optical_visible = optical_points[visible_mask]
        valid_indices = np.flatnonzero(finite_mask)[visible_mask]
        visible_depth = depth[visible_mask]

        camera_matrix = self._scaled_camera_matrix(current_width, current_height)
        projected, _ = cv2.projectPoints(
            optical_visible.reshape(-1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            camera_matrix,
            self.dist_coeffs,
        )
        uv = projected.reshape(-1, 2)
        uv_finite = np.isfinite(uv).all(axis=1)
        u = np.zeros(uv.shape[0], dtype=np.int32)
        v = np.zeros(uv.shape[0], dtype=np.int32)
        if np.any(uv_finite):
            u[uv_finite] = np.rint(uv[uv_finite, 0]).astype(np.int32)
            v[uv_finite] = np.rint(uv[uv_finite, 1]).astype(np.int32)
        in_image = uv_finite & (u >= 0) & (u < current_width) & (v >= 0) & (v < current_height)

        if self.enable_occlusion_filter and np.any(in_image):
            visible_depth_finite = visible_depth[uv_finite]
            u_in = u[in_image]
            v_in = v[in_image]
            depth_in = visible_depth_finite[in_image]
            pixel_ids = v_in.astype(np.int64) * int(current_width) + u_in.astype(np.int64)
            nearest_depth = np.full(current_width * current_height, np.inf, dtype=np.float32)
            np.minimum.at(nearest_depth, pixel_ids, depth_in.astype(np.float32))
            min_depth_at_pixel = nearest_depth[pixel_ids].astype(np.float64)
            occlusion_keep = depth_in <= (min_depth_at_pixel + self.occlusion_depth_tolerance_m)
            temp_mask = in_image.copy()
            temp_mask[np.flatnonzero(in_image)] = occlusion_keep
            in_image = temp_mask

        if self.drop_out_of_image_points:
            keep_indices = valid_indices[in_image]
            output_array = output_array[keep_indices]
            valid_indices = np.arange(output_array.shape[0], dtype=np.int64)
            u = u[in_image]
            v = v[in_image]
            in_image = np.ones(valid_indices.shape[0], dtype=bool)

        if np.any(in_image):
            sampled_bgr = image[v[in_image], u[in_image]]
            rgb_uint32 = (
                (sampled_bgr[:, 2].astype(np.uint32) << 16)
                | (sampled_bgr[:, 1].astype(np.uint32) << 8)
                | sampled_bgr[:, 0].astype(np.uint32)
            )
            rgb_float = rgb_uint32.view(np.float32)
            output_array["rgb"][valid_indices[in_image]] = rgb_float

        self.output_pub.publish(array_to_pointcloud2(output_array, cloud_msg.header))

    def _to_optical_frame(self, points: np.ndarray) -> np.ndarray:
        if self.projection_model == "optical_z_forward":
            return points

        if self.projection_model == "vehicle_neg_x_front":
            optical = np.empty_like(points)
            optical[:, 0] = points[:, 1]
            optical[:, 1] = -points[:, 2]
            optical[:, 2] = -points[:, 0]
            return optical

        raise ValueError(f"Unsupported projection model: {self.projection_model}")


def main() -> None:
    rospy.init_node("front_lidar_colorize_node")
    LidarColorizeNode()
    rospy.spin()


if __name__ == "__main__":
    main()
