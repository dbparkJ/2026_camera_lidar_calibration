#!/usr/bin/env python3

import math
import threading
from typing import List, Tuple

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import QuaternionStamped, Vector3Stamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


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

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


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


def geodetic_radii(latitude_deg: float) -> Tuple[float, float]:
    lat = math.radians(latitude_deg)
    sin_lat = math.sin(lat)
    denom = math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    prime_vertical = WGS84_A / denom
    meridian = WGS84_A * (1.0 - WGS84_E2) / (denom ** 3)
    return meridian, prime_vertical


def lla_to_enu(latitude: float, longitude: float, altitude: float,
               origin_lat: float, origin_lon: float, origin_alt: float) -> np.ndarray:
    meridian, prime_vertical = geodetic_radii(origin_lat)
    d_lat = math.radians(latitude - origin_lat)
    d_lon = math.radians(longitude - origin_lon)
    lat0 = math.radians(origin_lat)
    east = d_lon * prime_vertical * math.cos(lat0)
    north = d_lat * meridian
    up = altitude - origin_alt
    return np.array([east, north, up], dtype=np.float64)


def enu_to_lla(east: np.ndarray, north: np.ndarray, up: np.ndarray,
               origin_lat: float, origin_lon: float, origin_alt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    meridian, prime_vertical = geodetic_radii(origin_lat)
    lat0 = math.radians(origin_lat)
    lat = origin_lat + np.degrees(north / meridian)
    lon = origin_lon + np.degrees(east / (prime_vertical * math.cos(lat0)))
    alt = origin_alt + up
    return lat.astype(np.float64), lon.astype(np.float64), alt.astype(np.float64)


class LidarGlobalTransformNode:
    def __init__(self) -> None:
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.lock = threading.Lock()

        self.input_topic = rospy.get_param("~colored_cloud_topic", "/fusion/front_lidar/colored_points")
        self.output_topic = rospy.get_param("~global_cloud_topic", "/fusion/front_lidar/colored_global_points")
        self.position_topic = rospy.get_param("~position_topic", "/filter/positionlla")
        self.orientation_topic = rospy.get_param("~orientation_topic", "/filter/quaternion")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.tf_timeout_sec = float(rospy.get_param("~tf_timeout_sec", 0.2))
        self.state_timeout_sec = float(rospy.get_param("~state_timeout_sec", 0.5))
        self.publish_lat_lon_alt_fields = bool(rospy.get_param("~publish_lat_lon_alt_fields", True))
        self.origin_mode = rospy.get_param("~origin_mode", "first_fix")
        self.origin_latitude = float(rospy.get_param("~origin_latitude", 0.0))
        self.origin_longitude = float(rospy.get_param("~origin_longitude", 0.0))
        self.origin_altitude = float(rospy.get_param("~origin_altitude", 0.0))

        self.latest_position = None
        self.latest_position_stamp = None
        self.latest_quaternion = None
        self.latest_quaternion_stamp = None

        if self.origin_mode != "first_fix":
            self.origin = np.array(
                [self.origin_latitude, self.origin_longitude, self.origin_altitude],
                dtype=np.float64,
            )
        else:
            self.origin = None

        self.output_pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        rospy.Subscriber(self.position_topic, Vector3Stamped, self.position_callback, queue_size=10)
        rospy.Subscriber(self.orientation_topic, QuaternionStamped, self.orientation_callback, queue_size=50)
        rospy.Subscriber(self.input_topic, PointCloud2, self.cloud_callback, queue_size=1)

        rospy.loginfo(
            "front_lidar_global_transform_node listening: cloud=%s position=%s orientation=%s output=%s",
            self.input_topic,
            self.position_topic,
            self.orientation_topic,
            self.output_topic,
        )

    def position_callback(self, msg: Vector3Stamped) -> None:
        with self.lock:
            self.latest_position = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=np.float64)
            self.latest_position_stamp = msg.header.stamp
            if self.origin is None:
                self.origin = self.latest_position.copy()
                rospy.loginfo(
                    "Map origin fixed at lat=%.8f lon=%.8f alt=%.3f",
                    self.origin[0],
                    self.origin[1],
                    self.origin[2],
                )

    def orientation_callback(self, msg: QuaternionStamped) -> None:
        with self.lock:
            self.latest_quaternion = np.array(
                [
                    msg.quaternion.x,
                    msg.quaternion.y,
                    msg.quaternion.z,
                    msg.quaternion.w,
                ],
                dtype=np.float64,
            )
            self.latest_quaternion_stamp = msg.header.stamp

    def cloud_callback(self, cloud_msg: PointCloud2) -> None:
        with self.lock:
            position = None if self.latest_position is None else self.latest_position.copy()
            quaternion = None if self.latest_quaternion is None else self.latest_quaternion.copy()
            position_stamp = self.latest_position_stamp
            quaternion_stamp = self.latest_quaternion_stamp
            origin = None if self.origin is None else self.origin.copy()

        if position is None or quaternion is None or origin is None:
            rospy.logwarn_throttle(5.0, "Waiting for fused position/orientation state before publishing global cloud.")
            return

        if position_stamp is None or quaternion_stamp is None:
            return

        cloud_time = cloud_msg.header.stamp.to_sec()
        if abs(cloud_time - position_stamp.to_sec()) > self.state_timeout_sec:
            rospy.logwarn_throttle(2.0, "Position state is too old for current cloud.")
            return
        if abs(cloud_time - quaternion_stamp.to_sec()) > self.state_timeout_sec:
            rospy.logwarn_throttle(2.0, "Orientation state is too old for current cloud.")
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
                rospy.Duration(self.tf_timeout_sec),
            )
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "TF lookup failed for global transform: %s", exc)
            return

        cloud_array = pointcloud2_to_array(cloud_msg)
        if cloud_array.size == 0:
            return

        xyz_sensor = np.stack(
            (
                cloud_array["x"].astype(np.float64, copy=False),
                cloud_array["y"].astype(np.float64, copy=False),
                cloud_array["z"].astype(np.float64, copy=False),
            ),
            axis=1,
        )
        finite_mask = np.isfinite(xyz_sensor).all(axis=1)
        if not np.any(finite_mask):
            return

        rotation_sensor_to_base = quaternion_to_rotation_matrix(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        )
        translation_sensor_to_base = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        xyz_base = xyz_sensor @ rotation_sensor_to_base.T + translation_sensor_to_base

        vehicle_enu = lla_to_enu(position[0], position[1], position[2], origin[0], origin[1], origin[2])
        rotation_base_to_enu = quaternion_to_rotation_matrix(*quaternion)
        xyz_enu = xyz_base @ rotation_base_to_enu.T + vehicle_enu

        if self.publish_lat_lon_alt_fields:
            lat, lon, alt = enu_to_lla(xyz_enu[:, 0], xyz_enu[:, 1], xyz_enu[:, 2], origin[0], origin[1], origin[2])
        else:
            lat = np.zeros(xyz_enu.shape[0], dtype=np.float64)
            lon = np.zeros(xyz_enu.shape[0], dtype=np.float64)
            alt = np.zeros(xyz_enu.shape[0], dtype=np.float64)

        output_dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("intensity", np.float32),
                ("ring", np.uint16),
                ("time", np.float32),
                ("rgb", np.float32),
                ("sensor_x", np.float32),
                ("sensor_y", np.float32),
                ("sensor_z", np.float32),
                ("latitude", np.float64),
                ("longitude", np.float64),
                ("altitude", np.float64),
            ]
        )
        output_array = np.zeros(cloud_array.shape[0], dtype=output_dtype)
        output_array["x"] = xyz_enu[:, 0].astype(np.float32)
        output_array["y"] = xyz_enu[:, 1].astype(np.float32)
        output_array["z"] = xyz_enu[:, 2].astype(np.float32)
        output_array["sensor_x"] = xyz_sensor[:, 0].astype(np.float32)
        output_array["sensor_y"] = xyz_sensor[:, 1].astype(np.float32)
        output_array["sensor_z"] = xyz_sensor[:, 2].astype(np.float32)
        output_array["latitude"] = lat
        output_array["longitude"] = lon
        output_array["altitude"] = alt

        if "intensity" in cloud_array.dtype.names:
            output_array["intensity"] = cloud_array["intensity"].astype(np.float32, copy=False)
        if "ring" in cloud_array.dtype.names:
            output_array["ring"] = cloud_array["ring"].astype(np.uint16, copy=False)
        if "time" in cloud_array.dtype.names:
            output_array["time"] = cloud_array["time"].astype(np.float32, copy=False)
        if "rgb" in cloud_array.dtype.names:
            output_array["rgb"] = cloud_array["rgb"].astype(np.float32, copy=False)

        output_header = Header(stamp=cloud_msg.header.stamp, frame_id="map")
        self.output_pub.publish(array_to_pointcloud2(output_array, output_header))


def main() -> None:
    rospy.init_node("front_lidar_global_transform_node")
    LidarGlobalTransformNode()
    rospy.spin()


if __name__ == "__main__":
    main()
