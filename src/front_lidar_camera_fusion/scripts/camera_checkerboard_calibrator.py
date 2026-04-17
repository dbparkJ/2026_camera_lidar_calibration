#!/usr/bin/env python3

import os
from datetime import datetime
from typing import List

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraCheckerboardCalibrator:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.image_topic = rospy.get_param("~image_topic", "/roof_clpe_ros/roof_cam_1/image_raw")
        self.camera_name = rospy.get_param("~camera_name", "roof_cam_1")
        self.output_yaml = os.path.expanduser(
            rospy.get_param("~camera_calibration_file", "~/.ros/front_cam_1_checkerboard.yaml")
        )
        self.debug_dir = os.path.expanduser(
            rospy.get_param("~calibration_debug_dir", "~/.ros/front_cam_1_checkerboard_samples")
        )
        self.square_size_m = float(rospy.get_param("~checkerboard_square_size_m", 0.03))
        self.board_squares_x = int(rospy.get_param("~checkerboard_squares_x", 14))
        self.board_squares_y = int(rospy.get_param("~checkerboard_squares_y", 10))
        self.min_samples = int(rospy.get_param("~calibration_min_samples", 12))
        self.target_samples = int(rospy.get_param("~calibration_target_samples", 25))
        self.capture_interval_sec = float(rospy.get_param("~calibration_capture_interval_sec", 0.5))
        self.min_sample_distance = float(rospy.get_param("~calibration_min_sample_distance", 0.08))
        self.auto_shutdown = bool(rospy.get_param("~calibration_auto_shutdown", True))

        self.inner_corners_x = self.board_squares_x - 1
        self.inner_corners_y = self.board_squares_y - 1
        if self.inner_corners_x <= 0 or self.inner_corners_y <= 0:
            raise ValueError("Checkerboard square counts must be at least 2x2.")
        self.pattern_size = (self.inner_corners_x, self.inner_corners_y)

        os.makedirs(os.path.dirname(self.output_yaml) or ".", exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

        self.object_template = np.zeros((self.inner_corners_x * self.inner_corners_y, 3), np.float32)
        grid = np.mgrid[0:self.inner_corners_x, 0:self.inner_corners_y].T.reshape(-1, 2)
        self.object_template[:, :2] = grid * self.square_size_m

        self.object_points: List[np.ndarray] = []
        self.image_points: List[np.ndarray] = []
        self.sample_descriptors: List[np.ndarray] = []
        self.last_capture_time = rospy.Time(0)
        self.image_size = None
        self.finished = False

        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.loginfo(
            "front_camera_checkerboard_calibrator listening: image=%s board=%dx%d squares -> %dx%d inner corners, square=%.3fm output=%s",
            self.image_topic,
            self.board_squares_x,
            self.board_squares_y,
            self.inner_corners_x,
            self.inner_corners_y,
            self.square_size_m,
            self.output_yaml,
        )

    def image_callback(self, msg: Image) -> None:
        if self.finished:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Failed to convert calibration image: %s", exc)
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = self._detect_checkerboard(gray)
        if not found:
            rospy.loginfo_throttle(
                5.0,
                "Waiting for checkerboard %dx%d squares (%dx%d inner corners).",
                self.board_squares_x,
                self.board_squares_y,
                self.inner_corners_x,
                self.inner_corners_y,
            )
            return

        descriptor = self._build_descriptor(corners, gray.shape)
        now = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        if not self._should_accept_sample(descriptor, now):
            return

        self.image_size = (gray.shape[1], gray.shape[0])
        self.object_points.append(self.object_template.copy())
        self.image_points.append(corners.astype(np.float32))
        self.sample_descriptors.append(descriptor)
        self.last_capture_time = now
        self._save_debug_image(image, corners, len(self.image_points))

        rospy.loginfo(
            "Accepted checkerboard sample %d/%d",
            len(self.image_points),
            self.target_samples,
        )

        if len(self.image_points) >= self.target_samples:
            self._run_calibration()

    def _detect_checkerboard(self, gray: np.ndarray):
        corners = None
        found = False

        if hasattr(cv2, "findChessboardCornersSB"):
            flags = cv2.CALIB_CB_NORMALIZE_IMAGE
            flags |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
            flags |= getattr(cv2, "CALIB_CB_ACCURACY", 0)
            found, corners = cv2.findChessboardCornersSB(gray, self.pattern_size, flags)

        if not found:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)
            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if not found or corners is None:
            return False, None

        return True, corners.reshape(-1, 1, 2)

    def _build_descriptor(self, corners: np.ndarray, image_shape) -> np.ndarray:
        points = corners.reshape(-1, 2)
        width = float(image_shape[1])
        height = float(image_shape[0])
        center = points.mean(axis=0)
        span = points.max(axis=0) - points.min(axis=0)
        return np.array(
            [
                center[0] / width,
                center[1] / height,
                span[0] / width,
                span[1] / height,
            ],
            dtype=np.float32,
        )

    def _should_accept_sample(self, descriptor: np.ndarray, stamp: rospy.Time) -> bool:
        if len(self.image_points) == 0:
            return True

        elapsed = (stamp - self.last_capture_time).to_sec()
        if elapsed < self.capture_interval_sec:
            return False

        distances = [np.linalg.norm(descriptor - sample) for sample in self.sample_descriptors]
        min_distance = min(distances)
        if min_distance < self.min_sample_distance:
            return False

        return True

    def _save_debug_image(self, image: np.ndarray, corners: np.ndarray, sample_index: int) -> None:
        annotated = image.copy()
        cv2.drawChessboardCorners(annotated, self.pattern_size, corners, True)
        label = f"sample {sample_index:02d}/{self.target_samples}"
        cv2.putText(
            annotated,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        filename = os.path.join(self.debug_dir, f"accepted_{sample_index:02d}.jpg")
        cv2.imwrite(filename, annotated)

    def _run_calibration(self) -> None:
        if len(self.image_points) < self.min_samples:
            rospy.logwarn(
                "Need at least %d accepted samples before calibration, only have %d.",
                self.min_samples,
                len(self.image_points),
            )
            return

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.image_size,
            None,
            None,
        )
        mean_error = self._compute_reprojection_error(camera_matrix, dist_coeffs, rvecs, tvecs)
        rectification_matrix = np.eye(3, dtype=np.float64)
        projection_matrix = np.zeros((3, 4), dtype=np.float64)
        projection_matrix[:3, :3] = camera_matrix

        calibration = {
            "image_width": int(self.image_size[0]),
            "image_height": int(self.image_size[1]),
            "camera_name": self.camera_name,
            "distortion_model": "plumb_bob",
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": [float(x) for x in camera_matrix.reshape(-1)],
            },
            "distortion_coefficients": {
                "rows": 1,
                "cols": int(dist_coeffs.size),
                "data": [float(x) for x in dist_coeffs.reshape(-1)],
            },
            "rectification_matrix": {
                "rows": 3,
                "cols": 3,
                "data": [float(x) for x in rectification_matrix.reshape(-1)],
            },
            "projection_matrix": {
                "rows": 3,
                "cols": 4,
                "data": [float(x) for x in projection_matrix.reshape(-1)],
            },
            "checkerboard": {
                "square_size_m": self.square_size_m,
                "squares_x": self.board_squares_x,
                "squares_y": self.board_squares_y,
                "inner_corners_x": self.inner_corners_x,
                "inner_corners_y": self.inner_corners_y,
                "accepted_samples": len(self.image_points),
                "rms_reprojection_error": float(rms),
                "mean_reprojection_error_px": float(mean_error),
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
        }

        with open(self.output_yaml, "w", encoding="utf-8") as stream:
            yaml.safe_dump(calibration, stream, sort_keys=False)

        self.finished = True
        rospy.loginfo(
            "Camera calibration saved to %s with RMS=%.5f mean_error=%.5f px",
            self.output_yaml,
            rms,
            mean_error,
        )

        if self.auto_shutdown:
            rospy.signal_shutdown("Camera checkerboard calibration completed.")

    def _compute_reprojection_error(self, camera_matrix, dist_coeffs, rvecs, tvecs) -> float:
        total_error = 0.0
        total_points = 0
        for obj_points, img_points, rvec, tvec in zip(self.object_points, self.image_points, rvecs, tvecs):
            projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
            error = cv2.norm(img_points, projected, cv2.NORM_L2)
            total_error += error * error
            total_points += len(obj_points)
        if total_points == 0:
            return 0.0
        return float(np.sqrt(total_error / total_points))


def main() -> None:
    rospy.init_node("front_camera_checkerboard_calibrator")
    CameraCheckerboardCalibrator()
    rospy.spin()


if __name__ == "__main__":
    main()
