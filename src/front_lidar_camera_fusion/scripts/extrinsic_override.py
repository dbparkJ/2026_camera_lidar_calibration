#!/usr/bin/env python3

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
import yaml


@dataclass
class ExtrinsicOverride:
    path: str
    translation_m: np.ndarray
    rotation_deg: np.ndarray
    loaded_from_file: bool = False


def _vector_from_value(value, labels: Tuple[str, str, str]) -> np.ndarray:
    if isinstance(value, dict):
        return np.array([float(value.get(label, 0.0)) for label in labels], dtype=np.float64)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return np.array([float(item) for item in value], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def default_extrinsic_override(path: str) -> ExtrinsicOverride:
    return ExtrinsicOverride(
        path=os.path.expanduser(path),
        translation_m=np.zeros(3, dtype=np.float64),
        rotation_deg=np.zeros(3, dtype=np.float64),
        loaded_from_file=False,
    )


def load_extrinsic_override(path: str) -> ExtrinsicOverride:
    override = default_extrinsic_override(path)
    if not override.path or not os.path.exists(override.path):
        return override

    with open(override.path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    override.translation_m = _vector_from_value(data.get("translation_m", {}), ("x", "y", "z"))
    override.rotation_deg = _vector_from_value(data.get("rotation_deg", {}), ("roll", "pitch", "yaw"))
    override.loaded_from_file = True
    return override


def save_extrinsic_override(
    path: str,
    camera_tf_frame: str,
    lidar_frame: str,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
) -> str:
    expanded_path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(expanded_path) or ".", exist_ok=True)
    payload = {
        "version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "camera_tf_frame": camera_tf_frame,
        "lidar_frame": lidar_frame,
        "translation_m": {
            "x": float(translation_m[0]),
            "y": float(translation_m[1]),
            "z": float(translation_m[2]),
        },
        "rotation_deg": {
            "roll": float(rotation_deg[0]),
            "pitch": float(rotation_deg[1]),
            "yaw": float(rotation_deg[2]),
        },
        "notes": "Applied after TF lookup in the camera frame before optical projection.",
    }
    with open(expanded_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
    return expanded_path


def rotation_matrix_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = np.deg2rad(float(roll_deg))
    pitch = np.deg2rad(float(pitch_deg))
    yaw = np.deg2rad(float(yaw_deg))

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def apply_extrinsic_override(points_camera_frame: np.ndarray, override: ExtrinsicOverride) -> np.ndarray:
    if points_camera_frame.size == 0:
        return points_camera_frame
    if np.allclose(override.translation_m, 0.0) and np.allclose(override.rotation_deg, 0.0):
        return points_camera_frame

    rotation = rotation_matrix_from_euler_deg(*override.rotation_deg)
    return points_camera_frame @ rotation.T + override.translation_m
