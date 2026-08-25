"""Align SfM camera poses to GPS trajectory via similarity transform."""

import csv

import numpy as np
from scipy.spatial.transform import Rotation

from src.geospatial.coordinate import CoordinateTransformer


class GPSAligner:
    def __init__(self, config: dict):
        self.config = config
        self.gps_weight = config.get("gps_weight", 1.0)
        self.transformer = CoordinateTransformer()

    def align(
        self,
        poses: np.ndarray,
        gps_path: str,
        keyframe_indices: list[int],
        timestamps: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        gps_points = self._load_gps(gps_path)
        gps_enu = self._gps_to_enu(gps_points, timestamps, keyframe_indices)

        sfm_positions = poses[:, :3, 3]

        transform, scale = self._estimate_similarity_transform(sfm_positions, gps_enu)

        aligned_poses = self._apply_transform(poses, transform, scale)

        return aligned_poses, transform

    def _estimate_similarity_transform(
        self, src: np.ndarray, dst: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Umeyama alignment: find s, R, t that minimizes ||dst - (s*R*src + t)||."""
        n = min(len(src), len(dst))
        src = src[:n]
        dst = dst[:n]

        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        src_var = np.sum(src_centered ** 2) / n

        H = (src_centered.T @ dst_centered) / n
        U, S, Vt = np.linalg.svd(H)

        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1, 1, d])

        R = Vt.T @ sign_matrix @ U.T
        scale = np.trace(np.diag(S) @ sign_matrix) / src_var
        t = dst_mean - scale * R @ src_mean

        transform = np.eye(4)
        transform[:3, :3] = scale * R
        transform[:3, 3] = t

        return transform, scale

    def _apply_transform(
        self, poses: np.ndarray, transform: np.ndarray, scale: float
    ) -> np.ndarray:
        aligned = np.zeros_like(poses)
        R_align = transform[:3, :3] / scale
        t_align = transform[:3, 3]

        for i, pose in enumerate(poses):
            R_cam = pose[:3, :3]
            t_cam = pose[:3, 3]

            aligned[i] = np.eye(4)
            aligned[i, :3, :3] = R_align @ R_cam
            aligned[i, :3, 3] = scale * (R_align @ t_cam) + t_align

        return aligned

    def _load_gps(self, gps_path: str) -> list[dict]:
        points = []
        with open(gps_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                points.append({
                    "timestamp": float(row["timestamp"]),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "alt": float(row["alt"]),
                })
        return points

    def _gps_to_enu(
        self,
        gps_points: list[dict],
        timestamps: list[float],
        keyframe_indices: list[int],
    ) -> np.ndarray:
        if not gps_points:
            raise ValueError("No GPS data available")

        self.transformer.set_origin(
            gps_points[0]["lat"], gps_points[0]["lon"], gps_points[0]["alt"]
        )

        gps_ts = np.array([p["timestamp"] for p in gps_points])
        gps_enu_all = np.array([
            self.transformer.geodetic_to_enu(p["lat"], p["lon"], p["alt"])
            for p in gps_points
        ])

        keyframe_ts = [timestamps[i] for i in keyframe_indices]
        enu_at_keyframes = np.zeros((len(keyframe_ts), 3))

        for i, t in enumerate(keyframe_ts):
            if t <= gps_ts[0]:
                enu_at_keyframes[i] = gps_enu_all[0]
            elif t >= gps_ts[-1]:
                enu_at_keyframes[i] = gps_enu_all[-1]
            else:
                idx = np.searchsorted(gps_ts, t) - 1
                alpha = (t - gps_ts[idx]) / (gps_ts[idx + 1] - gps_ts[idx])
                enu_at_keyframes[i] = (
                    (1 - alpha) * gps_enu_all[idx] + alpha * gps_enu_all[idx + 1]
                )

        return enu_at_keyframes

    def compute_residuals(
        self, aligned_poses: np.ndarray, gps_enu: np.ndarray
    ) -> np.ndarray:
        n = min(len(aligned_poses), len(gps_enu))
        positions = aligned_poses[:n, :3, 3]
        residuals = np.linalg.norm(positions - gps_enu[:n], axis=1)
        return residuals
