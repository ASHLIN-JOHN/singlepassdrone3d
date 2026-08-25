"""Metric scale estimation from GPS baselines."""

import csv

import numpy as np

from src.geospatial.coordinate import CoordinateTransformer


class ScaleEstimator:
    def __init__(self, config: dict):
        self.method = config.get("scale_method", "gps_baseline")

    def estimate(
        self,
        poses: np.ndarray,
        gps_path: str,
        keyframe_indices: list[int],
        timestamps: list[float],
    ) -> float:
        if self.method == "gps_baseline":
            return self._estimate_from_gps_baseline(poses, gps_path, keyframe_indices, timestamps)
        elif self.method == "altitude":
            return self._estimate_from_altitude(poses, gps_path, keyframe_indices, timestamps)
        return 1.0

    def _estimate_from_gps_baseline(
        self,
        poses: np.ndarray,
        gps_path: str,
        keyframe_indices: list[int],
        timestamps: list[float],
    ) -> float:
        if not gps_path:
            return 1.0

        gps_positions = self._load_gps_enu(gps_path, timestamps, keyframe_indices)
        sfm_positions = poses[:, :3, 3]

        n = min(len(sfm_positions), len(gps_positions))
        if n < 2:
            return 1.0

        sfm_dists = []
        gps_dists = []
        for i in range(n - 1):
            sfm_dists.append(np.linalg.norm(sfm_positions[i + 1] - sfm_positions[i]))
            gps_dists.append(np.linalg.norm(gps_positions[i + 1] - gps_positions[i]))

        sfm_dists = np.array(sfm_dists)
        gps_dists = np.array(gps_dists)

        valid = sfm_dists > 1e-6
        if not valid.any():
            return 1.0

        ratios = gps_dists[valid] / sfm_dists[valid]
        scale = float(np.median(ratios))

        return scale

    def _estimate_from_altitude(
        self,
        poses: np.ndarray,
        gps_path: str,
        keyframe_indices: list[int],
        timestamps: list[float],
    ) -> float:
        gps_positions = self._load_gps_enu(gps_path, timestamps, keyframe_indices)
        sfm_positions = poses[:, :3, 3]

        gps_alt_range = gps_positions[:, 2].max() - gps_positions[:, 2].min()
        sfm_alt_range = sfm_positions[:, 2].max() - sfm_positions[:, 2].min()

        if sfm_alt_range < 1e-6:
            return 1.0

        return gps_alt_range / sfm_alt_range

    def _load_gps_enu(
        self, gps_path: str, timestamps: list[float], keyframe_indices: list[int]
    ) -> np.ndarray:
        gps_points = []
        with open(gps_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gps_points.append({
                    "timestamp": float(row["timestamp"]),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "alt": float(row["alt"]),
                })

        if not gps_points:
            return np.zeros((len(keyframe_indices), 3))

        transformer = CoordinateTransformer()
        transformer.set_origin(gps_points[0]["lat"], gps_points[0]["lon"], gps_points[0]["alt"])

        gps_ts = np.array([p["timestamp"] for p in gps_points])
        gps_enu = np.array([
            transformer.geodetic_to_enu(p["lat"], p["lon"], p["alt"])
            for p in gps_points
        ])

        keyframe_ts = [timestamps[i] for i in keyframe_indices]
        positions = np.zeros((len(keyframe_ts), 3))
        for i, t in enumerate(keyframe_ts):
            idx = max(0, np.searchsorted(gps_ts, t) - 1)
            idx = min(idx, len(gps_ts) - 2)
            if idx < len(gps_ts) - 1:
                alpha = (t - gps_ts[idx]) / max(gps_ts[idx + 1] - gps_ts[idx], 1e-9)
                alpha = np.clip(alpha, 0, 1)
                positions[i] = (1 - alpha) * gps_enu[idx] + alpha * gps_enu[idx + 1]
            else:
                positions[i] = gps_enu[-1]

        return positions
