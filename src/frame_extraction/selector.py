"""Keyframe selection with GPS-aware spatial diversity."""

import numpy as np
from typing import Optional


class KeyframeSelector:
    def __init__(self, config: dict):
        self.min_frames = config.get("min_frames", 50)
        self.max_frames = config.get("max_frames", 200)
        self.diversity_weight = config.get("diversity_weight", 0.3)
        self.gps_spacing_m = config.get("gps_spacing_m", 2.0)

    def select(
        self,
        frames: list[np.ndarray],
        scores: list[dict],
        timestamps: list[float],
        gps_data: Optional[str] = None,
    ) -> tuple[list[np.ndarray], list[int]]:
        n = len(frames)
        if n <= self.min_frames:
            return frames, list(range(n))

        quality_scores = np.array([s["composite"] for s in scores])
        quality_mask = np.array([s["is_sharp"] for s in scores])

        if gps_data:
            gps_positions = self._load_gps_positions(gps_data, timestamps)
            selected = self._select_with_gps(quality_scores, quality_mask, gps_positions)
        else:
            selected = self._select_by_quality_and_spacing(quality_scores, quality_mask, timestamps)

        selected_frames = [frames[i] for i in selected]
        return selected_frames, selected

    def _select_with_gps(
        self,
        scores: np.ndarray,
        mask: np.ndarray,
        positions: np.ndarray,
    ) -> list[int]:
        selected = [0]
        for i in range(1, len(scores)):
            if not mask[i]:
                continue
            last_pos = positions[selected[-1]]
            curr_pos = positions[i]
            dist = np.linalg.norm(curr_pos - last_pos)
            if dist >= self.gps_spacing_m:
                selected.append(i)

        if len(selected) > self.max_frames:
            quality_at_selected = scores[selected]
            top_indices = np.argsort(quality_at_selected)[-self.max_frames:]
            top_indices.sort()
            selected = [selected[i] for i in top_indices]

        if len(selected) < self.min_frames:
            remaining = [i for i in range(len(scores)) if i not in selected and mask[i]]
            remaining.sort(key=lambda i: scores[i], reverse=True)
            needed = self.min_frames - len(selected)
            selected.extend(remaining[:needed])
            selected.sort()

        return selected

    def _select_by_quality_and_spacing(
        self,
        scores: np.ndarray,
        mask: np.ndarray,
        timestamps: list[float],
    ) -> list[int]:
        target = min(self.max_frames, sum(mask))
        if target <= 0:
            target = self.min_frames

        valid_indices = np.where(mask)[0]
        if len(valid_indices) <= target:
            return valid_indices.tolist()

        step = len(valid_indices) / target
        selected = [valid_indices[int(i * step)] for i in range(target)]
        return selected

    def _load_gps_positions(self, gps_data: str, timestamps: list[float]) -> np.ndarray:
        """Load GPS and interpolate to frame timestamps.

        Expects CSV with columns: timestamp, lat, lon, alt
        Returns Nx3 array in local ENU meters.
        """
        import csv
        from src.geospatial.coordinate import CoordinateTransformer

        gps_points = []
        with open(gps_data) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gps_points.append({
                    "timestamp": float(row["timestamp"]),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "alt": float(row["alt"]),
                })

        if not gps_points:
            return np.zeros((len(timestamps), 3))

        transformer = CoordinateTransformer()
        origin = (gps_points[0]["lat"], gps_points[0]["lon"], gps_points[0]["alt"])
        transformer.set_origin(*origin)

        gps_ts = np.array([p["timestamp"] for p in gps_points])
        gps_enu = np.array([
            transformer.geodetic_to_enu(p["lat"], p["lon"], p["alt"])
            for p in gps_points
        ])

        positions = np.zeros((len(timestamps), 3))
        for i, t in enumerate(timestamps):
            positions[i] = self._interpolate_position(t, gps_ts, gps_enu)

        return positions

    def _interpolate_position(
        self, t: float, gps_ts: np.ndarray, gps_enu: np.ndarray
    ) -> np.ndarray:
        if t <= gps_ts[0]:
            return gps_enu[0]
        if t >= gps_ts[-1]:
            return gps_enu[-1]

        idx = np.searchsorted(gps_ts, t) - 1
        alpha = (t - gps_ts[idx]) / (gps_ts[idx + 1] - gps_ts[idx])
        return (1 - alpha) * gps_enu[idx] + alpha * gps_enu[idx + 1]
