"""Fuse multiple depth maps into a unified 3D point cloud."""

import numpy as np
import cv2


class DepthFusion:
    def __init__(self, config: dict):
        self.depth_trunc = config.get("depth_trunc", 50.0)
        self.min_confidence = config.get("min_confidence", 0.3)

    def fuse(
        self,
        frames: list[np.ndarray],
        depth_maps: list[np.ndarray],
        poses: np.ndarray,
        intrinsics: dict,
        scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_points = []
        all_colors = []

        cam_intrinsics = self._get_intrinsic_matrix(intrinsics)

        for i, (frame, depth, pose) in enumerate(zip(frames, depth_maps, poses)):
            points_cam, colors = self._unproject_depth(frame, depth, cam_intrinsics)

            points_world = self._transform_points(points_cam, pose, scale)

            valid = np.linalg.norm(points_cam, axis=1) < self.depth_trunc
            all_points.append(points_world[valid])
            all_colors.append(colors[valid])

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)

        return points, colors

    def _unproject_depth(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = depth.shape[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.astype(np.float32)
        v = v.astype(np.float32)

        z = depth.astype(np.float32)
        valid = z > 0

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x[valid], y[valid], z[valid]], axis=-1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[:2] != (h, w):
            frame_rgb = cv2.resize(frame_rgb, (w, h))
        colors = frame_rgb[valid].astype(np.float64) / 255.0

        # Subsample to keep point count manageable
        stride = max(1, len(points) // 100000)
        points = points[::stride]
        colors = colors[::stride]

        return points, colors

    def _transform_points(
        self, points: np.ndarray, pose: np.ndarray, scale: float
    ) -> np.ndarray:
        R = pose[:3, :3]
        t = pose[:3, 3]
        points_world = (scale * (R @ points.T)).T + t
        return points_world

    def _get_intrinsic_matrix(self, intrinsics: dict) -> np.ndarray:
        if not intrinsics:
            return np.array([
                [800, 0, 640],
                [0, 800, 360],
                [0, 0, 1],
            ], dtype=np.float64)

        cam = list(intrinsics.values())[0]
        params = cam["params"]
        w, h = cam["width"], cam["height"]

        model = cam.get("model", cam.get("model_id", "PINHOLE"))
        if model in ("PINHOLE", 1):
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model in ("SIMPLE_PINHOLE", 0):
            f, cx, cy = params[0], params[1], params[2]
            fx = fy = f
        elif model in ("SIMPLE_RADIAL", 2):
            f, cx, cy = params[0], params[1], params[2]
            fx = fy = f
        else:
            fx = fy = params[0]
            cx, cy = w / 2, h / 2

        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float64)
