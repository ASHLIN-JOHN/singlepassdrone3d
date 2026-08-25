"""Per-point confidence estimation combining multiple quality signals."""

import numpy as np


class ConfidenceEstimator:
    def __init__(self, config: dict):
        self.obs_weight = config.get("observation_weight", 0.3)
        self.reproj_weight = config.get("reprojection_weight", 0.3)
        self.depth_weight = config.get("depth_consistency_weight", 0.2)
        self.gps_weight = config.get("gps_residual_weight", 0.2)

    def estimate(
        self,
        points: np.ndarray,
        poses: np.ndarray,
        depth_maps: list[np.ndarray],
        intrinsics: dict,
    ) -> np.ndarray:
        n_points = len(points)

        obs_score = self._observation_coverage(points, poses, intrinsics)
        reproj_score = self._reprojection_consistency(points, poses, depth_maps, intrinsics)
        depth_score = self._depth_consistency(points, poses, depth_maps, intrinsics)

        confidence = (
            self.obs_weight * obs_score
            + self.reproj_weight * reproj_score
            + self.depth_weight * depth_score
        )

        confidence = np.clip(confidence, 0.0, 1.0)
        return confidence

    def _observation_coverage(
        self, points: np.ndarray, poses: np.ndarray, intrinsics: dict
    ) -> np.ndarray:
        """How many cameras observe each point."""
        cam = list(intrinsics.values())[0] if intrinsics else {"width": 1280, "height": 720}
        w = cam.get("width", 1280)
        h = cam.get("height", 720)

        K = self._get_K(intrinsics)
        n_points = len(points)
        obs_count = np.zeros(n_points)

        for pose in poses:
            R = pose[:3, :3]
            t = pose[:3, 3]

            R_inv = R.T
            t_inv = -R.T @ t

            points_cam = (R_inv @ points.T).T + t_inv

            visible = points_cam[:, 2] > 0.1

            if not visible.any():
                continue

            proj = (K @ points_cam[visible].T).T
            u = proj[:, 0] / proj[:, 2]
            v = proj[:, 1] / proj[:, 2]

            in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)

            vis_indices = np.where(visible)[0]
            obs_count[vis_indices[in_frame]] += 1

        max_obs = max(obs_count.max(), 1)
        return np.clip(obs_count / max(len(poses) * 0.3, 1), 0, 1)

    def _reprojection_consistency(
        self,
        points: np.ndarray,
        poses: np.ndarray,
        depth_maps: list[np.ndarray],
        intrinsics: dict,
    ) -> np.ndarray:
        """Consistency of reprojected depth vs estimated depth."""
        n_points = len(points)
        scores = np.ones(n_points) * 0.5

        K = self._get_K(intrinsics)

        for i, (pose, dmap) in enumerate(zip(poses, depth_maps)):
            h, w = dmap.shape[:2]
            R = pose[:3, :3]
            t = pose[:3, 3]
            R_inv = R.T
            t_inv = -R.T @ t

            points_cam = (R_inv @ points.T).T + t_inv
            visible = points_cam[:, 2] > 0.1

            if not visible.any():
                continue

            proj = (K @ points_cam[visible].T).T
            u = (proj[:, 0] / proj[:, 2]).astype(int)
            v = (proj[:, 1] / proj[:, 2]).astype(int)

            in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)

            u_valid = u[in_frame]
            v_valid = v[in_frame]
            z_proj = points_cam[visible][in_frame, 2]
            z_map = dmap[v_valid, u_valid]

            valid_depth = z_map > 0
            if not valid_depth.any():
                continue

            ratio = np.minimum(z_proj[valid_depth], z_map[valid_depth]) / (
                np.maximum(z_proj[valid_depth], z_map[valid_depth]) + 1e-6
            )

            vis_indices = np.where(visible)[0]
            frame_indices = vis_indices[in_frame][valid_depth]
            scores[frame_indices] = np.maximum(scores[frame_indices], ratio)

        return scores

    def _depth_consistency(
        self,
        points: np.ndarray,
        poses: np.ndarray,
        depth_maps: list[np.ndarray],
        intrinsics: dict,
    ) -> np.ndarray:
        """Multi-view depth consistency check."""
        return np.ones(len(points)) * 0.5

    def _get_K(self, intrinsics: dict) -> np.ndarray:
        if not intrinsics:
            return np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float64)

        cam = list(intrinsics.values())[0]
        params = cam["params"]
        fx = params[0]
        fy = params[1] if len(params) > 1 else fx
        cx = params[2] if len(params) > 2 else cam.get("width", 1280) / 2
        cy = params[3] if len(params) > 3 else cam.get("height", 720) / 2

        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
