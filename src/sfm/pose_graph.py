"""Camera pose graph management and interpolation."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PoseGraph:
    def __init__(self, poses: np.ndarray, timestamps: list[float] = None):
        self.poses = poses
        self.timestamps = timestamps or list(range(len(poses)))
        self._positions = poses[:, :3, 3]

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def rotations(self) -> np.ndarray:
        return self.poses[:, :3, :3]

    def trajectory_length(self) -> float:
        diffs = np.diff(self._positions, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def interpolate(self, t: float) -> np.ndarray:
        ts = np.array(self.timestamps)
        if t <= ts[0]:
            return self.poses[0]
        if t >= ts[-1]:
            return self.poses[-1]

        idx = np.searchsorted(ts, t) - 1
        alpha = (t - ts[idx]) / (ts[idx + 1] - ts[idx])

        pos = (1 - alpha) * self._positions[idx] + alpha * self._positions[idx + 1]

        r0 = Rotation.from_matrix(self.poses[idx, :3, :3])
        r1 = Rotation.from_matrix(self.poses[idx + 1, :3, :3])
        slerp = Slerp([0, 1], Rotation.concatenate([r0, r1]))
        rot = slerp(alpha).as_matrix()

        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos
        return pose

    def apply_transform(self, transform: np.ndarray) -> "PoseGraph":
        transformed = np.array([transform @ p for p in self.poses])
        return PoseGraph(transformed, self.timestamps)

    def apply_scale(self, scale: float) -> "PoseGraph":
        scaled = self.poses.copy()
        scaled[:, :3, 3] *= scale
        return PoseGraph(scaled, self.timestamps)

    def compute_baselines(self) -> np.ndarray:
        diffs = np.diff(self._positions, axis=0)
        return np.linalg.norm(diffs, axis=1)
