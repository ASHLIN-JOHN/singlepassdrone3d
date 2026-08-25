"""Point cloud construction, filtering, and export using plyfile."""

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


class PointCloudBuilder:
    def __init__(self, config: dict):
        self.voxel_size = config.get("voxel_size", 0.05)
        self.outlier_nb = config.get("statistical_outlier_nb", 20)
        self.outlier_std = config.get("statistical_outlier_std", 2.0)

    def build(self, points: np.ndarray, colors: np.ndarray = None) -> dict:
        pcd = {"points": points.astype(np.float32)}
        if colors is not None:
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)
            pcd["colors"] = colors
        return pcd

    def filter_outliers(self, pcd: dict) -> dict:
        points = pcd["points"]
        if len(points) < self.outlier_nb + 1:
            return pcd

        from scipy.spatial import KDTree
        tree = KDTree(points)
        dists, _ = tree.query(points, k=self.outlier_nb + 1)
        mean_dists = dists[:, 1:].mean(axis=1)

        threshold = mean_dists.mean() + self.outlier_std * mean_dists.std()
        mask = mean_dists < threshold

        result = {"points": points[mask]}
        if "colors" in pcd:
            result["colors"] = pcd["colors"][mask]
        if "confidence" in pcd:
            result["confidence"] = pcd["confidence"][mask]
        return result

    def downsample(self, pcd: dict) -> dict:
        points = pcd["points"]
        if self.voxel_size <= 0:
            return pcd

        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)

        result = {"points": points[unique_idx]}
        if "colors" in pcd:
            result["colors"] = pcd["colors"][unique_idx]
        if "confidence" in pcd:
            result["confidence"] = pcd["confidence"][unique_idx]
        return result

    def save(self, pcd: dict, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        points = pcd["points"]
        n = len(points)

        if "colors" in pcd:
            colors = pcd["colors"]
            vertex = np.zeros(n, dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ])
            vertex['x'] = points[:, 0]
            vertex['y'] = points[:, 1]
            vertex['z'] = points[:, 2]
            vertex['red'] = colors[:, 0]
            vertex['green'] = colors[:, 1]
            vertex['blue'] = colors[:, 2]
        else:
            vertex = np.zeros(n, dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ])
            vertex['x'] = points[:, 0]
            vertex['y'] = points[:, 1]
            vertex['z'] = points[:, 2]

        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(str(path))

    def save_with_confidence(self, pcd: dict, confidence: np.ndarray, path: Path):
        colors = self._confidence_to_colormap(confidence)
        conf_pcd = {
            "points": pcd["points"],
            "colors": colors,
        }
        self.save(conf_pcd, path)

    def _confidence_to_colormap(self, confidence: np.ndarray) -> np.ndarray:
        """Map confidence [0,1] to green-yellow-red colormap."""
        colors = np.zeros((len(confidence), 3), dtype=np.uint8)
        high = confidence > 0.66
        med = (confidence > 0.33) & (~high)
        low = ~high & ~med

        colors[high] = [51, 204, 51]    # green
        colors[med] = [230, 204, 26]    # yellow
        colors[low] = [230, 51, 26]     # red
        return colors

    def compute_measurements(self, pcd: dict) -> dict:
        points = pcd["points"]
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        extent = bbox_max - bbox_min

        return {
            "num_points": len(points),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "width": float(extent[0]),
            "depth": float(extent[1]),
            "height": float(extent[2]),
            "centroid": points.mean(axis=0).tolist(),
        }
