"""Main reconstruction pipeline orchestrator."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from src.frame_extraction.extractor import FrameExtractor
from src.frame_extraction.quality import QualityScorer
from src.frame_extraction.selector import KeyframeSelector
from src.sfm.colmap_wrapper import ColmapRunner
from src.depth.estimator import DepthEstimator
from src.geospatial.gps_alignment import GPSAligner
from src.geospatial.metric_scale import ScaleEstimator
from src.reconstruction.point_cloud import PointCloudBuilder
from src.reconstruction.fusion import DepthFusion
from src.confidence.estimator import ConfidenceEstimator


@dataclass
class PipelineResult:
    point_cloud_path: Optional[Path] = None
    mesh_path: Optional[Path] = None
    camera_poses: Optional[np.ndarray] = None
    georef_transform: Optional[np.ndarray] = None
    metrics: dict = field(default_factory=dict)
    confidence_map: Optional[np.ndarray] = None


class ReconstructionPipeline:
    def __init__(self, config_path: str = "config/default.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path(self.config["pipeline"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.workspace = Path(self.config["pipeline"]["workspace_dir"])
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        video_path: str,
        gps_data: Optional[str] = None,
        camera_params: Optional[dict] = None,
    ) -> PipelineResult:
        result = PipelineResult()

        # Stage 1: Frame extraction
        print("[1/7] Extracting frames...")
        extractor = FrameExtractor(self.config["frame_extraction"])
        frames, timestamps = extractor.extract(video_path)
        print(f"       Extracted {len(frames)} raw frames")

        # Stage 2: Quality scoring and keyframe selection
        print("[2/7] Scoring frame quality...")
        scorer = QualityScorer(self.config["frame_extraction"])
        scores = scorer.score_batch(frames)

        print("[3/7] Selecting keyframes...")
        selector = KeyframeSelector(self.config["frame_extraction"])
        keyframes, keyframe_indices = selector.select(frames, scores, timestamps, gps_data)
        print(f"       Selected {len(keyframes)} keyframes")

        self._save_keyframes(keyframes, keyframe_indices)

        # Stage 3: Structure from Motion
        print("[4/7] Running SfM (COLMAP)...")
        colmap = ColmapRunner(self.config["sfm"], self.workspace)
        poses, sparse_points, intrinsics = colmap.run(
            self.workspace / "frames",
            camera_params=camera_params,
        )
        result.camera_poses = poses
        print(f"       Recovered {len(poses)} camera poses, {len(sparse_points)} sparse points")

        # Stage 4: Depth estimation
        print("[5/7] Estimating depth...")
        depth_estimator = DepthEstimator(self.config["depth"])
        depth_maps = depth_estimator.estimate_batch(keyframes)

        # Stage 5: Geospatial alignment
        print("[6/7] Aligning to GPS / metric scale...")
        aligner = GPSAligner(self.config["geospatial"])
        scale_estimator = ScaleEstimator(self.config["geospatial"])

        if gps_data:
            georef_poses, transform = aligner.align(poses, gps_data, keyframe_indices, timestamps)
            result.georef_transform = transform
        else:
            georef_poses = poses
            print("       WARNING: No GPS data — model will not be georeferenced")

        scale = scale_estimator.estimate(georef_poses, gps_data, keyframe_indices, timestamps)
        result.metrics["estimated_scale"] = scale

        # Stage 6: Dense reconstruction
        print("[7/7] Building point cloud...")
        fusion = DepthFusion(self.config["reconstruction"])
        dense_points, dense_colors = fusion.fuse(
            keyframes, depth_maps, georef_poses, intrinsics, scale
        )

        # Build and save point cloud
        builder = PointCloudBuilder(self.config["reconstruction"])
        pcd = builder.build(dense_points, dense_colors)
        pcd = builder.filter_outliers(pcd)
        pcd = builder.downsample(pcd)

        pc_path = self.output_dir / "reconstruction.ply"
        builder.save(pcd, pc_path)
        result.point_cloud_path = pc_path

        # Confidence estimation
        confidence_est = ConfidenceEstimator(self.config["confidence"])
        result.confidence_map = confidence_est.estimate(
            pcd["points"], georef_poses, depth_maps, intrinsics
        )

        confidence_path = self.output_dir / "confidence.ply"
        builder.save_with_confidence(pcd, result.confidence_map, confidence_path)

        # Metrics
        result.metrics.update({
            "num_keyframes": len(keyframes),
            "num_poses": len(poses),
            "num_points": len(pcd["points"]),
            "sparse_points": len(sparse_points),
        })

        print(f"\nReconstruction complete.")
        print(f"  Points: {len(pcd['points']):,}")
        print(f"  Output: {pc_path}")
        print(f"  Scale:  {scale:.4f}")

        return result

    def _save_keyframes(self, keyframes, indices):
        frames_dir = self.workspace / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        import cv2
        for i, (frame, idx) in enumerate(zip(keyframes, indices)):
            path = frames_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(path), frame)
