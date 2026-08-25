"""Demo script that runs the pipeline stages available without COLMAP.

Demonstrates: frame extraction, quality scoring, keyframe selection,
simulated poses from GPS, depth estimation (mock), point cloud generation.
"""

import sys
import os
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.frame_extraction.extractor import FrameExtractor
from src.frame_extraction.quality import QualityScorer
from src.frame_extraction.selector import KeyframeSelector
from src.geospatial.coordinate import CoordinateTransformer
from src.geospatial.metric_scale import ScaleEstimator
from src.reconstruction.point_cloud import PointCloudBuilder
from src.reconstruction.fusion import DepthFusion
from src.confidence.estimator import ConfidenceEstimator


def main():
    config_path = "config/default.yaml"
    video_path = "data/synthetic/synthetic_flight.mp4"
    gps_path = "data/synthetic/gps_track.csv"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  AEROTWIN — Single-Pass UAV Reconstruction Demo")
    print("=" * 60)
    print()

    # Stage 1: Frame extraction
    t0 = time.time()
    print("[1/6] Extracting frames from video...")
    extractor = FrameExtractor(config["frame_extraction"])
    frames, timestamps = extractor.extract(video_path)
    print(f"       ✓ Extracted {len(frames)} frames ({time.time()-t0:.1f}s)")

    # Video metadata
    meta = extractor.extract_with_metadata(video_path)
    print(f"       Resolution: {meta['width']}x{meta['height']}")
    print(f"       Duration: {meta['duration']:.1f}s")
    print(f"       FPS: {meta['fps']:.1f}")
    print()

    # Stage 2: Quality scoring
    t0 = time.time()
    print("[2/6] Scoring frame quality...")
    scorer = QualityScorer(config["frame_extraction"])
    scores = scorer.score_batch(frames)

    sharp_count = sum(1 for s in scores if s["is_sharp"])
    avg_score = np.mean([s["composite"] for s in scores])
    print(f"       ✓ Sharp frames: {sharp_count}/{len(frames)}")
    print(f"       ✓ Average quality: {avg_score:.3f}")
    print(f"       ✓ Scored in {time.time()-t0:.1f}s")
    print()

    # Stage 3: Keyframe selection
    t0 = time.time()
    print("[3/6] Selecting optimal keyframes...")
    selector = KeyframeSelector(config["frame_extraction"])
    keyframes, keyframe_indices = selector.select(frames, scores, timestamps, gps_path)
    print(f"       ✓ Selected {len(keyframes)} keyframes from {len(frames)} candidates")
    print(f"       ✓ Selection in {time.time()-t0:.1f}s")
    print()

    # Stage 4: Simulate camera poses from GPS (since we don't have COLMAP)
    t0 = time.time()
    print("[4/6] Computing camera poses from GPS trajectory...")
    poses, intrinsics = simulate_poses_from_gps(gps_path, timestamps, keyframe_indices, meta)
    print(f"       ✓ {len(poses)} camera poses computed")

    # Compute trajectory stats
    positions = poses[:, :3, 3]
    diffs = np.diff(positions, axis=0)
    trajectory_length = np.sum(np.linalg.norm(diffs, axis=1))
    print(f"       ✓ Trajectory length: {trajectory_length:.1f} m")
    print(f"       ✓ Poses in {time.time()-t0:.1f}s")
    print()

    # Stage 5: Depth estimation (simplified — use disparity from structure)
    t0 = time.time()
    print("[5/6] Estimating depth maps...")
    depth_maps = estimate_depth_simple(keyframes)
    print(f"       ✓ {len(depth_maps)} depth maps estimated")
    print(f"       ✓ Depth in {time.time()-t0:.1f}s")
    print()

    # Stage 6: Point cloud generation
    t0 = time.time()
    print("[6/6] Building 3D point cloud...")
    fusion = DepthFusion(config["reconstruction"])
    points, colors = fusion.fuse(keyframes, depth_maps, poses, intrinsics, scale=1.0)
    print(f"       ✓ Fused {len(points):,} raw points")

    builder = PointCloudBuilder(config["reconstruction"])
    pcd = builder.build(points, colors)
    pcd = builder.filter_outliers(pcd)
    pcd = builder.downsample(pcd)
    print(f"       ✓ After filtering: {len(pcd['points']):,} points")

    # Save point cloud
    pc_path = output_dir / "reconstruction.ply"
    builder.save(pcd, pc_path)
    print(f"       ✓ Saved: {pc_path}")

    # Confidence estimation
    confidence_est = ConfidenceEstimator(config["confidence"])
    confidence = confidence_est.estimate(pcd["points"], poses, depth_maps, intrinsics)

    conf_path = output_dir / "confidence.ply"
    builder.save_with_confidence(pcd, confidence, conf_path)
    print(f"       ✓ Confidence map: {conf_path}")

    # Measurements
    measurements = builder.compute_measurements(pcd)
    print(f"       ✓ Built in {time.time()-t0:.1f}s")
    print()

    # Summary
    print("=" * 60)
    print("  RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"  Points:       {measurements['num_points']:,}")
    print(f"  Width (X):    {measurements['width']:.2f} m")
    print(f"  Depth (Y):    {measurements['depth']:.2f} m")
    print(f"  Height (Z):   {measurements['height']:.2f} m")
    print(f"  Centroid:     [{measurements['centroid'][0]:.1f}, {measurements['centroid'][1]:.1f}, {measurements['centroid'][2]:.1f}]")
    print(f"  Confidence:   {confidence.mean():.1%} avg")
    print(f"  Output:       {pc_path.absolute()}")
    print(f"  Confidence:   {conf_path.absolute()}")
    print()

    high = (confidence > 0.66).sum()
    med = ((confidence > 0.33) & (confidence <= 0.66)).sum()
    low = (confidence <= 0.33).sum()
    total = len(confidence)
    print(f"  Confidence Distribution:")
    print(f"    High (>66%):    {high:,} points ({high/total:.0%})")
    print(f"    Medium (33-66%): {med:,} points ({med/total:.0%})")
    print(f"    Low (<33%):     {low:,} points ({low/total:.0%})")
    print()
    print("  To view: open the PLY file in the web viewer (cd viewer && npm run dev)")
    print("=" * 60)


def simulate_poses_from_gps(gps_path, timestamps, keyframe_indices, video_meta):
    """Create camera poses directly from GPS when COLMAP is unavailable."""
    import csv

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

    transformer = CoordinateTransformer()
    transformer.set_origin(gps_points[0]["lat"], gps_points[0]["lon"], gps_points[0]["alt"])

    gps_ts = np.array([p["timestamp"] for p in gps_points])
    gps_enu = np.array([
        transformer.geodetic_to_enu(p["lat"], p["lon"], p["alt"])
        for p in gps_points
    ])

    poses = []
    for idx in keyframe_indices:
        t = timestamps[idx]
        # Interpolate GPS position
        i = max(0, np.searchsorted(gps_ts, t) - 1)
        i = min(i, len(gps_ts) - 2)
        alpha = np.clip((t - gps_ts[i]) / max(gps_ts[i+1] - gps_ts[i], 1e-9), 0, 1)
        pos = (1 - alpha) * gps_enu[i] + alpha * gps_enu[i+1]

        # Camera looks toward origin (building center)
        forward = -pos / (np.linalg.norm(pos) + 1e-9)
        up = np.array([0, 0, 1.0])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, forward)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = -up
        pose[:3, 2] = forward
        pose[:3, 3] = pos
        poses.append(pose)

    # Intrinsics from video metadata
    w = video_meta.get("width", 1280)
    h = video_meta.get("height", 720)
    focal = 0.8 * max(w, h)
    intrinsics = {
        1: {
            "model": "PINHOLE",
            "width": w,
            "height": h,
            "params": [focal, focal, w/2, h/2],
        }
    }

    return np.array(poses), intrinsics


def estimate_depth_simple(frames):
    """Simple gradient-based depth proxy when neural depth models aren't available."""
    depth_maps = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Use image gradient magnitude as a rough depth proxy
        # (edges/texture → closer, smooth → farther)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(gx**2 + gy**2)

        # Invert and normalize: high gradient → close, low → far
        depth = 1.0 / (gradient / gradient.max() + 0.02)
        # Scale to reasonable range (5-50m)
        depth = 5.0 + 45.0 * (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
        depth_maps.append(depth)

    return depth_maps


if __name__ == "__main__":
    main()
