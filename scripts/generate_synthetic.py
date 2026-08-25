"""Generate synthetic drone data for testing (no Blender required).

Creates a simple scene with known ground-truth geometry and a simulated
circular drone trajectory with GPS coordinates.
"""

import csv
import math
from pathlib import Path

import cv2
import numpy as np


def generate_synthetic_scene(output_dir: str = "data/synthetic"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Scene: a textured box (building) on a ground plane
    width, height = 1280, 720
    n_frames = 60
    fps = 2.0

    # Ground truth building dimensions (meters)
    building = {
        "width": 20.0,
        "depth": 15.0,
        "height": 12.0,
        "center": np.array([0.0, 0.0, 6.0]),
    }

    # Drone trajectory: circular path around building
    radius = 40.0
    altitude = 30.0
    origin_lat, origin_lon = 13.0827, 80.2707  # Chennai

    # Generate frames and GPS
    video_path = out / "synthetic_flight.mp4"
    gps_path = out / "gps_track.csv"
    gt_path = out / "ground_truth.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    gps_track = []

    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        # Camera position on circular path
        cam_x = radius * math.cos(angle)
        cam_y = radius * math.sin(angle)
        cam_z = altitude

        # Render simple synthetic frame
        frame = _render_frame(width, height, cam_x, cam_y, cam_z, building)
        writer.write(frame)

        # GPS (approximate offset from origin)
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(origin_lat))
        lat = origin_lat + cam_y / meters_per_deg_lat
        lon = origin_lon + cam_x / meters_per_deg_lon
        alt = cam_z

        gps_track.append({
            "timestamp": i / fps,
            "lat": lat,
            "lon": lon,
            "alt": alt,
        })

    writer.release()

    # Save GPS CSV
    with open(gps_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "lat", "lon", "alt"])
        w.writeheader()
        w.writerows(gps_track)

    # Save ground truth
    import json
    with open(gt_path, "w") as f:
        json.dump({
            "building": building | {"center": building["center"].tolist()},
            "trajectory_radius_m": radius,
            "trajectory_altitude_m": altitude,
            "origin": {"lat": origin_lat, "lon": origin_lon},
        }, f, indent=2)

    print(f"Synthetic data generated:")
    print(f"  Video:  {video_path}")
    print(f"  GPS:    {gps_path}")
    print(f"  GT:     {gt_path}")
    print(f"  Frames: {n_frames}")


def _render_frame(
    width: int, height: int,
    cam_x: float, cam_y: float, cam_z: float,
    building: dict,
) -> np.ndarray:
    """Render a simple wireframe/colored view of the building from camera position."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient
    for row in range(height // 2):
        intensity = int(180 + 75 * row / (height // 2))
        frame[row] = [intensity, 140, 80]  # sky blue (BGR)

    # Ground
    frame[height // 2 :] = [60, 120, 60]  # green ground

    # Simple perspective projection of building corners
    bw, bd, bh = building["width"], building["depth"], building["height"]
    bc = building["center"]

    corners = np.array([
        [bc[0] - bw/2, bc[1] - bd/2, 0],
        [bc[0] + bw/2, bc[1] - bd/2, 0],
        [bc[0] + bw/2, bc[1] + bd/2, 0],
        [bc[0] - bw/2, bc[1] + bd/2, 0],
        [bc[0] - bw/2, bc[1] - bd/2, bh],
        [bc[0] + bw/2, bc[1] - bd/2, bh],
        [bc[0] + bw/2, bc[1] + bd/2, bh],
        [bc[0] - bw/2, bc[1] + bd/2, bh],
    ])

    cam_pos = np.array([cam_x, cam_y, cam_z])
    look_at = bc.copy()
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 0, 1]))
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, forward)

    R = np.stack([right, -up, forward], axis=0)

    focal = 800
    projected = []
    for corner in corners:
        p = R @ (corner - cam_pos)
        if p[2] > 0.1:
            u = int(focal * p[0] / p[2] + width / 2)
            v = int(focal * p[1] / p[2] + height / 2)
            projected.append((u, v))
        else:
            projected.append(None)

    # Draw edges
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # bottom
        (4,5), (5,6), (6,7), (7,4),  # top
        (0,4), (1,5), (2,6), (3,7),  # vertical
    ]

    for a, b in edges:
        if projected[a] and projected[b]:
            cv2.line(frame, projected[a], projected[b], (200, 200, 200), 2)

    # Fill visible faces with color
    faces = [
        ([0,1,5,4], (180, 160, 140)),  # front
        ([1,2,6,5], (160, 140, 120)),  # right
        ([2,3,7,6], (140, 120, 100)),  # back
        ([3,0,4,7], (150, 130, 110)),  # left
        ([4,5,6,7], (100, 100, 120)),  # roof
    ]

    for face_idx, color in faces:
        pts = [projected[i] for i in face_idx]
        if all(p is not None for p in pts):
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(frame, [pts_arr], color)

    # Add texture noise
    noise = np.random.randint(0, 15, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)

    return frame


if __name__ == "__main__":
    generate_synthetic_scene()
