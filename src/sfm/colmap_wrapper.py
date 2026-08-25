"""COLMAP CLI wrapper for feature extraction, matching, and reconstruction."""

import subprocess
import struct
from pathlib import Path
from typing import Optional

import numpy as np


class ColmapRunner:
    def __init__(self, config: dict, workspace: Path):
        self.config = config
        self.workspace = workspace
        self.database_path = workspace / "database.db"
        self.sparse_dir = workspace / "sparse"
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        image_dir: Path,
        camera_params: Optional[dict] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        self._feature_extraction(image_dir, camera_params)
        self._feature_matching()
        self._sparse_reconstruction(image_dir)

        poses = self._read_poses()
        points = self._read_points()
        intrinsics = self._read_intrinsics()

        return poses, points, intrinsics

    def _feature_extraction(self, image_dir: Path, camera_params: Optional[dict]):
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(image_dir),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1",
        ]

        if camera_params:
            model = camera_params.get("model", "PINHOLE")
            cmd.extend(["--ImageReader.camera_model", model])
            if "params" in camera_params:
                params_str = ",".join(str(p) for p in camera_params["params"])
                cmd.extend(["--ImageReader.camera_params", params_str])

        self._run_cmd(cmd, "Feature extraction")

    def _feature_matching(self):
        matcher = self.config.get("matcher_type", "sequential")

        if matcher == "sequential":
            cmd = [
                "colmap", "sequential_matcher",
                "--database_path", str(self.database_path),
                "--SequentialMatching.overlap", str(self.config.get("sequential_overlap", 10)),
            ]
        else:
            cmd = [
                "colmap", "exhaustive_matcher",
                "--database_path", str(self.database_path),
            ]

        self._run_cmd(cmd, "Feature matching")

    def _sparse_reconstruction(self, image_dir: Path):
        cmd = [
            "colmap", "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(image_dir),
            "--output_path", str(self.sparse_dir),
            "--Mapper.min_num_matches", str(self.config.get("min_num_matches", 15)),
            "--Mapper.ba_refine_focal_length",
            str(int(self.config.get("ba_refine_focal_length", True))),
            "--Mapper.ba_refine_extra_params",
            str(int(self.config.get("ba_refine_extra_params", True))),
        ]
        self._run_cmd(cmd, "Sparse reconstruction")

    def _read_poses(self) -> np.ndarray:
        model_dir = self._find_best_model()
        images_path = model_dir / "images.bin"

        if not images_path.exists():
            images_path = model_dir / "images.txt"
            return self._read_poses_txt(images_path)

        return self._read_poses_bin(images_path)

    def _read_poses_bin(self, path: Path) -> np.ndarray:
        poses = []
        with open(path, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                image_id = struct.unpack("<I", f.read(4))[0]
                qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                tx, ty, tz = struct.unpack("<3d", f.read(24))
                camera_id = struct.unpack("<I", f.read(4))[0]

                name_len = 0
                while True:
                    ch = f.read(1)
                    if ch == b"\x00":
                        break
                    name_len += 1

                num_points = struct.unpack("<Q", f.read(8))[0]
                f.read(num_points * 24)  # skip point2D entries

                R = self._qvec_to_rotmat(np.array([qw, qx, qy, qz]))
                t = np.array([tx, ty, tz])
                # Camera center in world coords
                center = -R.T @ t

                pose = np.eye(4)
                pose[:3, :3] = R.T
                pose[:3, 3] = center
                poses.append(pose)

        return np.array(poses)

    def _read_poses_txt(self, path: Path) -> np.ndarray:
        poses = []
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])

                R = self._qvec_to_rotmat(np.array([qw, qx, qy, qz]))
                t = np.array([tx, ty, tz])
                center = -R.T @ t

                pose = np.eye(4)
                pose[:3, :3] = R.T
                pose[:3, 3] = center
                poses.append(pose)
                next(f, None)  # skip points2D line

        return np.array(poses)

    def _read_points(self) -> np.ndarray:
        model_dir = self._find_best_model()
        points_path = model_dir / "points3D.bin"

        if not points_path.exists():
            points_path = model_dir / "points3D.txt"
            return self._read_points_txt(points_path)

        return self._read_points_bin(points_path)

    def _read_points_bin(self, path: Path) -> np.ndarray:
        points = []
        with open(path, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_points):
                point_id = struct.unpack("<Q", f.read(8))[0]
                x, y, z = struct.unpack("<3d", f.read(24))
                r, g, b = struct.unpack("<3B", f.read(3))
                error = struct.unpack("<d", f.read(8))[0]
                track_len = struct.unpack("<Q", f.read(8))[0]
                f.read(track_len * 8)
                points.append([x, y, z])

        return np.array(points)

    def _read_points_txt(self, path: Path) -> np.ndarray:
        points = []
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 7:
                    x, y, z = map(float, parts[1:4])
                    points.append([x, y, z])
        return np.array(points) if points else np.zeros((0, 3))

    def _read_intrinsics(self) -> dict:
        model_dir = self._find_best_model()
        cameras_path = model_dir / "cameras.txt"

        if not cameras_path.exists():
            cameras_path = model_dir / "cameras.bin"
            return self._read_intrinsics_bin(cameras_path)

        intrinsics = {}
        with open(cameras_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    cam_id = int(parts[0])
                    model = parts[1]
                    width = int(parts[2])
                    height = int(parts[3])
                    params = list(map(float, parts[4:]))
                    intrinsics[cam_id] = {
                        "model": model,
                        "width": width,
                        "height": height,
                        "params": params,
                    }
        return intrinsics

    def _read_intrinsics_bin(self, path: Path) -> dict:
        if not path.exists():
            return {}
        intrinsics = {}
        with open(path, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                cam_id = struct.unpack("<I", f.read(4))[0]
                model_id = struct.unpack("<i", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]
                num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5}.get(model_id, 4)
                params = struct.unpack(f"<{num_params}d", f.read(num_params * 8))
                intrinsics[cam_id] = {
                    "model_id": model_id,
                    "width": width,
                    "height": height,
                    "params": list(params),
                }
        return intrinsics

    def _find_best_model(self) -> Path:
        candidates = sorted(self.sparse_dir.iterdir())
        if not candidates:
            raise RuntimeError("No COLMAP reconstruction found")
        return candidates[0]

    def _qvec_to_rotmat(self, qvec: np.ndarray) -> np.ndarray:
        w, x, y, z = qvec
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
        ])
        return R

    def _run_cmd(self, cmd: list[str], stage: str):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"COLMAP {stage} failed:\n{e.stderr}") from e
        except FileNotFoundError:
            raise RuntimeError(
                "COLMAP not found. Install from https://colmap.github.io/"
            )
