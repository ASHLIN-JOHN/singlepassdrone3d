"""Extract frames from drone video at configurable rates."""

import cv2
import numpy as np


class FrameExtractor:
    def __init__(self, config: dict):
        self.target_fps = config.get("target_fps", 2.0)
        self.max_frames = config.get("max_frames", 200)

    def extract(self, video_path: str) -> tuple[list[np.ndarray], list[float]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        frame_interval = max(1, int(video_fps / self.target_fps))

        frames = []
        timestamps = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frames.append(frame)
                timestamps.append(frame_idx / video_fps)

                if len(frames) >= self.max_frames:
                    break

            frame_idx += 1

        cap.release()
        return frames, timestamps

    def extract_with_metadata(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        cap.release()

        metadata["duration"] = metadata["total_frames"] / metadata["fps"]
        return metadata
