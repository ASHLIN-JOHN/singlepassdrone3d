"""Monocular depth estimation using Depth Anything V2 or MiDAS."""

import numpy as np
import torch
import cv2


class DepthEstimator:
    def __init__(self, config: dict):
        self.model_name = config.get("model", "depth_anything_v2")
        self.encoder = config.get("encoder", "vitl")
        self.input_size = config.get("input_size", 518)
        self.batch_size = config.get("batch_size", 4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return

        if self.model_name == "depth_anything_v2":
            self.model = self._load_depth_anything_v2()
        else:
            self.model = self._load_midas()

        self.model.to(self.device)
        self.model.eval()

    def _load_depth_anything_v2(self):
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor

        model_id = f"depth-anything/Depth-Anything-V2-{self.encoder.capitalize()}-hf"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        return model

    def _load_midas(self):
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        return model

    def estimate_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        self._load_model()

        depth_maps = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i : i + self.batch_size]
            batch_depths = self._process_batch(batch)
            depth_maps.extend(batch_depths)

        return depth_maps

    def _process_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if self.model_name == "depth_anything_v2":
            return self._process_depth_anything(frames)
        return self._process_midas(frames)

    def _process_depth_anything(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        from PIL import Image

        results = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                depth = outputs.predicted_depth

            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(frame.shape[0], frame.shape[1]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            results.append(depth.cpu().numpy())

        return results

    def _process_midas(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        results = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(rgb).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_tensor)

            depth = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(frame.shape[0], frame.shape[1]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            results.append(depth.cpu().numpy())

        return results

    def estimate_single(self, frame: np.ndarray) -> np.ndarray:
        return self.estimate_batch([frame])[0]
