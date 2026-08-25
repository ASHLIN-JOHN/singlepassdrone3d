"""Frame quality scoring — blur, exposure, feature richness."""

import cv2
import numpy as np


class QualityScorer:
    def __init__(self, config: dict):
        self.blur_threshold = config.get("blur_threshold", 100.0)
        self.min_feature_count = config.get("min_feature_count", 500)

    def score_batch(self, frames: list[np.ndarray]) -> list[dict]:
        return [self.score_single(frame) for frame in frames]

    def score_single(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = self._blur_score(gray)
        exposure = self._exposure_score(gray)
        features = self._feature_score(gray)

        composite = (
            0.4 * min(blur / self.blur_threshold, 1.0)
            + 0.3 * exposure
            + 0.3 * min(features / self.min_feature_count, 1.0)
        )

        return {
            "blur": blur,
            "exposure": exposure,
            "features": features,
            "composite": composite,
            "is_sharp": blur >= self.blur_threshold,
        }

    def _blur_score(self, gray: np.ndarray) -> float:
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _exposure_score(self, gray: np.ndarray) -> float:
        mean = gray.mean()
        optimal = 127.0
        deviation = abs(mean - optimal) / optimal
        return max(0.0, 1.0 - deviation)

    def _feature_score(self, gray: np.ndarray) -> int:
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints = orb.detect(gray, None)
        return len(keypoints)
