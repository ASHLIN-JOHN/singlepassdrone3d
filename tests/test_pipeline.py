"""Integration test for the full pipeline (uses synthetic data)."""

import pytest
from pathlib import Path


@pytest.fixture
def synthetic_data(tmp_path):
    """Generate minimal synthetic test data."""
    import cv2
    import csv
    import numpy as np

    video_path = tmp_path / "test.mp4"
    gps_path = tmp_path / "gps.csv"

    # Create a short test video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10, (640, 480))
    for i in range(30):
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        # Add some structure
        cv2.rectangle(frame, (200 + i, 150), (400 + i, 350), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    # Create GPS track
    with open(gps_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "lat", "lon", "alt"])
        w.writeheader()
        for i in range(30):
            w.writerow({
                "timestamp": i / 10.0,
                "lat": 13.0827 + i * 0.00001,
                "lon": 80.2707 + i * 0.00001,
                "alt": 50.0,
            })

    return {"video": str(video_path), "gps": str(gps_path)}


def test_frame_extraction(synthetic_data):
    from src.frame_extraction.extractor import FrameExtractor

    extractor = FrameExtractor({"target_fps": 2.0, "max_frames": 100})
    frames, timestamps = extractor.extract(synthetic_data["video"])
    assert len(frames) > 0
    assert len(frames) == len(timestamps)
    assert frames[0].shape == (480, 640, 3)


def test_quality_scoring(synthetic_data):
    from src.frame_extraction.extractor import FrameExtractor
    from src.frame_extraction.quality import QualityScorer

    extractor = FrameExtractor({"target_fps": 2.0, "max_frames": 100})
    frames, _ = extractor.extract(synthetic_data["video"])

    scorer = QualityScorer({"blur_threshold": 100.0, "min_feature_count": 500})
    scores = scorer.score_batch(frames)

    assert len(scores) == len(frames)
    assert all(0 <= s["composite"] <= 1 for s in scores)


def test_keyframe_selection(synthetic_data):
    from src.frame_extraction.extractor import FrameExtractor
    from src.frame_extraction.quality import QualityScorer
    from src.frame_extraction.selector import KeyframeSelector

    extractor = FrameExtractor({"target_fps": 5.0, "max_frames": 100})
    frames, timestamps = extractor.extract(synthetic_data["video"])

    scorer = QualityScorer({"blur_threshold": 50.0, "min_feature_count": 100})
    scores = scorer.score_batch(frames)

    selector = KeyframeSelector({
        "min_frames": 3,
        "max_frames": 10,
        "diversity_weight": 0.3,
        "gps_spacing_m": 2.0,
    })
    selected, indices = selector.select(frames, scores, timestamps)
    assert 3 <= len(selected) <= 10
