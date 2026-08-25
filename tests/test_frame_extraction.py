"""Tests for frame extraction and quality scoring."""

import numpy as np
import pytest

from src.frame_extraction.quality import QualityScorer


@pytest.fixture
def scorer():
    return QualityScorer({"blur_threshold": 100.0, "min_feature_count": 500})


def test_blur_score_sharp_image(scorer):
    sharp = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Add edges to make it "sharp"
    sharp[100:200, 100:200] = 255
    score = scorer.score_single(sharp)
    assert score["blur"] > 0


def test_blur_score_blank_image(scorer):
    blank = np.ones((480, 640, 3), dtype=np.uint8) * 128
    score = scorer.score_single(blank)
    assert score["blur"] < 1.0
    assert not score["is_sharp"]


def test_exposure_score_normal(scorer):
    normal = np.ones((480, 640, 3), dtype=np.uint8) * 127
    score = scorer.score_single(normal)
    assert score["exposure"] > 0.9


def test_exposure_score_dark(scorer):
    dark = np.ones((480, 640, 3), dtype=np.uint8) * 20
    score = scorer.score_single(dark)
    assert score["exposure"] < 0.5


def test_composite_score_range(scorer):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    score = scorer.score_single(frame)
    assert 0.0 <= score["composite"] <= 1.0


def test_batch_scoring(scorer):
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
    scores = scorer.score_batch(frames)
    assert len(scores) == 5
    assert all("composite" in s for s in scores)
