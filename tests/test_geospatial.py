"""Tests for coordinate transformations."""

import numpy as np
import pytest

from src.geospatial.coordinate import CoordinateTransformer


@pytest.fixture
def transformer():
    t = CoordinateTransformer()
    t.set_origin(13.0827, 80.2707, 0.0)  # Chennai
    return t


def test_origin_is_zero(transformer):
    enu = transformer.geodetic_to_enu(13.0827, 80.2707, 0.0)
    np.testing.assert_allclose(enu, [0, 0, 0], atol=0.01)


def test_north_positive(transformer):
    enu = transformer.geodetic_to_enu(13.0837, 80.2707, 0.0)
    assert enu[1] > 0  # north
    assert abs(enu[0]) < 1.0  # minimal east drift


def test_east_positive(transformer):
    enu = transformer.geodetic_to_enu(13.0827, 80.2717, 0.0)
    assert enu[0] > 0  # east
    assert abs(enu[1]) < 1.0  # minimal north drift


def test_altitude_positive(transformer):
    enu = transformer.geodetic_to_enu(13.0827, 80.2707, 100.0)
    assert enu[2] > 90  # up
    assert enu[2] < 110


def test_roundtrip(transformer):
    lat, lon, alt = 13.0850, 80.2750, 50.0
    enu = transformer.geodetic_to_enu(lat, lon, alt)
    lat2, lon2, alt2 = transformer.enu_to_geodetic(*enu)
    assert abs(lat2 - lat) < 1e-5
    assert abs(lon2 - lon) < 1e-5
    assert abs(alt2 - alt) < 1.0


def test_distance_approximate(transformer):
    # ~111m per 0.001 degree latitude
    enu = transformer.geodetic_to_enu(13.0837, 80.2707, 0.0)
    dist = np.linalg.norm(enu[:2])
    assert 100 < dist < 120


def test_utm_zone():
    t = CoordinateTransformer()
    assert t._compute_utm_zone(80.0) == 44
    assert t._compute_utm_zone(-73.0) == 18
    assert t._compute_utm_zone(0.0) == 31
