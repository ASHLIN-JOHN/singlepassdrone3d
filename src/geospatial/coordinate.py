"""Coordinate system transformations: WGS84, UTM, ENU."""

import math

import numpy as np
from pyproj import Proj, Transformer


class CoordinateTransformer:
    def __init__(self, epsg: int = 4326):
        self.source_epsg = epsg
        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt = None
        self._utm_zone = None
        self._utm_proj = None

    def set_origin(self, lat: float, lon: float, alt: float = 0.0):
        self.origin_lat = lat
        self.origin_lon = lon
        self.origin_alt = alt
        self._utm_zone = self._compute_utm_zone(lon)
        hemisphere = "north" if lat >= 0 else "south"
        epsg = 32600 + self._utm_zone if hemisphere == "north" else 32700 + self._utm_zone
        self._utm_proj = Proj(f"EPSG:{epsg}")
        self._wgs84_to_utm = Transformer.from_crs(
            f"EPSG:{self.source_epsg}", f"EPSG:{epsg}", always_xy=True
        )
        self._utm_to_wgs84 = Transformer.from_crs(
            f"EPSG:{epsg}", f"EPSG:{self.source_epsg}", always_xy=True
        )

    def geodetic_to_enu(self, lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
        if self.origin_lat is None:
            raise ValueError("Origin not set. Call set_origin() first.")

        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        lat0_r = math.radians(self.origin_lat)
        lon0_r = math.radians(self.origin_lon)

        a = 6378137.0  # WGS84 semi-major axis
        f = 1 / 298.257223563
        e2 = 2 * f - f * f

        sin_lat = math.sin(lat_r)
        cos_lat = math.cos(lat_r)
        N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)

        x = (N + alt) * cos_lat * math.cos(lon_r)
        y = (N + alt) * cos_lat * math.sin(lon_r)
        z = (N * (1 - e2) + alt) * sin_lat

        sin_lat0 = math.sin(lat0_r)
        cos_lat0 = math.cos(lat0_r)
        N0 = a / math.sqrt(1 - e2 * sin_lat0 * sin_lat0)

        x0 = (N0 + self.origin_alt) * cos_lat0 * math.cos(lon0_r)
        y0 = (N0 + self.origin_alt) * cos_lat0 * math.sin(lon0_r)
        z0 = (N0 * (1 - e2) + self.origin_alt) * sin_lat0

        dx = x - x0
        dy = y - y0
        dz = z - z0

        sin_lon0 = math.sin(lon0_r)
        cos_lon0 = math.cos(lon0_r)

        east = -sin_lon0 * dx + cos_lon0 * dy
        north = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
        up = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz

        return np.array([east, north, up])

    def enu_to_geodetic(self, east: float, north: float, up: float) -> tuple[float, float, float]:
        if self.origin_lat is None:
            raise ValueError("Origin not set. Call set_origin() first.")

        lat0_r = math.radians(self.origin_lat)
        lon0_r = math.radians(self.origin_lon)

        a = 6378137.0
        f = 1 / 298.257223563
        e2 = 2 * f - f * f

        sin_lat0 = math.sin(lat0_r)
        cos_lat0 = math.cos(lat0_r)
        sin_lon0 = math.sin(lon0_r)
        cos_lon0 = math.cos(lon0_r)

        N0 = a / math.sqrt(1 - e2 * sin_lat0 * sin_lat0)
        x0 = (N0 + self.origin_alt) * cos_lat0 * cos_lon0
        y0 = (N0 + self.origin_alt) * cos_lat0 * sin_lon0
        z0 = (N0 * (1 - e2) + self.origin_alt) * sin_lat0

        dx = -sin_lon0 * east - sin_lat0 * cos_lon0 * north + cos_lat0 * cos_lon0 * up
        dy = cos_lon0 * east - sin_lat0 * sin_lon0 * north + cos_lat0 * sin_lon0 * up
        dz = cos_lat0 * north + sin_lat0 * up

        x = x0 + dx
        y = y0 + dy
        z = z0 + dz

        lon = math.atan2(y, x)
        p = math.sqrt(x * x + y * y)
        lat = math.atan2(z, p * (1 - e2))

        for _ in range(5):
            N = a / math.sqrt(1 - e2 * math.sin(lat) * math.sin(lat))
            lat = math.atan2(z + e2 * N * math.sin(lat), p)

        N = a / math.sqrt(1 - e2 * math.sin(lat) * math.sin(lat))
        alt = p / math.cos(lat) - N

        return math.degrees(lat), math.degrees(lon), alt

    def geodetic_to_utm(self, lat: float, lon: float) -> tuple[float, float]:
        if self._wgs84_to_utm is None:
            self.set_origin(lat, lon, 0)
        easting, northing = self._wgs84_to_utm.transform(lon, lat)
        return easting, northing

    def utm_to_geodetic(self, easting: float, northing: float) -> tuple[float, float]:
        lon, lat = self._utm_to_wgs84.transform(easting, northing)
        return lat, lon

    def _compute_utm_zone(self, lon: float) -> int:
        return int((lon + 180) / 6) + 1
