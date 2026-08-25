"""Parse drone video metadata — GPS, IMU, camera info from video containers."""

import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


class MetadataParser:
    """Extract embedded metadata from drone video files.

    Supports DJI SRT subtitles and generic EXIF/FFprobe metadata.
    """

    def parse(self, video_path: str) -> dict:
        metadata = {}

        metadata["video"] = self._parse_video_info(video_path)
        metadata["camera"] = self._parse_camera_params(video_path)

        srt_path = Path(video_path).with_suffix(".SRT")
        if srt_path.exists():
            metadata["gps_track"] = self._parse_dji_srt(str(srt_path))

        return metadata

    def _parse_video_info(self, video_path: str) -> dict:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format", "-show_streams",
                    video_path,
                ],
                capture_output=True,
                text=True,
            )
            probe = json.loads(result.stdout)
            video_stream = next(
                (s for s in probe.get("streams", []) if s["codec_type"] == "video"),
                {},
            )
            return {
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": eval(video_stream.get("r_frame_rate", "30/1")),
                "duration": float(probe.get("format", {}).get("duration", 0)),
                "codec": video_stream.get("codec_name", ""),
            }
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _parse_camera_params(self, video_path: str) -> Optional[dict]:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_entries", "format_tags",
                    video_path,
                ],
                capture_output=True,
                text=True,
            )
            probe = json.loads(result.stdout)
            tags = probe.get("format", {}).get("tags", {})

            fov = None
            for key in tags:
                if "fov" in key.lower() or "focal" in key.lower():
                    try:
                        fov = float(tags[key])
                    except ValueError:
                        pass

            if fov:
                return {"fov_degrees": fov}
            return None
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _parse_dji_srt(self, srt_path: str) -> list[dict]:
        """Parse DJI SRT subtitle files containing GPS telemetry."""
        gps_track = []

        with open(srt_path, encoding="utf-8") as f:
            content = f.read()

        import re
        gps_pattern = re.compile(
            r"\[latitude:\s*([-\d.]+)\]\s*\[longitude:\s*([-\d.]+)\]\s*"
            r"\[altitude:\s*([-\d.]+)\]",
            re.IGNORECASE,
        )
        time_pattern = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")

        blocks = content.split("\n\n")
        for block in blocks:
            time_match = time_pattern.search(block)
            gps_match = gps_pattern.search(block)

            if time_match and gps_match:
                h, m, s, ms = time_match.groups()
                timestamp = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

                lat = float(gps_match.group(1))
                lon = float(gps_match.group(2))
                alt = float(gps_match.group(3))

                gps_track.append({
                    "timestamp": timestamp,
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                })

        return gps_track

    def export_gps_csv(self, gps_track: list[dict], output_path: str):
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "lat", "lon", "alt"])
            writer.writeheader()
            writer.writerows(gps_track)
