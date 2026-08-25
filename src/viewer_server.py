"""Flask server to serve point cloud data to the web viewer."""

from pathlib import Path

from flask import Flask, send_file, send_from_directory
from flask_cors import CORS


def create_app(point_cloud_path: str = None, viewer_dir: str = "viewer/dist") -> Flask:
    app = Flask(__name__, static_folder=viewer_dir)
    CORS(app)

    @app.route("/")
    def index():
        return send_from_directory(viewer_dir, "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        return send_from_directory(viewer_dir, path)

    @app.route("/api/pointcloud")
    def get_pointcloud():
        if point_cloud_path and Path(point_cloud_path).exists():
            return send_file(point_cloud_path, mimetype="application/octet-stream")
        return {"error": "No point cloud available"}, 404

    @app.route("/api/metadata")
    def get_metadata():
        if point_cloud_path and Path(point_cloud_path).exists():
            from plyfile import PlyData
            import numpy as np
            ply = PlyData.read(point_cloud_path)
            vertex = ply['vertex']
            points = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            extent = bbox_max - bbox_min
            return {
                "num_points": len(points),
                "bbox": {
                    "min": bbox_min.tolist(),
                    "max": bbox_max.tolist(),
                },
                "dimensions": {
                    "width": float(extent[0]),
                    "depth": float(extent[1]),
                    "height": float(extent[2]),
                },
            }
        return {"error": "No point cloud available"}, 404

    return app
