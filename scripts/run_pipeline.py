"""CLI entry point for the AeroTwin reconstruction pipeline."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="AeroTwin — Single-Pass UAV Video to Georeferenced 3D Model"
    )
    parser.add_argument("video", help="Path to drone video file")
    parser.add_argument("--gps", help="Path to GPS CSV (timestamp,lat,lon,alt)")
    parser.add_argument("--config", default="config/default.yaml", help="Pipeline config YAML")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument(
        "--camera-model", default="PINHOLE",
        help="Camera model: PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL"
    )
    parser.add_argument("--focal-length", type=float, help="Focal length in pixels")
    parser.add_argument("--no-depth", action="store_true", help="Skip AI depth estimation")
    parser.add_argument("--viewer", action="store_true", help="Launch web viewer after reconstruction")

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    camera_params = None
    if args.focal_length:
        camera_params = {
            "model": args.camera_model,
            "params": [args.focal_length],
        }

    from src.pipeline import ReconstructionPipeline

    pipeline = ReconstructionPipeline(config_path=args.config)
    result = pipeline.run(
        video_path=args.video,
        gps_data=args.gps,
        camera_params=camera_params,
    )

    print(f"\n{'='*50}")
    print("RECONSTRUCTION SUMMARY")
    print(f"{'='*50}")
    print(f"  Point cloud: {result.point_cloud_path}")
    print(f"  Points:      {result.metrics.get('num_points', 0):,}")
    print(f"  Keyframes:   {result.metrics.get('num_keyframes', 0)}")
    print(f"  Scale:       {result.metrics.get('estimated_scale', 1.0):.4f}")

    if result.georef_transform is not None:
        print(f"  Georeferenced: Yes")
    else:
        print(f"  Georeferenced: No (no GPS data)")

    if args.viewer and result.point_cloud_path:
        launch_viewer(result.point_cloud_path)


def launch_viewer(point_cloud_path: Path):
    import webbrowser
    from src.viewer_server import create_app

    app = create_app(str(point_cloud_path))
    print(f"\nLaunching viewer at http://localhost:8080")
    webbrowser.open("http://localhost:8080")
    app.run(port=8080)


if __name__ == "__main__":
    main()
