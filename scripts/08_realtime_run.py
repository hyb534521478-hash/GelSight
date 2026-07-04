#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gelsight_force_calib.config import load_config
from gelsight_force_calib.realtime import run_realtime


def main():
    parser = argparse.ArgumentParser(description="GelSight实时力方向预测软件，支持r键复位Img0")
    parser.add_argument("--checkpoint", required=True, help="训练输出的best.pt")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    camera = int(cfg["realtime"].get("camera_index", 0)) if args.camera is None else args.camera
    run_realtime(
        checkpoint_path=args.checkpoint,
        camera_index=camera,
        width=int(cfg["realtime"].get("width", 640)),
        height=int(cfg["realtime"].get("height", 480)),
        fps=int(cfg["realtime"].get("fps", 30)),
    )


if __name__ == "__main__":
    main()
