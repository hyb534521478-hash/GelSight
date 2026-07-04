#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gelsight_force_calib.config import ensure_output_dirs, load_config
from gelsight_force_calib.train import train_image_method


def main():
    parser = argparse.ArgumentParser(description="方法1：[Img_0, Img_n] -> [F, theta, alpha]")
    parser.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    run_dir = train_image_method(cfg, mode="method1_pair", run_name="method1_img0_imgn", window_size=1)
    print(f"方法1训练完成: {run_dir}")


if __name__ == "__main__":
    main()
