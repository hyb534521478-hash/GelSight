#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gelsight_force_calib.config import ensure_output_dirs, load_config
from gelsight_force_calib.train import train_image_method


def main():
    parser = argparse.ArgumentParser(description="方法4：[Img_0, 时间窗口] -> [F, theta, alpha]")
    parser.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    parser.add_argument("--window", type=int, default=3, choices=[3, 5, 7], help="时间窗口长度，建议分别跑3/5/7")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    run_dir = train_image_method(cfg, mode="method4_ref_temporal", run_name=f"method4_ref_temporal_w{args.window}", window_size=args.window)
    print(f"方法4训练完成: {run_dir}")


if __name__ == "__main__":
    main()
