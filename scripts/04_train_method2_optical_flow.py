#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gelsight_force_calib.config import ensure_output_dirs, load_config
from gelsight_force_calib.train import train_flow_method


def main():
    parser = argparse.ArgumentParser(description="方法2：光流点位偏移 -> [F, theta, alpha]")
    parser.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    run_dir = train_flow_method(cfg, run_name="method2_optical_flow")
    print(f"方法2训练完成: {run_dir}")


if __name__ == "__main__":
    main()
