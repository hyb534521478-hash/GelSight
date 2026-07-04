#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gelsight_force_calib.config import load_config
from gelsight_force_calib.evaluate import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description="复评某个模型checkpoint")
    parser.add_argument("--checkpoint", required=True, help="runs目录下的best.pt路径")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate_checkpoint(cfg, args.checkpoint, split=args.split)


if __name__ == "__main__":
    main()
