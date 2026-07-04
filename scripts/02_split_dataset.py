#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gelsight_force_calib.config import ensure_output_dirs, load_config
from gelsight_force_calib.split import split_by_episode


def main():
    parser = argparse.ArgumentParser(description="按片段8:2划分训练集和验证集")
    parser.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    df, train_eps, val_eps = split_by_episode(cfg)
    print(f"划分完成: {cfg['split_csv']}  总样本={len(df)}")
    print(f"训练片段数={len(train_eps)} 验证片段数={len(val_eps)}")
    print(f"训练片段文件: {cfg['train_episodes']}")
    print(f"验证片段文件: {cfg['val_episodes']}")


if __name__ == "__main__":
    main()
