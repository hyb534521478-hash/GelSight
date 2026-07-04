#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gelsight_force_calib.config import ensure_output_dirs, load_config
from gelsight_force_calib.preprocess import build_alignment_table, save_alignment


def main():
    parser = argparse.ArgumentParser(description="GelSight图像-拉力计时间对齐预处理")
    parser.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    parser.add_argument("--no-debug-plots", action="store_true", help="不保存接触检测曲线图")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    aligned, unmatched = build_alignment_table(cfg, debug_plots=not args.no_debug_plots)
    save_alignment(cfg, aligned, unmatched)
    print(f"对齐完成: {cfg['aligned_csv']}  样本数={len(aligned)}")
    print(f"异常/未匹配样本: {cfg['unmatched_csv']}  数量={len(unmatched)}")


if __name__ == "__main__":
    main()
