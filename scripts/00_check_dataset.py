#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gelsight_force_calib.config import load_config
from gelsight_force_calib.io_utils import list_episode_dirs, list_images, read_force_txt


def main():
    cfg = load_config(ROOT / "config/default.yaml")
    episodes = list_episode_dirs(cfg["image_root"], cfg.get("episode_count"))
    print(f"图片根目录: {cfg['image_root']}")
    print(f"力数据目录: {cfg['force_root']}")
    print(f"发现片段数: {len(episodes)}")
    for eid, ep_dir in episodes[:5]:
        images = list_images(ep_dir, cfg.get("image_exts", [".png"]))
        force_path = cfg["force_root"] / f"{eid}.txt"
        print(f"episode={eid} images={len(images)} force_file={force_path.exists()}")
        if force_path.exists():
            df = read_force_txt(force_path)
            print(f"  force rows={len(df)} time=[{df.force_time_sec.min():.3f}, {df.force_time_sec.max():.3f}]")


if __name__ == "__main__":
    main()
