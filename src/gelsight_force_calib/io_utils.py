from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

IMAGE_NAME_RE = re.compile(
    r"^gelsight_"
    r"(?P<file_F>-?\d+(?:\.\d+)?)_"
    r"(?P<theta>-?\d+(?:\.\d+)?)_"
    r"(?P<alpha>-?\d+(?:\.\d+)?)_"
    r"(?P<date>\d{8})_"
    r"(?P<time>\d{6})_"
    r"(?P<ms>\d+)"
    r"(?:\.[^.]+)?$"
)


def parse_episode_id(name: str) -> Optional[int]:
    """把 1、2、003 这类片段名转成整数，非数字目录直接跳过。"""
    try:
        return int(str(name).strip())
    except ValueError:
        return None


def parse_image_name(path: Path) -> Optional[Dict[str, object]]:
    """从图片名解析力档位、方向角和绝对时间戳；解析失败时返回None并交给调用方记录。"""
    m = IMAGE_NAME_RE.match(path.name)
    if not m:
        return None
    ms_raw = m.group("ms")
    ms = int(ms_raw[:3].ljust(3, "0"))
    base_time = datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")
    return {
        "file_F": float(m.group("file_F")),
        "theta": float(m.group("theta")),
        "alpha": float(m.group("alpha")),
        "timestamp": base_time + timedelta(milliseconds=ms),
        "timestamp_str": (base_time + timedelta(milliseconds=ms)).isoformat(timespec="milliseconds"),
    }


def list_episode_dirs(image_root: Path, episode_count: int | None = None) -> List[Tuple[int, Path]]:
    """扫描new_output下的片段目录，按数字顺序返回，保证1.txt对应1号图片目录。"""
    episodes: List[Tuple[int, Path]] = []
    if not image_root.exists():
        raise FileNotFoundError(f"找不到图片根目录: {image_root}")
    for p in image_root.iterdir():
        if p.is_dir():
            eid = parse_episode_id(p.name)
            if eid is not None:
                episodes.append((eid, p))
    episodes.sort(key=lambda x: x[0])
    if episode_count:
        episodes = [(eid, p) for eid, p in episodes if 1 <= eid <= episode_count]
    return episodes


def list_images(episode_dir: Path, image_exts: Iterable[str]) -> List[Path]:
    """读取一个片段内所有图片，并优先按文件名中的绝对时间排序。"""
    exts = {e.lower() for e in image_exts}
    images = [p for p in episode_dir.iterdir() if p.suffix.lower() in exts]

    def sort_key(p: Path):
        parsed = parse_image_name(p)
        if parsed is not None:
            return parsed["timestamp"]
        return datetime.fromtimestamp(p.stat().st_mtime)

    return sorted(images, key=sort_key)


def read_force_txt(path: Path) -> pd.DataFrame:
    """读取拉力计txt，兼容空格、tab、逗号分隔以及带表头的情况，只保留时间和力值两列。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到力数据文件: {path}")
    rows = []
    float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            vals = [float(x) for x in float_re.findall(line)]
            if len(vals) >= 2:
                rows.append((vals[0], vals[1]))
    if not rows:
        raise ValueError(f"力数据文件没有解析出两列数值: {path}")
    df = pd.DataFrame(rows, columns=["force_time_sec", "force_raw"])
    df = df.dropna().sort_values("force_time_sec").drop_duplicates("force_time_sec")
    return df.reset_index(drop=True)


def interpolate_force(force_df: pd.DataFrame, query_times: np.ndarray) -> np.ndarray:
    """把图片对齐后的相对时间映射到拉力计曲线上，输出同一时刻的力值。"""
    return np.interp(
        query_times,
        force_df["force_time_sec"].to_numpy(dtype=np.float64),
        force_df["force_raw"].to_numpy(dtype=np.float64),
    )


def write_text_list(path: Path, values: Iterable[int]) -> None:
    """保存训练/验证片段编号，便于复现实验划分。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{int(v)}\n")
