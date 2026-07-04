from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .io_utils import write_text_list


def split_by_episode(cfg: Dict) -> Tuple[pd.DataFrame, list[int], list[int]]:
    """按片段做8:2划分，避免同一个按压过程同时出现在训练集和验证集造成数据泄漏。"""
    aligned_csv: Path = cfg["aligned_csv"]
    if not aligned_csv.exists():
        raise FileNotFoundError(f"缺少对齐表，请先运行预处理: {aligned_csv}")

    df = pd.read_csv(aligned_csv)
    episodes = sorted(df["episode"].dropna().astype(int).unique().tolist())
    rng = np.random.default_rng(int(cfg["split"].get("seed", 42)))
    rng.shuffle(episodes)

    train_count = max(1, int(round(len(episodes) * float(cfg["split"].get("train_ratio", 0.8)))))
    train_eps = sorted(episodes[:train_count])
    val_eps = sorted(episodes[train_count:])
    if not val_eps and len(train_eps) > 1:
        val_eps = [train_eps.pop()]

    split_map = {e: "train" for e in train_eps}
    split_map.update({e: "val" for e in val_eps})
    df["split"] = df["episode"].astype(int).map(split_map)
    df = df[df["split"].notna()].copy()

    cfg["split_csv"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["split_csv"], index=False, encoding="utf-8-sig")
    write_text_list(cfg["train_episodes"], train_eps)
    write_text_list(cfg["val_episodes"], val_eps)
    return df, train_eps, val_eps
