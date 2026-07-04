from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def expand_path(value: str | Path) -> Path:
    """把配置里的 ~ 和环境变量展开，保证Ubuntu桌面中文路径可直接使用。"""
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def load_config(config_path: str | Path = "config/default.yaml") -> Dict[str, Any]:
    """读取工程配置，统一处理关键路径，避免脚本里到处写死目录。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    for key in ["project_root", "force_root", "image_root", "output_root"]:
        cfg[key] = expand_path(cfg[key])

    cfg["aligned_csv"] = cfg["output_root"] / "aligned" / "alignment_table.csv"
    cfg["unmatched_csv"] = cfg["output_root"] / "aligned" / "unmatched_table.csv"
    cfg["debug_dir"] = cfg["output_root"] / "debug_alignment"
    cfg["split_csv"] = cfg["output_root"] / "splits" / "dataset_split.csv"
    cfg["train_episodes"] = cfg["output_root"] / "splits" / "train_episodes.txt"
    cfg["val_episodes"] = cfg["output_root"] / "splits" / "val_episodes.txt"
    cfg["runs_dir"] = cfg["output_root"] / "runs"
    return cfg


def ensure_output_dirs(cfg: Dict[str, Any]) -> None:
    """提前创建输出目录，训练、预处理和可视化结果都归档到工程outputs下。"""
    for path in [
        cfg["output_root"],
        cfg["aligned_csv"].parent,
        cfg["unmatched_csv"].parent,
        cfg["debug_dir"],
        cfg["split_csv"].parent,
        cfg["runs_dir"],
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)
