from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .flow_features import compute_flow_feature
from .transforms import stack_images_as_channels

TARGET_COLUMNS = ["F", "theta", "alpha"]


class AlignmentTable:
    """把对齐后的CSV整理成按片段索引的数据结构，方便取Img0和历史窗口。"""

    def __init__(self, split_csv: str | Path, split: str):
        df = pd.read_csv(split_csv)
        df = df[df["split"] == split].copy()
        if df.empty:
            raise ValueError(f"{split}集合为空，请检查8:2划分结果")
        df["episode"] = df["episode"].astype(int)
        df["img_index"] = df["img_index"].astype(int)
        df = df.sort_values(["episode", "img_index"]).reset_index(drop=True)
        self.df = df
        self.groups: Dict[int, pd.DataFrame] = {
            int(ep): g.sort_values("img_index").reset_index(drop=True) for ep, g in df.groupby("episode")
        }
        self.row_lookup: Dict[Tuple[int, int], int] = {}
        for ep, g in self.groups.items():
            for local_pos, row in g.iterrows():
                self.row_lookup[(ep, int(row["img_index"]))] = int(local_pos)

    def ref_image(self, episode: int) -> str:
        """返回该片段第一张未受力/初始图，训练时等价于实时软件里的复位Img0。"""
        return str(self.groups[int(episode)].iloc[0]["image_path"])

    def window_images(self, episode: int, img_index: int, window_size: int) -> List[str]:
        """返回以当前帧结尾的历史窗口，不足时用最早帧补齐，保证输入通道固定。"""
        g = self.groups[int(episode)]
        pos = self.row_lookup[(int(episode), int(img_index))]
        start = max(0, pos - window_size + 1)
        rows = g.iloc[start : pos + 1]
        paths = rows["image_path"].astype(str).tolist()
        while len(paths) < window_size:
            paths.insert(0, paths[0])
        return paths[-window_size:]


class ImageRegressionDataset(Dataset):
    """四种训练方式中的图像输入数据集：支持双帧、时间窗口、Img0+时间窗口。"""

    def __init__(self, split_csv: str | Path, split: str, mode: str, image_size: int, window_size: int = 3):
        self.table = AlignmentTable(split_csv, split)
        self.rows = self.table.df.reset_index(drop=True)
        self.mode = mode
        self.image_size = int(image_size)
        self.window_size = int(window_size)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def in_channels(self) -> int:
        if self.mode == "method1_pair":
            return 6
        if self.mode == "method3_temporal":
            return 3 * self.window_size
        if self.mode == "method4_ref_temporal":
            return 3 * (self.window_size + 1)
        raise ValueError(f"未知图像模式: {self.mode}")

    def _input_paths(self, row: pd.Series) -> List[str]:
        episode = int(row["episode"])
        img_index = int(row["img_index"])
        if self.mode == "method1_pair":
            return [self.table.ref_image(episode), str(row["image_path"])]
        if self.mode == "method3_temporal":
            return self.table.window_images(episode, img_index, self.window_size)
        if self.mode == "method4_ref_temporal":
            return [self.table.ref_image(episode)] + self.table.window_images(episode, img_index, self.window_size)
        raise ValueError(f"未知图像模式: {self.mode}")

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        x = stack_images_as_channels(self._input_paths(row), self.image_size)
        y = torch.tensor(row[TARGET_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32)
        meta = {
            "episode": int(row["episode"]),
            "img_index": int(row["img_index"]),
            "image_path": str(row["image_path"]),
        }
        return x, y, meta


class FlowRegressionDataset(Dataset):
    """光流训练方式：把Img0到Img_n的点阵位移作为输入特征。"""

    def __init__(self, split_csv: str | Path, split: str, image_size: int, flow_cfg: Dict):
        self.table = AlignmentTable(split_csv, split)
        self.rows = self.table.df.reset_index(drop=True)
        self.image_size = int(image_size)
        self.flow_cfg = flow_cfg
        sample_ref = self.table.ref_image(int(self.rows.iloc[0]["episode"]))
        sample_cur = str(self.rows.iloc[0]["image_path"])
        self.feature_dim = int(compute_flow_feature(sample_ref, sample_cur, self.image_size, self.flow_cfg).shape[0])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        ref = self.table.ref_image(int(row["episode"]))
        cur = str(row["image_path"])
        feat = compute_flow_feature(ref, cur, self.image_size, self.flow_cfg)
        x = torch.tensor(feat, dtype=torch.float32)
        y = torch.tensor(row[TARGET_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32)
        meta = {
            "episode": int(row["episode"]),
            "img_index": int(row["img_index"]),
            "image_path": str(row["image_path"]),
        }
        return x, y, meta
