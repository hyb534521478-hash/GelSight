from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np


def _read_gray_resized(path: str | Path, image_size: int) -> np.ndarray:
    """读取光流输入图，压到固定尺寸，保证每个样本的特征维度一致。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"图片读取失败: {path}")
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img


def compute_flow_feature(ref_path: str | Path, cur_path: str | Path, image_size: int, flow_cfg: Dict) -> np.ndarray:
    """用Farneback光流估计点阵位移，把形变场压缩成可训练的数值特征。"""
    ref = _read_gray_resized(ref_path, image_size)
    cur = _read_gray_resized(cur_path, image_size)
    flow = cv2.calcOpticalFlowFarneback(
        ref,
        cur,
        None,
        pyr_scale=float(flow_cfg.get("farneback_pyr_scale", 0.5)),
        levels=int(flow_cfg.get("farneback_levels", 3)),
        winsize=int(flow_cfg.get("farneback_winsize", 21)),
        iterations=int(flow_cfg.get("farneback_iterations", 3)),
        poly_n=int(flow_cfg.get("farneback_poly_n", 5)),
        poly_sigma=float(flow_cfg.get("farneback_poly_sigma", 1.2)),
        flags=0,
    )

    gx = int(flow_cfg.get("grid_x", 15))
    gy = int(flow_cfg.get("grid_y", 15))
    xs = np.linspace(0, image_size - 1, gx).astype(np.int32)
    ys = np.linspace(0, image_size - 1, gy).astype(np.int32)
    sampled = flow[np.ix_(ys, xs)].reshape(-1, 2)
    mag = np.sqrt(np.sum(flow**2, axis=2))

    stats = np.asarray(
        [
            float(mag.mean()),
            float(mag.std()),
            float(mag.max()),
            float(np.percentile(mag, 95)),
            float(sampled[:, 0].mean()),
            float(sampled[:, 1].mean()),
            float(sampled[:, 0].std()),
            float(sampled[:, 1].std()),
        ],
        dtype=np.float32,
    )
    feat = np.concatenate([sampled.reshape(-1).astype(np.float32), stats], axis=0)
    return feat
