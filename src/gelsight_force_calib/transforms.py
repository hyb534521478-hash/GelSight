from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image


def load_rgb_tensor(path: str | Path, image_size: int) -> torch.Tensor:
    """读取单张GelSight图片，统一尺寸和归一化，输出3通道张量。"""
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # 使用ImageNet均值方差是为了让ResNet输入尺度稳定，即使不加载预训练权重也更好收敛。
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def stack_images_as_channels(paths: List[str | Path], image_size: int) -> torch.Tensor:
    """把多帧图片沿通道拼接，形成[Img0, Imgn]或时间窗口输入。"""
    tensors = [load_rgb_tensor(p, image_size) for p in paths]
    return torch.cat(tensors, dim=0)
