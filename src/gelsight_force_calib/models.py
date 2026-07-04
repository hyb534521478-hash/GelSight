from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class MLPRegressor(nn.Module):
    """光流点位偏移到三维标定量的映射网络，适合数值特征回归。"""

    def __init__(self, input_dim: int, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_resnet18_regressor(in_channels: int, pretrained: bool = False, output_dim: int = 3) -> nn.Module:
    """构建ResNet18回归主干，并把第一层改成多帧拼接输入需要的通道数。"""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    if pretrained:
        with torch.no_grad():
            base = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(base.repeat(1, in_channels, 1, 1) * (3.0 / float(in_channels)))
    model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, output_dim)
    return model


def build_model_from_checkpoint_meta(meta: dict) -> nn.Module:
    """实时推理时根据checkpoint里的元信息恢复正确网络结构。"""
    model_type = meta.get("model_type")
    if model_type == "resnet18":
        return build_resnet18_regressor(int(meta["in_channels"]), pretrained=False, output_dim=3)
    if model_type == "mlp":
        return MLPRegressor(int(meta["feature_dim"]), output_dim=3)
    raise ValueError(f"未知模型类型: {model_type}")
