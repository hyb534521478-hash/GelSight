from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import FlowRegressionDataset, ImageRegressionDataset, TARGET_COLUMNS
from .models import MLPRegressor, build_resnet18_regressor


def _collate_keep_meta(batch):
    """DataLoader批处理：保留样本来源，方便保存验证预测明细。"""
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), list(metas)


def _collect_target_stats(dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """只用训练集统计标签均值方差，避免F、theta、alpha量纲不同导致训练不稳。"""
    ys = []
    for _, y, _ in DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=_collate_keep_meta):
        ys.append(y)
    y_all = torch.cat(ys, dim=0)
    mean = y_all.mean(dim=0)
    std = y_all.std(dim=0).clamp_min(1e-6)
    return mean, std


def _denorm(pred: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """把网络输出从标准化空间还原回真实物理量。"""
    return pred * std.to(pred.device) + mean.to(pred.device)


def _make_loaders(train_ds, val_ds, batch_size: int, num_workers: int):
    """统一创建训练/验证加载器，验证集不打乱，便于定位具体误差样本。"""
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_keep_meta,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_keep_meta,
    )
    return train_loader, val_loader


def _plot_loss(log_df: pd.DataFrame, path: Path) -> None:
    """输出训练曲线，快速判断是否欠拟合、过拟合或学习率异常。"""
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train_loss")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE in normalized target space")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _evaluate(model, loader, mean, std, device, criterion) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """验证集评估：同时保存归一化loss和真实量纲MAE。"""
    model.eval()
    losses = []
    preds_all = []
    targets_all = []
    meta_rows = []
    with torch.no_grad():
        for x, y, metas in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_norm = (y - mean.to(device)) / std.to(device)
            pred_norm = model(x)
            loss = criterion(pred_norm, y_norm)
            losses.append(float(loss.item()))
            pred = _denorm(pred_norm, mean, std).detach().cpu()
            preds_all.append(pred)
            targets_all.append(y.detach().cpu())
            meta_rows.extend(metas)

    preds = torch.cat(preds_all, dim=0).numpy()
    targets = torch.cat(targets_all, dim=0).numpy()
    abs_err = np.abs(preds - targets)
    metrics = {f"MAE_{name}": float(abs_err[:, i].mean()) for i, name in enumerate(TARGET_COLUMNS)}
    pred_df = pd.DataFrame(meta_rows)
    for i, name in enumerate(TARGET_COLUMNS):
        pred_df[f"pred_{name}"] = preds[:, i]
        pred_df[f"gt_{name}"] = targets[:, i]
        pred_df[f"abs_err_{name}"] = abs_err[:, i]
    return float(np.mean(losses)), metrics, pred_df


def train_image_method(cfg: Dict, mode: str, run_name: str, window_size: int = 3) -> Path:
    """训练图像类方案：方法1、方法3、方法4共用这条训练流水线。"""
    image_size = int(cfg["training"].get("image_size", 224))
    train_ds = ImageRegressionDataset(cfg["split_csv"], "train", mode=mode, image_size=image_size, window_size=window_size)
    val_ds = ImageRegressionDataset(cfg["split_csv"], "val", mode=mode, image_size=image_size, window_size=window_size)
    model = build_resnet18_regressor(train_ds.in_channels, pretrained=bool(cfg["training"].get("pretrained_resnet", False)))
    meta = {
        "model_type": "resnet18",
        "mode": mode,
        "window_size": int(window_size),
        "image_size": int(image_size),
        "in_channels": int(train_ds.in_channels),
        "target_columns": TARGET_COLUMNS,
    }
    return train_model(cfg, model, train_ds, val_ds, run_name, meta, int(cfg["training"].get("batch_size", 2)))


def train_flow_method(cfg: Dict, run_name: str) -> Path:
    """训练光流方案：先提取点阵位移，再用MLP学习位移到力方向的映射。"""
    image_size = int(cfg["training"].get("image_size", 224))
    train_ds = FlowRegressionDataset(cfg["split_csv"], "train", image_size=image_size, flow_cfg=cfg["flow"])
    val_ds = FlowRegressionDataset(cfg["split_csv"], "val", image_size=image_size, flow_cfg=cfg["flow"])
    model = MLPRegressor(train_ds.feature_dim, output_dim=3)
    meta = {
        "model_type": "mlp",
        "mode": "method2_optical_flow",
        "window_size": 1,
        "image_size": int(image_size),
        "feature_dim": int(train_ds.feature_dim),
        "target_columns": TARGET_COLUMNS,
        "flow_cfg": cfg["flow"],
    }
    return train_model(cfg, model, train_ds, val_ds, run_name, meta, int(cfg["training"].get("flow_batch_size", 64)))


def train_model(cfg: Dict, model: nn.Module, train_ds, val_ds, run_name: str, meta: Dict, batch_size: int) -> Path:
    """核心训练循环：标准化标签、保存best.pt、日志、曲线和验证预测。"""
    if not Path(cfg["split_csv"]).exists():
        raise FileNotFoundError(f"缺少数据划分文件，请先运行 scripts/02_split_dataset.py: {cfg['split_csv']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["runs_dir"]) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    mean, std = _collect_target_stats(train_ds)

    train_loader, val_loader = _make_loaders(
        train_ds,
        val_ds,
        batch_size=batch_size,
        num_workers=int(cfg["training"].get("num_workers", 4)),
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["training"].get("epochs", 80)))
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = max(1, int(cfg["training"].get("accum_steps", 1)))

    best_val = float("inf")
    log_rows = []
    best_pred_df = None

    for epoch in range(1, int(cfg["training"].get("epochs", 80)) + 1):
        model.train()
        train_losses = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"{run_name} epoch {epoch}")
        for step, (x, y, _) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_norm = (y - mean.to(device)) / std.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_norm = model(x)
                loss = criterion(pred_norm, y_norm) / accum_steps
            scaler.scale(loss).backward()
            if step % accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.item() * accum_steps))
            pbar.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        scheduler.step()
        val_loss, metrics, pred_df = _evaluate(model, val_loader, mean, std, device, criterion)
        train_loss = float(np.mean(train_losses))
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"], **metrics}
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(run_dir / "training_log.csv", index=False, encoding="utf-8-sig")

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_pred_df = pred_df
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "meta": meta,
                "target_mean": mean.tolist(),
                "target_std": std.tolist(),
                "best_val_loss": best_val,
                "epoch": epoch,
            }
            torch.save(checkpoint, run_dir / "best.pt")
            pred_df.to_csv(run_dir / "val_predictions_best.csv", index=False, encoding="utf-8-sig")

        if epoch % int(cfg["training"].get("save_every", 10)) == 0:
            torch.save({"model_state_dict": model.state_dict(), "meta": meta, "target_mean": mean.tolist(), "target_std": std.tolist(), "epoch": epoch}, run_dir / f"epoch_{epoch:03d}.pt")

        print(f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    log_df = pd.DataFrame(log_rows)
    _plot_loss(log_df, run_dir / "loss_curve.png")
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "target_mean": mean.tolist(), "target_std": std.tolist(), "best_val_loss": best_val}, f, ensure_ascii=False, indent=2)
    if best_pred_df is not None:
        best_pred_df.to_csv(run_dir / "val_predictions_best.csv", index=False, encoding="utf-8-sig")
    return run_dir
