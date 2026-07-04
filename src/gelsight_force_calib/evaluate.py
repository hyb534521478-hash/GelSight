from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .datasets import FlowRegressionDataset, ImageRegressionDataset, TARGET_COLUMNS
from .models import build_model_from_checkpoint_meta


def _collate_keep_meta(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), list(metas)


def evaluate_checkpoint(cfg: Dict, checkpoint_path: str | Path, split: str = "val") -> Path:
    """复评某个best.pt，输出验证/训练集预测表，便于四种方法横向对比。"""
    checkpoint_path = Path(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    meta = ckpt["meta"]
    mode = meta["mode"]
    image_size = int(meta.get("image_size", 224))

    if meta["model_type"] == "mlp":
        ds = FlowRegressionDataset(cfg["split_csv"], split, image_size=image_size, flow_cfg=meta.get("flow_cfg", cfg["flow"]))
    else:
        ds = ImageRegressionDataset(cfg["split_csv"], split, mode=mode, image_size=image_size, window_size=int(meta.get("window_size", 3)))

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=_collate_keep_meta)
    model = build_model_from_checkpoint_meta(meta).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = torch.tensor(ckpt["target_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(ckpt["target_std"], dtype=torch.float32, device=device)

    rows = []
    with torch.no_grad():
        for x, y, metas in loader:
            pred = model(x.to(device)) * std + mean
            pred_np = pred.detach().cpu().numpy()
            y_np = y.numpy()
            for i, meta_row in enumerate(metas):
                row = dict(meta_row)
                for j, name in enumerate(TARGET_COLUMNS):
                    row[f"pred_{name}"] = float(pred_np[i, j])
                    row[f"gt_{name}"] = float(y_np[i, j])
                    row[f"abs_err_{name}"] = float(abs(pred_np[i, j] - y_np[i, j]))
                rows.append(row)

    out = checkpoint_path.parent / f"{split}_predictions_eval.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    mae = {f"MAE_{name}": float(df[f"abs_err_{name}"].mean()) for name in TARGET_COLUMNS}
    print("评估完成:", out)
    print(mae)
    return out
