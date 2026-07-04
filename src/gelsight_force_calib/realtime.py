from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

from .flow_features import compute_flow_feature
from .models import build_model_from_checkpoint_meta
from .transforms import load_rgb_tensor


def _frame_to_tmp_png(frame_bgr: np.ndarray, path: Path) -> Path:
    """把实时帧临时落盘，复用训练阶段的图像/光流处理逻辑，减少在线离线不一致。"""
    cv2.imwrite(str(path), frame_bgr)
    return path


def _frame_to_tensor(frame_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    """实时帧转成和训练完全一致的3通道输入张量。"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    tmp = np.asarray(img.resize((image_size, image_size), Image.BILINEAR), dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tmp = (tmp - mean) / std
    return torch.from_numpy(tmp).permute(2, 0, 1).contiguous()


def _predict_from_tensors(model, tensors: List[torch.Tensor], mean, std, device) -> np.ndarray:
    """多帧图像通道拼接后执行一次回归推理。"""
    x = torch.cat(tensors, dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = model(x)
        pred = pred_norm * std.to(device) + mean.to(device)
    return pred.squeeze(0).detach().cpu().numpy()


def run_realtime(checkpoint_path: str | Path, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
    """实时运行软件：r复位Img0，q退出，观察模型在真实GelSight视频流上的预测稳定性。"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    meta: Dict = ckpt["meta"]
    model = build_model_from_checkpoint_meta(meta).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = torch.tensor(ckpt["target_mean"], dtype=torch.float32)
    std = torch.tensor(ckpt["target_std"], dtype=torch.float32)

    mode = meta["mode"]
    image_size = int(meta.get("image_size", 224))
    window_size = int(meta.get("window_size", 3))
    frame_window = deque(maxlen=max(1, window_size))
    tmp_dir = checkpoint_path.parent / "realtime_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(int(camera_index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机: {camera_index}")

    ref_frame = None
    last_pred = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    print("实时窗口按键：r=复位当前帧为Img0，q=退出")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("相机读取失败，退出")
            break
        if ref_frame is None:
            ref_frame = frame.copy()
            frame_window.clear()

        frame_window.append(frame.copy())
        try:
            if mode == "method1_pair":
                tensors = [_frame_to_tensor(ref_frame, image_size), _frame_to_tensor(frame, image_size)]
                last_pred = _predict_from_tensors(model, tensors, mean, std, device)
            elif mode == "method3_temporal":
                frames = list(frame_window)
                while len(frames) < window_size:
                    frames.insert(0, frames[0])
                tensors = [_frame_to_tensor(f, image_size) for f in frames[-window_size:]]
                last_pred = _predict_from_tensors(model, tensors, mean, std, device)
            elif mode == "method4_ref_temporal":
                frames = list(frame_window)
                while len(frames) < window_size:
                    frames.insert(0, frames[0])
                tensors = [_frame_to_tensor(ref_frame, image_size)] + [_frame_to_tensor(f, image_size) for f in frames[-window_size:]]
                last_pred = _predict_from_tensors(model, tensors, mean, std, device)
            elif mode == "method2_optical_flow":
                ref_path = _frame_to_tmp_png(ref_frame, tmp_dir / "ref.png")
                cur_path = _frame_to_tmp_png(frame, tmp_dir / "cur.png")
                feat = compute_flow_feature(ref_path, cur_path, image_size, meta.get("flow_cfg", {}))
                x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_norm = model(x)
                    last_pred = (pred_norm * std.to(device) + mean.to(device)).squeeze(0).detach().cpu().numpy()
            else:
                raise ValueError(f"checkpoint模式不支持实时运行: {mode}")
        except Exception as e:
            cv2.putText(frame, f"predict error: {e}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        show = frame.copy()
        cv2.putText(show, f"mode: {mode}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(show, f"F={last_pred[0]:.3f} N  theta={last_pred[1]:.2f}  alpha={last_pred[2]:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(show, "r: reset Img0   q: quit", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("GelSight force realtime", show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            ref_frame = frame.copy()
            frame_window.clear()
            frame_window.append(frame.copy())
            print("已复位：当前帧已设置为新的Img0")

    cap.release()
    cv2.destroyAllWindows()
