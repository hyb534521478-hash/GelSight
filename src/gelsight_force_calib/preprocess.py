from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .io_utils import (
    interpolate_force,
    list_episode_dirs,
    list_images,
    parse_image_name,
    read_force_txt,
)


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """平滑突变曲线，削弱单帧噪声对接触点判断的影响。"""
    if window <= 1 or len(values) < window:
        return values.astype(np.float64)
    return pd.Series(values).rolling(window, center=True, min_periods=1).median().to_numpy()


def _robust_detect_first_change(
    values: np.ndarray,
    baseline_count: int,
    threshold_k: float,
    min_jump: float,
    consecutive: int,
) -> Tuple[int, float, float]:
    """通用首次突变检测：用前段静止区估计背景，找到首次连续超过阈值的位置。"""
    n = len(values)
    if n == 0:
        raise ValueError("空序列无法检测接触点")
    baseline_count = max(3, min(int(baseline_count), max(3, n // 3)))
    base = values[:baseline_count]
    base_med = float(np.median(base))
    mad = float(np.median(np.abs(base - base_med)))
    robust_std = 1.4826 * mad
    threshold = base_med + max(float(min_jump), float(threshold_k) * (robust_std + 1e-6))
    consecutive = max(1, int(consecutive))

    for idx in range(baseline_count, max(baseline_count + 1, n - consecutive + 1)):
        if np.all(values[idx : idx + consecutive] >= threshold):
            return idx, threshold, base_med

    # 数据噪声较大或接触过早时，用最大上升沿兜底，保证预处理不中断。
    if n >= 2:
        grad = np.diff(values)
        fallback = int(np.argmax(grad) + 1)
        return fallback, threshold, base_med
    return 0, threshold, base_med


def compute_image_change_curve(images: List[Path], cfg: Dict[str, Any]) -> np.ndarray:
    """计算 d_t=max(|I_t-I_0|)，用第一张未受力图作为参考背景。"""
    if not images:
        raise ValueError("图片列表为空")
    resize_width = int(cfg["alignment"].get("image_resize_width", 160))

    def read_gray(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"图片读取失败: {path}")
        if resize_width > 0 and img.shape[1] != resize_width:
            scale = resize_width / float(img.shape[1])
            img = cv2.resize(img, (resize_width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        return img.astype(np.int16)

    ref = read_gray(images[0])
    diffs = []
    for p in images:
        cur = read_gray(p)
        diffs.append(float(np.max(np.abs(cur - ref))))
    diffs_np = np.asarray(diffs, dtype=np.float64)
    return _rolling_median(diffs_np, int(cfg["alignment"].get("image_smooth_window", 5)))


def detect_image_contact(images: List[Path], cfg: Dict[str, Any]) -> Tuple[int, np.ndarray, float, float]:
    """检测图像首次接触帧：接触后GelSight纹理变化会让d_t明显突增。"""
    curve = compute_image_change_curve(images, cfg)
    idx, threshold, baseline = _robust_detect_first_change(
        curve,
        baseline_count=int(cfg["alignment"].get("image_baseline_frames", 10)),
        threshold_k=float(cfg["alignment"].get("image_threshold_k", 8.0)),
        min_jump=float(cfg["alignment"].get("image_min_jump", 8.0)),
        consecutive=int(cfg["alignment"].get("consecutive_frames", 2)),
    )
    return idx, curve, threshold, baseline


def detect_force_contact(force_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[int, float, np.ndarray, float, float, float]:
    """检测力曲线首次接触点：力值离开零点后的第一个稳定突变位置。"""
    raw = force_df["force_raw"].to_numpy(dtype=np.float64)
    baseline_samples = int(cfg["alignment"].get("force_baseline_samples", 20))
    baseline_samples = max(3, min(baseline_samples, max(3, len(raw) // 3)))
    zero = float(np.median(raw[:baseline_samples]))
    signal = raw - zero
    if bool(cfg["alignment"].get("force_use_abs", True)):
        signal = np.abs(signal)
    smooth = _rolling_median(signal, int(cfg["alignment"].get("force_smooth_window", 5)))
    idx, threshold, base = _robust_detect_first_change(
        smooth,
        baseline_count=baseline_samples,
        threshold_k=float(cfg["alignment"].get("force_threshold_k", 8.0)),
        min_jump=float(cfg["alignment"].get("force_min_jump_n", 0.03)),
        consecutive=1,
    )
    t0 = float(force_df.iloc[idx]["force_time_sec"])
    return idx, t0, smooth, threshold, base, zero


def _save_debug_plot(
    debug_dir: Path,
    episode: int,
    image_curve: np.ndarray,
    image_idx: int,
    image_threshold: float,
    force_curve: np.ndarray,
    force_idx: int,
    force_threshold: float,
) -> None:
    """保存每个片段的接触检测曲线，方便人工排查误检片段。"""
    debug_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(image_curve)
    ax1.axvline(image_idx, linestyle="--")
    ax1.axhline(image_threshold, linestyle=":")
    ax1.set_title(f"Episode {episode} image contact")
    ax1.set_ylabel("max abs diff")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(force_curve)
    ax2.axvline(force_idx, linestyle="--")
    ax2.axhline(force_threshold, linestyle=":")
    ax2.set_title("force contact")
    ax2.set_xlabel("sample index")
    ax2.set_ylabel("force delta")
    fig.tight_layout()
    fig.savefig(debug_dir / f"episode_{episode:03d}_contact_debug.png", dpi=150)
    plt.close(fig)


def build_alignment_table(cfg: Dict[str, Any], debug_plots: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """主预处理入口：完成图像绝对时间与力相对时间的接触点对齐。"""
    image_root: Path = cfg["image_root"]
    force_root: Path = cfg["force_root"]
    episodes = list_episode_dirs(image_root, cfg.get("episode_count"))

    rows: List[Dict[str, object]] = []
    unmatched: List[Dict[str, object]] = []

    for episode, ep_dir in tqdm(episodes, desc="对齐109个片段"):
        try:
            images = list_images(ep_dir, cfg.get("image_exts", [".png"]))
            if len(images) < 3:
                raise ValueError("图片数量过少")
            force_path = force_root / f"{episode}.txt"
            force_df = read_force_txt(force_path)

            img_contact_idx, img_curve, img_th, img_base = detect_image_contact(images, cfg)
            force_contact_idx, force_contact_time, force_curve, force_th, force_base, force_zero = detect_force_contact(force_df, cfg)

            img_contact_meta = parse_image_name(images[img_contact_idx])
            if img_contact_meta is None:
                raise ValueError(f"接触帧文件名无法解析时间: {images[img_contact_idx].name}")
            img_contact_time = img_contact_meta["timestamp"]

            min_force_t = float(force_df["force_time_sec"].min())
            max_force_t = float(force_df["force_time_sec"].max())

            parsed_cache = []
            for img_idx, img_path in enumerate(images):
                meta = parse_image_name(img_path)
                if meta is None:
                    unmatched.append({"episode": episode, "path": str(img_path), "reason": "图片文件名无法解析"})
                    continue
                rel_from_contact = (meta["timestamp"] - img_contact_time).total_seconds()
                query_t = force_contact_time + rel_from_contact
                if query_t < min_force_t or query_t > max_force_t:
                    unmatched.append({"episode": episode, "path": str(img_path), "reason": "对齐后超出力曲线时间范围"})
                    continue
                parsed_cache.append((img_idx, img_path, meta, query_t, rel_from_contact))

            if not parsed_cache:
                raise ValueError("该片段没有任何图片能对齐到力曲线")

            query_times = np.asarray([x[3] for x in parsed_cache], dtype=np.float64)
            interp_raw = interpolate_force(force_df, query_times)
            F_values = np.abs(interp_raw - force_zero) if bool(cfg["alignment"].get("force_use_abs", True)) else (interp_raw - force_zero)

            for (img_idx, img_path, meta, query_t, rel_from_contact), raw_force, F_value in zip(parsed_cache, interp_raw, F_values):
                rows.append(
                    {
                        "episode": episode,
                        "img_index": img_idx,
                        "image_path": str(img_path.resolve()),
                        "image_name": img_path.name,
                        "image_abs_time": meta["timestamp_str"],
                        "image_rel_sec_from_contact": rel_from_contact,
                        "image_contact_index": img_contact_idx,
                        "image_contact_time": img_contact_time.isoformat(timespec="milliseconds"),
                        "force_txt": str(force_path.resolve()),
                        "force_time_sec": query_t,
                        "force_raw_interpolated": float(raw_force),
                        "force_zero_baseline": float(force_zero),
                        "F": float(F_value),
                        "theta": float(meta["theta"]),
                        "alpha": float(meta["alpha"]),
                        "filename_F": float(meta["file_F"]),
                    }
                )

            if debug_plots:
                _save_debug_plot(cfg["debug_dir"], episode, img_curve, img_contact_idx, img_th, force_curve, force_contact_idx, force_th)

        except Exception as e:
            unmatched.append({"episode": episode, "path": str(ep_dir), "reason": f"片段处理失败: {e}"})

    aligned = pd.DataFrame(rows)
    unmatched_df = pd.DataFrame(unmatched)
    return aligned, unmatched_df


def save_alignment(cfg: Dict[str, Any], aligned: pd.DataFrame, unmatched: pd.DataFrame) -> None:
    """保存对齐总表和异常样本表，后续训练只依赖alignment_table.csv。"""
    cfg["aligned_csv"].parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(cfg["aligned_csv"], index=False, encoding="utf-8-sig")
    unmatched.to_csv(cfg["unmatched_csv"], index=False, encoding="utf-8-sig")
