#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
官方 *_iq.npz → 四节点 2×2 频谱拼图 PNG（512×512）

与训练时完全一致的流程：
  1) 每个节点 STFT → dB → 5/95 分位归一化 → Viridis 伪彩 → 560×560
  2) 2×2 拼成 1120×1120（node0 左上 / node1 右上 / node2 左下 / node3 右下）
  3) LANCZOS 缩放到 512×512（Qwen3-VL SFT 实际喂入尺寸）

缺失节点：全黑 560×560 瓦片。

用法示例：
  # 单文件
  python iq_npz_to_mosaic_png.py --npz path/to/00000002_e149d585_iq.npz -o out.png

  # 按官方 index.csv 批量（输出 <stem>.png）
  python iq_npz_to_mosaic_png.py --dataset-root /path/to/train --out-dir ./pngs --limit 1
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import signal

# ---------- 训练时固定参数（勿改） ----------
TILE = 560  # 单节点边长；马赛克 2*TILE = 1120
MOSAIC = TILE * 2
OUT_SIZE = 512  # Qwen3-VL 训练输入边长
PCT_LOW, PCT_HIGH = 5, 95
NPERSEG_CAP = 1024
# 布局：
#   [node0 | node1]
#   [node2 | node3]


def load_iq_node(raw: np.ndarray) -> np.ndarray:
    """int16 交错 I/Q → complex64。"""
    if raw.size == 0:
        return np.zeros(0, dtype=np.complex64)
    raw = raw[: len(raw) // 2 * 2].astype(np.float32)
    return (raw[::2] + 1j * raw[1::2]).astype(np.complex64)


def iq_to_spectrogram(iq: np.ndarray, fs_hz: float, out_hw: tuple[int, int]) -> np.ndarray:
    """
    complex IQ → BGR Viridis uint8。
    STFT: scipy.signal.spectrogram, return_onesided=False
    nperseg = min(1024, max(32, len(iq)//100)), noverlap = nperseg//2
    频率轴 fftshift；功率 10*log10；分位 [5,95] 归一化到 [0,1]
    频率方向：fftshift 后低频在中间、高频在上下边缘（与训练一致）
    """
    h, w = out_hw
    if iq.size < 16 or fs_hz <= 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    nperseg = min(NPERSEG_CAP, max(32, len(iq) // 100))
    noverlap = nperseg // 2
    _, _, Sxx = signal.spectrogram(
        iq, fs=fs_hz, nperseg=nperseg, noverlap=noverlap, return_onesided=False
    )
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    p_low, p_high = np.percentile(Sxx_db, [PCT_LOW, PCT_HIGH])
    if p_high <= p_low:
        p_low, p_high = float(Sxx_db.min()), float(Sxx_db.max() + 1e-6)
    Sxx_norm = np.clip((Sxx_db - p_low) / (p_high - p_low), 0, 1)
    gray = (Sxx_norm * 255).astype(np.uint8)
    gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)


def make_mosaic_bgr(npz_path: Path, node_flags: list[int] | None = None) -> np.ndarray:
    """读官方 *_iq.npz，返回 1120×1120 BGR 马赛克。"""
    tiles = []
    with np.load(npz_path) as data:
        for i in range(4):
            raw = data[f"iq_node{i}"]
            sr = float(data[f"sr_node{i}"])
            flag_ok = True if node_flags is None else bool(int(node_flags[i]))
            present = flag_ok and raw.size > 0 and sr > 0
            if not present:
                tiles.append(np.zeros((TILE, TILE, 3), dtype=np.uint8))
                continue
            iq = load_iq_node(raw)
            tiles.append(iq_to_spectrogram(iq, sr, (TILE, TILE)))

    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    return np.concatenate([top, bot], axis=0)


def mosaic_to_png_bytes_rgb(mosaic_bgr: np.ndarray, size: int = OUT_SIZE) -> Image.Image:
    """BGR 马赛克 → RGB，再 LANCZOS 缩到 size×size（与训练 build_dataset 一致）。"""
    rgb = cv2.cvtColor(mosaic_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    if im.size != (size, size):
        im = im.resize((size, size), Image.Resampling.LANCZOS)
    return im


def convert_one(
    npz_path: Path,
    out_png: Path,
    node_flags: list[int] | None = None,
    size: int = OUT_SIZE,
) -> Path:
    mosaic = make_mosaic_bgr(npz_path, node_flags)
    im = mosaic_to_png_bytes_rgb(mosaic, size=size)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(out_png), format="PNG", optimize=True)
    return out_png


def stem_from_iq_relpath(rel: str) -> str:
    # e.g. iq_sample/00000002_e149d585_iq.npz → 00000002_e149d585
    return Path(rel).stem.replace("_iq", "")


def _worker(task: dict) -> str:
    convert_one(
        Path(task["npz"]),
        Path(task["out"]),
        node_flags=task.get("flags"),
        size=int(task["size"]),
    )
    return task["out"]


def batch_from_index(
    dataset_root: Path,
    out_dir: Path,
    *,
    size: int = OUT_SIZE,
    limit: int | None = None,
    workers: int = 1,
) -> int:
    index_csv = dataset_root / "index.csv"
    with open(index_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]

    tasks = []
    for row in rows:
        rel = row["iq_npz_relpath"].replace("\\", "/")
        stem = stem_from_iq_relpath(rel)
        flags = [int(row.get(f"iq_node{i}", 1)) for i in range(4)]
        tasks.append(
            {
                "npz": str(dataset_root / rel),
                "out": str(out_dir / f"{stem}.png"),
                "flags": flags,
                "size": size,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        for t in tasks:
            _worker(t)
            print(f"[ok] {t['out']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            for fut in as_completed(futs):
                print(f"[ok] {fut.result()}")
    return len(tasks)


def main():
    p = argparse.ArgumentParser(description="IQ npz → 512×512 四节点 2×2 频谱拼图 PNG")
    p.add_argument("--npz", type=Path, help="单个 *_iq.npz")
    p.add_argument("-o", "--output", type=Path, help="输出 PNG 路径（单文件模式）")
    p.add_argument("--dataset-root", type=Path, help="官方 train 或 test_public 根目录（含 index.csv）")
    p.add_argument("--out-dir", type=Path, default=Path("./mosaic_png"), help="批量输出目录")
    p.add_argument("--size", type=int, default=OUT_SIZE, help="输出边长，默认 512")
    p.add_argument("--limit", type=int, default=None, help="批量时只处理前 N 条")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    if args.npz:
        out = args.output
        if out is None:
            stem = args.npz.stem.replace("_iq", "")
            out = Path(f"{stem}.png")
        convert_one(args.npz, out, size=args.size)
        print(f"[ok] {args.npz} -> {out} ({args.size}x{args.size})")
        return

    if args.dataset_root:
        n = batch_from_index(
            args.dataset_root,
            args.out_dir,
            size=args.size,
            limit=args.limit,
            workers=args.workers,
        )
        print(f"done. wrote {n} pngs to {args.out_dir}")
        return

    p.error("请指定 --npz 或 --dataset-root")


if __name__ == "__main__":
    main()
