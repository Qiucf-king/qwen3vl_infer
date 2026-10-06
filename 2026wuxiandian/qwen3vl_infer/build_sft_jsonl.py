#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把官方 train 标签 + 2x2 mosaic 频谱图 转成 ms-swift SFT jsonl。

官方 train 目录一般是:
  train/label/<stem>_label.json
  （图不在官方包里，要用和推理相同的 2x2 mosaic png）

图放在 --spec-dir，文件名必须是 <stem>.png，例如:
  images/train_raw/00000002_e149d585.png
  对应 train/label/00000002_e149d585_label.json

输出（相对本脚本目录）:
  train.jsonl  valid.jsonl
  images/train/*.png  images/valid/*.png   # 默认缩到 512x512
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
COORD_DECIMALS = 4
QWEN_FACTOR = 32
DEFAULT_SIZE = 512

SYSTEM_PROMPT = (
    "You are an expert multimodal assistant for radio sensing and drone detection. "
    "Given a 2x2 mosaic spectrogram from four IQ receiver nodes (ISM 2.4/5.8 GHz), "
    "identify non-allowlisted drones and localize each target in the ENU frame. "
    "Always reply with a single JSON object only, no markdown."
)

USER_TEMPLATE = (
    "<image>"
    "This image is a 2x2 mosaic spectrogram from four IQ nodes "
    "(node0 top-left, node1 top-right, node2 bottom-left, node3 bottom-right). "
    "Task: detect all target drones that are NOT on the allowlist, and predict each "
    "drone's model_id (integer 0-7) and ENU position in meters.\n"
    "Allowlist model_ids for this sample (do NOT predict these): {allowlist}\n"
    "Output JSON schema:\n"
    '{{"drones":[{{"model_id":0,"position_enu":{{"e_m":0.0,"n_m":0.0,"u_m":0.0}}}}]}}\n'
    "Rules:\n"
    "- drones may contain 1~3 targets\n"
    "- do not include allowlisted model_ids\n"
    "- use meters; keep ENU coordinates to 4 decimal places\n"
    "- output JSON only"
)


def round_coord(x, ndigits: int = COORD_DECIMALS) -> float:
    return round(float(x), ndigits)


def fmt_coord(x, ndigits: int = COORD_DECIMALS) -> str:
    return f"{round_coord(x, ndigits):.{ndigits}f}"


def fmt_allowlist(allowlist) -> str:
    ids = sorted({int(x) for x in (allowlist or [])})
    return "[]" if not ids else "[" + ", ".join(str(i) for i in ids) + "]"


def fmt_answer(drones: list) -> str:
    out = []
    for d in drones or []:
        pe = d.get("position_enu") if isinstance(d, dict) else None
        if not isinstance(pe, dict):
            pe = d if isinstance(d, dict) else {}
        out.append(
            {
                "model_id": int(d["model_id"]),
                "e": round_coord(pe.get("e_m", d.get("e_m"))),
                "n": round_coord(pe.get("n_m", d.get("n_m"))),
                "u": round_coord(pe.get("u_m", d.get("u_m"))),
            }
        )
    out.sort(key=lambda x: (x["model_id"], x["e"]))
    parts = [
        '{"model_id":%d,"position_enu":{"e_m":%s,"n_m":%s,"u_m":%s}}'
        % (x["model_id"], fmt_coord(x["e"]), fmt_coord(x["n"]), fmt_coord(x["u"]))
        for x in out
    ]
    return '{"drones":[' + ",".join(parts) + "]}"


def save_resized(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if size <= 0:
        if dst.resolve() != src.resolve():
            shutil.copy2(src, dst)
        return
    if dst.exists():
        try:
            with Image.open(dst) as im:
                if im.size == (size, size):
                    return
        except Exception:
            pass
        dst.unlink(missing_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        im.save(dst, format="PNG", optimize=True)


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[ok] {path}: {len(rows)}")


def load_from_official_labels(label_dir: Path, spec_dir: Path) -> list[dict]:
    samples = []
    missing = 0
    files = sorted(label_dir.glob("*_label.json"))
    if not files:
        raise SystemExit(f"no *_label.json in {label_dir}")
    for lp in tqdm(files, desc="scan labels"):
        stem = lp.name[: -len("_label.json")]
        png = spec_dir / f"{stem}.png"
        if not png.exists():
            missing += 1
            continue
        obj = json.loads(lp.read_text(encoding="utf-8"))
        try:
            sample_id = int(stem.split("_", 1)[0])
        except Exception:
            sample_id = len(samples)
        samples.append(
            {
                "sample_id": sample_id,
                "stem": stem,
                "src_png": png,
                "allowlist": obj.get("allowlist") or [],
                "drones": obj.get("drones") or [],
            }
        )
    if not samples:
        raise SystemExit(
            f"found labels but no matching png in {spec_dir} "
            f"(need <stem>.png). missing={missing}"
        )
    print(f"[ok] paired {len(samples)} samples (missing png {missing})")
    return samples


def load_from_ann_json(ann_path: Path, images_root: Path) -> list[dict]:
    obj = json.loads(ann_path.read_text(encoding="utf-8"))
    rows = obj.get("samples") if isinstance(obj, dict) else obj
    samples = []
    missing = 0
    for s in rows:
        name = s.get("file_name") or f"{s.get('stem')}.png"
        png = images_root / name
        if not png.exists():
            png = images_root / Path(name).name
        if not png.exists():
            missing += 1
            continue
        samples.append(
            {
                "sample_id": int(s["sample_id"]),
                "stem": s.get("stem") or Path(name).stem,
                "src_png": png,
                "allowlist": s.get("allowlist") or [],
                "drones": s.get("drones") or [],
            }
        )
    if not samples:
        raise SystemExit(f"no images found under {images_root}, missing={missing}")
    print(f"[ok] ann {ann_path.name}: {len(samples)} (missing png {missing})")
    return samples


def to_record(sample: dict, rel_img: str) -> dict:
    user = USER_TEMPLATE.format(allowlist=fmt_allowlist(sample.get("allowlist")))
    return {
        "sample_id": int(sample["sample_id"]),
        "stem": sample.get("stem", ""),
        "images": [rel_img],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": fmt_answer(sample.get("drones") or [])},
        ],
    }


def materialize(samples: list[dict], pack: str, size: int) -> list[dict]:
    rows = []
    for s in tqdm(samples, desc=pack):
        rel = Path("images") / pack / f"{s['stem']}.png"
        save_resized(Path(s["src_png"]), ROOT / rel, size)
        rec = to_record(s, rel.as_posix())
        rec["split"] = pack
        rows.append(rec)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labels-dir",
        type=Path,
        help="官方 train/label 目录（*_label.json）",
    )
    ap.add_argument(
        "--spec-dir",
        type=Path,
        help="与 label 的 stem 同名的 mosaic png 目录",
    )
    ap.add_argument("--train-ann", type=Path, help="可选：已有 annotations/train.json")
    ap.add_argument("--valid-ann", type=Path, help="可选：已有 annotations/valid.json")
    ap.add_argument(
        "--ann-images",
        type=Path,
        help="train-ann/valid-ann 对应的原图目录",
    )
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE, help="0=不缩放")
    args = ap.parse_args()

    size = int(args.size)
    if size > 0 and size % QWEN_FACTOR != 0:
        raise SystemExit(f"--size 须整除 {QWEN_FACTOR}，推荐 512")

    if args.train_ann:
        img_root = args.ann_images
        if img_root is None:
            raise SystemExit("--train-ann 需要同时给 --ann-images")
        train_s = load_from_ann_json(args.train_ann, img_root)
        if args.valid_ann:
            valid_s = load_from_ann_json(args.valid_ann, img_root)
        else:
            rng = random.Random(args.seed)
            rng.shuffle(train_s)
            n_val = max(1, int(len(train_s) * args.val_ratio))
            valid_s = train_s[:n_val]
            train_s = train_s[n_val:]
    else:
        if not args.labels_dir or not args.spec_dir:
            raise SystemExit("请给 --labels-dir 和 --spec-dir，或给 --train-ann/--ann-images")
        all_s = load_from_official_labels(args.labels_dir, args.spec_dir)
        rng = random.Random(args.seed)
        rng.shuffle(all_s)
        n_val = max(1, int(len(all_s) * args.val_ratio))
        valid_s = all_s[:n_val]
        train_s = all_s[n_val:]

    train_rows = materialize(train_s, "train", size)
    valid_rows = materialize(valid_s, "valid", size)
    write_jsonl(ROOT / "train.jsonl", train_rows)
    write_jsonl(ROOT / "valid.jsonl", valid_rows)
    print("[hint] 然后: export DATASET=./train.jsonl VAL_DATASET=./valid.jsonl && bash train_sft.sh")


if __name__ == "__main__":
    main()
