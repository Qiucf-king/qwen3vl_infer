#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Infer infer_public.jsonl with LoRA ckpt -> submission jsonl (ENU 4dp)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

COORD_DECIMALS = 4
ROOT = Path(__file__).resolve().parent


def round4(x) -> float:
    return round(float(x), COORD_DECIMALS)


def parse_answer(text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            obj = json.loads(re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", m.group(0))))
        except json.JSONDecodeError:
            return []
    drones = obj.get("drones") if isinstance(obj, dict) else None
    if not isinstance(drones, list):
        return []
    out = []
    for d in drones:
        if not isinstance(d, dict):
            continue
        try:
            mid = int(d["model_id"])
        except Exception:
            continue
        pe = d.get("position_enu") if isinstance(d.get("position_enu"), dict) else d
        try:
            e = round4(pe.get("e_m"))
            n = round4(pe.get("n_m"))
            u = round4(pe.get("u_m"))
        except Exception:
            continue
        out.append({"model_id": mid, "e_m": e, "n_m": n, "u_m": u})
    return out


def resolve_image(path: str, root: Path) -> str:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    cand = root / path
    if cand.exists():
        return str(cand.resolve())
    raise FileNotFoundError(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LoRA checkpoint dir")
    ap.add_argument(
        "--model",
        default=str(ROOT / "../../pretrain_model/Qwen3-VL-8B-Instruct"),
    )
    ap.add_argument("--dataset", default=str(ROOT / "infer_public.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "submits" / "infer_submit.jsonl"))
    ap.add_argument("--raw-out", default=str(ROOT / "submits" / "infer_raw.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N")
    args = ap.parse_args()

    os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")

    model_path = Path(args.model).resolve()
    ckpt = Path(args.ckpt).resolve()
    dataset = Path(args.dataset).resolve()
    out_path = Path(args.out)
    raw_path = Path(args.raw_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[infer] model={model_path}", flush=True)
    print(f"[infer] ckpt={ckpt}", flush=True)
    print(f"[infer] dataset={dataset}", flush=True)

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(ckpt))
    model.eval()
    print("[infer] model+lora loaded", flush=True)

    rows = []
    with dataset.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    print(f"[infer] n={len(rows)}", flush=True)

    raw_f = raw_path.open("w", encoding="utf-8")
    out_f = out_path.open("w", encoding="utf-8")

    ok = 0
    empty = 0
    parse_fail = 0
    for row in tqdm(rows, desc="infer", dynamic_ncols=True, file=sys.stdout):
        sid = int(row["sample_id"])
        imgs = [resolve_image(p, ROOT) for p in (row.get("images") or [])]
        messages = []
        for m in row["messages"]:
            if m["role"] not in ("system", "user"):
                continue
            if m["role"] == "user":
                text = m["content"].replace("<image>", "").strip()
                content = [{"type": "image", "image": ip} for ip in imgs]
                content.append({"type": "text", "text": text})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "system", "content": m["content"]})

        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            prompt_len = inputs["input_ids"].shape[1]
            out_ids = gen[0, prompt_len:]
            text = processor.tokenizer.decode(out_ids, skip_special_tokens=True)
        except Exception as e:
            parse_fail += 1
            raw_f.write(
                json.dumps(
                    {"sample_id": sid, "raw": "", "error": repr(e), "drones": []},
                    ensure_ascii=False,
                )
                + "\n"
            )
            raw_f.flush()
            out_f.write(json.dumps({"sample_id": sid, "drones": []}, ensure_ascii=False) + "\n")
            out_f.flush()
            continue

        drones = parse_answer(text)
        if not drones:
            if text.strip():
                parse_fail += 1
            else:
                empty += 1
        else:
            ok += 1

        raw_f.write(
            json.dumps({"sample_id": sid, "raw": text, "drones": drones}, ensure_ascii=False)
            + "\n"
        )
        raw_f.flush()
        out_f.write(json.dumps({"sample_id": sid, "drones": drones}, ensure_ascii=False) + "\n")
        out_f.flush()

    raw_f.close()
    out_f.close()
    print(
        f"[done] out={out_path} raw={raw_path} ok={ok} empty={empty} parse_fail={parse_fail}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
