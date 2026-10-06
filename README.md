# Qwen3-VL 推理 + 训练

**train / infer 都是同一条链路：官方数据 → 输入 jsonl + mosaic 图 → 模型 → JSON。**

```
官方 train/label/*_label.json  +  <stem>.png
        │
        ▼  python build_official.py train
 train.jsonl / valid.jsonl + images/train|valid
        │
        ▼  bash train_sft.sh
 LoRA checkpoint
        │
官方 test_public/label/*_label.json  +  <stem>.png
        │
        ▼  python build_official.py infer
 infer_public.jsonl + images/infer
        │
        ▼  bash infer.sh
 submits/infer_submit.jsonl     ← 交榜用这个
```

目录必须和训练时一样，基座才是 `../../pretrain_model/Qwen3-VL-8B-Instruct`。

```
<本仓库根>/
  pretrain_model/Qwen3-VL-8B-Instruct/
  2026wuxiandian/qwen3vl_infer/     # 在这里转换 / 训练 / 推理
    build_official.py               # 官方数据 -> jsonl + 图
    train_sft.sh
    run_infer_submit.py             # 推理并解析成提交 jsonl
    infer.sh
```

```bash
cd 2026wuxiandian/qwen3vl_infer
```

## 1. 官方预训练模型

- Hugging Face：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- ModelScope：https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct

在**仓库根**执行：

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
  --local_dir ./pretrain_model/Qwen3-VL-8B-Instruct
```

## 2. 官方数据 → 输入 jsonl + 图片

官方包里是 `label/<stem>_label.json`（以及 IQ）。  
模型吃的图是四节点 **2x2 mosaic**，文件名必须是 `<stem>.png`，和 label 对齐。

| 官方 | label 内容 | 本仓库产出 |
|------|------------|------------|
| `train/` | `drones` + `allowlist` | `train.jsonl` `valid.jsonl` + `images/train` `images/valid`（带 assistant JSON） |
| `test_public/`（A 榜） | 只有 `allowlist` | `infer_public.jsonl` + `images/infer`（无 assistant） |
| B 榜目录 | 同样 `*_label.json` | `inferB.jsonl` + `images/inferB` |

```bash
pip install -r requirements-train.txt

# 训练输入
python build_official.py train \
  --official-dir /path/to/train \
  --spec-dir /path/to/mosaic_png \
  --val-ratio 0.2 --seed 42 --size 512

# 推理输入（A 榜）
python build_official.py infer \
  --official-dir /path/to/test_public \
  --spec-dir /path/to/mosaic_png \
  --pack infer --size 512
```

`--official-dir` 可以是数据集根（自动找 `label/`），也可以直接指到 `label/`。  
`--spec-dir` 里 png 名为 `00000000_1a5a96b8.png`，对应 `00000000_1a5a96b8_label.json`。

输入 jsonl 一行（训练多一段 assistant）：

```json
{
  "sample_id": 2,
  "images": ["images/train/00000002_e149d585.png"],
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>..."},
    {"role": "assistant", "content": "{\"drones\":[{\"model_id\":3,\"position_enu\":{\"e_m\":3.4324,\"n_m\":31.0193,\"u_m\":59.4936}}]}"}
  ]
}
```

换 **inferB**：把官方 B 榜目录和对应 mosaic 转进去即可。

```bash
python build_official.py infer \
  --official-dir /path/to/inferB \
  --spec-dir /path/to/inferB_png \
  --pack inferB
```

## 3. 训练（ms-swift）

训练目标就是上面的 **assistant JSON**（ENU 四位小数，带 `position_enu`）。

```bash
export MODEL_PATH=../../pretrain_model/Qwen3-VL-8B-Instruct
export DATASET=./train.jsonl
export VAL_DATASET=./valid.jsonl
export OUTPUT_DIR=./outputs/sft_lora
bash train_sft.sh
tail -f "$(ls -t logs/sft_lora_*.log | head -1)"
```

默认：LoRA r=16 α=32 all-linear、bf16、batch 1×accum 8、lr=1e-4、warmup 0.05、max_length=768、`IMAGE_MAX_TOKEN_NUM=256`、每 20 step log、每 200 step eval/save、最多留 5 个 ckpt。

训完把 `outputs/sft_lora/.../checkpoint-xxxx` 拷到 `lora_ckpt/`。

## 4. 推理 → 提交 jsonl

`run_infer_submit.py` 解析模型 JSON，写成**官方交榜格式**（扁平 `e_m/n_m/u_m`，四位小数）：

```json
{"sample_id":0,"drones":[{"model_id":1,"e_m":2.4766,"n_m":-7.7424,"u_m":59.9733}]}
```

和训练标签是同一套字段，只是交榜不要嵌套 `position_enu`，并带上 `sample_id`。

```bash
pip install -r requirements.txt
export CKPT=./lora_ckpt
export DATASET=./infer_public.jsonl
export OUT=./submits/infer_submit.jsonl
bash infer.sh
```

| 变量 | 默认 | 含义 |
|------|------|------|
| `OUT` | `./submits/infer_submit.jsonl` | **交榜文件** |
| `RAW` | `./submits/infer_raw.jsonl` | 模型原文，排查用 |
| `DATASET` | `./infer_public.jsonl` | 上一步生成的推理 jsonl |
| `CKPT` | 必填 | LoRA 目录 |

B 榜：

```bash
export DATASET=./inferB.jsonl
export OUT=./submits/inferB_submit.jsonl
bash infer.sh
```

## 5. 我们自己的 LoRA（约 166MB）

放到 `lora_ckpt/`（`adapter_config.json` + `adapter_model.safetensors`）。

```
链接：<填写百度网盘分享链接>
提取码：<填写>
文件说明：<填写，例如 pseudo77 checkpoint-1800 LoRA>
```
