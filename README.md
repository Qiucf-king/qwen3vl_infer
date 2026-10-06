# Qwen3-VL 纯推理

目录必须和训练时一样，基座路径才是 `../../pretrain_model/Qwen3-VL-8B-Instruct`。

```
<本仓库根>/
  pretrain_model/
    Qwen3-VL-8B-Instruct/          # 官方基座，约 16GB，不要 git
  2026wuxiandian/
    qwen3vl_infer/                 # 在这里跑推理
      run_infer_submit.py          # 推理 + 解析成提交 jsonl
      infer.sh
      requirements.txt
      lora_ckpt/                   # 百度网盘 LoRA
      infer_public.jsonl           # 当前 A 榜；换 B 榜换成对应 jsonl
      images/
        infer/                     # A 榜图；B 榜见下文
          00000000_xxxxxxxx.png
      submits/
        infer_submit.jsonl         # 提交文件（脚本自动写出）
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

## 2. 我们自己的 LoRA（约 166MB）

放到 `2026wuxiandian/qwen3vl_infer/lora_ckpt/`（`adapter_config.json` + `adapter_model.safetensors`）。

```
链接：<填写百度网盘分享链接>
提取码：<填写>
文件说明：<填写，例如 pseudo77 checkpoint-1800 LoRA>
```

## 3. 图片目录（换 inferB 时改这里）

脚本从 **jsonl 的 `images` 字段** 找图，路径相对 `2026wuxiandian/qwen3vl_infer/`。

当前 A 榜：

```
images/infer/00000000_1a5a96b8.png
images/infer/00000001_3b3c7a43.png
...
```

jsonl 一行示例：

```json
{
  "sample_id": 0,
  "images": ["images/infer/00000000_1a5a96b8.png"],
  "messages": [{"role":"system","content":"..."},{"role":"user","content":"<image>..."}]
}
```

换 **inferB** 时保持同一约定即可，例如：

```
images/inferB/*.png
inferB.jsonl          # 里面 images 写成 ["images/inferB/xxxx.png"]
```

然后：

```bash
export DATASET=./inferB.jsonl
export OUT=./submits/inferB_submit.jsonl
bash infer.sh
```

规则：

- 一张样本一张 2x2 mosaic 频谱图（png）
- jsonl 的 `images[0]` 必须能相对本目录打开
- 子目录名可换（`infer` / `inferB`），但 jsonl 路径要和磁盘一致
- 不要用绝对路径，便于换机器

## 4. 提交文件（已在 py / sh 里写出）

`run_infer_submit.py` 会解析模型 JSON，并写成官方提交 jsonl（每行一个样本，ENU 保留 4 位小数）：

```json
{"sample_id":0,"drones":[{"model_id":1,"e_m":2.4766,"n_m":-7.7424,"u_m":59.9733}]}
```

`infer.sh` 默认：

| 变量 | 默认 | 含义 |
|------|------|------|
| `OUT` | `./submits/infer_submit.jsonl` | **提交文件** |
| `RAW` | `./submits/infer_raw.jsonl` | 原始模型文本（排查解析失败） |
| `DATASET` | `./infer_public.jsonl` | 待推理列表 |
| `CKPT` | 必填 | LoRA 目录 |

交榜用 `OUT` 那个 jsonl，不要交 `RAW`。

## 5. 跑推理

```bash
cd 2026wuxiandian/qwen3vl_infer
pip install -r requirements.txt

export CKPT=./lora_ckpt
export DATASET=./infer_public.jsonl
export OUT=./submits/infer_submit.jsonl
bash infer.sh
```
