# Qwen3-VL 纯推理

目录必须和训练时一样，基座路径才是 `../../pretrain_model/Qwen3-VL-8B-Instruct`。

```
<本仓库根>/
  pretrain_model/
    Qwen3-VL-8B-Instruct/     # 官方基座，约 16GB，不要 git
  2026wuxiandian/
    qwen3vl_infer/            # 推理脚本（在这里跑）
      run_infer_submit.py
      infer.sh
      lora_ckpt/              # 百度网盘下来的 LoRA
      infer_public.jsonl
      images/
```

clone 后：

```bash
cd 2026wuxiandian/qwen3vl_infer
# 此时 ../../pretrain_model/Qwen3-VL-8B-Instruct 正好指向仓库根下的基座
```

AutoDL 现有布局相同：`<fs>/pretrain_model` 与 `<fs>/2026wuxiandian/qwen3vl_infer` 并列。

## 1. 官方预训练模型

- Hugging Face：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- ModelScope：https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct

在**仓库根**执行：

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
  --local_dir ./pretrain_model/Qwen3-VL-8B-Instruct

# 或 Hugging Face（可用镜像）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir ./pretrain_model/Qwen3-VL-8B-Instruct
```

## 2. 我们自己的 LoRA（约 166MB）

放到 `2026wuxiandian/qwen3vl_infer/lora_ckpt/`（`adapter_config.json` + `adapter_model.safetensors`）。

```
链接：<填写百度网盘分享链接>
提取码：<填写>
文件说明：<填写，例如 pseudo77 checkpoint-1800 LoRA>
```

## 3. 依赖与推理

```bash
cd 2026wuxiandian/qwen3vl_infer
pip install -r requirements.txt

export CKPT=./lora_ckpt
export DATASET=./infer_public.jsonl
bash infer.sh
```

默认 `MODEL_PATH=../../pretrain_model/Qwen3-VL-8B-Instruct`。
输出 `submits/infer_submit.jsonl`（坐标 4 位小数）。
