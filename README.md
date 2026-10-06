# Qwen3-VL 纯推理

相对路径。图片用 `images/...`，基座默认 `../../pretrain_model/Qwen3-VL-8B-Instruct`。

## 1. 官方预训练模型（Qwen3-VL-8B-Instruct）

约 16GB 级，不要放进 git。放到与本仓库平级的 `pretrain_model/`：

```
<pretrain_model>/Qwen3-VL-8B-Instruct
<本仓库>/qwen3vl_infer
```

**Hugging Face**

- 页面：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

```bash
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir ../../pretrain_model/Qwen3-VL-8B-Instruct
```

国内可加镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir ../../pretrain_model/Qwen3-VL-8B-Instruct
```

**ModelScope（国内通常更稳）**

- 页面：https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
  --local_dir ../../pretrain_model/Qwen3-VL-8B-Instruct
```

下完后目录里应有 `config.json`、`model.safetensors` / 分片权重、`tokenizer` 等。

## 2. 我们自己的 LoRA 权重（约 166MB）

不进 git，用百度网盘。下载后放到例如 `./lora_ckpt/`（需含 `adapter_config.json` 和 `adapter_model.safetensors`）。

```
链接：<填写百度网盘分享链接>
提取码：<填写>
文件说明：<填写，例如 pseudo77 checkpoint-1800 LoRA>
```

## 3. 依赖

```bash
pip install -r requirements.txt
```

## 4. 跑推理

把 `infer_public.jsonl` 和 `images/` 放在本目录（或改环境变量）。

```bash
export MODEL_PATH=../../pretrain_model/Qwen3-VL-8B-Instruct
export CKPT=./lora_ckpt
export DATASET=./infer_public.jsonl
bash infer.sh
```

可选：

```bash
export OUT=./submits/infer_submit.jsonl
export LIMIT=20          # 只跑前 N 条
```

输出：`submits/infer_submit.jsonl`（`sample_id` + `drones[{model_id,e_m,n_m,u_m}]`，坐标 4 位小数）。

## 5. 推到 git（不含权重）

远程 `git账号密码.txt` 两行用户名/密码，第三行写仓库地址；或改 `push_to_git.sh` 顶部 `GIT_URL`：

```bash
cd /root/autodl-fs/2026wuxiandian/qwen3vl_infer
bash push_to_git.sh
```

`git账号密码.txt` 不要提交进仓库。GitHub 请用 Token。
