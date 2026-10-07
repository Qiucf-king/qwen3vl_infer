# LoRA 权重目录

把百度网盘的 `ckp_best.zip` 解压到本目录，得到：

```
lora_ckpt/
  ckp_best/
    adapter_config.json
    adapter_model.safetensors
```

推理：

```bash
export CKPT=./lora_ckpt/ckp_best
bash infer.sh
```
