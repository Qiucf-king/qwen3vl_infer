# images 目录

相对 `2026wuxiandian/qwen3vl_infer/`。jsonl 里的路径必须和这里一致。

## A 榜（当前）

```
images/infer/<stem>.png
```

对应 jsonl：`"images": ["images/infer/<stem>.png"]`

## B 榜（换数据集）

```
images/inferB/<stem>.png
```

jsonl 改成 `"images": ["images/inferB/<stem>.png"]`，然后：

```bash
export DATASET=./inferB.jsonl
export OUT=./submits/inferB_submit.jsonl
bash infer.sh
```

## 训练图（build_sft_jsonl.py 生成）

```
images/train/<stem>.png
images/valid/<stem>.png
```

对应 jsonl：`"images": ["images/train/<stem>.png"]`

每张图是四节点 2x2 mosaic 频谱图。不要用绝对路径。
