# images 目录

相对 `2026wuxiandian/qwen3vl_infer/`。由 `build_official.py` 从官方 `label/*_label.json` + 同名 mosaic png 生成。jsonl 的 `images` 必须和这里一致。

```
images/train/<stem>.png     # 官方 train 划分出的训练集
images/valid/<stem>.png
images/infer/<stem>.png     # 官方 A 榜 test_public
images/inferB/<stem>.png    # B 榜（换数据集时 --pack inferB）
```

png 文件名 = label 去 `_label.json` 的 stem，例如：

- `train/label/00000002_e149d585_label.json`
- `images/train/00000002_e149d585.png`

每张图是四节点 2x2 mosaic。不要用绝对路径。
