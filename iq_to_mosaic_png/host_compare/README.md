# host_compare

本目录用于对比「本地新脚本生成」与「线上远端已有」的 mosaic PNG。

## 目录说明

| 目录 | 含义 |
|------|------|
| `local_new/` | 由 `iq_npz_to_mosaic_png.py` 从 infer npz 本地生成的 PNG |
| `remote_online/` | 从 SSH 远端同路径下载的同名 PNG（`/root/autodl-fs/2026wuxiandian/qwen3vl_sft/images/infer`） |

两侧文件名完全一致，便于逐张对照。

## 校验结果

以下 3 个 stem 经 SHA256 比对，均为 **EXACT_MATCH**（本地与远端字节级一致）：

- `00000002_c056455a.png`
- `00000000_1a5a96b8.png`
- `00000004_249d4493.png`

文件来源：已有 `compare_out/`（本地 PNG 在根目录，远端副本在 `compare_out/remote/`），复制到本目录对应子文件夹。
