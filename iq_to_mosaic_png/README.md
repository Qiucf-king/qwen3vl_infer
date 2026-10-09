# IQ → 四节点 2×2 频谱拼图 PNG（训练同款）

本目录脚本用于从官方 `*_iq.npz` 复现训练时使用的频谱输入图。

## 训练时实际流程（必须一致）

```
*_iq.npz
  → 每节点 STFT → dB → 5%/95% 分位归一化 → Viridis 伪彩 → 560×560
  → 2×2 拼成 1120×1120
       node0 左上 | node1 右上
       node2 左下 | node3 右下
  → LANCZOS 缩放到 512×512   ← Qwen3-VL SFT 实际喂入尺寸
  → <stem>.png
```

缺失节点：对应瓦片全黑。

### STFT / 归一化 / 配色 / 频率方向（固定）

| 项 | 值 |
|----|-----|
| IQ 格式 | `iq_node{i}` int16，I/Q 交错；`sr_node{i}` 采样率 |
| STFT | `scipy.signal.spectrogram`，`return_onesided=False` |
| `nperseg` | `min(1024, max(32, len(iq)//100))` |
| `noverlap` | `nperseg // 2` |
| 频率轴 | `np.fft.fftshift`（低频居中） |
| 功率 | `10 * log10(Sxx + 1e-12)` |
| 归一化 | 分位数 clip `[P5, P95]` → `[0,1]` |
| 配色 | OpenCV `COLORMAP_VIRIDIS` |
| 单节点尺寸 | 560×560（`cv2.INTER_LINEAR` resize） |
| 马赛克 | 1120×1120 |
| 最终 PNG | 512×512，`PIL.Image.Resampling.LANCZOS` |

原始实现位置（完整数据集制作）：`../code/make_spec_dataset.py`  
Qwen 包缩放：`../qwen3vl_sft/build_dataset.py`（`--size 512`）

## 依赖

```bash
pip install -r requirements.txt
```

## 运行命令

### 1）单文件（推荐用来生成主办方核对样例）

```bash
python iq_npz_to_mosaic_png.py \
  --npz /path/to/train/iq_sample/00000002_e149d585_iq.npz \
  -o ./sample_00000002_e149d585.png
```

输出：`sample_00000002_e149d585.png`，尺寸 **512×512**。

### 2）按官方 `index.csv` 批量

```bash
# 训练集
python iq_npz_to_mosaic_png.py \
  --dataset-root /path/to/data_and_code_quarter/train \
  --out-dir ./mosaic_png_train \
  --workers 8

# 只生成 1 张做核对
python iq_npz_to_mosaic_png.py \
  --dataset-root /path/to/data_and_code_quarter/train \
  --out-dir ./mosaic_png_sample \
  --limit 1
```

文件名规则：`iq_sample/<stem>_iq.npz` → `<stem>.png`  
例如：`00000002_e149d585_iq.npz` → `00000002_e149d585.png`

### 3）若只需 1120 原分辨率马赛克（不做 512 缩放）

```bash
python iq_npz_to_mosaic_png.py --npz xxx_iq.npz -o out_1120.png --size 1120
```

## 标准样例 PNG

请将本脚本对某一官方 `*_iq.npz` 生成的 PNG（512×512）随本目录一并提交，用于像素级核对。  
建议样例：训练集任意一条四节点齐全的样本（`iq_node0~3` 均为 1）。
