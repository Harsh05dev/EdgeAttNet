# EdgeAttNet Setup — Harsh's Working Notes

## Overview

This repo was extended from a demonstration stub into a **runnable EdgeAttNet pipeline** for solar filament segmentation on the MAGFILO dataset.

### What was broken before
- `edgeattnet_model.py` was an eval stub (no actual model class)
- `main.py` had broken imports
- `utils.py` was incomplete (syntax error + missing metric functions)
- No training script existed
- `data_loader.py` expected PNG masks + nested image folders

### What works now
- COCO JSON + flat image directory loading
- Full `UNetEdgeTransformer` with EG-MHSA bottleneck
- Training (`train.py`) and evaluation (`main.py`)
- Paper-aligned metrics (pairwise mIoU, multiscale IoU)

---

## Data paths

| Asset | Path |
|-------|------|
| COCO JSON | `/media/data/magfilo_dataset/magfilo_2024_v1.0.json` |
| Images | `/media/data/magfilo_dataset/images/` |
| Checkpoints | `EdgeAttNet/models/best_model.pth` |

Images use GONG filenames (e.g. `20110109104734Ch.jpeg`).  
Masks are rasterized on-the-fly from COCO polygon annotations.

---

## Year-based split (default)

| Split | Years |
|-------|-------|
| Train | 2011–2019 |
| Val   | 2020 |
| Test  | 2021–2022 |

---

## File changes

| File | Purpose |
|------|---------|
| `model/data_loader.py` | `CocoFilamentDataset`, COCO mask rasterization, year splits |
| `model/edgeattnet_model.py` | `UNetEdgeTransformer`, `EGMHSA`, edge branch |
| `model/train.py` | End-to-end training (BCE + Dice, Adam lr=1e-4) |
| `model/utils.py` | Pairwise/multiscale IoU, prediction collection |
| `model/main.py` | Load checkpoint and run evaluation |
| `model/evaluation.py` | Clean metric imports |
| `model/generate_plots.py` | Fixed utils imports, removed auto-run example |

---

## Environment

```bash
cd /media/project/harsh/EdgeAttNet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use `../.venv/bin/python` from the `model/` directory if not activating the venv.

---

## Results vs paper

Source: [EdgeAttNet paper](https://arxiv.org/abs/2509.02964) (MAGFILO test split).  
Our run: 50 epochs, CUDA, lr `1e-4`, image size `512`, batch size `4`, checkpoint `models/best_model.pth` (best val Dice at epoch 28).

### Setup comparison

| | Paper (EdgeAttNet) | This run |
|--|-------------------|----------|
| Train / Val / Test | 1295 / 45 / 99 | 1412 / 62 / 119 |
| Split strategy | Paper-defined holdout | Year-based (train 2011–2019, val 2020, test 2021–2022) |
| Input images | Preprocessed Hα (limb correction, CLAHE, etc.) | Raw GONG JPEGs |
| Trainable parameters | 22,658,891 | 13,849,483 |
| Epochs | 50 | 50 |
| Learning rate | 1e-4 | 1e-4 |
| Data augmentation | None | None |

### Segmentation metrics (paper-aligned)

Metrics reported in the paper on the **99-image test split**:

| Metric | Paper | This run | Δ (ours − paper) |
|--------|------:|---------:|-----------------:|
| Pairwise mIoU | 0.6451 | 0.5832 | −0.0619 |
| Multiscale mIoU | 0.7032 | 0.8461 | +0.1429 |

Our pairwise / multiscale numbers come from `main.py` on **all 1593 images** (years 2011–2022), not the paper's 99-image test set — see bullets below.

### Pixel-level metrics (training script)

From `train.py` — standard per-pixel Dice / IoU, **not** reported in the paper:

| Split | Images | Dice | IoU |
|-------|-------:|-----:|----:|
| Train (epoch 50) | 1412 | 0.7881 | 0.6503 |
| Val (epoch 50) | 62 | 0.4805 | 0.3162 |
| Test (held-out) | 119 | 0.6105 | 0.4394 |
| Best val (epoch 28) | 62 | 0.4853 | — |

### Other metrics (`main.py`, full dataset)

| Metric | This run | Paper |
|--------|---------:|-------|
| Average Precision (AP) | 0.7930 | — |
| Average Recall | 6.5882 | — |

Average Recall > 1 indicates a bug in our metric code; do not use it for comparison.

### Why results differ from the paper

- **Different train/val/test split** — we use year-based splits (1412/62/119); the paper uses 1295/45/99 with a specific holdout list we do not have.
- **Different evaluation set for mIoU** — paper metrics are on 99 test images; our `main.py` evaluates on all 1593 annotated images, so pairwise / multiscale numbers are not directly comparable.
- **No preprocessing pipeline** — the paper trains on limb-corrected, disk-masked, CLAHE-enhanced FITS-derived images; we use raw GONG JPEGs from the dataset download.
- **Reimplemented architecture** — our model has ~13.8M parameters vs ~22.7M in the paper; the released repo did not include the official `UNetEdgeTransformer` weights or full architecture.
- **Different metrics mixed together** — paper Table II uses pairwise / multiscale mIoU; our `train.py` test Dice (0.61) / IoU (0.44) are pixel-overlap metrics and should not be compared to paper mIoU values.
- **Metric implementation gaps** — Average Recall in our `utils.py` is likely incorrect; multiscale IoU being higher than the paper may partly reflect implementation or evaluation-scope differences, not necessarily better segmentation.
- **No official pretrained checkpoint** — we trained from scratch; the paper results used their own trained weights on preprocessed data.

### Commands used

**Train:**
```bash
cd /media/project/harsh/EdgeAttNet/model
../.venv/bin/python train.py \
  --coco-json /media/data/magfilo_dataset/magfilo_2024_v1.0.json \
  --image-dir /media/data/magfilo_dataset/images \
  --out-dir ../models \
  --epochs 50 \
  --batch-size 4 \
  --lr 1e-4 \
  --image-size 512 \
  --num-workers 4
```

**Evaluate (full dataset, ~8 min):**
```bash
cd /media/project/harsh/EdgeAttNet/model
../.venv/bin/python main.py \
  --model-path ../models/best_model.pth \
  --batch-size 4
```

---

## Visualize a single prediction

Run from the `model/` directory (venv is at `EdgeAttNet/.venv`, not one level up from the repo root):

```bash
cd /media/project/harsh/EdgeAttNet/model
../.venv/bin/python main.py \
  --model-path ../models/best_model.pth \
  --visualize-id 040401-20220714185352Th \
  --skip-eval
```

**From repo root (alternative):**
```bash
cd /media/project/harsh/EdgeAttNet
.venv/bin/python model/main.py \
  --model-path models/best_model.pth \
  --visualize-id 040401-20220714185352Th \
  --skip-eval
```

`--visualize-id` must be a COCO image `id` (e.g. `040401-20220714185352Th`), not the GONG filename alone.

PNG saved to `models/visualizations/<visualize-id>.png` by default. Use `--save-viz path/to/out.png` for a custom path.

---

## Output files

```text
models/
  best_model.pth      # best val dice (epoch 28)
  last_model.pth      # final epoch
  metrics.csv         # per-epoch train/val metrics
  config.json         # training config
  test_metrics.json   # test split dice/iou after training
  visualizations/     # saved PNG overlays from --visualize-id
```

---

## Quick reference

| Task | Command |
|------|---------|
| Train | `cd model && ../.venv/bin/python train.py ...` |
| Evaluate | `cd model && ../.venv/bin/python main.py --model-path ../models/best_model.pth` |
| Visualize | `cd model && ../.venv/bin/python main.py --model-path ../models/best_model.pth --visualize-id <COCO_ID>` |

---

## Known limitations

1. Paper preprocessing pipeline (limb darkening, CLAHE, etc.) not implemented — uses raw GONG JPEGs
2. Paper's exact train/val/test split (1295/45/99) not used — year-based split instead
3. Do not commit `models/` checkpoints or `/media/data/magfilo_dataset/` to git
