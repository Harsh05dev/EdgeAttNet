# phase3_fair_raw_year512_ep50 (EdgeAttNet)

Fair comparison vs U-Net. Phase 3 run on raw GONG JPEGs with a larger validation set — **complete (50/50 epochs)**.

## Setup

- Phase: 3 (fair, no preprocessing, expanded val years)
- Images: raw GONG JPEGs (`/media/data/magfilo_dataset/images/`)
- Split: `splits/year_2011-2017_val2018-2020_test2021-2022/` (train 1281 / val 193 / test 119)
- image_size: **512**, epochs: 50, batch_size: 4, lr: 1e-4, optimizer: Adam, AMP: on
- Augmentation: none. Best checkpoint: val Dice. Test: evaluated once at end on held-out test split.

Run: `CUDA_VISIBLE_DEVICES=1 bash model/run_phase3_fair_raw.sh`

## Results

| Metric | Best val (epoch 25) | Final test (best ckpt) |
| --- | ---: | ---: |
| Dice | 0.6218 | **0.6113** |
| IoU | 0.4512 | **0.4402** |

## Phase 3 comparison (512×512, expanded val split)

| Model | Best val epoch | Val Dice | Test Dice | Test IoU |
| --- | ---: | ---: | ---: | ---: |
| EdgeAttNet (this run) | 25 | **0.6218** | **0.6113** | **0.4402** |
| U-Net | 32 | 0.6108 | 0.6009 | 0.4295 |

EdgeAttNet reaches higher val and test Dice/IoU with the expanded 2018–2020 validation split.

## Outputs

| File | Description |
| --- | --- |
| `config.json` | Training hyperparameters |
| `metrics.csv` | Per-epoch train + val metrics |
| `test_metrics.json` | Final held-out test evaluation |
| `best_model.pth` / `last_model.pth` | Checkpoints (gitignored) |
| `comparison.png` | Slide viz: Input \| GT \| prediction for `040401-20220714185352Th` (same test image as Phase 1) |

Regenerate slide viz:

```bash
cd /media/project/harsh/EdgeAttNet/model
../.venv/bin/python main.py \
  --model-path ../runs/phase3_fair_raw_year512_ep50/best_model.pth \
  --image-size 512 \
  --visualize-id 040401-20220714185352Th \
  --skip-eval \
  --save-viz ../runs/phase3_fair_raw_year512_ep50/comparison.png
```
