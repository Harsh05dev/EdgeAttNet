# phase1_fair_raw_year512_ep50 (EdgeAttNet)

Fair comparison vs U-Net. Phase 1 run on raw GONG JPEGs — **complete (50/50 epochs)**.

## Setup

- Phase: 1 (fair, no preprocessing)
- Images: raw GONG JPEGs (`/media/data/magfilo_dataset/images/`)
- Split: `splits/year_2011-2019_val2020_test2021-2022/` (train 1412 / val 62 / test 119)
- image_size: **512**, epochs: 50, batch_size: 4, lr: 1e-4, optimizer: Adam, AMP: on
- Augmentation: none. Best checkpoint: val Dice. Test: evaluated once at end on held-out test split.

Run: `CUDA_VISIBLE_DEVICES=1 bash model/run_phase1_fair_raw.sh`

## Results

| Metric | Best val (epoch 47) | Final test (best ckpt) |
| --- | ---: | ---: |
| Dice | 0.5001 | **0.6265** |
| IoU | 0.3335 | **0.4561** |

## Phase 1 comparison (512×512, same split)

| Model | Best val epoch | Val Dice | Test Dice | Test IoU |
| --- | ---: | ---: | ---: | ---: |
| EdgeAttNet (this run) | 47 | 0.5001 | **0.6265** | **0.4561** |
| U-Net | 17 | **0.5527** | 0.6015 | 0.4301 |

EdgeAttNet reaches higher test Dice/IoU; U-Net reaches higher validation Dice on this phase.

## Outputs

| File | Description |
| --- | --- |
| `config.json` | Training hyperparameters |
| `metrics.csv` | Per-epoch train + val metrics |
| `test_metrics.json` | Final held-out test evaluation |
| `best_model.pth` / `last_model.pth` | Checkpoints (gitignored) |
| `comparison.png` | Slide viz: Input \| GT \| prediction for `040401-20220714185352Th` (gitignored) |

Regenerate slide viz:

```bash
cd /media/project/harsh/EdgeAttNet/model
../.venv/bin/python main.py \
  --model-path ../runs/phase1_fair_raw_year512_ep50/best_model.pth \
  --image-size 512 \
  --visualize-id 040401-20220714185352Th \
  --skip-eval \
  --save-viz ../runs/phase1_fair_raw_year512_ep50/comparison.png
```
