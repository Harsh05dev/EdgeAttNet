# phase1_fair_raw_year2048_ep50 (EdgeAttNet) — PARTIAL / CANCELLED

Aborted 2048×50 attempt. Phase 1 fair comparison was completed at **512** in `phase1_fair_raw_year512_ep50/`.

## What was attempted

- Phase: 1 (fair, no preprocessing)
- Images: raw GONG JPEGs (`/media/data/magfilo_dataset/images/`)
- Split: `splits/year_2011-2019_val2020_test2021-2022/` (train 1412 / val 62 / test 119)
- image_size: **2048**, epochs: 50, batch_size: 1, lr: 1e-4, optimizer: Adam, AMP: on
- Stopped early: EdgeAttNet ~18 h/epoch at 2048 (attention OOM at batch 2); U-Net partial run stopped at epoch 5

## Why 512 was used instead

| | 2048 | 512 |
|--|------|-----|
| EdgeAttNet | ~18 h/epoch, batch 1 | ~22 min/epoch, batch 4 |
| U-Net | ~5 h total | ~15 min total |
| Fair match | Yes, but impractical on 32 GB GPUs | **Chosen** — matched both models |

## Outputs (if any)

Partial `config.json` / `metrics.csv` may exist from cancelled runs. No final `test_metrics.json` for this folder.
