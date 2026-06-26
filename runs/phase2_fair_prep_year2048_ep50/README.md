# phase2_fair_prep_year2048_ep50 (EdgeAttNet)

Fair comparison vs U-Net on **preprocessed** Hα at **2048×2048** (paper-native resolution).

## Setup

- Phase: 2b (fair, preprocessed @ full res)
- Images: `EdgeAttNet/data/processed-H-alpha/`
- Split: `splits/year_2011-2019_val2020_test2021-2022_prep/` (train 1322 / val 18 / test 99)
- image_size: **2048**, epochs: 50, batch_size: **1**, lr: 1e-4, optimizer: Adam, AMP: on
- Augmentation: none. Best checkpoint: val Dice.

Run: `CUDA_VISIBLE_DEVICES=1 bash model/run_phase2_fair_prep_2048.sh`

## Phase 2b comparison (2048×2048, prep-filtered split)

| Model | Best val epoch | Val Dice | Test Dice | Test IoU |
| --- | ---: | ---: | ---: | ---: |
| U-Net | 26 | 0.5718 | **0.6152** | **0.4442** |
| EdgeAttNet (this run) | — | — | — | — |

U-Net complete. EdgeAttNet run in progress.

## Status

**U-Net done. EdgeAttNet in progress.**
