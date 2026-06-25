# phase2_fair_prep_year512_ep50 (EdgeAttNet)

Fair comparison vs U-Net on **preprocessed** H-alpha images at **512×512**.

## Setup

- Phase: 2 (fair, paper preprocessing)
- Images: `EdgeAttNet/data/processed-H-alpha/` (authors' pipeline output; not copied)
- Split: `splits/year_2011-2019_val2020_test2021-2022_prep/` — year split filtered to images present in processed data (train 1322 / val 18 / test 99)
- image_size: **512**, epochs: 50, batch_size: 4, lr: 1e-4, optimizer: Adam, AMP: on
- Augmentation: none. Best checkpoint: val Dice. Test: evaluated once at end.

Same hyperparameters as Phase 1; only the image source changes (preprocessed vs raw).

## Run

```bash
cd /media/project/harsh/EdgeAttNet
CUDA_VISIBLE_DEVICES=1 bash model/run_phase2_fair_prep.sh
```

Run U-Net in parallel on GPU 0: `cd filament && CUDA_VISIBLE_DEVICES=0 bash run_phase2_fair_prep.sh`

## Phase 2 comparison (512×512, prep-filtered split)

| Model | Best val epoch | Val Dice | Test Dice | Test IoU |
| --- | ---: | ---: | ---: | ---: |
| U-Net | 15 | **0.6277** | **0.5990** | **0.4275** |
| EdgeAttNet (this run) | — | — | — | — |

U-Net complete. EdgeAttNet run pending.

## Status

**U-Net done. EdgeAttNet not started yet.**
