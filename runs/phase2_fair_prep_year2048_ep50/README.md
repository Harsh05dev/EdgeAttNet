# phase2_fair_prep_year2048_ep50 (EdgeAttNet)

Fair comparison vs U-Net on **preprocessed** Hα at **2048×2048** (paper-native resolution) — **complete (50/50 epochs)**.

## Setup

- Phase: 2b (fair, preprocessed @ full res)
- Images: `EdgeAttNet/data/processed-H-alpha/`
- Split: `splits/year_2011-2019_val2020_test2021-2022_prep/` (train 1322 / val 18 / test 99)
- image_size: **2048**, epochs: 50, batch_size: **1**, lr: 1e-4, optimizer: Adam, AMP: on
- Augmentation: none. Best checkpoint: val Dice. Test: evaluated on held-out test split (batch_size 1; batch 4 OOMs at 2048).

Run: `CUDA_VISIBLE_DEVICES=1 bash model/run_phase2_fair_prep_2048.sh`

Wall-clock: ~15 min/epoch (~12–13 h total for 50 epochs).

## Results

| Metric | Best val (epoch 5) | Final test (best ckpt) |
| --- | ---: | ---: |
| Dice | 0.5297 | 0.0396 |
| IoU | 0.3603 | 0.0202 |

Val metrics are noisy (only 18 val images). Best checkpoint was saved at epoch 5; training continued but val/test performance did not recover. Test eval at 2048 requires `batch_size=1` (attention OOM at batch 4).

## Phase 2b comparison (2048×2048, prep-filtered split)

| Model | Best val epoch | Val Dice | Test Dice | Test IoU |
| --- | ---: | ---: | ---: | ---: |
| U-Net | 26 | 0.5718 | **0.6152** | **0.4442** |
| EdgeAttNet (this run) | 5 | 0.5297 | 0.0396 | 0.0202 |

U-Net clearly outperforms EdgeAttNet at native 2048 resolution on this split. EdgeAttNet @ 512 (Phase 2) reached test Dice 0.6209 — full-res training did not transfer.

## Phase 2 resolution comparison (EdgeAttNet, preprocessed)

| Resolution | Test Dice | Test IoU |
| --- | ---: | ---: |
| 512 | **0.6209** | **0.4502** |
| 2048 | 0.0396 | 0.0202 |

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
../.venv/bin/python -c "
from pathlib import Path
import torch
from data_loader import CocoFilamentDataset, ImageMaskTransform
from edgeattnet_model import UNetEdgeTransformer
from visualize import visualize_prediction_by_filename

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = CocoFilamentDataset(
    '../splits/year_2011-2019_val2020_test2021-2022_prep/test.json',
    '../data/processed-H-alpha',
    transform=ImageMaskTransform((2048, 2048)),
)
model = UNetEdgeTransformer().to(device)
model.load_state_dict(torch.load('../runs/phase2_fair_prep_year2048_ep50/best_model.pth', map_location=device, weights_only=True))
model.eval()
visualize_prediction_by_filename(
    model, ds, device, '040401-20220714185352Th',
    save_path=Path('../runs/phase2_fair_prep_year2048_ep50/comparison.png'),
)
"
```
