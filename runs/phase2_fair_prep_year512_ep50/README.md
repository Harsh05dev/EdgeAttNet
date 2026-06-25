# phase2_fair_prep_year512_ep50 (EdgeAttNet)

Fair comparison vs U-Net on **preprocessed** H-alpha images at **512×512** — **complete (50/50 epochs)**.

## Setup

- Phase: 2 (fair, paper preprocessing)
- Images: `EdgeAttNet/data/processed-H-alpha/` (authors' pipeline output; not copied)
- Split: `splits/year_2011-2019_val2020_test2021-2022_prep/` (train 1322 / val 18 / test 99)
- image_size: **512**, epochs: 50, batch_size: 4, lr: 1e-4, optimizer: Adam, AMP: on
- Augmentation: none. Best checkpoint: val Dice. Test: evaluated once at end on held-out test split.

Run: `CUDA_VISIBLE_DEVICES=1 bash model/run_phase2_fair_prep.sh`

## Results

| Metric | Best val (epoch 35) | Final test (best ckpt) |
| --- | ---: | ---: |
| Dice | 0.6325 | **0.6209** |
| IoU | 0.4625 | **0.4502** |

## Phase 2 comparison (512×512, prep-filtered split)

| Model | Best val epoch | Val Dice | Test Dice | Test IoU |
| --- | ---: | ---: | ---: | ---: |
| EdgeAttNet (this run) | 35 | **0.6325** | **0.6209** | **0.4502** |
| U-Net | 15 | 0.6277 | 0.5990 | 0.4275 |

EdgeAttNet reaches higher val and test Dice/IoU on preprocessed inputs for this phase.

## Outputs

| File | Description |
| --- | --- |
| `config.json` | Training hyperparameters |
| `metrics.csv` | Per-epoch train + val metrics |
| `test_metrics.json` | Final held-out test evaluation |
| `best_model.pth` / `last_model.pth` | Checkpoints (gitignored) |
| `comparison.png` | Slide viz: Input \| GT \| prediction for `040401-20220714185352Th` (gitignored, same test image as Phase 1) |

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
    transform=ImageMaskTransform((512, 512)),
)
model = UNetEdgeTransformer().to(device)
model.load_state_dict(torch.load('../runs/phase2_fair_prep_year512_ep50/best_model.pth', map_location=device))
model.eval()
visualize_prediction_by_filename(
    model, ds, device, '040401-20220714185352Th',
    save_path=Path('../runs/phase2_fair_prep_year512_ep50/comparison.png'),
)
"
```
