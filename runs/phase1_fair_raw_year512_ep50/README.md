# phase1_fair_raw_year512_ep50 (EdgeAttNet)

Fair comparison vs U-Net. Output folder for the Phase 1 run.

- Phase: 1 (fair, no preprocessing)
- Images: raw GONG JPEGs (`/media/data/magfilo_dataset/images/`)
- Split: `splits/year_2011-2019_val2020_test2021-2022/` (train 1412 / val 62 / test 119)
- image_size: 512, epochs: 50, batch_size: 4, lr: 1e-4, optimizer: Adam, AMP: on
- Augmentation: none. Best checkpoint: val Dice. Test: evaluated once at end.

Run: `CUDA_VISIBLE_DEVICES=1 bash model/run_phase1_fair_raw.sh`

Outputs written here: config.json, metrics.csv, test_metrics.json, best_model.pth, last_model.pth
(checkpoints are gitignored; commit metrics/config/test_metrics only).

## Visual comparisons (Phase 1 vs U-Net)

After training, generate side-by-side test images (same input, GT, U-Net, EdgeAttNet):

```bash
cd /media/project/harsh/EdgeAttNet/model
../.venv/bin/python generate_phase1_comparisons.py
```

Outputs in `comparisons/` (identical copy also under `filament/runs/phase1_fair_raw_year512_ep50/comparisons/`):

| Path | Description |
|------|-------------|
| `summary_grid.png` | 8-image slide-ready grid (EdgeAttNet wins + U-Net wins) |
| `per_image/{id}_compare.png` | 4-panel: Input \| GT \| U-Net \| EdgeAttNet |
| `unet/`, `edgeattnet/` | Per-model overlays (GT green, pred red) |
| `manifest.json` | Per-image Dice/IoU for both models |

PNG files are gitignored; `manifest.json` is committed.
