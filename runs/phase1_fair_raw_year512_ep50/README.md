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

Note: `phase1_fair_raw_year2048_ep50/` holds a partial 2048 attempt (cancelled; EdgeAttNet ~18h/epoch).
