# phase0_legacy_raw_year512_ep50 (ARCHIVE — EdgeAttNet)

Original EdgeAttNet run, kept for reference. Not comparable to UNet baseline.

- Phase: 0 (legacy)
- Images: raw GONG JPEGs (`/media/data/magfilo_dataset/images/`)
- Preprocessing: none
- Split: year-based (train 2011-2019, val 2020, test 2021-2022), ~1412/62/119
- image_size: 512
- epochs: 50, batch_size: 4, lr: 1e-4, optimizer: Adam
- Test (held-out): Dice 0.6105, IoU 0.4394 (last-epoch weights)
- Files: config.json, metrics.csv, test_metrics.json, best_model.pth, last_model.pth, visualizations/
- Was located at: EdgeAttNet/models/ before restructure
