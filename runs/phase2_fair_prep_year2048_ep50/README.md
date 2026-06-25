# phase2_fair_prep_year2048_ep50 (EdgeAttNet) - RESERVED

Reserved for the preprocessing rerun. Do NOT run yet.

- Phase: 2 (fair, WITH preprocessing)
- Same split / 2048 / 50 epochs / Adam lr=1e-4 as Phase 1 - only the images change.
- Preprocessed images already on disk: `EdgeAttNet/data/processed-H-alpha/`
  (layout: year/month/day/<coco-id>.jpg). Phase 2 will point `--image-dir` there
  after confirming filename-to-COCO-id resolution.

Blocked on: Phase 1 results review by mentor.
