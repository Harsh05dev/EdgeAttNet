#!/usr/bin/env bash
# Phase 2 fair comparison - EdgeAttNet on preprocessed H-alpha (paper pipeline output).
# Same year split / 512 / 50 epochs as Phase 1; only images change.
#
# Run on GPU 1 while U-Net uses GPU 0:
#   CUDA_VISIBLE_DEVICES=1 bash model/run_phase2_fair_prep.sh
set -euo pipefail
cd "$(dirname "$0")"

SPLIT=../splits/year_2011-2019_val2020_test2021-2022_prep
IMAGE_DIR=../data/processed-H-alpha
OUT=../runs/phase2_fair_prep_year512_ep50
PY=../.venv/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" train.py \
  --train-json "$SPLIT/train.json" \
  --val-json   "$SPLIT/val.json" \
  --test-json  "$SPLIT/test.json" \
  --image-dir  "$IMAGE_DIR" \
  --out-dir    "$OUT" \
  --image-size 512 \
  --epochs 50 \
  --batch-size 4 \
  --lr 1e-4 \
  --num-workers 4 \
  --seed 42 \
  --amp
