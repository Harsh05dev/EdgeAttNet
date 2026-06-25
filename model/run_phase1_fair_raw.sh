#!/usr/bin/env bash
# Phase 1 fair comparison - EdgeAttNet on RAW GONG JPEGs.
# Same split / epochs / resolution as U-Net. No preprocessing.
#
# EdgeAttNet attention at 2048 needs batch_size=1 on 32GB GPUs (OOM at batch 2).
# Run on GPU 1 while U-Net uses GPU 0:
#   CUDA_VISIBLE_DEVICES=1 bash model/run_phase1_fair_raw.sh
set -euo pipefail
cd "$(dirname "$0")"

SPLIT=../splits/year_2011-2019_val2020_test2021-2022
IMAGE_DIR=/media/data/magfilo_dataset/images
OUT=../runs/phase1_fair_raw_year2048_ep50
PY=../.venv/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" train.py \
  --train-json "$SPLIT/train.json" \
  --val-json   "$SPLIT/val.json" \
  --test-json  "$SPLIT/test.json" \
  --image-dir  "$IMAGE_DIR" \
  --out-dir    "$OUT" \
  --image-size 2048 \
  --epochs 50 \
  --batch-size 1 \
  --lr 1e-4 \
  --num-workers 4 \
  --seed 42 \
  --amp
