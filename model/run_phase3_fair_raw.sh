#!/usr/bin/env bash
# Phase 3 fair comparison - EdgeAttNet on RAW GONG JPEGs.
# Larger val set: 2018+2019+2020 (193 images). Train 2011-2017, test 2021-2022.
# Same split / settings as U-Net phase 3.
#
# Run on GPU 1 while U-Net uses GPU 0:
#   CUDA_VISIBLE_DEVICES=1 bash model/run_phase3_fair_raw.sh
set -euo pipefail
cd "$(dirname "$0")"

SPLIT=../splits/year_2011-2017_val2018-2020_test2021-2022
IMAGE_DIR=/media/data/magfilo_dataset/images
OUT=../runs/phase3_fair_raw_year512_ep50
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
