#!/usr/bin/env bash
# train_melting_point_chemprop.sh
#
# Trains a real Chemprop molecule-level regression checkpoint predicting
# melting point (Kelvin) from scripts/prepare_melting_point_training_data.py's
# output (the Bradley Double Plus Good dataset, ~3,022 molecules after
# cleaning). Same architecture convention as every other checkpoint in
# this project's registry (hiddenSize=300, depth=3, SCAFFOLD_BALANCED
# 80/10/10 split).
#
# NOTE: see prepare_melting_point_training_data.py's own docstring --
# this trains a NEW checkpoint from this real public source with a fresh
# split, not a reproduction of the currently-shipped model/melting-point/
# checkpoint's exact original training data (whose curation/split isn't
# fully known -- see that model's registry.json notes).
#
# Usage:
#   scripts/train_melting_point_chemprop.sh [data_csv] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_CSV="${1:-data/melting-point/melting_point.csv}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"

mkdir -p "$CKPT_DIR"

echo "=== training melting-point (target=mp, regression, Kelvin) ==="
conda run -n cov-chemprop chemprop train \
  --data-path "$DATA_CSV" \
  --smiles-columns smiles \
  --target-columns mp \
  --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
  --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 --ffn-num-layers 1 \
  --dropout 0.0 \
  --warmup-epochs 2 --init-lr 0.0001 --max-lr 0.001 --final-lr 0.0001 \
  --batch-size 64 \
  --epochs "$EPOCHS" \
  --output-dir "$CKPT_DIR/melting-point"

echo "done -- checkpoint in $CKPT_DIR/melting-point/"
