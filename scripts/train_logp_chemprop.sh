#!/usr/bin/env bash
# train_logp_chemprop.sh
#
# Trains a real Chemprop molecule-level regression checkpoint predicting
# logP (octanol-water partition coefficient) from
# scripts/prepare_logp_training_data.py's output (EPA/NIEHS OPERA's own
# curated LogP training set, ~3,550 molecules after cleaning). Same
# architecture convention as every other checkpoint in this project's
# registry (hiddenSize=300, depth=3, SCAFFOLD_BALANCED 80/10/10 split),
# but with Chemprop's MVE (Mean-Variance Estimation) head
# (-t regression-mve -l mve) instead of plain MSE regression -- a single
# model that outputs both a prediction AND its own predictive variance
# per molecule, giving real (aleatoric) per-prediction uncertainty rather
# than just a bare number. See js/chemprop-model.js's applyHead() for how
# the browser-side inference reads this back out.
#
# Usage:
#   scripts/train_logp_chemprop.sh [data_csv] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_CSV="${1:-data/logp-opera/logp_opera.csv}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"

mkdir -p "$CKPT_DIR"

echo "=== training logp-v1 (target=logP, regression-mve) ==="
conda run -n cov-chemprop chemprop train \
  --data-path "$DATA_CSV" \
  --smiles-columns smiles \
  --target-columns logP \
  -t regression-mve -l mve \
  --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
  --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 --ffn-num-layers 1 \
  --dropout 0.0 \
  --warmup-epochs 2 --init-lr 0.0001 --max-lr 0.001 --final-lr 0.0001 \
  --batch-size 64 \
  --epochs "$EPOCHS" \
  --output-dir "$CKPT_DIR/logp-v1"

echo "done -- checkpoint in $CKPT_DIR/logp-v1/"
