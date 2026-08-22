#!/usr/bin/env bash
# train_aqsoldb_chemprop.sh
#
# Trains a real Chemprop molecule-level regression checkpoint predicting
# aqueous solubility (logS, log10 mol/L) from
# scripts/prepare_aqsoldb_training_data.py's output (AqSolDB, ~9,980
# molecules after cleaning). Same architecture convention as every other
# checkpoint in this project's registry (hiddenSize=300, depth=3,
# SCAFFOLD_BALANCED 80/10/10 split), but now with Chemprop's MVE
# (Mean-Variance Estimation) head (-t regression-mve -l mve) instead of
# plain MSE regression -- a single model that outputs both a prediction
# AND its own predictive variance per molecule, giving real (aleatoric)
# per-prediction uncertainty. See js/chemprop-model.js's applyHead().
#
# NOTE: see prepare_aqsoldb_training_data.py's own docstring -- this is
# very likely (strong file-name evidence, not just dataset-name
# similarity) but not CONFIRMED to be the same data the currently-shipped
# model/logs-aqsoldb/ checkpoint was trained on. Trains a fresh
# checkpoint from this same real public source with its own split.
#
# Usage:
#   scripts/train_aqsoldb_chemprop.sh [data_csv] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_CSV="${1:-data/logs-aqsoldb/aqsoldb.csv}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"

mkdir -p "$CKPT_DIR"

echo "=== training logs-aqsoldb (target=logS, regression-mve) ==="
conda run -n cov-chemprop chemprop train \
  --data-path "$DATA_CSV" \
  --smiles-columns smiles \
  --target-columns logS \
  -t regression-mve -l mve \
  --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
  --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 --ffn-num-layers 1 \
  --dropout 0.0 \
  --warmup-epochs 2 --init-lr 0.0001 --max-lr 0.001 --final-lr 0.0001 \
  --batch-size 64 \
  --epochs "$EPOCHS" \
  --output-dir "$CKPT_DIR/logs-aqsoldb"

echo "done -- checkpoint in $CKPT_DIR/logs-aqsoldb/"
