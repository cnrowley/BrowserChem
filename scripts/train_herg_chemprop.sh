#!/usr/bin/env bash
# train_herg_chemprop.sh
#
# Trains a real Chemprop molecule-level BINARY CLASSIFICATION checkpoint
# predicting hERG (KCNH2) channel inhibition -- scripts/prepare_herg_
# training_data.py's output, sourced from PubChem AID 588834 (NCGC/NIEHS/
# NTP qHTS, confirmatory tier).
#
# Hyperparameters match this project's other binary-classification
# checkpoints (electrophile-reactivity-v1, the CYP450 panel -- see
# model/registry.json's training.hyperparameters for those entries)
# rather than being retuned for this endpoint -- hiddenSize=300, depth=3,
# ffnHiddenDim=300, batchSize=256, dropout=0, SCAFFOLD_BALANCED 80/10/10
# split, warmup 2 epochs, initLr=1e-4, maxLr=1e-3, finalLr=1e-4, no
# batch-norm.
#
# Usage:
#   scripts/train_herg_chemprop.sh [data_dir] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_DIR="${1:-data/herg}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"

mkdir -p "$CKPT_DIR"

echo "=== training herg (target=label, classification) ==="
conda run -n cov-chemprop chemprop train \
  --data-path "$DATA_DIR/herg.csv" \
  --smiles-columns smiles \
  --target-columns label \
  --task-type classification \
  --loss-function bce \
  --metrics roc prc accuracy f1 \
  --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
  --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 --ffn-num-layers 1 \
  --dropout 0.0 \
  --warmup-epochs 2 --init-lr 0.0001 --max-lr 0.001 --final-lr 0.0001 \
  --batch-size 256 \
  --epochs "$EPOCHS" \
  --output-dir "$CKPT_DIR/herg"

echo "done -- checkpoint in $CKPT_DIR/herg/"
