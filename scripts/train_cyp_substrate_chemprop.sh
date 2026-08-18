#!/usr/bin/env bash
# train_cyp_substrate_chemprop.sh
#
# Trains six real Chemprop molecule-level BINARY CLASSIFICATION
# checkpoints predicting whether a molecule is a cytochrome P450
# SUBSTRATE (metabolized by that isoform), one per isoform (CYP1A2,
# CYP2C9, CYP2C19, CYP2D6, CYP2E1, CYP3A4) -- scripts/prepare_cyp_
# substrate_training_data.py's output, sourced from the Figshare
# "Curated CYP450 Interaction Dataset" (Xu et al., Scientific Data
# 2025). Six separate single-task checkpoints, not one multi-task model
# and not a second head on the existing inhibition checkpoints --
# substrate status and inhibition are different biochemical properties,
# matching this project's one-checkpoint-per-registry-entry convention.
#
# Hyperparameters match train_cyp_chemprop.sh (the inhibition panel)
# and this project's other binary-classification checkpoints rather
# than being retuned per isoform -- hiddenSize=300, depth=3,
# ffnHiddenDim=300, batchSize=256, dropout=0, SCAFFOLD_BALANCED 80/10/10
# split, warmup 2 epochs, initLr=1e-4, maxLr=1e-3, finalLr=1e-4.
#
# Usage:
#   scripts/train_cyp_substrate_chemprop.sh [data_dir] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_DIR="${1:-data/cyp_substrate}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"

mkdir -p "$CKPT_DIR"

train_one() {
  local isoform="$1"
  local data_csv="$DATA_DIR/cyp_substrate_${isoform}.csv"

  echo "=== training cyp-substrate-${isoform} (target=label, classification) ==="
  conda run -n cov-chemprop chemprop train \
    --data-path "$data_csv" \
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
    --output-dir "$CKPT_DIR/cyp-substrate-${isoform}"
}

for isoform in 1a2 2c9 2c19 2d6 2e1 3a4; do
  train_one "$isoform"
done

echo "done -- checkpoints in $CKPT_DIR/cyp-substrate-{1a2,2c9,2c19,2d6,2e1,3a4}/"
