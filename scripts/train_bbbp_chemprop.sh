#!/usr/bin/env bash
# train_bbbp_chemprop.sh
#
# Trains a real Chemprop molecule-level BINARY CLASSIFICATION checkpoint
# predicting blood-brain barrier penetration -- scripts/
# prepare_bbbp_training_data.py's output (Martins et al. 2012 /
# MoleculeNet BBBP, 1965 compounds after RDKit canonicalization/dedup).
#
# Hyperparameters match this project's other binary-classification
# checkpoints -- hiddenSize=300, depth=3, ffnHiddenDim=300, batchSize=256,
# dropout=0, SCAFFOLD_BALANCED 80/10/10 split, warmup 2 epochs,
# initLr=1e-4, maxLr=1e-3, finalLr=1e-4.
#
# Usage:
#   scripts/train_bbbp_chemprop.sh [data_csv] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_CSV="${1:-data/bbbp/bbbp.csv}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"

mkdir -p "$CKPT_DIR"

conda run -n cov-chemprop chemprop train \
  --data-path "$DATA_CSV" \
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
  --output-dir "$CKPT_DIR/bbbp"

echo "done -- checkpoint in $CKPT_DIR/bbbp/"
