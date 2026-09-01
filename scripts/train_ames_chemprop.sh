#!/usr/bin/env bash
# train_ames_chemprop.sh
#
# Trains a real Chemprop molecule-level BINARY CLASSIFICATION checkpoint
# predicting Ames mutagenicity -- scripts/prepare_ames_training_data.py's
# output (Hansen et al. 2009 benchmark, 6505 compounds after RDKit
# canonicalization/dedup).
#
# Hyperparameters match this project's other binary-classification
# checkpoints (the CYP450/hERG panels) rather than being retuned --
# hiddenSize=300, depth=3, ffnHiddenDim=300, batchSize=256, dropout=0,
# SCAFFOLD_BALANCED 80/10/10 split, warmup 2 epochs, initLr=1e-4,
# maxLr=1e-3, finalLr=1e-4.
#
# Usage:
#   scripts/train_ames_chemprop.sh [data_csv] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_CSV="${1:-data/ames/ames.csv}"
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
  --output-dir "$CKPT_DIR/ames-mutagenicity"

echo "done -- checkpoint in $CKPT_DIR/ames-mutagenicity/"
