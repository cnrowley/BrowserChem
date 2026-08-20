#!/usr/bin/env bash
# train_electrophile_reactivity_chemprop.sh
#
# Trains a real Chemprop molecule-level BINARY CLASSIFICATION checkpoint
# predicting protein reactivity (electrophile / covalent warhead) --
# scripts/prepare_electrophile_reactivity_training_data.py's output,
# sourced from RowleyGroup/covalent-classifier (Cano Gil & Rowley,
# Digital Discovery 2024, DOI:10.1039/D4DD00038B), using the LARGER
# positive set (original CovInDB/DrugBank positives plus additional
# electrophiles from a 2026 Nature Cell Biology paper).
#
# Hyperparameters match this project's other binary-classification
# checkpoints (the CYP450 panel, hERG -- see model/registry.json's
# training.hyperparameters) rather than being retuned for this endpoint --
# hiddenSize=300, depth=3, ffnHiddenDim=300, batchSize=256, dropout=0,
# SCAFFOLD_BALANCED 80/10/10 split, warmup 2 epochs, initLr=1e-4,
# maxLr=1e-3, finalLr=1e-4, no batch-norm.
#
# Usage:
#   scripts/train_electrophile_reactivity_chemprop.sh [data_dir] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_DIR="${1:-data/electrophile-reactivity}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"

mkdir -p "$CKPT_DIR"

echo "=== training electrophile-reactivity (target=label, classification) ==="
conda run -n cov-chemprop chemprop train \
  --data-path "$DATA_DIR/electrophile_reactivity.csv" \
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
  --output-dir "$CKPT_DIR/electrophile-reactivity"

echo "done -- checkpoint in $CKPT_DIR/electrophile-reactivity/"
