#!/usr/bin/env bash
# train_b3db_chemprop.sh
#
# Trains a real Chemprop molecule-level BINARY CLASSIFICATION checkpoint
# predicting blood-brain barrier penetration from B3DB -- scripts/
# prepare_b3db_training_data.py's output (Meng et al. 2021, 7805
# compounds after RDKit canonicalization/dedup). Replaced this
# project's original MoleculeNet-BBBP-trained bbbp-v1 checkpoint after a
# real 3-seed experiment showed a dramatic improvement from the larger,
# more recent dataset alone (mean test ROC-AUC 0.798 -> 0.904, before
# even adding the 9-descriptor X_d fusion on top) -- see
# model/registry.json's bbbp-v1 entry for the full comparison.
#
# Hyperparameters match this project's other binary-classification
# checkpoints -- hiddenSize=300, depth=3, ffnHiddenDim=300, batchSize=256,
# dropout=0, SCAFFOLD_BALANCED 80/10/10 split, warmup 2 epochs,
# initLr=1e-4, maxLr=1e-3, finalLr=1e-4.
#
# For the shipped bbbp-v1 checkpoint, run this against data/b3db/
# b3db_with_descriptors.csv (scripts/join_admet_x9_descriptors.py's
# output after joining scripts/pka-physical-baseline-harness/
# compute_admet_x9_descriptors.js's 9-descriptor set) with
# --descriptors-columns logp logd pka_acidic has_acidic pka_basic
# has_basic nagl_min nagl_max nagl_mean -- this script alone only
# reproduces the SMILES-only baseline number.
#
# Usage:
#   scripts/train_b3db_chemprop.sh [data_csv] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_CSV="${1:-data/b3db/b3db.csv}"
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
  --output-dir "$CKPT_DIR/b3db"

echo "done -- checkpoint in $CKPT_DIR/b3db/"
