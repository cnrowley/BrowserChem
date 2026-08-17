#!/usr/bin/env bash
# train_cyp_chemprop.sh
#
# Trains five real Chemprop molecule-level BINARY CLASSIFICATION
# checkpoints predicting cytochrome P450 inhibition, one per isoform
# (CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4) -- scripts/prepare_cyp_
# training_data.py's output, sourced from PubChem AID 1851 (Veith et
# al. 2009). Five separate single-task checkpoints, not one multi-task
# model, matching this project's convention elsewhere of one checkpoint
# per registry entry (e.g. the two separate QM9 checkpoints from
# train_qm9_chemprop.sh) and this project's JS inference engine, which
# only implements single-task regression/binary-classification heads
# (see convert_chemprop_checkpoint.py's docstring).
#
# Hyperparameters match this project's other binary-classification
# checkpoint (electrophile-reactivity-v1, see model/registry.json's
# training.hyperparameters for that entry) rather than being retuned
# per isoform -- hiddenSize=300, depth=3, ffnHiddenDim=300,
# batchSize=256, dropout=0, SCAFFOLD_BALANCED 80/10/10 split, warmup 2
# epochs, initLr=1e-4, maxLr=1e-3, finalLr=1e-4, no batch-norm.
#
# Usage:
#   scripts/train_cyp_chemprop.sh [data_dir] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_DIR="${1:-data/cyp}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"

mkdir -p "$CKPT_DIR"

train_one() {
  local isoform="$1"
  local data_csv="$DATA_DIR/cyp_${isoform}.csv"

  echo "=== training cyp-${isoform} (target=label, classification) ==="
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
    --output-dir "$CKPT_DIR/cyp-${isoform}"
}

for isoform in cyp1a2 cyp2c9 cyp2c19 cyp2d6 cyp3a4; do
  train_one "$isoform"
done

echo "done -- checkpoints in $CKPT_DIR/cyp-{cyp1a2,cyp2c9,cyp2c19,cyp2d6,cyp3a4}/"
