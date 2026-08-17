#!/usr/bin/env bash
# train_qm9_chemprop.sh
#
# Trains two real Chemprop molecule-level regression checkpoints on QM9
# (scripts/prepare_qm9_training_data.py's output): isotropic
# polarizability (`alpha`, Angstrom^3) and HOMO-LUMO gap (`gap`, eV).
# Same architecture convention as every other checkpoint in this
# project's registry (hiddenSize=300, depth=3) and the same
# SCAFFOLD_BALANCED 80/10/10 split convention the NMR checkpoints use.
#
# Usage:
#   scripts/train_qm9_chemprop.sh [data_csv] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_CSV="${1:-data/qm9/qm9_prepared.csv}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-30}"

mkdir -p "$CKPT_DIR"

train_one() {
  local name="$1"
  local target_col="$2"

  echo "=== training $name (target=$target_col) ==="
  conda run -n cov-chemprop chemprop train \
    --data-path "$DATA_CSV" \
    --smiles-columns smiles \
    --target-columns "$target_col" \
    --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
    --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 \
    --epochs "$EPOCHS" \
    --output-dir "$CKPT_DIR/qm9-$name"
}

train_one "polarizability" "alpha"
train_one "homo-lumo-gap" "gap"

echo "done -- checkpoints in $CKPT_DIR/qm9-{polarizability,homo-lumo-gap}/"
