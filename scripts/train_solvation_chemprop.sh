#!/usr/bin/env bash
# train_solvation_chemprop.sh
#
# Trains 21 real Chemprop molecule-level REGRESSION checkpoints, one per
# solvent in scripts/prepare_solvation_training_data.py's SOLVENTS panel,
# predicting COSMO-RS solvation free energy (dGsolv, kcal/mol) of a
# solute SMILES in that FIXED solvent. Fixed-solvent single-task models,
# not a two-molecule solute+solvent model -- see the prep script's
# docstring for why (this project's JS D-MPNN only supports one
# molecular graph per checkpoint).
#
# Same architecture convention as every other checkpoint in this
# project's registry -- hiddenSize=300, depth=3, ffnHiddenDim=300,
# SCAFFOLD_BALANCED 80/10/10 split (each solvent's ~3200-4200 rows is
# comparable in size to this project's CYP panels, which used the same
# split/architecture successfully).
#
# Usage:
#   scripts/train_solvation_chemprop.sh [data_dir] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_DIR="${1:-data/solvation}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-60}"

mkdir -p "$CKPT_DIR"

train_one() {
  local solvent="$1"
  local data_csv="$DATA_DIR/solv_${solvent}.csv"

  echo "=== training solv-${solvent} (target=dgsolv, regression) ==="
  conda run -n cov-chemprop chemprop train \
    --data-path "$data_csv" \
    --smiles-columns smiles \
    --target-columns dgsolv \
    --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
    --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 --ffn-num-layers 1 \
    --dropout 0.0 \
    --warmup-epochs 2 --init-lr 0.0001 --max-lr 0.001 --final-lr 0.0001 \
    --batch-size 64 \
    --epochs "$EPOCHS" \
    --output-dir "$CKPT_DIR/solv-${solvent}"
}

for solvent in water methanol ethanol 2-propanol 1-octanol acetone acetonitrile \
               dmso dmf thf dioxane diethyl-ether ethyl-acetate dcm chloroform \
               toluene hexane heptane pyridine acetic-acid cyclohexane; do
  train_one "$solvent"
done

echo "done -- checkpoints in $CKPT_DIR/solv-{water,methanol,...}/"
