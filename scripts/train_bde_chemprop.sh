#!/usr/bin/env bash
# train_bde_chemprop.sh
#
# Trains a Chemprop bond-level regression checkpoint predicting bond
# dissociation enthalpy (BDE, kcal/mol) on the CSV
# scripts/prepare_bde_training_data.py produces from patonlab/BDE-db2, in
# the `cov-chemprop` conda env (chemprop 2.2.3).
#
# This is the app's first BOND-level (not atom- or molecule-level)
# Chemprop checkpoint -- confirmed via chemprop's own installed source
# (chemprop/models/mol_atom_bond.py) that its bond-predictor fingerprint
# is `concat([H_e[edge], H_e[reverse_edge]])` fed through the FFN head,
# then averaged pairwise over the two directions
# (`(preds[::2] + preds[1::2]) / 2`) -- js/chemprop-model.js's
# runOneBondLevel() replicates this exactly.
#
# --add-h (every hydrogen its own graph node) matches how BDE-db2's own
# `bond_index` column was confirmed (by direct RDKit cross-check) to be
# ordered -- Chem.AddHs(mol).GetBonds() order, same convention already
# used for the 1H NMR checkpoint. --splits-column reuses BDE-db2's own
# train/valid/test split (remapped valid->val) rather than re-splitting,
# so held-out numbers are comparable to the source paper's own reported
# accuracy.
#
# hiddenSize=300, depth=3 match every other checkpoint in this project's
# registry -- not a technical requirement, just the path of least
# surprise, same reasoning as train_nmr_chemprop.sh.
#
# Usage:
#   scripts/train_bde_chemprop.sh [data_csv] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_CSV="${1:-data/bde/bde_bond_targets.csv}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-40}"

mkdir -p "$CKPT_DIR"

echo "=== training bde ($DATA_CSV, target=bde) ==="
conda run -n cov-chemprop chemprop train \
  --data-path "$DATA_CSV" \
  --smiles-columns smiles \
  --bond-target-columns bde \
  --splits-column split \
  --add-h \
  --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 \
  --bond-ffn-hidden-dim 300 \
  --epochs "$EPOCHS" \
  --output-dir "$CKPT_DIR/bde-chemprop"

echo "done -- checkpoint in $CKPT_DIR/bde-chemprop/"
