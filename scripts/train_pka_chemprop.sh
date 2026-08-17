#!/usr/bin/env bash
# train_pka_chemprop.sh
#
# Trains a real Chemprop atom-level regression checkpoint predicting
# aqueous pKa at ANY ionizable heavy atom -- one unified model covering
# both acidic (e.g. carboxylic acid, phenol) and basic (e.g. amine,
# pyridine) sites, trained on the real Baltruschat & Czodrowski dataset
# (scripts/prepare_pka_training_data.py's output). This app's own
# js/pka-microstates.js SMARTS detector decides WHICH atoms are
# ionizable and what class (acid/base) they are; this model just answers
# "what's the pKa at this specific atom" once asked.
#
# graphType is "heavy" (no --add-h): every labeled atom in the source
# dataset is a heavy atom (O/N/S, occasionally C), never an explicit
# hydrogen, confirmed directly from the source SDF's own molblocks
# (heavy-atom-only, no explicit H atoms present).
#
# hiddenSize=300, depth=3 match every other checkpoint in this project's
# registry.
#
# Usage:
#   scripts/train_pka_chemprop.sh [data_csv] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_CSV="${1:-data/pka/pka_prepared.csv}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-60}"

mkdir -p "$CKPT_DIR"

echo "=== training aqueous-pka (target=pka) ==="
conda run -n cov-chemprop chemprop train \
  --data-path "$DATA_CSV" \
  --smiles-columns smiles \
  --atom-target-columns pka \
  --splits-column split \
  --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 \
  --atom-ffn-hidden-dim 300 \
  --epochs "$EPOCHS" \
  --output-dir "$CKPT_DIR/aqueous-pka"

echo "done -- checkpoint in $CKPT_DIR/aqueous-pka/"
