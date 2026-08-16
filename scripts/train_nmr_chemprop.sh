#!/usr/bin/env bash
# train_nmr_chemprop.sh
#
# Trains four Chemprop atom-level regression checkpoints (13C, 15N, 19F,
# 1H) on the CSVs scripts/prepare_nmr_training_data.py produces, in the
# `cov-chemprop` conda env (chemprop 2.2.3 -- confirmed installed on this
# machine alongside torch/lightning/rdkit).
#
# hiddenSize=300, depth=3 match every other checkpoint in this project's
# registry -- not a technical requirement, just the path of least
# surprise. Flags NOT passed (message-bias, atom-ffn-num-layers,
# multi-hot-atom-featurizer-mode, reorder-atoms) are left at chemprop
# 2.2.3's own defaults specifically because those defaults already match
# what scripts/convert_chemprop_checkpoint.py requires (bias=False,
# n_layers=1, V2 featurizer, atom order preserved) -- confirmed by
# inspecting the installed package's own argparse defaults, not assumed.
#
# 1H gets --add-h (every hydrogen is its own graph node, matching how
# nmr_1h.csv's labels were built) -- every other nucleus does not, since
# NMRShiftDB2/NMRexp label heavy atoms directly for 13C/15N/19F. Running
# the 1H checkpoint in the browser needs a new explicit-H D-MPNN graph
# builder that doesn't exist yet in this project -- out of scope here,
# see the plan this script was written from.
#
# Usage:
#   scripts/train_nmr_chemprop.sh [data_dir] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_DIR="${1:-data/nmr}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"

train_one() {
  local nucleus="$1"
  local csv="$2"
  local target_col="$3"
  local extra_flags="$4"

  echo "=== training $nucleus ($csv, target=$target_col) ==="
  conda run -n cov-chemprop chemprop train \
    --data-path "$DATA_DIR/$csv" \
    --smiles-columns smiles \
    --atom-target-columns "$target_col" \
    --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
    --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 \
    --atom-ffn-hidden-dim 300 \
    --epochs "$EPOCHS" \
    --output-dir "$CKPT_DIR/nmr-$nucleus" \
    $extra_flags
}

mkdir -p "$CKPT_DIR"

train_one "13c" "nmr_13c.csv" "shift_13c" ""
train_one "15n" "nmr_15n.csv" "shift_15n" ""
train_one "19f" "nmr_19f.csv" "shift_19f" ""
train_one "1h"  "nmr_1h.csv"  "shift_1h"  "--add-h"

echo "done -- checkpoints in $CKPT_DIR/nmr-{13c,15n,19f,1h}/"
