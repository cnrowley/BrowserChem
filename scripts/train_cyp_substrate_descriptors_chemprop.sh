#!/usr/bin/env bash
# train_cyp_substrate_descriptors_chemprop.sh
#
# Retrains all six CYP450 substrate/metabolism checkpoints (see
# train_cyp_substrate_chemprop.sh for the original SMILES-only version)
# with real X_d feature fusion: logP, LogD(pH7), most-acidic/most-basic
# site pKa (+ has-site flags), and NAGL-MBIS charge min/max/mean --
# --descriptors-columns logp logd pka_acidic has_acidic pka_basic
# has_basic nagl_min nagl_max nagl_mean, chemprop's own native support
# (no custom training loop needed, unlike train_pka_microstate_
# freeenergy.py's paired thermodynamic-cycle case).
#
# This is NOT a speculative retrain -- a real offline experiment
# (scripts/pka-physical-baseline-harness/compute_cyp_descriptors.js +
# scripts/join_cyp_descriptors.py, 3 seeds, on CYP3A4 substrate) showed
# a consistent test-set gain over the SMILES-only baseline (mean
# ROC-AUC 0.805 -> 0.823, F1 0.698 -> 0.731), while a real 3D-SASA and
# BDE follow-up on the same task showed BDE added nothing and 3D SASA's
# gain wasn't judged worth its computation cost for this panel -- see
# model/registry.json's cyp{isoform}-substrate-v1 entries' own `notes`
# once regenerated. The CYP450 INHIBITION panel is deliberately NOT
# touched by this script -- that panel's real gain came mostly from the
# (rejected) 3D SASA feature, not the 2D set alone, so it stays on its
# original SMILES-only checkpoints.
#
# Fixed --data-seed/--pytorch-seed 0 (unlike the original SMILES-only
# checkpoints, trained with chemprop's default unfixed seed) -- matches
# every seed used in the offline feature-selection experiment itself,
# so this is the exact configuration already shown to win, not a fresh
# unseeded roll of the dice.
#
# Prerequisite: scripts/join_cyp_descriptors.py must have already been
# run to produce data/cyp_substrate/cyp_substrate_<isoform>_with_
# descriptors.csv for every isoform (see that script's own header).
#
# Usage:
#   scripts/train_cyp_substrate_descriptors_chemprop.sh [data_dir] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_DIR="${1:-data/cyp_substrate}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"

DESC_COLS="logp logd pka_acidic has_acidic pka_basic has_basic nagl_min nagl_max nagl_mean"

mkdir -p "$CKPT_DIR"

train_one() {
  local isoform="$1"
  local data_csv="$DATA_DIR/cyp_substrate_${isoform}_with_descriptors.csv"

  echo "=== training cyp-substrate-${isoform}-descriptors (target=label, classification, X_d fusion) ==="
  conda run -n cov-chemprop chemprop train \
    --data-path "$data_csv" \
    --smiles-columns smiles \
    --target-columns label \
    --descriptors-columns $DESC_COLS \
    --task-type classification \
    --loss-function bce \
    --metrics roc prc accuracy f1 \
    --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
    --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 --ffn-num-layers 1 \
    --dropout 0.0 \
    --warmup-epochs 2 --init-lr 0.0001 --max-lr 0.001 --final-lr 0.0001 \
    --batch-size 256 \
    --data-seed 0 --pytorch-seed 0 --save-smiles-splits \
    --epochs "$EPOCHS" \
    --output-dir "$CKPT_DIR/cyp-substrate-${isoform}-descriptors"
}

for isoform in 1a2 2c9 2c19 2d6 2e1 3a4; do
  train_one "$isoform"
done

echo "done -- checkpoints in $CKPT_DIR/cyp-substrate-{1a2,2c9,2c19,2d6,2e1,3a4}-descriptors/"
