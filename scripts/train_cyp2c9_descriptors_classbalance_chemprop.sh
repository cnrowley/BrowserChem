#!/usr/bin/env bash
# train_cyp2c9_descriptors_classbalance_chemprop.sh
#
# Reproduces the shipped cyp2c9-substrate-v1 checkpoint: the same
# ADMET-9 X_d fusion train_cyp_substrate_descriptors_chemprop.sh already
# uses for all 6 substrate isoforms, PLUS --class-balance (equal
# positive/negative per training batch) -- added specifically for
# CYP2C9 after its ADMET-9-only checkpoint never crossed the 0.5
# decision threshold for the minority/substrate class (F1=0 despite a
# real ROC-AUC signal). See model/registry.json's cyp2c9-substrate-v1
# entry (metrics.note) for the full comparison this configuration won:
# mean test ROC-AUC 0.707 / F1 0.407 / MCC 0.233, vs. 0.733 ROC / F1=0
# for ADMET-9 alone. This is CYP2C9-SPECIFIC -- a multitask-pretrained-
# encoder-transplant approach (see train_cyp2e1_multitask_transplant_
# chemprop.sh) won instead for CYP2E1; don't assume this same recipe is
# the right one for the other 4 substrate isoforms, which are untouched
# by this investigation and still trained via
# train_cyp_substrate_descriptors_chemprop.sh's plain ADMET-9 config.
#
# Prerequisite: scripts/join_cyp_descriptors.py already run to produce
# data/cyp_substrate/cyp_substrate_2c9_with_descriptors.csv.
#
# Usage:
#   scripts/train_cyp2c9_descriptors_classbalance_chemprop.sh [data_csv] [checkpoints_dir] [epochs] [data_seed]

set -euo pipefail

DATA_CSV="${1:-data/cyp_substrate/cyp_substrate_2c9_with_descriptors.csv}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"
SEED="${4:-0}"

DESC_COLS="logp logd pka_acidic has_acidic pka_basic has_basic nagl_min nagl_max nagl_mean"

mkdir -p "$CKPT_DIR"

conda run -n cov-chemprop chemprop train \
  --data-path "$DATA_CSV" \
  --smiles-columns smiles \
  --target-columns label \
  --descriptors-columns $DESC_COLS \
  --task-type classification \
  --loss-function bce \
  --class-balance \
  --metrics roc prc accuracy f1 \
  --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
  --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 --ffn-num-layers 1 \
  --dropout 0.0 \
  --warmup-epochs 2 --init-lr 0.0001 --max-lr 0.001 --final-lr 0.0001 \
  --batch-size 256 \
  --data-seed "$SEED" --pytorch-seed "$SEED" \
  --epochs "$EPOCHS" \
  --output-dir "$CKPT_DIR/cyp2c9-substrate-classbalance"

echo "done -- checkpoint in $CKPT_DIR/cyp2c9-substrate-classbalance/"
