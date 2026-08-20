#!/usr/bin/env bash
# train_vapor_pressure_chemprop.sh
#
# Trains a real Chemprop molecule-level regression checkpoint predicting
# log10 vapor pressure (mmHg, 25 C) from
# scripts/prepare_vapor_pressure_training_data.py's output (the
# EPA/NICEATM OPERA endpoint via gkxiao/vapor-pressure, confirmed exact
# match to the shipped checkpoint's own documented split -- see that
# script's docstring).
#
# UNLIKE every other train_*.sh in this project, this does NOT use
# SCAFFOLD_BALANCED -- it reproduces model/registry.json's documented
# methodology exactly: the pre-defined test.csv is held out via
# --separate-test-path (never touched by the split below), and the
# train.csv partition gets its own internal RANDOM 90/10 train/val split
# (--split-sizes 0.9 0.1 0.0, --data-seed 0), matching "further internal
# 90/10 train/val split of the train partition during training (RANDOM,
# chemprop split-sizes=[0.9,0.1,0.0], data-seed=0)" from that entry's
# dataset.splitStrategy field.
#
# Usage:
#   scripts/train_vapor_pressure_chemprop.sh [train_csv] [test_csv] [checkpoints_dir] [epochs]

set -euo pipefail

TRAIN_CSV="${1:-data/vapor-pressure/vapor_pressure_train.csv}"
TEST_CSV="${2:-data/vapor-pressure/vapor_pressure_test.csv}"
CKPT_DIR="${3:-checkpoints}"
EPOCHS="${4:-50}"

mkdir -p "$CKPT_DIR"

echo "=== training vapor-pressure (target=logVP, regression) ==="
conda run -n cov-chemprop chemprop train \
  --data-path "$TRAIN_CSV" \
  --separate-test-path "$TEST_CSV" \
  --smiles-columns smiles \
  --target-columns logVP \
  --split random --split-sizes 0.9 0.1 0.0 --data-seed 0 \
  --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 --ffn-num-layers 1 \
  --dropout 0.0 \
  --warmup-epochs 2 --init-lr 0.0001 --max-lr 0.001 --final-lr 0.0001 \
  --batch-size 64 \
  --epochs "$EPOCHS" \
  --output-dir "$CKPT_DIR/vapor-pressure"

echo "done -- checkpoint in $CKPT_DIR/vapor-pressure/"
