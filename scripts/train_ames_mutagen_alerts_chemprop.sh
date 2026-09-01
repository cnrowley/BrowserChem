#!/usr/bin/env bash
# train_ames_mutagen_alerts_chemprop.sh
#
# Reproduces the shipped ames-mutagenicity-v1 checkpoint: SMILES +
# X_d fusion of 9 real Benigni/Bossa genotoxicity structural alerts
# (data/mutagenicity_alerts_benigni_bossa.json, computed via
# scripts/compute_mutagenicity_alert_features.py) -- see
# model/registry.json's ames-mutagenicity-v1 entry for the full 7-way
# comparison this configuration won (mean test ROC-AUC 0.850, tied with
# combining these alerts + electrophile-reactivity-v1's score at 0.851,
# but with one fewer model dependency at inference time).
#
# Prerequisite: scripts/compute_mutagenicity_alert_features.py +
# scripts/join_admet_x9_descriptors.py must have already been run to
# produce data/ames/ames_mutagen9_electro.csv (or any CSV with the same
# alert_sa_* columns joined onto data/ames/ames.csv).
#
# Usage:
#   scripts/train_ames_mutagen_alerts_chemprop.sh [data_csv] [checkpoints_dir] [epochs]

set -euo pipefail

DATA_CSV="${1:-data/ames/ames_mutagen9_electro.csv}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"

DESC_COLS="alert_sa_1 alert_sa_6 alert_sa_7 alert_sa_12 alert_sa_13 alert_sa_14 alert_sa_16 alert_sa_21 alert_sa_22"

mkdir -p "$CKPT_DIR"

conda run -n cov-chemprop chemprop train \
  --data-path "$DATA_CSV" \
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
  --data-seed 0 --pytorch-seed 0 \
  --epochs "$EPOCHS" \
  --output-dir "$CKPT_DIR/ames-mutagenicity-alerts"

echo "done -- checkpoint in $CKPT_DIR/ames-mutagenicity-alerts/"
