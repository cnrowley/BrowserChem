#!/usr/bin/env bash
# train_ames_mutagen24_admet9_electro_chemprop.sh
#
# Reproduces the CURRENTLY shipped ames-mutagenicity-v1 checkpoint
# (updated 2026-09-01, superseding scripts/train_ames_mutagen_alerts_
# chemprop.sh's 9-alert-only config): SMILES + X_d fusion of 34
# descriptors -- 22 corrected/expanded Benigni/Bossa genotoxicity SMARTS
# alerts + 2 custom ring-fusion alerts (data/mutagenicity_alerts_
# benigni_bossa.json, computed via scripts/compute_mutagenicity_alert_
# features.py) + the same 9-descriptor ADMET set js/admet-x9-
# descriptors.js already validated for CYP450-substrate/BBBP + electro-
# phile-reactivity-v1's own predicted probability. See model/registry.json's
# ames-mutagenicity-v1 entry (metrics.note) for the complete before/after
# comparison this configuration won: mean test ROC-AUC 0.862 across 5
# fixed scaffold-balanced seeds, vs. 0.852 for the previous 9-alert-only
# config retrained on the identical 5 splits.
#
# The shipped checkpoint uses --data-seed 4 --pytorch-seed 4, chosen by
# LOWEST VALIDATION LOSS among 5 candidates (seeds 0-4) -- not by
# cherry-picking the best test score. To reproduce the full comparison,
# rerun this script across seeds 0-4 and compare val_loss from each
# checkpoint filename (chemprop names it into the .ckpt file itself,
# e.g. "best-epoch=48-val_loss=0.45.ckpt").
#
# Prerequisite: scripts/compute_mutagenicity_alert_features.py (on the
# corrected data/mutagenicity_alerts_benigni_bossa.json) +
# scripts/join_admet_x9_descriptors.py (joining in ADMET-9 descriptors
# and electrophile-reactivity-v1's score, both computed via
# scripts/pka-physical-baseline-harness/compute_admet_x9_descriptors.js
# and compute_electrophile_reactivity_feature.js respectively) must have
# already been run to produce data/ames/ames_mutagen24_admet9_electro.csv.
#
# Usage:
#   scripts/train_ames_mutagen24_admet9_electro_chemprop.sh [data_csv] [checkpoints_dir] [epochs] [data_seed]

set -euo pipefail

DATA_CSV="${1:-data/ames/ames_mutagen24_admet9_electro.csv}"
CKPT_DIR="${2:-checkpoints}"
EPOCHS="${3:-50}"
SEED="${4:-4}"

DESC_COLS="alert_sa_1 alert_sa_6 alert_sa_7 alert_sa_12 alert_sa_13 alert_sa_14 alert_sa_16 alert_sa_21 alert_sa_22 alert_sa_4 alert_sa_10 alert_sa_11 alert_sa_15 alert_sa_17 alert_sa_30 alert_sa_31b alert_sa_31c alert_sa_42 alert_sa_44 alert_sa_47 alert_sa_49 alert_sa_50 alert_sa_18 alert_sa_19 logp logd pka_acidic has_acidic pka_basic has_basic nagl_min nagl_max nagl_mean electrophileReactivity"

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
  --data-seed "$SEED" --pytorch-seed "$SEED" \
  --epochs "$EPOCHS" \
  --output-dir "$CKPT_DIR/ames-mutagenicity-admet9-electro-seed$SEED"

echo "done -- checkpoint in $CKPT_DIR/ames-mutagenicity-admet9-electro-seed$SEED/"
