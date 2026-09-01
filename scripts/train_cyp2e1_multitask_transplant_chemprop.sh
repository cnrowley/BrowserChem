#!/usr/bin/env bash
# train_cyp2e1_multitask_transplant_chemprop.sh
#
# Reproduces the shipped cyp2e1-substrate-v1 checkpoint: a multitask
# CYP450-substrate encoder (trained jointly across all 6 isoforms) is
# pretrained first, its message_passing weights are spliced into a
# fresh single-task skeleton, then fine-tuned single-task on CYP2E1
# alone with --class-balance. See model/registry.json's
# cyp2e1-substrate-v1 entry (metrics.note) for the full comparison this
# configuration won: mean test ROC-AUC 0.743 / F1 0.312 / MCC 0.240,
# vs. 0.671 ROC / F1=0 for the previous ADMET-9-fused single-task config.
#
# chemprop's own --checkpoint flag can't do a task-count-changing
# encoder transplant directly (it copies the WHOLE model including the
# task-count-mismatched predictor head, which crashes on shape
# mismatch); --model-frzn is deprecated and crashes outright in
# chemprop 2.3.1 when used alone. Hence the two-step splice below.
#
# Usage:
#   scripts/train_cyp2e1_multitask_transplant_chemprop.sh [checkpoints_dir] [data_seed]

set -euo pipefail

CKPT_DIR="${1:-checkpoints}"
SEED="${2:-1}"
MULTITASK_CSV="${MULTITASK_CSV:-data/cyp_substrate/cyp_substrate_multitask_full.csv}"  # smiles + cyp{iso}_substrate columns, see step 1
CYP2E1_CSV="${CYP2E1_CSV:-data/cyp_substrate/cyp_substrate_2e1.csv}"

mkdir -p "$CKPT_DIR"

# Step 1: build the multitask union CSV if it doesn't already exist.
# Columns: smiles, cyp1a2_substrate, cyp2c9_substrate, cyp2c19_substrate,
# cyp2d6_substrate, cyp2e1_substrate, cyp3a4_substrate (blank where a
# molecule wasn't tested for that isoform -- chemprop masks blanks out
# of the loss automatically, no flag needed).
if [ ! -f "$MULTITASK_CSV" ]; then
  echo "Build $MULTITASK_CSV first: union data/cyp_substrate/cyp_substrate_{1a2,2c9,2c19,2d6,2e1,3a4}.csv by canonical SMILES." >&2
  exit 1
fi

# Step 2: train the multitask encoder (NOT deployable directly --
# scripts/convert_chemprop_checkpoint.py refuses multi-task checkpoints).
conda run -n cov-chemprop chemprop train \
  --data-path "$MULTITASK_CSV" \
  --smiles-columns smiles \
  --target-columns cyp1a2_substrate cyp2c9_substrate cyp2c19_substrate cyp2d6_substrate cyp2e1_substrate cyp3a4_substrate \
  --task-type classification --loss-function bce \
  --metrics roc prc accuracy f1 \
  --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
  --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 --ffn-num-layers 1 \
  --dropout 0.0 --warmup-epochs 2 --init-lr 0.0001 --max-lr 0.001 --final-lr 0.0001 \
  --batch-size 256 --data-seed 0 --pytorch-seed 0 --epochs 50 \
  --output-dir "$CKPT_DIR/cyp-multitask-encoder"

# Step 3: build a real single-task CYP2E1 skeleton (2 epochs is enough --
# only its architecture/shapes matter, its weights get overwritten next).
conda run -n cov-chemprop chemprop train \
  --data-path "$CYP2E1_CSV" \
  --smiles-columns smiles --target-columns label \
  --task-type classification --loss-function bce \
  --metrics roc prc accuracy f1 \
  --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
  --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 --ffn-num-layers 1 \
  --dropout 0.0 --warmup-epochs 1 --init-lr 0.0001 --max-lr 0.001 --final-lr 0.0001 \
  --batch-size 256 --data-seed 0 --pytorch-seed 0 --epochs 2 \
  --output-dir "$CKPT_DIR/cyp2e1-skeleton"

# Step 4: splice the multitask encoder's message_passing + shared
# predictor.ffn.0.0 weights into the skeleton (both are n_tasks-
# independent; only predictor.ffn.1.2/criterion/metrics.*.task_weights
# are task-count-dependent and are left as the skeleton's own values).
conda run -n cov-chemprop python3 - "$CKPT_DIR/cyp-multitask-encoder/model_0/best.pt" "$CKPT_DIR/cyp2e1-skeleton/model_0/best.pt" "$CKPT_DIR/cyp2e1-frankenstein.pt" <<'PYEOF'
import sys
import torch

multitask_path, skeleton_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
mt = torch.load(multitask_path, map_location="cpu", weights_only=False)
sk = torch.load(skeleton_path, map_location="cpu", weights_only=False)

TRANSPLANT_PREFIXES = ("message_passing.", "predictor.ffn.0.0.")
for key in list(sk["state_dict"].keys()):
    if key.startswith(TRANSPLANT_PREFIXES):
        assert mt["state_dict"][key].shape == sk["state_dict"][key].shape
        sk["state_dict"][key] = mt["state_dict"][key].clone()

torch.save(sk, out_path)
print(f"wrote {out_path}")
PYEOF

# Step 5: fine-tune single-task, unfrozen, at a REDUCED learning rate
# (avoids the catastrophic-forgetting/negative-transfer result this
# project already saw once with a from-scratch-tuned LR schedule --
# see ames-mutagenicity-v1's registry history), plus --class-balance.
conda run -n cov-chemprop chemprop train \
  --data-path "$CYP2E1_CSV" \
  --smiles-columns smiles --target-columns label \
  --task-type classification --loss-function bce --class-balance \
  --checkpoint "$CKPT_DIR/cyp2e1-frankenstein.pt" \
  --metrics roc prc accuracy f1 \
  --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
  --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 --ffn-num-layers 1 \
  --dropout 0.0 --warmup-epochs 2 --init-lr 0.00003 --max-lr 0.0003 --final-lr 0.00003 \
  --batch-size 256 --data-seed "$SEED" --pytorch-seed "$SEED" --epochs 50 \
  --output-dir "$CKPT_DIR/cyp2e1-substrate-final"

echo "done -- checkpoint in $CKPT_DIR/cyp2e1-substrate-final/"
