#!/usr/bin/env bash
# train_cyp_chemeleon_chemprop.sh
#
# Reproduces the shipped cyp2c9-substrate-v1/cyp2e1-substrate-v1
# checkpoints: fine-tuned from the CHEMELEON foundation model
# (Burns 2025, github.com/JacksonBurns/chemeleon -- a D-MPNN pretrained
# on 1M PubChem molecules to predict ~200 Mordred descriptors) via
# chemprop's native `--from-foundation CHEMELEON` (auto-downloads
# chemeleon_mp.pt from Zenodo record 15460715 to ~/.chemprop/ on first
# use) + --class-balance, SMILES-only. See model/registry.json's
# cyp2c9-substrate-v1/cyp2e1-substrate-v1 entries (metrics.note) for the
# complete before/after comparison this configuration won on both
# isoforms, including a real head-to-head loss against a from-scratch-
# fine-tuned ChemBERTa-2 baseline (see finetune_chemberta_baseline.py).
#
# Real, disclosed cost: CHEMELEON's encoder is d_h=2048/depth=6 (vs.
# this project's usual d_h=300/depth=3), ~9.3M params, ~36MB per
# checkpoint after conversion -- ~30x this project's typical model size.
# Uses MeanAggregation (not this project's usual NormAggregation) --
# requires js/pooling.js's poolMean() wired into js/chemprop-model.js
# (dims.aggregationType manifest field) and scripts/convert_chemprop_
# checkpoint.py's aggregation check to accept MeanAggregation, both
# already shipped as of this same commit.
#
# GOTCHA: compute_applicability_domain.py's own pure-numpy D-MPNN
# reimplementation is impractically slow at this encoder size (39+
# minutes and still not finished on a 2500-molecule set, confirmed) --
# use scripts/fast_embeddings.py (real chemprop batched GPU tensor ops)
# + compute_applicability_domain.py's --embeddings-csv flag instead of
# running that script's default (slow) path directly.
#
# Usage:
#   scripts/train_cyp_chemeleon_chemprop.sh [checkpoints_dir] [data_seed]

set -euo pipefail

CKPT_DIR="${1:-checkpoints}"
SEED="${2:-1}"

mkdir -p "$CKPT_DIR"

for iso in 2c9 2e1; do
  echo "=== training cyp${iso}-substrate CHEMELEON+class-balance (seed $SEED) ==="
  conda run -n cov-chemprop chemprop train \
    --data-path "data/cyp_substrate/cyp_substrate_${iso}.csv" \
    --smiles-columns smiles --target-columns label \
    --task-type classification --loss-function bce --class-balance \
    --from-foundation CHEMELEON \
    --metrics roc prc accuracy f1 \
    --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
    --warmup-epochs 2 --init-lr 0.0001 --max-lr 0.001 --final-lr 0.0001 \
    --batch-size 256 --data-seed "$SEED" --pytorch-seed "$SEED" --epochs 50 \
    --output-dir "$CKPT_DIR/cyp${iso}-substrate-chemeleon"
done

echo "done -- checkpoints in $CKPT_DIR/cyp{2c9,2e1}-substrate-chemeleon/"
echo ""
echo "Next steps to reproduce the shipped registry entries:"
echo "  1. python3 scripts/convert_chemprop_checkpoint.py <ckpt>/model_0/best.pt model/cyp-substrate-<iso>/ --task-key cyp<iso>substrate --name cyp<iso>-substrate"
echo "  2. python3 scripts/fast_embeddings.py <ckpt>/model_0/best.pt data/cyp_substrate/cyp_substrate_<iso>.csv /tmp/embeddings_<iso>.csv"
echo "  3. python3 scripts/compute_applicability_domain.py cyp<iso>-substrate-v1 data/cyp_substrate/cyp_substrate_<iso>.csv --embeddings-csv /tmp/embeddings_<iso>.csv --output model/cyp-substrate-<iso>/applicability-domain.json"
