# electrophile-reactivity-v1

Predicts the probability that a molecule is protein-reactive (a covalent
modifier / electrophilic warhead) -- a DMPNN (Chemprop D-MPNN) binary
classifier, `propertyKey: "label"`, see `model/registry.json` for the
full registry entry (dataset provenance, metrics, hyperparameters,
applicability-domain vocabulary).

## Files in this directory

- `manifest.json` + `weights.bin` -- the converted checkpoint this app's
  JS loads at runtime (produced by `scripts/convert_chemprop_checkpoint.py`,
  see below). Never hand-edited.
- `applicability-domain.json` -- training-vocabulary gate + embedding-domain
  confidence-tier data for this model (produced by
  `scripts/compute_applicability_domain.py`, see below). Never hand-edited.

## Dataset

Source: [RowleyGroup/covalent-classifier](https://github.com/RowleyGroup/covalent-classifier),
the code + data release behind

> Cano Gil, V. H.; Rowley, C. N. Graph neural networks for identifying
> protein reactive electrophiles. *Digital Discovery* **2024**, *3*, 1776.
> https://doi.org/10.1039/D4DD00038B

Positive set: `data/SMILES_training/trainingset_covalent_smiles_larger_set.csv`
(the larger positive set -- original CovInDB/DrugBank-derived positives
plus additional electrophiles from a 2026 *Nature Cell Biology* paper, per
that repo's own `data/SMILES_training/README`). Negative set:
`data/SMILES_training/trainingset_noncovalent_smiles.csv` (BindingDB +
DrugBank non-reactive compounds). 62,678 molecules after RDKit
canonicalization and dedup (16,946 reactive / 45,732 non-reactive).

## Reproducing this checkpoint from scratch

Three steps, run from the repo root. Steps 1-2 need `pandas`, `rdkit`,
`requests`; step 2 additionally needs `torch`, `chemprop==2.2.3`,
`lightning` (see `scripts/convert_chemprop_checkpoint.py`'s own docstring
for why those aren't part of this repo's runtime dependencies).

```bash
# 1. Download + clean the dataset (fetches both CSVs directly from
#    RowleyGroup/covalent-classifier's raw GitHub URLs -- no local
#    clone of that repo needed). Writes
#    data/electrophile-reactivity/electrophile_reactivity.csv.
python3 scripts/prepare_electrophile_reactivity_training_data.py

# 2. Train. Exact command (also in scripts/train_electrophile_reactivity_chemprop.sh):
scripts/train_electrophile_reactivity_chemprop.sh
#   which runs:
#   chemprop train \
#     --data-path data/electrophile-reactivity/electrophile_reactivity.csv \
#     --smiles-columns smiles --target-columns label \
#     --task-type classification --loss-function bce \
#     --metrics roc prc accuracy f1 \
#     --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
#     --message-hidden-dim 300 --depth 3 --ffn-hidden-dim 300 --ffn-num-layers 1 \
#     --dropout 0.0 \
#     --warmup-epochs 2 --init-lr 0.0001 --max-lr 0.001 --final-lr 0.0001 \
#     --batch-size 256 --epochs 50 \
#     --output-dir checkpoints/electrophile-reactivity
# Produces checkpoints/electrophile-reactivity/model_0/best.pt (a
# Chemprop v2 Lightning checkpoint).

# 3. Convert to the manifest.json + weights.bin pair this app loads,
#    and recompute the applicability-domain sidecar from the same CSV
#    training used:
python3 scripts/convert_chemprop_checkpoint.py \
    checkpoints/electrophile-reactivity/model_0/best.pt \
    /tmp/electrophile-conv
cp /tmp/electrophile-conv/label-manifest.json model/electrophile-reactivity/manifest.json
cp /tmp/electrophile-conv/label.bin model/electrophile-reactivity/weights.bin

python3 scripts/compute_applicability_domain.py \
    electrophile-reactivity-v1 \
    data/electrophile-reactivity/electrophile_reactivity.csv
```

`chemprop train`'s SCAFFOLD_BALANCED split is randomized per run and not
saved to disk (see `scripts/compute_applicability_domain.py`'s docstring),
so a fresh run will train/test on a different split and produce a
checkpoint with slightly different weights and metrics than the one
currently shipped here -- not a bit-for-bit reproduction, but the same
architecture, hyperparameters, and real dataset.

After step 3, run `python3 scripts/validate_registry.py model/registry.json`
to confirm the registry's `dataset`/`metrics`/`applicabilityDomain` fields
are still consistent with whatever you just produced (update them by hand
if you retrained with different data/metrics).
