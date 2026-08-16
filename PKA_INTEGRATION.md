# Carbon acid (C-H) pKa predictor — integration notes

## What this is

A from-scratch, client-side port of [jensengroup/pKalculator](https://github.com/jensengroup/pKalculator)
(Ree et al., "pKalculator: A pKa predictor for C-H bonds", ChemRxiv 2024,
[10.26434/chemrxiv-2024-56h5h](https://chemrxiv.org/doi/10.26434/chemrxiv-2024-56h5h),
MIT-licensed) — predicts the pKa of individual C-H bonds (deprotonation to
a stabilized carbanion), per candidate carbon site, entirely in the
browser. Scoped to carbon acids only; a general titratable-group (O-H,
N-H, etc.) pKa predictor is a separate, not-yet-built feature.

## Why it isn't the original model

pKalculator's real ML pipeline is LightGBM (easy to port — tree
traversal, no matrix math) trained on a "graph charge shell" descriptor
built from **CM5 atomic charges**, which the original tool computes by
shelling out to a real `xtb` binary (`xtb --gfn 1 ... --lmo`,
GFN1-xTB semiempirical QM). There is no WASM build of xtb, and porting
GFN1-xTB itself was ruled out as out of scope — so CM5 is a hard
architectural blocker for browser deployment.

The user chose (via an explicit options discussion) to retrain the same
model architecture on a charge source this app can already compute
entirely client-side: **this app's own NAGL-MBIS partial charge model**
(`model/nagl-mbis-charges/`, `js/nagl-model.js`) instead of Gasteiger
charges (the initial plan) — NAGL is trained to reproduce real QM (MBIS)
charges and captures conjugation effects much better than a
Gasteiger-style equilibration.

## Validating NAGL-MBIS first (a real, separate fix)

Before trusting NAGL charges as a foundation, `js/nagl-model.js`'s own
header documented that it had **never been validated against real
ground truth** (DGL wouldn't build in the original dev sandbox). A
working `nagl`/`naglmbis`/DGL conda environment was found on this
machine, making it possible to run the actual
`jthorton/nagl-mbis` `MBISGraphModel.compute_properties()` (real DGL
SAGEConv, not a reimplementation) and diff it against this project's JS
port across 28 molecules — every trained element, charged species,
fused rings, ring sizes 3-6, degree 1-4 edge cases. Result: bit-exact
(~2e-7 max error, float32 rounding only). This also surfaced that the
shipped `model/nagl-mbis-charges/weights.bin` was converted from
`nagl-v1-mbis-dipole.ckpt`, not the plain-named checkpoint — harmless,
now documented in `js/nagl-model.js`'s header and `registry.json`.

## Training pipeline (offline, not part of the shipped app)

1. **Dataset**: `dataset_full.pkl` (775 molecules, per-site QM-derived
   pKa targets, paper-defined train/test split and 5-fold CV columns),
   downloaded from pKalculator's own ERDA sharelink with the user's
   explicit permission.
2. **Charges**: a Node.js harness loads `@rdkit/rdkit`'s WASM build
   directly in Node (not just the browser) and this app's own
   `molecule.js`/`molfile.js`/`graph-builder.js`/`nagl-features.js`/
   `nagl-model.js` unmodified, to compute NAGL-MBIS charges for every
   atom (heavy + explicit H) across all 775 molecules using the exact
   same code the browser runs. 16 molecules (containing Se, Si, or I)
   fail — outside NAGL-MBIS's trained element vocabulary — and are
   dropped; this is an honest dataset limitation, not a bug.
3. **Atom ordering**: cross-checked that the JS-computed atom sequence
   (heavy atoms in original order, then synthetic explicit-H nodes
   grouped by parent heavy atom) matches Python's
   `Chem.AddHs(Chem.MolFromSmiles(smiles))` atom sequence exactly, for
   all 759 molecules and all 4290 deprotonation sites — zero mismatches.
4. **Descriptor**: pKalculator's own "graph charge shell" algorithm
   (`smi2gcs/DescriptorCreator/GraphChargeShell.py`) ported to Python
   (`scripts/`-adjacent training pipeline, not shipped) reading NAGL
   charges instead of CM5. One deliberate deviation from upstream: their
   own CIP-tiebreak code crashes on every currently-installable RDKit
   version (2021.09-2025.09, confirmed by running their unmodified code
   — a Python negative-list-index that only "worked" under whatever
   RDKit version they originally pinned) and, even when it doesn't
   crash, applies a confusing indirect permutation to charge-tied
   groups. Replaced with a plain stable-sort-by-charge within each
   tied-priority run — simpler, well-defined, and since this is a
   freshly-trained model rather than a reproduction of the original
   paper's numbers, there's no value in preserving that upstream quirk.
5. **Training**: LightGBM/DART, reusing the original paper's own tuned
   hyperparameters (`params_optuna_dart_reg.pkl`) unchanged. Boost round
   count picked from a fold1-internal train/val curve, then retrained on
   the full official train split; evaluated once on the untouched
   official test split.

## Honest accuracy comparison

| | MAE | RMSE | test sites |
|---|---|---|---|
| Original (real CM5/GFN1-xTB charges) | 1.24 | 2.15 | 789 |
| This app (NAGL-MBIS charges) | 1.40 | 2.49 | 754 |

About 13-16% worse than the original — the real, expected cost of
swapping a semiempirical-QM-derived charge for a GNN-predicted
approximation of one. Test site count differs slightly because 16
molecules (Se/Si/I) can't be scored by NAGL at all.

## JS runtime pieces

- `js/pka-descriptor.js` — `CC.PKA.buildDescriptor()`, the graph-charge-
  shell construction (shell traversal, CIP-priority + charge tiebreak),
  cross-checked bit-identical against the Python training-time builder
  across all 4290 real training sites before this was trusted.
- `js/pka-model.js` — `CC.PKA.predict()`. Trees are flattened into one
  shared node pool (parallel typed arrays), self-checked in
  `scripts/convert_pka_lightgbm.py`'s pipeline to reproduce
  `Booster.predict()` exactly (0.0 max error) before being written out.
  Candidate sites are any carbon bearing ≥1 hydrogen (implicit or
  explicit) — generalizes pKalculator's own site-enumeration SMARTS
  (`qm_pkalculator/modify_smiles.py`'s `rm_proton`: `[CX4;H1-4]`,
  `[CX3;H1-2]`, `[CX2;H1]`).
- `js/nagl-model.js`'s new `CC.NAGL.predictAll()` — the shipped charge
  model's public `predict()` only returns heavy-atom charges
  (truncating synthetic explicit-H node charges); the pKa descriptor
  needs charges for every graph node including H's, so this adds a
  permanent, non-truncated API rather than hacking around it.
- Wired into `model-registry.json`/`model-registry.js`/
  `gnn-inference.js`'s existing engine-dispatch and atom-property-merge
  pattern (new `"engine": "pka"`) — requires its declared
  `descriptor.chargeSource` NAGL model to already be loaded, surfaced as
  a clear warning (not a crash) if it isn't, consistent with how one
  incompatible model already doesn't sink every other loaded model's
  predictions.
- `model/pka-ch-nagl/` — `manifest.json` + `weights.bin`, produced by
  `scripts/convert_pka_lightgbm.py` from a LightGBM text model file.

## Known limitations

- Se, Si, I: cannot be scored at all (NAGL-MBIS's trained vocabulary is
  H/C/N/O/F/P/S/Cl/Br).
- C-H acidity only — not titratable heteroatom groups.
- Accuracy is honestly worse than the original CM5-based tool (see
  table above) — a real tradeoff for running fully client-side.
