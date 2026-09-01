# Carbon acid (C-H) pKa predictor — integration notes

## What this is

A from-scratch, client-side port of [jensengroup/pKalculator](https://github.com/jensengroup/pKalculator)
(Borup, Ree & Jensen, "pKalculator: A pKa predictor for C-H bonds",
*Beilstein J. Org. Chem.* 2024, 20, 1614-1622,
[10.3762/bjoc.20.144](https://doi.org/10.3762/bjoc.20.144), MIT-licensed)
— predicts the pKa of individual C-H bonds (deprotonation to
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

Against the rest of the published C-H-acidity-specific ML literature
(not general aqueous pKa, a much easier problem most other pKa models
target -- see below), checked against the real papers, not abstracts,
2026-08-30:

| | Charge descriptor | MAE | RMSE |
|---|---|---|---|
| Grzybowski/Roszak GCNN (*J. Am. Chem. Soc.* 2019, 141, 17142-17149, 10.1021/jacs.9b05895) | Gasteiger (empirical equilibration, no QM) | 2.1 | -- |
| An/Liu/Cai/Shao EEGpKa (*J. Chem. Inf. Model.* 2024, 64, 2383-2392, 10.1021/acs.jcim.3c00958), DMSO set | Gasteiger partial charge (one of several plain atom features) | 2.03 | 2.81 |
| **This app (NAGL-MBIS charges)** | NAGL-MBIS (GNN-predicted approximation of real ab initio MBIS charges) | **1.40** | **2.49** |
| pKalculator (this app's own source model) | real CM5 (GFN1-xTB semiempirical QM, via an actual `xtb` run) | 1.24 | 2.15 |

Solidly mid-pack against dedicated published methods for this specific,
harder-than-average subproblem. The other direction worth being honest
about: none of the general-purpose aqueous pKa models in the broader
literature (e.g. GraFpKa, or this app's own `pka-microstate-freeenergy`
and its Uni-pKa-lineage peers -- see
`PKA_MICROSTATE_FREEENERGY_INTEGRATION.md`) are a fair comparison here at
all, MAE 0.4-0.6 and all -- general O-H/N-H pKa is a substantially easier
target (smaller dynamic range, far more experimental training data,
usually measured directly in water) than DMSO-scale carbanion acidity.

## Could this be improved further?

Checked against the real papers (not secondhand summaries) on
2026-08-30 -- one of the two ideas floated earlier turned out not to
hold up, which is itself worth recording rather than quietly dropping.

**Data augmentation does NOT look promising here, on closer reading.**
An/Liu et al.'s EEGpKa isn't literally synthesizing new (structure, pKa)
training examples -- it's a self-supervised PRETRAINING step: H atoms on
CH3/CH2/CH groups get replaced with 30 real substituents spanning
known +I/-I/+C/-C electronic effects (their own Table 2), and the
message-passing unit is pretrained to predict four auxiliary targets
(substituent-to-ionization-site distance, which of the 30 substituents,
whether it strengthens/weakens acidity, and I- vs. C-type effect) before
being fine-tuned on the real pKa task. It DOES measurably help in a
low-data regime -- their own ablation (Figure 9) shows a real accuracy
gain over the same architecture without pretraining at 20%/33% of
their training set. But their absolute numbers on the DMSO benchmark
this app's own model targets are MAE 2.03 / RMSE 2.81 (their own
dataset1, 671 iBonD compounds -- not pKalculator's dataset, no direct
train/test overlap with this app's own split either) -- worse than what
`pka-ch-nagl` already gets (1.40), and only marginally better than
Grzybowski's original 2019 GCNN it was implicitly benchmarked against.
Layering this pretraining step onto the existing LightGBM + NAGL-MBIS
pipeline is not obviously going to help when the technique's own
published ceiling is already behind this app's current number.

**Charge-descriptor quality is the better-supported lever, now with a
real three-point trend instead of a two-point coincidence.** All three
papers checked in full report which atomic-charge descriptor they used,
and the ranking tracks charge quality exactly:

- Gasteiger (crude, empirical, no QM at all) → MAE 2.03-2.1
- NAGL-MBIS (this app; a GNN's learned approximation of real ab initio MBIS charges) → MAE 1.40
- CM5 (real semiempirical-QM charges, an actual GFN1-xTB calculation) → MAE 1.24

Roszak et al.'s own ablation makes this causal, not just correlational,
within a single fixed architecture: removing their Gasteiger-charge
feature (keeping everything else the same) costs them 0.4 pKa units of
MAE on its own. The concrete, scoped next step this points to:
**recalibrate/fine-tune NAGL-MBIS's own weights specifically against
real CM5 charges on this dataset's 3910 real C-H sites**, rather than
trusting NAGL's existing generic MBIS-charge training (fit on a broader,
unrelated molecule population) to transfer well to this specific
downstream task untouched. This keeps the deployed model exactly as
browser-compatible as today (NAGL-MBIS still runs client-side at
inference time -- only its training weights would change, the same way
`pka-microstate-freeenergy`'s own checkpoint gets retrained without
touching runtime code) while directly targeting the one variable that
now has real, converging evidence behind it. Not yet attempted, and not
free: it needs a real `xtb` install (not present on this dev machine
as of this check -- would need `conda install -c conda-forge xtb` or
equivalent) to generate the real CM5 reference charges to fine-tune
against, then an actual retraining pass with its own real risk of not
paying off (see this app's own `metrics.note` fields elsewhere in
`model/registry.json` for two disclosed examples, on this project's
OTHER pKa model, of a principled-looking fix making things worse).

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
