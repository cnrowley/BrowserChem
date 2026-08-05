# NMR chemical shift predictors — integration notes

## What this is

Three from-scratch-trained Chemprop D-MPNN checkpoints predicting NMR
chemical shift per atom: **13C**, **19F**, and **1H** (aggregated — see
below). All three run entirely client-side through this app's existing
Chemprop inference pipeline (`js/chemprop-model.js`), the same engine
`logp-v1`/`logs-aqsoldb`/`melting-point`/`electrophile-reactivity-v1`
already use for molecule-level predictions — these are the first
**atom-level** Chemprop checkpoints this app has ever shipped through
that engine (`nagl-mbis-charges` and `pka-ch-nagl` are atom-level too,
but use entirely separate engines).

A fourth nucleus, **15N**, was also trained but deliberately **not
shipped** — NMRShiftDB2 (the training data source) only has 84 15N
spectra total (vs. 43,258 for 13C), and the resulting checkpoint's real
held-out test RMSE was ~164 ppm — not a usable model, just an honest
negative result.

## Datasets and training

See `scripts/download_nmr_datasets.py`, `scripts/prepare_nmr_training_data.py`,
and `scripts/train_nmr_chemprop.sh` for the full pipeline, and their own
header comments for the real gotchas hit along the way:

- **13C / 1H / 15N**: NMRShiftDB2 (`nmrshiftdb.nmr.uni-koeln.de`), a
  peer-reviewed open NMR database with per-atom-indexed shift
  assignments baked into its bulk SD file's own properties.
- **19F**: NMRShiftDB2's own (small, ~1000-record) native 19F data,
  combined with NMRexp (Zenodo, DOI 10.5281/zenodo.17296666) — a much
  larger 2025 literature-mined database, restricted to molecules with
  exactly one reported 19F shift (unambiguous atom assignment; NMRexp's
  format doesn't map shifts to specific atoms the way NMRShiftDB2's
  does, so multi-fluorine molecules were dropped rather than guessed).
- Trained with real Chemprop 2.2.3 (`cov-chemprop` conda env), the same
  `hiddenSize=300, depth=3` convention every other checkpoint in this
  project's registry uses.
- Two real RDKit round-trip bugs were found and fixed by the data-prep
  pipeline's own atom-order self-check (every accepted molecule's
  written SMILES is re-parsed and diffed against the label array before
  being trusted) — `Chem.MolToSmiles(mol, canonical=False)` does **not**
  preserve input atom order on re-parse (the `_smilesAtomOutputOrder`
  property is the correct tool for this), and RDKit's default
  SMILES-parse sanitization silently strips explicit H atoms baked into
  a SMILES string back into implicit H counts.

Held-out test-set results (real, from `chemprop train`'s own logged
`test/atom/mse`, not estimated):

| Nucleus | Test RMSE | Molecules | Atom labels |
|---|---|---|---|
| 1H (per-hydrogen, before aggregation) | 0.65 ppm | 13,085 | 137,926 |
| 13C | 3.98 ppm | 30,322 | 311,760 |
| 19F | 27.5 ppm | 49,738 | 53,402 |
| 15N (not shipped) | ~164 ppm | 81 | 125 |

19F's larger error reflects NMRexp being LLM-extracted from paper text
(not curated/reviewed the way NMRShiftDB2 is) and combining two
heterogeneous sources whose solvent/referencing conventions aren't
accounted for — a real, not-yet-fixed source of label noise, not a
training bug.

## The 1H explicit-hydrogen problem

Every other atom-level model in this app (NAGL charges, C-H pKa, 13C/19F
shift) predicts a value per **heavy** atom — this app's own molecule
representation never has explicit hydrogen atoms. 1H shift is
fundamentally different: it's a property of the hydrogen itself, and a
heavy atom can carry several chemically distinct hydrogens.

The 1H checkpoint was trained with chemprop's `--add-h` flag: every
hydrogen gets its own D-MPNN graph node, using
`js/chemprop-features-explicit-h.js` — a new, separate graph builder
(not a modification of `graph-builder.js`'s normal heavy-atom-only
`buildMolGraphChemprop`), mirroring the same heavy-atoms-then-appended-H
convention `js/nagl-features.js` and `js/pka-descriptor.js` already
established elsewhere in this project. Feature encoding for the explicit
H nodes was confirmed directly against real RDKit (not assumed): after
`Chem.AddHs()`, a heavy atom's `GetTotalNumHs()` drops to 0 (no more
*implicit* H's — they're real neighbors now) while `GetTotalDegree()`
stays numerically the same; a synthetic H atom's own
`GetHybridization()` is `HybridizationType.UNSPECIFIED`, which isn't in
`chemprop-features.js`'s hybridization choice list at all, so it lands
in that feature's pad/"unknown" bucket exactly the way real Chemprop's
own encoder handles it.

The model genuinely predicts a distinct value per hydrogen atom
internally. But this app's atom-heatmap UI colors real *drawn* (heavy)
atoms only — the same reason `nagl-model.js`'s public `predict()`
doesn't surface its own synthetic H nodes' charges individually — so
`chemprop-model.js`'s `runOneAtomLevelExplicitH()` aggregates (mean)
each heavy atom's own attached-H predictions back down to one displayed
number. Chemically equivalent protons (a CH3's three H's) genuinely
share one true shift, so this is exact for the common case;
diastereotopic (inequivalent) protons on the same heavy atom collapse to
their average instead of their true distinct values — a real, bounded
display-layer simplification, on top of an equivalent one already made
during training-label construction (NMRShiftDB2 itself only
heavy-atom-indexes 1H shifts, so multiple reported values on one heavy
atom were already averaged before training — see
`scripts/prepare_nmr_training_data.py`'s header for that earlier step).

## New manifest fields (`chemprop-model.js`, `convert_chemprop_checkpoint.py`)

- `manifest.graphType`: `"heavy"` (default) | `"explicit-h"` — which
  graph builder a given atom-level checkpoint needs. Only 1H uses
  `"explicit-h"` today.
- `manifest.applicableElement`: e.g. `"C"` for 13C, `"F"` for 19F — gates
  which heavy atoms actually get annotated (a 13C model should never
  label an oxygen), the same masking pattern `pka-model.js` already used
  for its own candidate-site gating. Not applied for `"explicit-h"`
  models, where "H" documents the target species rather than a literal
  heavy-atom filter (hydrogen is never a heavy atom in this app's
  molecule model — filtering by `element === 'H'` would exclude
  everything).
- `convert_chemprop_checkpoint.py` also gained a hard `d_v == 72 and
  d_e == 14` assertion — a checkpoint trained with a non-default
  featurizer would previously convert "successfully" and silently feed
  `chemprop-model.js`'s forward pass wrong-shaped input; it now aborts
  loudly at conversion time instead.

## Two real bugs found during integration testing (not NMR-specific, fixed project-wide)

- **`js/chemprop-model.js`'s `elementMask`** initially checked
  `heavyAtom.element === model.applicableElement` unconditionally,
  including for the 1H explicit-h model (`applicableElement: "H"`) —
  since hydrogen is never a heavy atom here, this silently zeroed out
  every 1H prediction. Fixed to skip element-masking for `"explicit-h"`
  models (the NaN check for heavy atoms with no attached H's already
  does the right filtering there).
- **`js/app.js`'s heatmap-property dropdown** populated itself from
  `Object.keys(result.atomProperties[0])` — only the *first* atom's own
  property keys. This silently missed any property that happened to be
  absent from atom index 0 (e.g. `shift_19f` on a molecule whose first
  heavy atom is a carbon) — a latent bug in the existing app, never
  triggered before because every prior sparse atom-level property
  (`pka-ch`) happened to test out on molecules where atom 0 had it.
  Fixed to union keys across every atom.

## Validation

Cross-checked against real literature values on fluorobenzene and
4-fluorophenylacetic acid: ipso-13C 163.4 ppm (lit. ~163), correct
ortho/meta ring symmetry, 19F -110.7 ppm (lit. ~-113.5), aromatic 1H
~7.1-7.3 ppm, CH2 ~3.6 ppm, COOH proton 10.8 ppm (lit. range 10-13) —
all chemically sensible, including molecular symmetry being correctly
respected across all three nuclei simultaneously and alongside every
other loaded model (NAGL charges, pKa, the four molecule-level
properties) with zero prediction warnings.
