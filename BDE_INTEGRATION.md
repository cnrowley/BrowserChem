# Bond dissociation enthalpy (BDE) — integration notes

## What this is

A real, from-scratch-trained Chemprop D-MPNN checkpoint predicting
**bond dissociation enthalpy** (kcal/mol) for every bond in a molecule —
this app's first **bond-level** (as opposed to atom- or molecule-level)
property, and its first new UI surface for coloring/labeling bonds
directly on the 2D canvas.

This was originally scoped as a port of
[ALFABET](https://github.com/NREL/alfabet) (St. John et al., *Nature
Communications* 2020) — reverse-engineering its real pretrained
TensorFlow + `nfp` model layer-by-layer (residual `EdgeUpdate`/
`NodeUpdate` blocks, a source/target-swap bug in `EdgeUpdate`'s concat
order, an off-by-one ring-size quirk in its vocabulary hashing) and
reaching bit-exact numeric parity against the real model in a from-
scratch numpy reconstruction. That approach was abandoned mid-flight —
not because it didn't work, but because it meant standing up a second,
genuinely different inference engine (TensorFlow-shaped weights, a
vocabulary-embedding featurizer, two separate legacy conda environments
just to extract weights) alongside this project's existing, already-
proven Chemprop D-MPNN pipeline, for a property that pipeline turns out
to support natively.

## Why Chemprop instead

Chemprop v2 (confirmed directly from its installed source,
`chemprop/models/mol_atom_bond.py`) has a native bond-level predictor:
its D-MPNN hidden states already live on directed bonds, so a bond-level
head is a small, well-trodden addition to the SAME engine already
powering every other GNN property in this app (`js/dmpnn.js`,
`js/chemprop-model.js`), not a new one. Concretely, Chemprop's own
bond-predictor fingerprint is `concat([H_e[edge], H_e[reverse_edge]])`
fed through the FFN head, with the final per-bond value averaging the
two directions' outputs — `js/chemprop-model.js`'s `runOneBondLevel()`
replicates this exactly, validated bit-exact (~1e-6, float32 precision)
against real chemprop output on ethane/toluene/methanol during
conversion (see `scripts/convert_chemprop_checkpoint.py`).

The one new piece of math needed: Chemprop's edge_finalize step
(`H_e = ReLU(W_eo · [bondFeat, h])`, confirmed from
`chemprop/nn/message_passing/mol_atom_bond.py`) — a dedicated edge-output
projection this project's D-MPNN forward pass (`js/dmpnn.js`) never
needed before, since every prior checkpoint only read atom embeddings.
`Wo`/`Weo` are now each independently optional in `runDMPNN()`: a
bond-level checkpoint has no atom-output layer at all (Chemprop never
builds `message_passing.W_vo` without a mol/atom predictor attached).

## Dataset and training

[BDE-db2](https://github.com/patonlab/BDE-db2) (S. V. Shree Sowndarya,
Kim, Kim, St. John, Paton — *Digital Discovery* 2023), "Model 3": the
final, broadest-coverage dataset from that paper, extending the original
ALFABET/BDE-db (C/H/N/O only) to also cover halogenated (F/Cl/Br/I) and
S/P-containing organic molecules — 65,740 molecules, 834,066 real
DFT-computed per-bond BDE/BDFE labels. See
`scripts/prepare_bde_training_data.py` for the reshape from BDE-db2's
long (one-row-per-bond) format into Chemprop's wide bond-target CSV
format.

`bond_index` in the raw data was confirmed (by direct RDKit cross-check,
not assumed) to index into `Chem.AddHs(mol)`'s own bond order — the same
explicit-hydrogen convention this project already uses for the 1H NMR
checkpoint (`js/chemprop-features-explicit-h.js`) — so training used
`--add-h`, and every molecule's full bond list (including C-H/N-H/O-H)
gets a real prediction, not just the heavy-heavy bonds that are actually
drawn.

Trained with real Chemprop 2.2.3 (`cov-chemprop` conda env),
`hiddenSize=300, depth=3` matching this project's convention, 40 epochs,
reusing BDE-db2's own train/valid/test split (63,740/1,000/1,000
molecules) rather than re-splitting, so held-out numbers are comparable
to the source paper's own.

**Real held-out test-set metrics** (12,586 individual bond labels across
1,000 held-out molecules, computed directly from this checkpoint's own
predictions):

| Metric | Value |
|---|---|
| MAE | 1.20 kcal/mol |
| RMSE | 2.11 kcal/mol |

Higher error than ALFABET's own published 0.58 kcal/mol MAE — expected,
not a red flag: this is a single untuned 40-epoch run (no ensembling, no
hyperparameter search) on a broader chemical space than ALFABET's
original C/H/N/O-only domain, not a like-for-like comparison.

## Display: bond heatmap + per-atom "weakest attached H" aggregate

Per-bond values for the drawn (heavy-heavy) bonds are shown via a new
bond heatmap (`js/bond-heatmap.js`, mirroring `js/atom-heatmap.js`'s
structure and reusing its exported `CC.heatColor`): color-coded bond
lines plus toggleable numeric labels at each bond's midpoint. Unlike the
atom heatmap's charge-specific zero-centering, bond BDE always uses the
reversed colormap (low/weak/reactive → red, high/strong → blue) — the
chemically useful "hot spot" framing regardless of which specific
bond-level property is loaded.

C-H/N-H/O-H bonds aren't individually drawn in this app's 2D editor
(hydrogens are implicit), so their BDE values can't be shown as colored
bonds the same way. Instead, each heavy atom's own attached-H bonds are
aggregated down to one number — the **minimum** (not mean, unlike the 1H
NMR shift aggregate) — surfaced as a `BDE-XH` property through the
existing atom heatmap. This is deliberate: several equivalent protons
genuinely share one true NMR shift, so averaging there is exact for the
common case, but several C-H's on the same carbon do NOT share one true
BDE, and the chemically useful number is "how easy is it to break the
weakest bond here" (this is literally ALFABET's own headline use case —
finding labile C-H sites), not their average.

## Known limitations (documented honestly, not glossed over)

- **No domain-of-validity gating yet.** Every other atom-level model in
  this project (`pka-model.js`'s `checkCompatibility`,
  `chemprop-features.js`'s `checkChempropCompatibility`) flags molecules
  containing elements/environments outside the model's training
  vocabulary. This checkpoint doesn't have an equivalent check yet —
  Chemprop's own `MultiHotAtomFeaturizer` degrades gracefully (pads
  unseen values into an "other" bucket) rather than hard-failing, so
  out-of-domain predictions won't error, but their accuracy is
  unverified outside BDE-db2's own element coverage.
- **BDFE not trained.** BDE-db2 provides bond dissociation free energy
  labels alongside BDE (ALFABET predicts both), but this first pass only
  trained the `bde` target — a natural follow-on using the identical
  pipeline (`scripts/prepare_bde_training_data.py --bdfe`, already
  wired to emit a `bdfe` column).
- **Single training run, no ensembling.** ALFABET's own reported
  accuracy comes from a more carefully tuned setup; see the metrics
  table above.
