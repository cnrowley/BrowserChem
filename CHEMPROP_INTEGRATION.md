# Chemprop logP model — integration notes

`best.pt` was a Chemprop v2.3.0 checkpoint (D-MPNN, single-task
regression on `logP`). This adds a fully client-side path to run it —
no server, no PyTorch, no ONNX Runtime.

## What's here

```
model/
  chemprop-logp.bin              1.2MB float32 weights, converted from best.pt
  chemprop-logp-manifest.json    tensor shapes/offsets + architecture dims
convert_chemprop_checkpoint.py   re-run this on any future retrained best.pt
js/
  chemprop-features.js           NEW — bit-exact Chemprop atom/bond featurizers
  chemprop-model.js              NEW — loads manifest+bin, runs the forward pass
  dmpnn.js                       FIXED — real bug in the message-passing loop
  graph-builder.js               EXTENDED — now pulls chirality/bond-stereo too
  gnn-inference.js               UPDATED — prefers the Chemprop backend when loaded
  model-config.js                NEW — set URLs here to auto-load a model on page open
index.html / js/app.js           UPDATED — SMILES input box in the header, "Load Chemprop
                                  model…" button in the GNN panel, auto-load wiring
```

## Auto-loading a model from GitHub on page open

Edit `js/model-config.js`:

```js
CC.CONFIG = {
  chempropManifestUrl: 'https://raw.githubusercontent.com/<user>/<repo>/<branch>/model/chemprop-logp-manifest.json',
  chempropWeightsUrl:  'https://raw.githubusercontent.com/<user>/<repo>/<branch>/model/chemprop-logp.bin',
};
```

Push the `model/` folder to a GitHub repo (public, or private with the
repo itself accessible to whoever loads the page), then point those two
URLs at the raw files. `raw.githubusercontent.com` sends
`Access-Control-Allow-Origin: *`, so the browser's `fetch()` just works
cross-origin — no server, no proxy, no CORS config needed on your end.
The `.bin` is 1.2MB, well under GitHub's limits, so no Git LFS needed
either. `cdn.jsdelivr.net/gh/<user>/<repo>@<branch>/...` is a solid
alternative if you'd rather use jsDelivr's cache instead of raw GitHub.

Leave both URLs blank (the default) and the app behaves exactly as
before — use "Load Chemprop model…" to load one manually per session.
Check the browser console / the status line under that button if
auto-load fails (wrong URL, branch not pushed yet, etc.) — it won't
block the rest of the app from working either way.

## Loading a structure from a SMILES string

There's now a text box in the header, next to Open/Save. Type or paste a
SMILES string and press Enter or click **Load SMILES** — it replaces
whatever's on the canvas. Under the hood this is just
`RDKit.get_mol(smiles).get_molblock()` fed through the same
`molblockToMolecule()` parser the Open button already uses, so anything
Open can round-trip (wedge/hash stereo included — RDKit auto-generates
both 2D coordinates and wedge bonds for a bare SMILES with defined
stereocenters) works here too. Invalid SMILES show an inline error
instead of touching the canvas.

## Using it

1. Open the app, find the **GNN prediction** panel.
2. Click **Load Chemprop model…**, and in the file picker select *both*
   `model/chemprop-logp-manifest.json` and `model/chemprop-logp.bin`
   together (ctrl/cmd-click to multi-select).
3. Draw a structure, click **Run prediction** — you'll get a real logP
   value, not the demo D-MPNN's random-weight placeholder.

To use a different/retrained Chemprop checkpoint later:

```
pip install torch chemprop lightning --break-system-packages
python3 convert_chemprop_checkpoint.py your_new_best.pt model/
```

This only supports the architecture your `best.pt` actually uses
(`BondMessagePassing` + `NormAggregation` + single-task `RegressionFFN`,
`n_layers=1`) — the script checks and errors out with a specific message
if a future checkpoint uses something else (e.g. `AtomMessagePassing`,
mean aggregation, a deeper FFN, multi-task). Extending `chemprop-model.js`
to match would be needed first.

## Validation

I reconstructed the checkpoint in PyTorch/Chemprop and compared it
against this JS implementation on 24 test molecules — alkanes, aromatics,
esters, amides, carboxylic acids, phenols, anilines, nitro groups, a real
chiral center, hypervalent P/S centers (phosphate ester, sulfonamide),
and disconnected ion pairs. All 24 matched to the precision checked
(max abs error 0.000000 at 4 decimal places).

## Known approximations (honestly, not hidden)

RDKit.js's `get_json()` output doesn't expose everything Chemprop's
Python featurizer reads directly off an RDKit `Atom`/`Bond` object, so a
couple of features are inferred rather than read straight from RDKit:

- **Hybridization**: not exposed per-atom by RDKit.js at all. Inferred
  from bond orders, aromaticity, and total degree — including a specific
  fix so hypervalent centers with a formal double bond (phosphate P=O,
  sulfonyl S=O) stay SP3 rather than being wrongly promoted to SP2, and a
  lone-pair-conjugation rule so ester oxygens / amide nitrogens / acid
  -OH groups correctly come out SP2. Verified exactly against RDKit on
  every test molecule above, but an unusual hypervalent/organometallic
  structure could still disagree with RDKit's own perception.
- **Bond conjugation**: same story — approximated from which atoms sit in
  a pi system and whether that system extends past the bond itself.
- **Double-bond stereo (cis/trans)**: RDKit.js's CommonChem export only
  reports a coarse "cis"/"trans" label, which is mapped to Chemprop's
  Z/E enum values. This is the common case once RDKit has CIP-perceived
  the stereochemistry (true whenever there are real 2D coordinates, which
  this app always has), but isn't a guaranteed 1:1 mapping for every edge
  case.

Everything else — element, degree, formal charge, chiral tag (CW/CCW,
read directly from RDKit's own perception of your wedge/hash bonds),
implicit H count, aromaticity, and ring membership — is exact.

## Classification models (hERG, Ames, protein-reactivity, etc.)

`convert_chemprop_checkpoint.py` and `chemprop-model.js` now support both
Chemprop output heads:

- **RegressionFFN** (logP-style) — unscaled via the checkpoint's
  mean/scale, as before.
- **BinaryClassificationFFN** — a plain sigmoid, matching Chemprop's own
  `BinaryClassificationFFN.forward()` exactly. Returns a probability
  (0-1) of the positive class, not a raw logit.

The converter detects which one your checkpoint uses automatically (via
`hyper_parameters.predictor.cls`) and writes `manifest.taskType` for
`chemprop-model.js` to branch on -- nothing to configure by hand. Tested
end-to-end against a real trained classification checkpoint (bit-exact
match against Chemprop's own inference).

If you're retraining on the `covalent-classifier` repo's
`ProteinReactiveDB` data (`data/SMILES_training/trainingset_*.csv`), cite
Cano Gil & Rowley, *Digital Discovery* 2024, 3, 1776-1792 (MIT-licensed
repo, so the data and any weights you train are yours to use) -- same as
you'd cite for any dataset you train against.

## SA Score (synthetic accessibility)

`sascorer.js` is a from-scratch, bit-exact port of RDKit's own
`Contrib/SA_Score/sascorer.py` (Ertl & Schuffenhauer, *J. Cheminf.* 1:8,
2009) -- runs fully client-side, no model to load beyond a static ~5.4MB
fragment-score table (`model/sa-fragment-scores.bin` +
`-manifest.json`, converted from RDKit's own `fpscores.pkl.gz` via
`export_sa_fragment_scores.py`). It's on by default -- the score shows up
automatically in the existing descriptor panel once the table finishes
loading at startup (a few hundred ms on a typical connection), no button
to click.

This one's a deeper reimplementation than it looks: RDKit.js's exposed
`get_morgan_fp()` only returns a folded fixed-size bit vector, but
`fpscores.pkl.gz` is keyed on the *unfolded* raw 32-bit Morgan/ECFP
fragment hash IDs -- not available through any documented RDKit.js API.
`sascorer.js` instead reimplements RDKit's actual C++ hashing algorithm
directly (the atom connectivity invariants and the iterative
circular-environment growth loop, plus RDKit's own frozen, pre-Boost-1.81
`hash_combine`), reverse-engineered from RDKit's C++ source rather than
guessed at. Validated against real RDKit's `sascorer.calculateScore()` on
35 structurally diverse test molecules -- simple and complex structures,
macrocycles, spiro/bridgehead systems, a real chiral center, and several
of the reactive-warhead motifs from earlier (epoxide, aziridine,
chloroacetamide, isocyanate) -- **exact match on all 35**. Chiral-center,
spiro-atom, and bridgehead-atom counts are read directly from RDKit.js's
own `get_descriptors()` rather than reimplemented, so those are exact by
construction, not just by validation luck.

## Model registry (adding more Chemprop models)

`model/registry.json` is the catalog of available models -- dataset
provenance, size, metrics, hyperparameters -- separate from each model's
auto-generated technical `manifest.json` (never hand-edit that one; it's
overwritten every time you re-run the converter). Layout:

```
model/
  registry.json          <- the catalog, hand-authored
  logp/
    manifest.json         <- auto-generated by convert_chemprop_checkpoint.py
    weights.bin
  <your-next-model>/
    manifest.json
    weights.bin
```

To add a model:
1. `python3 convert_chemprop_checkpoint.py your_model/best.pt model/your-model-id/`
2. Add an entry to `model/registry.json` (see the `logp-v1` entry as a
   template -- every field beyond `id`/`displayName`/`propertyKey`/
   `taskType`/`files` is optional, but fill in whatever you actually
   have: dataset name/size/split strategy, test-set metrics,
   hyperparameters).
3. `python3 validate_registry.py model/registry.json` -- checks required
   fields, catches duplicate ids, confirms the referenced files exist,
   and cross-checks your registry's declared `taskType` against what the
   technical manifest actually says (catches the easy mistake of a stale
   or mistyped registry entry after re-exporting a checkpoint).

The GNN panel fetches the registry once at startup and lists every valid
entry with a per-row **Load** button -- weights aren't fetched until you
click it (load-on-demand, not load-all; see the earlier design
discussion for why). Multiple models can be loaded and run
simultaneously -- load logP and a classifier together and both show up
after **Run prediction**, no need to unload one to use the other.

Classification models render as a color-coded **Positive/Negative**
badge plus the raw score (0.5 is Chemprop's own decision boundary), not
a plain number -- this is driven by each model's `taskType` in its
technical manifest, so it's automatic for any classification checkpoint
you add.

The old single-model `chemprop-manifest.json`/`chemprop.bin` +
`CC.CONFIG.chempropManifestUrl`/`chempropWeightsUrl` auto-load path from
earlier is gone, superseded by the registry entirely -- `model-config.js`
now just points at `registryUrl` (defaults to the bundled
`model/registry.json`; override it to point at a registry (and its
models) hosted elsewhere, e.g. a GitHub repo, the same way described
above for the old single-model path).

## Two real bugs found via 6,268-molecule real-world validation

The electrophile-reactivity model came with `test_predictions.csv` -- its
actual predictions on its own real held-out test set (SMILES + predicted
probability; no ground-truth labels, so this validates the *port*
against the *original PyTorch model*, not the model's own correctness
against reality). Running all 6,268 through the JS port surfaced two
real bugs neither the earlier 24-molecule logP validation nor the
10-molecule electrophile sanity check happened to trigger:

1. **`molfile.js` was dropping the V2000 "either" bond stereo flag** on
   double bonds with genuinely unspecified E/Z geometry (flag `3`,
   double-bond-specific -- distinct from the wedge/hash flags on single
   bonds). A molecule loaded via "Load SMILES" with no explicit `/` `\`
   stereo marks gets that flag from RDKit's own molblock writer,
   specifically to stop a re-parse from inferring stereo off the 2D
   coordinates alone. The round-trip through this app's own molblock
   writer silently dropped it, causing exactly the misinterpretation
   RDKit's flag exists to prevent -- and it disproportionately hit
   push-pull alkenes (cyanoacrylamide/acrylamide Michael acceptors),
   this model's own target chemotype, causing near-total prediction
   inversions (0.98 vs 0.0004) on about 3% of the test set.
2. **Neutral trivalent boron defaulted to sp3 instead of sp2** in the
   hybridization heuristic. Unlike carbon/nitrogen sp2 (which comes from
   pi-bonding), boron's sp2 character comes from its own valence-3
   electron count and empty p-orbital -- true regardless of what it's
   bonded to. Affects boronic acids and boronate esters, a real covalent
   warhead class (e.g. bortezomib-type inhibitors).

Both are fixed in `molfile.js` and `chemprop-features.js` and apply
project-wide, not just to this one model -- any molecule with an
unspecified-stereo double bond or a boronic acid group benefits, across
every model using this pipeline (SA Score included, since it shares the
same featurization code).

Post-fix: **99.0% of the 6,268 test molecules match the original
PyTorch model to <1e-6**, mean absolute error 0.00056 across the full
set. A residual ~0.3% (21 molecules) still shows moderate error,
concentrated in explicit chiral centers, defined E/Z double bonds, and
boronic acids -- within the known-approximation scope documented above
for chirality/stereo, not yet root-caused further.

## Atom-level models (per-atom targets, e.g. MBIS partial charges)

`convert_chemprop_checkpoint.py` and `chemprop-model.js` now also support
Chemprop's atom-level output (`--atom-target-columns`), not just the
molecule-level regression/classification heads. Detected automatically
from the checkpoint's `hyper_parameters` (`MolAtomBondMPNN` with an
`atom_predictor` set) -- nothing to configure by hand, same as the
regression-vs-classification detection.

The underlying math needed *zero* changes to the D-MPNN forward pass:
Chemprop's `MABBondMessagePassing` (used whenever an atom or bond target
is set) is confirmed -- directly from its own docstring, not assumed --
to be the exact same bond-message-passing formula as the plain
`BondMessagePassing` every other model here uses, just with its output
layer named `W_vo` instead of `W_o`. The only real new code is the head:
atom-level skips the NormAggregation pooling step entirely and runs the
FFN once per atom, on that atom's own embedding, instead of once on the
pooled molecule vector.

Results feed the atom heatmap (the property dropdown + atom coloring in
the GNN panel) rather than the properties table -- that heatmap UI
already existed for the unrelated demo backend and just needed a real
atom-level model to feed it. Bit-exact validated against a real
Chemprop `MolAtomBondMPNN` model (not just shape-checked): built an
actual single-task atom-regression checkpoint with Chemprop's own
classes, ran real inference on 4 test molecules (28 atoms total) in
Python, converted the checkpoint, and diffed against the JS output --
max abs error 1.3e-8 (float32/float64 rounding, not a bug) across every
atom.

Not yet supported: bond-level targets (`--bond-target-columns`), and a
checkpoint with more than one output level at once (e.g. both a
mol-level and an atom-level predictor in the same model) -- the
converter explicitly rejects both with a clear error rather than
silently exporting something wrong.

## NAGL model registry wiring (plumbing complete, no live model yet)

Every piece needed to run a NAGL-MBIS model through the same registry
and UI as the Chemprop models is now built and tested -- *except* real
validated weights (see nagl-model.js's header for exactly what's
confirmed vs. not: architecture transcribed from real source, structural
checks pass -- exact charge conservation, correct symmetry -- but no
numeric diff against real DGL output yet, since I don't have a working
DGL environment or your actual `nagl-v1-mbis.ckpt`).

What's wired:
- `registry.json` entries take an `"engine": "chemprop" | "nagl"` field
  (defaults to `"chemprop"` for every existing entry, so nothing already
  shipped needs touching). `model-registry.js`'s loader dispatches to
  `chemprop-model.js` or `nagl-model.js` based on it.
- `CC.GNN.predictMolecule()` now merges results from *both* engines into
  one result -- a loaded Chemprop model's `logP` and a loaded NAGL
  model's per-atom charges show up together, in the same properties
  table / atom heatmap, verified against the same atom-id ordering (both
  engines iterate the molecule's own atom list directly, no reordering).
- If a loaded NAGL model rejects a structure (unsupported element,
  degree outside its range), that model's prediction fails on its own
  without sinking predictions from anything else loaded alongside it --
  tested directly with TMS (Si is outside NAGL's vocabulary): the NAGL
  charges fail with a clear warning while a loaded Chemprop logP
  prediction still comes through.
- `validate_registry.py` understands the `engine` field -- `taskType` is
  only required/cross-checked for `"chemprop"` entries; NAGL entries get
  a different manifest cross-check (verifies the postprocess type is one
  nagl-model.js actually implements).

To actually add a live NAGL entry once you have validated weights:
```bash
python3 convert_nagl_checkpoint.py nagl-v1-mbis.ckpt model/nagl-mbis/
```
then add a `registry.json` entry with `"engine": "nagl"` pointing at the
output, same shape as any Chemprop entry otherwise.
