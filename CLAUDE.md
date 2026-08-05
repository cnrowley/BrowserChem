# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ChemCanvas: a browser-based 2D/3D molecule editor with client-side ML
property prediction. No server, no Python runtime in the browser —
RDKit.js (WASM) plus hand-rolled JavaScript inference engines run
entirely client-side. Python only comes in via `scripts/`, offline, to
convert/validate model checkpoints before they're committed as static
assets.

## Running / testing changes

There is no build step, no bundler, no package.json, and no JS test
runner. Serve the directory root with any static file server and open
`index.html`:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

`file://` will not work — RDKit's WASM module and the JSON model/data
files must be fetched over HTTP. After any change, actually load the
page and exercise the feature (draw a structure, check the console/status
bar) rather than assuming it works — there's no automated test suite to
fall back on.

Model-tooling scripts under `scripts/` are ordinary argparse CLIs, e.g.:

```bash
python3 scripts/validate_registry.py model/registry.json
python3 scripts/convert_chemprop_checkpoint.py your_model/best.pt model/your-model-id/
python3 scripts/convert_nagl_checkpoint.py nagl-v1-mbis.ckpt model/nagl-mbis/
```

`convert_chemprop_checkpoint.py` needs `torch`, `chemprop`, `lightning`
installed (not part of this repo's runtime). Some scripts
(`extract_mbis_charges.py`, `qca_step1_probe.py`) need `qcportal` and live
network access to QCArchive; `nagl_probe.py` expects to run inside a local
nagl-mbis/PyTorch environment the sandbox doesn't have — these are meant
to be run by the user on their own machine, not executed here.

## Architecture

### No modules, no globals collision — the `CC` namespace

Every JS file attaches to a single global `window.CC` namespace (e.g.
`CC.Molecule`, `CC.render`, `CC.GNN`). There is no ES module system or
bundler; load order in `index.html` **is** the dependency graph — a file
can only reference `CC.X` if a script defining `CC.X` appears earlier in
the `<script>` list. When adding a new file, add its `<script>` tag in
the right position relative to what it depends on and what depends on it.
Third-party libraries (RDKit, ONNX Runtime, 3Dmol, xlsx, jsPDF) are
loaded as UMD globals from CDNs at the very top of `index.html`, then
normalized into `window.chemCanvasLibs` by `js/lib-loader.js`.

### Core layering (roughly load order)

1. **Model layer** — `molecule.js` (pure graph: atoms/bonds Map, no DOM
   or chemistry validation), `geometry.js`, `history.js` (undo/redo
   stack), `molfile.js` (V2000 molfile read/write).
2. **Chemistry via RDKit.js** — `chemistry.js` wraps RDKit calls
   (validation, descriptors, canonical SMILES); `qed.js` and
   `sascorer.js` are from-scratch, bit-exact ports of RDKit's own QED and
   SA Score algorithms (validated against RDKit's reference
   implementations, not just approximated).
3. **2D editing** — `tools.js` (the interaction controller: tool state
   machine driving atom/bond/chain/ring/charge/erase tools) and
   `render.js` (SVG rendering) sit on top of the model layer.
   `stereo2d.js` handles wedge/hash stereo perception.
4. **3D** — `embed3d.js` (implicit-H addition, a from-scratch
   UFF/MMFF-style force field: harmonic bonds/angles, periodic torsions,
   out-of-plane terms, LJ nonbonded, staged minimization, torsion-driven
   conformer search) and `viewer3d.js` (3Dmol.js wiring). `vec3.js`
   and `elements.js` are shared low-level utilities.
5. **GNN inference pipeline** — this is the most involved part of the
   codebase; see below.
6. **UI glue** — `app.js` is the only file that touches every other
   piece together (DOM wiring, event handlers). It is intentionally the
   "everything" file; other modules stay independently testable/reasoned
   about. `export.js` (CSV/XLSX/PDF), `radar-chart.js`, `hrms.js`,
   `atom-heatmap.js`, `smarts-filters.js` are feature-specific UI panels
   `app.js` wires up.

### GNN inference pipeline (Chemprop D-MPNN + NAGL)

This is a hand-rolled, from-scratch reimplementation of trained PyTorch
models, not a generic ONNX/ORT path (ONNX Runtime is loaded but only used
for a legacy manual-load demo path). Understanding it requires reading
multiple files together:

- `atom-features.js` / `bond-features.js` / `chemprop-features.js` —
  Chemprop-compatible featurizers built on RDKit.js's `get_json()`
  output. Several features (hybridization, conjugation, cis/trans) aren't
  directly exposed by RDKit.js and are *inferred* — see
  `CHEMPROP_INTEGRATION.md`'s "Known approximations" section before
  touching these; the inference rules encode real, previously-debugged
  edge cases (hypervalent P/S, boron sp2, unspecified-stereo double
  bonds).
- `graph-builder.js` — turns a `CC.Molecule` into the atom/bond graph
  tensors the D-MPNN forward pass consumes.
- `dmpnn.js` — the bond-centered D-MPNN message-passing forward pass
  (`BondMessagePassing`/`MABBondMessagePassing` — same math, used
  whichever Chemprop head is active).
- `pooling.js` — `NormAggregation` for molecule-level output (skipped
  entirely for atom-level targets).
- `gnn-heads.js` / `chemprop-model.js` — output heads: regression
  (unscale by checkpoint mean/scale), binary classification (sigmoid),
  and atom-level (FFN run per-atom instead of on the pooled vector).
  `chemprop-model.js` loads a manifest.json + weights.bin pair produced
  by `scripts/convert_chemprop_checkpoint.py` and detects task type from
  the manifest — never hand-edit a `manifest.json`, it's fully
  regenerated by the converter.
- `nagl-features.js` / `nagl-model.js` — separate engine (GraphSAGE +
  electronegativity equalization) for NAGL-MBIS per-atom partial charges,
  produced by `scripts/convert_nagl_checkpoint.py`.
- `model-registry.js` — fetches `model/registry.json` at startup, lists
  available models, dispatches load-on-demand (weights aren't fetched
  until the user clicks Load) to either the Chemprop or NAGL engine based
  on each entry's `"engine"` field.
- `gnn-inference.js` — `CC.GNN.predictMolecule()` merges results from
  every currently-loaded model (Chemprop + NAGL together) into one result
  set; a failing model (e.g. NAGL rejecting an out-of-vocabulary element)
  doesn't sink predictions from other loaded models.
- `atom-heatmap.js` — renders atom-level predictions (e.g. per-atom
  charges) as canvas coloring, separate from the molecule-level
  properties table.
- `model-config.js` — optional auto-load URLs (`registryUrl`, or the
  legacy single-model URLs) for auto-loading a model on page open instead
  of requiring a manual file picker.

### Model registry (`model/registry.json`)

The hand-authored catalog of available models (dataset provenance, size,
metrics, hyperparameters) — deliberately separate from each model's
auto-generated `manifest.json` (tensor shapes/offsets/architecture dims;
regenerated by the converter scripts, never hand-edited). Each entry
declares `"engine": "chemprop"` or `"nagl"`; `taskType` is only
required/cross-checked for `"chemprop"` entries. Run
`scripts/validate_registry.py model/registry.json` after any registry
edit or after adding a new model directory — it catches duplicate ids,
missing files, and registry/manifest taskType mismatches.

### Structural alerts

`smarts-filters.js` + `data/smarts_filters.json` — ~1250 medicinal-
chemistry SMARTS filters (PAINS, Glaxo, Dundee, BMS, SureChEMBL, MLSMR,
Inpharmatica, LINT) run against RDKit.js substructure matching, with
match highlighting on the 2D canvas.

## Conventions and gotchas

- **This project documents its own validation status honestly** —
  `CHEMPROP_INTEGRATION.md` and `registry.json`'s `notes` fields state
  plainly what's bit-exact-validated against real ground truth vs. what's
  a reasonable-but-unverified approximation. Read the relevant section
  before changing featurization or model code, and preserve this honesty
  norm in any new code/comments/registry entries you add — don't upgrade
  a documented approximation to sound more certain than it is.
- `js/index.html` and `js/styles.css` are stray unused duplicates of the
  real `index.html` and `css/styles.css` — not referenced by anything.
  Edit the top-level `index.html` / `css/styles.css`, not the copies
  under `js/`.
- No linter or formatter is configured; match the existing style
  (4-space-ish, semicolons, `CC.Namespace.thing` conventions) in
  whichever file you're editing.
- `chemcanvas-repo/` at the repo root is currently empty — not part of
  the working app.
