# ChemCanvas

A browser-based 2D/3D molecule editor with client-side machine-learning
property prediction — no server, no Python runtime in the browser.
Everything runs as WASM (RDKit.js) and hand-rolled JavaScript inference
engines for the trained models.

## What's in here

- **Structure validation layer** — 14 real structural checks (valence,
  charges, radicals, salts/counterions, duplicates, stereochemistry,
  aromaticity, metals, hypervalent atoms, unusual isotopes, and more) run
  before any property model, feeding a three-tier per-model verdict
  (compatible / outside applicability domain — warned / incompatible —
  blocked) shown in a dedicated Validation tab and as badges next to
  every model in the Properties panel — see `VALIDATION.md`.
- **2D structure editor** — ChemDraw-style drawing tools, ring templates,
  keyboard shortcuts, standard-bond-length "clean up" via RDKit's own
  coordinate generator.
- **3D structure generation** — implicit hydrogens, a from-scratch
  UFF/MMFF-*style* force field (harmonic bonds/angles, periodic torsions,
  out-of-plane terms, 12-6 Lennard-Jones nonbonded, staged minimization,
  real torsion-driven conformer search), rendered via 3Dmol.js.
- **RDKit-backed 2D properties** — standard descriptors, a bit-exact
  from-scratch port of RDKit's SA Score, and a from-scratch port of QED
  validated against RDKit's own reference doctests.
- **Client-side GNN property prediction** — a hand-rolled D-MPNN forward
  pass (Chemprop-compatible, validated to <1e-6 against real PyTorch
  output) for logP, logS, melting point, and a protein-reactivity
  classifier, plus a from-scratch NAGL-MBIS port (GraphSAGE +
  electronegativity equalization) for per-atom partial charges.
- **Per-atom prediction** — NAGL-MBIS partial charges, a from-scratch C-H
  pKa predictor (`PKA_INTEGRATION.md`), and 13C/19F/1H NMR chemical shift
  (`NMR_INTEGRATION.md`), all via the same atom-heatmap UI.
- **ANI-2x neural network potential** — a from-scratch AEV + per-element
  ensemble-network forward pass, with analytic backprop for forces, to
  drive 3D geometry optimization and report total energy (H/C/N/O/F/S/Cl,
  neutral molecules only) — see `ANI2X_INTEGRATION.md`.
- **OpenFF Sage (SMIRNOFF) force field** — a real published small-molecule
  force field (`openff-2.1.0.offxml`), typed via SMARTS matching against
  RDKit.js, as a third 3D generation method alongside Classical and
  GeoMol; electrostatics use this app's own NAGL-MBIS charges in place of
  Sage's official AM1-BCC protocol — see `OPENFF_INTEGRATION.md`.
- **Conformer search** — generates diverse rotamer-sampled starting
  geometries, optimizes each with a chosen energy model (Classical/
  OpenFF Sage/ANI-2x), then prunes to distinct conformers using CREST's
  own real CREGEN thresholds (6 kcal/mol energy window, 0.125 Å RMSD
  duplicate cutoff) — CREST-inspired, not a literal port (CREST is a
  native binary and can't run in a browser) — see `CONFORMER_SEARCH.md`.
- **Bond dissociation enthalpy (per bond)** — this app's first bond-level
  property: every bond gets a predicted BDE, color-coded and labeled
  directly on the 2D structure, plus a per-atom "weakest attached C-H"
  value through the existing atom heatmap. A real trained Chemprop
  bond-level D-MPNN checkpoint (own dataset/training, not a port of
  ALFABET's published weights) — see `BDE_INTEGRATION.md`.
- **Structural alerts** — ~1250 real medicinal-chemistry SMARTS filters
  (PAINS, Glaxo, Dundee, BMS, SureChEMBL, MLSMR, Inpharmatica, LINT),
  with substructure highlighting.
- **Property radar, HRMS adduct calculator**, and more — see
  `CHEMPROP_INTEGRATION.md` for the full technical writeup.

## Running it

No build step. Serve the directory with any static file server and open
`index.html`:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

(Opening `index.html` directly via `file://` will not work — the WASM
modules and JSON data files need to be fetched over HTTP.)

## Project layout

```
index.html
css/            stylesheet
js/             all application logic (no build/bundle step)
model/          converted model weights + registry.json
data/           SMARTS filter data, etc.
scripts/        Python conversion/validation tooling (checkpoint
                converters, registry validator, dataset extraction) --
                not needed to run the app, only to add/update models
```

## Adding a model

See `scripts/convert_chemprop_checkpoint.py` and
`scripts/convert_nagl_checkpoint.py` for converting a trained checkpoint
into the browser-loadable manifest + weights format,
`scripts/convert_ani2x_checkpoint.py` for re-exporting the ANI-2x neural
network potential (see `ANI2X_INTEGRATION.md`), and
`scripts/validate_registry.py` for checking `model/registry.json` before
shipping a change.

## Status / honesty notes

This project documents its own validation status throughout —
`CHEMPROP_INTEGRATION.md` and the registry entries' `notes` fields say
plainly what's been validated against real ground truth vs. what's a
reasonable-but-unverified approximation. Worth reading before trusting
any specific number out of it.

## License

See `LICENSE`.
