# Third-Party Notices

This project bundles or depends on third-party data, models, and
libraries not covered by this repository's own LICENSE. This list is a
starting point, not a substitute for your own due diligence before
public release (e.g. before a JOSS or journal submission, which will
generally require this to be accurate and complete).

## Confirmed

- **RDKit.js** (`@rdkit/rdkit` npm package) — BSD 3-Clause. Used
  throughout for 2D chemistry, descriptors, SMARTS matching, and 2D
  coordinate generation.
- **3Dmol.js** — BSD 3-Clause. Used for the 3D viewer.
- **rd_filters SMARTS collection** (`data/smarts_filters.json`) —
  sourced from github.com/PatWalters/rd_filters, MIT licensed. That
  repository itself aggregates ChEMBL's structural_alerts table; if you
  redistribute this data, attribute both.
- **SA Score fragment table** (`model/sa-fragment-scores*`) — derived
  from RDKit's own `fpscores.pkl.gz` (RDKit Contrib), which ships under
  RDKit's BSD license.

## Needs your own verification before public release

- **NAGL-MBIS model weights** (`model/nagl-mbis-charges/`) — the
  checkpoint (`nagl-v1-mbis-dipole.ckpt`) came from
  github.com/jthorton/nagl-mbis. Check that repository's own LICENSE
  file and any dataset/model-specific terms before redistributing the
  converted weights here.
- **Chemprop-trained model weights** (logP, logS, melting point,
  electrophile-reactivity) — trained as part of this project using the
  Chemprop architecture (MIT licensed upstream), on datasets with mixed
  provenance (see each entry's `dataset` field in
  `model/registry.json`). Some training datasets may carry their own
  redistribution terms independent of the code that trained on them --
  check each `dataset.sourceUrl` / `citationKey` before treating the
  *weights* as freely redistributable, even though the code is yours.

## Not a lawyer, not legal advice

This file is meant to save you a first pass, not replace checking the
actual license text of each upstream project yourself.
