# Structure validation layer — integration notes

## What this is

A dedicated validation step that runs **before** any property model, on
every 2D edit — not each model independently discovering (or worse,
silently mispredicting on) a problem. Two pieces:

- `js/structure-validation.js` — `CC.Validate.checkStructure(molecule)`
  runs 14 real structural checks (valence, charges, radicals, fragments/
  salts/counterions, duplicates, stereochemistry, aromaticity, explicit
  hydrogens, metals, hypervalent atoms, unusual isotopes) and returns a
  severity-tagged issue list.
- `CC.Validate.checkModelCompatibility(molecule, structureReport)` folds
  that report together with every model's own hard compatibility check
  (reusing `CC.NAGL.checkCompatibility`, `CC.ANI.checkCompatibility`,
  `CC.GeoMol.checkCompatibility`, `CC.GNN.checkChempropCompatibility` —
  all pre-existing, not duplicated) into one three-tier verdict per model:

  1. **Structure valid → model compatible → prediction available**
     (`tier: 'compatible'`)
  2. **Structure valid → model outside applicability domain → prediction
     permitted with warning** (`tier: 'warning'`, with reasons)
  3. **Structure incompatible → prediction blocked** (`tier: 'blocked'`,
     with reasons)

Surfaced in a new "Validation" side-tab (structural issue list with
canvas highlighting, full model-compatibility table) and as compact
tier badges directly in the Properties panel's model list, next to each
model's own "Load" button.

## The 14 checks

| Check | Severity | Method |
|---|---|---|
| Valence violation | error | RDKit sanitization failure (`get_mol()` returns null/invalid) |
| Unusual formal charge | warning | `|charge| ≥ 2` on any atom, or `|net charge| ≥ 2` |
| Radical | warning | RDKit's `nRad` (unpaired electrons) > 0 |
| Disconnected fragments | info | `get_frags()`, > 1 component |
| Salt | warning | > 1 fragment, non-parent components present |
| Counterion | info | small non-parent fragment, name-matched against a curated common-ion table |
| Duplicate atoms | warning | two atoms within ~15% of a bond length of each other (2D coordinates) |
| Duplicate fragments | info | two disconnected components with identical canonical SMILES |
| Undefined stereocenter | info | RDKit's `get_stereo_tags()` CIP tag `"(?)"` |
| Undefined E/Z | info | non-ring double bond, real substituents on both ends (via RDKit's own `cipRanks`), no defined stereo — see file header for why this needs its own logic, not just `get_stereo_tags()` |
| Aromaticity inconsistency | warning | SSSR ring with a fully alternating bond pattern that RDKit does *not* perceive as aromatic |
| Explicit hydrogens | info | atom with `element === 'H'` among the app's own drawn atoms |
| Metal | warning | element in a curated common-metals set |
| Hypervalent atom | info | explicit bond-order sum exceeds (standard valence + formal charge) while RDKit still sanitizes it |
| Unusual isotope | info | RDKit's `isotope` field ≠ 0 |

All field names (`chg`, `nRad`, `isotope`, `bo`, `stereo`, `cipRanks`,
`get_frags()`'s MolList iterator shape, `get_stereo_tags()`'s `"(?)"`
convention) were confirmed directly against a live RDKit.js instance
before writing this file, not assumed from a different RDKit build or
from memory.

## Two real bugs found and fixed along the way

Building and testing this surfaced two pre-existing, unrelated bugs —
fixed rather than worked around, since both directly undermined checks
this feature needed to be honest about:

1. **This app had no way to represent a radical or an explicit isotope
   at all.** `CC.Molecule`'s atom shape had no `radical`/`isotope`
   fields, and `molfile.js` never read or wrote the MDL `M  RAD`/`M  ISO`
   property lines — so a radical or isotope-labeled structure loaded via
   SMILES would silently lose that information the moment it touched
   this app's own molecule model, making the radical/isotope checks
   permanently unable to fire. Fixed by adding both fields to
   `CC.Molecule.addAtom()` and round-tripping them in `molfile.js`.
2. **`moleculeToMolblock`/`atoms3DToMolblock` wrote one character too few
   per atom line** (`'0  0  0  ...'` — 12 fields all 3-wide — instead of
   the V2000 spec's 2-wide first field + 11 3-wide fields), silently
   shifting every subsequent fixed-column field left by one when RDKit
   re-parsed a molblock this app had written. Found via this feature's
   own counterion-recognition test (a `[Na+]` fragment's canonical SMILES
   came back as `[Na+:0]` instead of `[Na+]` — a spurious atom-map
   number from the misaligned columns), confirmed byte-for-byte against
   a real RDKit-written reference molblock, and fixed in both molblock
   writers in `molfile.js`.
3. (Related, found the same way) **`CC.elementData()` fell back to
   carbon's valence (4) for any element outside its ~10-element table**,
   so a bare metal ion like `[Na+]` displayed as `NaH4+` on the 2D canvas
   — 4 invented implicit hydrogens. Fixed in `elements.js`: the fallback
   now keeps carbon's *radius* (for reasonable bond spacing) but uses
   `valence: 0` (assume no implicit hydrogens for an untabulated
   element) instead.

## Applicability-domain heuristic — what it is and isn't

The WARNING tier for models without their own hard compatibility check
(most Chemprop-family property models) is a documented heuristic, not a
per-checkpoint verified domain-of-applicability study: it reflects
general, well-known practice (most public QSAR/ADMET training sets are
curated to single, neutral-ish, standard-isotope, non-radical organic
structures), applied per model *engine family* — the same granularity
`CC.GNN.checkChempropCompatibility` already uses. It is not a claim that
any specific loaded checkpoint's real training set has been inspected
structure-by-structure. See `js/structure-validation.js`'s own header for
the full honesty note.

## Validation status

Every check was tested directly against real molecules (not just read
over) via both the JS API and the actual UI: valence violations
(pentavalent carbon, built directly via the app's own Molecule API,
bypassing SMILES since RDKit itself would refuse to parse it), radicals
(methyl radical), isotopes (¹³C), salts/counterions (sodium acetate,
correctly identifies "sodium" and blocks NAGL specifically while warning
the rest), stereocenters (alanine, defined vs. undefined), E/Z
(2-butene, defined vs. undefined; correctly *not* flagging symmetric
2,3-dimethyl-2-butene or terminal ethylene), aromaticity inconsistency
(cyclooctatetraene correctly flagged; benzene correctly not), hypervalent
phosphorus, and a clean molecule (aspirin) correctly showing zero issues
and all 22 registry models "Compatible."
