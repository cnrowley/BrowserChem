# Conformer search — integration notes

## What this is

`js/conformer-search.js`: generates a diverse set of starting 3D
geometries for a drawn molecule, locally optimizes every one of them with
a chosen energy model, then prunes the resulting ensemble down to
distinct, low-energy conformers — a ranked list of real structures with
real (model-relative) energies, not just the single best structure this
app's individual optimizers already return on their own. Reachable from
the 3D view tab's "Conformer search" section: pick an energy model, click
"Run conformer search…".

## Energy models

One pulldown selects which of this app's three 3D optimizers actually
computes energies/gradients during the search:

| Model | What it is | Notes |
|---|---|---|
| Classical force field | `embed3d.js`'s hand-tuned harmonic/LJ force field | Fast; energies are arbitrary units, not real kcal/mol |
| OpenFF Sage (SMIRNOFF) | Real published force field, SMARTS-typed | See `OPENFF_INTEGRATION.md`; needs the NAGL-MBIS model loaded for electrostatics (auto-loaded on click) |
| ANI-2x | Neural network potential | H/C/N/O/F/S/Cl, neutral molecules only; each seed gets a quick classical pre-relax first (load-bearing, not optional — see `js/conformer-search.js`'s header) |

All three reuse that engine's own already-validated single-seed optimizer
(`CC.Embed3DShared.optimizeSeedClassical`, `CC.OpenFF.optimizeSeed`,
`CC.ANI.optimizeGeometry`) — this module adds seed generation and
ensemble pruning on top, not a fourth energy calculation.

## Why "CREST-inspired" and not literally CREST

This was originally scoped as using [CREST](https://github.com/crest-lab/crest)
directly. CREST is a real, widely-used conformer search tool, but it's a
compiled Fortran/C binary built around semiempirical tight-binding QM
(GFN-xTB) and metadynamics — neither is available here: this project is a
static browser page with no server and no native binary execution (see
`CLAUDE.md`), and there's no WASM build of CREST or xtb. A real CREST
checkout was cloned specifically to check its source before writing this
module (not worked from memory), and what's actually reused from it:

- **The pipeline shape**: generate many candidate geometries → locally
  optimize each → prune to a distinct, low-energy ensemble. Rotatable-bond
  torsion sampling (systematic for few rotatable bonds, randomized
  otherwise) stands in for CREST's metadynamics step, which needs real QM
  forces to bias against that this project doesn't have.
- **CREGEN's real default thresholds**, confirmed directly against
  `src/confparse.f90` in that checkout:
  - `ewin = 6.0` kcal/mol — final energy window above the global minimum
  - `rthr = 0.125` Å — RMSD duplicate threshold, after best-fit (Kabsch)
    alignment, **all-atom by default** (heavy-atom-only is an opt-in
    CREST flag, `heavyrmsd`, defaulting to false in CREST's own source —
    matched here as `opts.heavyOnlyRmsd`, also defaulting to false)

Simplified relative to real CREGEN: no `ethr` energy pre-filter before
computing RMSD (a performance optimization for CREST's typical
thousands-of-structures ensembles; this app's tens-of-conformers scale
makes full pairwise RMSD comparison cheap regardless), and no secondary
"RMSD < 2×rthr + matching rotational constants" fallback duplicate check.

## Known limitation

Seed diversity comes entirely from acyclic rotatable-bond torsion states
(`embed3d.js`'s `findRotatableBonds`, which excludes ring bonds — the same
definition the classical model's own multi-attempt search already relies
on). **Ring-puckering conformers are not sampled** (e.g. cyclohexane
chair/boat/twist-boat) — this was already true of every conformer-
generating path in this project before this feature existed, not a new
gap.

## RMSD alignment

Reuses `geomol-assembly.js`'s existing Kabsch/SVD implementation
(`CC.GeoMol._internal`, already validated there for stitching predicted
conformer fragments together) via a new `CC.Shape.rmsd()` helper in
`molecular-shape.js`, rather than a second from-scratch SVD.

## Validation status

The seed-generation and CREGEN-pruning logic in `conformer-search.js` is
new and was tested directly (not just read over) against several real
molecules (ethanol, 1-propanol, ibuprofen) via both its JS API and the
actual UI: energies ranked correctly, real RMSD-based deduplication
confirmed (e.g. 10 optimized ibuprofen seeds pruned to 5 genuinely
distinct conformers within the energy window), no NaN/crashes across
classical and SMIRNOFF energy models, and the "View" conformer-selection
UI round-trips correctly through `getCurrent3DGeometry()` to other panels
(confirmed via the SASA panel). It has not been cross-validated against
real CREST/CREGEN output on the same molecules (no CREST binary available
to run in this environment) — treat the *ensemble structure* (how many
distinct conformers, their relative energies) as a real result of the
implemented algorithm, not a bit-exact match to what real CREST+GFN-xTB
would report for the same input.
