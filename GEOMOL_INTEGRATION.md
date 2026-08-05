# GeoMol — real learned 3D conformer generation — integration notes

Adds a from-scratch, fully client-side reimplementation of GeoMol (Ganea,
Pattanaik et al., *"GeoMol: Torsional Geometric Generation of Molecular 3D
Conformer Ensembles"*, NeurIPS 2021 — github.com/PattanaikL/GeoMol, MIT
license) as an alternative 3D structure generator, alongside the existing
hand-tuned classical force field (`embed3d.js`) and the ANI-2x neural
network potential (`ani2x-model.js`). Unlike those two — which only ever
*refine* a geometry that already exists — GeoMol predicts a conformer
directly from the 2D molecular graph: a message-passing GNN encodes the
graph, then per-atom local geometry and per-rotatable-bond torsion angles
are predicted and assembled into a full 3D structure in one pass, without
needing any classical force field at all.

This picks the state-of-the-field review the user pointed to (Baillif,
Cole, McCabe & Bender, *Curr. Opin. Struct. Biol.* 2023) and specifically
chose GeoMol over that review's other candidates (Torsional Diffusion,
GeoDiff): those are iterative diffusion models needing dozens of
sequential score-network evaluations per conformer plus an SDE sampler.
GeoMol is a single forward pass — a fundamentally smaller, more tractable
port, and its GNN encoder is architecturally close to the D-MPNN this
project already has from-scratch in `dmpnn.js`.

## Status: complete -- full pipeline built, validated end-to-end, and usable from the app's 3D view panel

This is a large, staged port (see the design discussion that led here —
GeoMol's real assembly algorithm involves per-bond rotation-frame
construction, a small linear-system solve per rotatable bond, and a
multi-start Kabsch-averaging ring-closure step, none of which resemble
anything already in this codebase). Each stage is validated against a
*live* run of the real PyTorch checkpoint before moving to the next,
same discipline `ANI2X_INTEGRATION.md` already established for ANI-2x.

### Done

- **`scripts/convert_geomol_checkpoint.py`** — real, working converter
  (mirrors `convert_ani2x_checkpoint.py`'s pattern: every tensor
  name/shape/hyperparameter is read directly off the live
  `trained_models/{drugs,qm9}/best_model.pt` state_dict +
  `model_parameters.yml`, nothing transcribed from memory). Only
  supports `global_transformer: false`, which is what both the published
  `drugs` and `qm9` checkpoints actually use.
- **`model/geomol-drugs/`** — the real `drugs` checkpoint (trained on
  GEOM-Drugs, i.e. actual drug-like molecules, not just QM9's small
  organics), converted: **1.1MB** total (`manifest.json` + `geomol.bin`)
  — far lighter than ANI-2x's 54MB, since GeoMol's `model_dim` is only 50.
- **`js/geomol-features.js`** — featurization exactly matching
  `model/featurization.py:featurize_mol_from_smiles`'s 74-dim node / 4-dim
  edge vectors for the `drugs` element table, run on an
  `add_hs_in_place()`'d molecule (RDKit.js does support this; confirmed
  directly that it preserves heavy-atom order and appends hydrogens
  after, identical to real RDKit's own `AddHs`). Validated **bit-exact**
  (zero mismatches) against a live PyTorch run on both plain butane and a
  46-atom molecule with two aromatic rings, a stereocenter, and a
  thiazole sulfur.
  - One real gap found and fixed: RDKit.js exposes no per-atom
    hybridization at all (same known gap `chemprop-features.js` already
    documented) — reuses that file's validated `guessHybridization()`
    (now exposed as `CC.GNN.guessHybridization`) rather than a second,
    independent guess, with one correction: a bare hydrogen's real RDKit
    hybridization is never one of GeoMol's five SP* choices and needs to
    land in the one-hot "other" bucket, not fall through to sp3 the way
    `guessHybridization` does when only ever exercised on heavy atoms.
  - Another confirmed-not-guessed fact: GeoMol's "implicit valence"
    feature block is computed *after* `AddHs`, where by construction no
    atom has any implicit valence left — verified directly against
    ground truth (constant one-hot "0" for every atom of every test
    molecule tried) and hardcoded as that constant rather than computed.
- **`js/geomol-model.js`** — the weight-tied 3-round MetaLayer GNN
  encoder (`CC.GeoMol.embed`) and the per-neighborhood local-structure
  head (`CC.GeoMol.predictLocalStructures`: a real from-scratch
  multi-head self-attention Transformer encoder layer — post-LN, ReLU
  feedforward, PyTorch's exact defaults — plus chirality-aware sign
  correction via signed tetrahedron volume, plus the symmetric
  double-evaluation distance MLP). Validated against a live PyTorch run
  with the *same* injected Gaussian noise (captured by monkey-patching
  `torch.distributions.normal.Normal.sample` during a real forward pass,
  so the comparison is apples-to-apples rather than comparing two
  independent random draws):
  - GNN encoder (`x1`/`x2`/`h_mol`): RMSE ~0.1–1% relative to each
    tensor's own typical magnitude, on both test molecules — consistent
    with float32 weight storage + JS float64 vs. PyTorch float32
    arithmetic accumulating over a multi-round message-passing
    computation, the same conclusion (and same order of magnitude) as
    this project's ANI-2x and NAGL validations reached, not a bug.
  - Local-structure prediction (`model_local_coords`): max absolute
    difference 0.004–0.03 Å against typical magnitudes of 0.5–1.3 Å,
    including on the molecule's real stereocenter (confirms the
    chirality-flip branch, which butane never exercises, is also
    correct).
- **`js/geomol-assembly.js`** — `CC.GeoMol.getDihedralPairs`, a direct
  port of `model/utils.py:get_dihedral_pairs` (+
  `cycle_utils.py:get_current_cycle_indices`): pick one representative
  directed edge per interior bond, then expand any bond touching a ring
  into a full traversal of that ring, popping each ring from the working
  set the first time any of its bonds is visited (so a fused ring
  system's shared atoms/bonds don't get reprocessed from scratch, though
  a bond shared between two SSSR rings — e.g. a benzofuran-style fusion
  bond — legitimately does appear twice in the output, once from each
  ring's own traversal; confirmed this is the *real* algorithm's
  behavior, not a bug to fix). Validated: **exact match** (same pairs,
  same order) on acyclic butane; on the 46-atom fused-ring test molecule,
  the exact same 28 pairs including the fusion-bond duplicate, but in a
  different order and with the fused ring's second cycle walked in the
  opposite rotational direction.
  - Honest, understood-not-hand-waved difference: this uses RDKit's SSSR
    (already available via `geomol-features.js`'s RDKit.js JSON parse)
    as the ring basis rather than reimplementing networkx's
    `cycle_basis` algorithm. For this test molecule both perceive the
    *same* two rings (same atom sets), just enumerate each ring's atom
    order starting from a different point/rotational direction — an
    inherently arbitrary, implementation-specific detail of ring
    perception that both libraries are free to choose differently.
    Confirmed directly (not assumed) that this doesn't change *which*
    bonds get treated as dihedral pairs, only the bookkeeping order they
    get processed in.

- **Torsion-angle prediction** (`js/geomol-model.js`'s
  `CC.GeoMol.predictTorsions`, porting `model.py`'s `model_pair_stats` /
  the embedding half of `align_dihedral_neighbors`): for each dihedral
  pair, the `alpha_mlp` predicts the target torsion angle
  (`v_star = [cos alpha, sin alpha]`, evaluated in both neighbor orders
  and summed) and the `c_mlp` predicts a weight for each of the 9
  neighbor-slot-pair combinations (`c_ij`). Proved by hand, and confirmed
  directly against the real source, that both are independent of the
  geometric rotation frame constructed during assembly (see below) —
  which is what makes it valid to predict them here, purely from GNN
  embeddings, separately from the frame/coordinate math. Validated
  against a live PyTorch run (same captured noise, same technique as
  Stage 1's validation): `v_star` matches to ~2e-9 (essentially exact);
  `c_ij` RMSE ~0.07–0.3% relative to its own typical magnitude (same
  float32-accumulation pattern as everything else validated so far) —
  confirmed on both butane (all 3 pairs, exact traversal-order match) and
  17 of the complex molecule's 28 pairs (the other 11 only differ in
  which of two possible directed-pair orderings they were traversed in —
  a bookkeeping artifact of the traversal-order difference already
  understood and documented above, not a computation difference).
- **Geometric assembly, acyclic case** (`js/geomol-assembly.js`'s
  `CC.GeoMol.constructConformersAcyclic`, porting
  `model/inference.py`'s `construct_conformers_acyclic` +
  `rotation_matrix_inf_v2` + `calculate_gamma`): walks dihedral pairs in
  order, building each fragment's local frame (Gram-Schmidt rotation
  aligning the shared bond to a common axis), mirroring and translating
  one side to attach at the other's predicted bond vector, then solving
  the 2x2 linear system that fits a "gamma" rotation-around-the-bond-axis
  correction reproducing the predicted torsion. One deliberate,
  understood difference from the reference implementation: the frame
  construction's perpendicular-axis choice is real GeoMol's own source of
  a harmless internal randomness (proved by hand that it doesn't affect
  the final physical geometry, since the gamma-fit step measures the
  "current" dihedral in whatever frame this constructs and corrects for
  it) — this port uses a deterministic Gram-Schmidt seed instead of
  matching PyTorch's RNG for it, which is valid precisely because that
  choice never reaches `c_ij`/`v_star` (confirmed above) or the final
  torsion angle. Validated on butane by measuring the actual physical
  dihedral angle at each of the 3 rotatable bonds from the assembled
  coordinates (not comparing raw coordinates directly, since the
  eta-choice difference does produce a different *global* orientation of
  the whole molecule, exactly as expected): matches ground truth's own
  measured angles to within 0.8-1.0 degrees at every bond, and bond
  lengths match ground truth's own range almost exactly (1.080-1.548 A
  vs. 1.080-1.548 A). Also exercised (no crash, no NaN, 45/46 atoms
  placed — the lone unplaced atom is only reachable through a ring bond,
  which this acyclic-only function correctly and deliberately skips) on
  the 46-atom fused-ring molecule.
- **Ring closure** (`js/geomol-assembly.js`'s `smoothCycleCoords` +
  `CC.GeoMol.constructConformers`, porting `model/inference.py`'s
  `smooth_cycle_coords` + the ring-aware branches of
  `construct_conformers`): for each ring, builds independent candidate
  assemblies starting at every possible rotational offset around the
  ring (same per-pair mechanics as the acyclic case, `cycleLen - 1`
  sequential steps per candidate, all offsets advanced in lockstep),
  Kabsch-aligns every candidate but the first onto the first (fitting the
  rotation using only genuine ring atoms, not their exocyclic
  substituents), and averages them — each candidate excluding its own
  "closing seam" atoms from its own contribution, since those were
  placed last in that candidate's own sequence and are the least
  cross-validated there. `constructConformers` is the full, ring-aware
  dispatcher: walks a *fresh* copy of the molecule's rings independently
  of whatever `getDihedralPairs`'s own traversal-ordering pass already
  consumed, routing each dihedral pair to the ring-closure path, the
  ring-entry path (an acyclic bond attaching a fresh ring), or the plain
  acyclic path.
  - New numerical primitive this port needed nowhere else: a 3x3 SVD (via
    Jacobi eigendecomposition of the cross-covariance matrix), for the
    Kabsch alignment itself. Self-tested independently before use:
    reconstructs an arbitrary 3x3 matrix to ~1e-15 error, and recovers a
    known rotation+translation applied to a test point set (both a full
    3D set and a deliberately-degenerate coplanar set, to exercise the
    reflection-correction guard) to ~1e-16 error.
  - Validated on the 46-atom fused-ring molecule (two rings sharing a
    fusion bond) by checking the *physical* outcome ring closure is
    actually for, rather than raw coordinates (which the eta-choice
    global-orientation difference already established won't match
    directly): all 48 bonds' lengths match ground truth's own range
    almost exactly (0.924-1.711 A vs. 0.924-1.707 A) with zero outliers
    outside a chemically sane range; zero nonbonded clashes anywhere in
    the molecule (closest approach 1.34 A); the aromatic 6-ring came out
    genuinely planar (max deviation from its own best-fit plane: 0.033
    A) without that being enforced anywhere -- it falls out of the
    torsion predictions and ring-closure averaging actually being
    correct. Also confirmed `constructConformers` is a strict superset of
    `constructConformersAcyclic`: on ring-free butane, the two produce
    byte-identical output (max difference exactly 0), so the ring-aware
    dispatcher introduces zero regression on molecules with no rings.
- **Registry/UI wiring**: a new `"geomol"` engine in
  `model-registry.js`/`registry.json`/`validate_registry.py`
  (`convert_geomol_checkpoint.py`'s output cross-checked the same way
  `ani2x`-engine entries are -- architecture string, required tensor
  groups, and that `encoder.dModel` is exactly `2*modelDim`, matching
  the local-structure Transformer's concat(neighbor, center) input),
  `CC.GeoMol.checkCompatibility` (mirrors `CC.ANI.checkCompatibility`'s
  shape -- flags molecules with fewer than 4 atoms or no bond with a
  real substituent on each side, matching the real featurizer's own
  `HasSubstructMatch([*]~[*]~[*]~[*])` + `N >= 4` scope), and
  `CC.GeoMol.generateConformer` (runs the full pipeline -- featurize,
  encode, predict local structure and torsions, assemble -- and returns
  the same `{atoms, bonds, energy, converged}` shape
  `CC.buildInitial3D`/`CC.optimize3D` already produce, so it's a drop-in
  alternative for `CC.render3D`).
  - The 3D view panel now has a "3D generation method" selector
    (classical force field / GeoMol) above the toolbar. Selecting GeoMol
    and clicking "Generate 3D structure" auto-loads the `geomol-drugs-v1`
    registry model on first use (same load-on-click pattern the ANI-2x
    button already established), runs the compatibility check, and
    renders a freshly-sampled conformer (each click draws new noise, so
    repeated clicks give genuinely different conformers -- the real
    model's own "ensemble" behavior). "Optimize geometry..." and
    "Optimize with ANI-2x..." both still work afterward, refining
    whatever GeoMol produced, since `lastInitial` (the metadata
    `CC.optimize3D` needs) is always computed from
    `CC.buildInitial3D` regardless of which method generated the
    on-screen structure -- verified directly: generated a conformer with
    GeoMol, then clicked "Optimize geometry...", and the classical
    staged minimizer ran correctly to completion on it.
  - Verified end-to-end through the real UI (not just the console): drew
    the 46-atom fused-ring test molecule, selected GeoMol, generated a
    conformer (bond lengths 0.913-1.73 A, zero nonbonded clashes),
    confirmed the Properties panel's model list correctly shows it as
    "Loaded", confirmed regenerating gives different coordinates each
    time, confirmed switching back to "Classical force field" still
    works unchanged, and confirmed a too-small molecule (methane) falls
    back to the classical seed with a clear message rather than
    producing a degenerate structure.

## Architecture

**GNN encoder.** Two independent weight-tied message-passing GNNs
(`gnn1`/`gnn2` in this port, `model.gnn`/`model.gnn2` in the real
checkpoint) — same `node_init`/`edge_init` MLPs run once, then the same
`EdgeModel`+`NodeModel` update (both residual: `new = (1+eps)*old +
delta`) applied `depth=3` times with *shared* weights. `gnn1`'s output
(`x1`) feeds local-structure prediction; `gnn2`'s output (`x2`, and its
sum-pooled+MLP'd molecule-level embedding `h_mol`) feeds torsion
prediction. Both take the real 74/4-dim features concatenated with a
10-dim Gaussian noise vector (std 1.0) — this is GeoMol's actual source
of conformer-to-conformer diversity; a fresh draw per requested conformer
gives genuinely different (but all locally low-energy) 3D structures,
the same "ensemble" behavior the paper is named for.

**Local-structure prediction.** For every atom with more than one
neighbor: gather up to 4 neighbor embeddings (zero-padded), run them
through a single `nn.TransformerEncoderLayer` (2-head self-attention,
padding-masked so padded slots are never attended *to*), then a small
MLP (`coord_pred`) predicts a unit direction per neighbor slot and
another (`d_mlp`, evaluated symmetrically in both neighbor/center orders
and summed) predicts a bond distance — multiplying the two gives that
neighbor's position in the central atom's own local frame. A genuine
stereocenter (exactly 4 neighbors, a nonzero chiral tag) gets a sign
correction on the local frame's out-of-plane axis, computed from the
signed volume of the 4 predicted directions compared against the parsed
CW/CCW tag.

**Dihedral pairs.** Real cheminformatics doesn't get to pick one
canonical spanning-tree traversal of a molecular graph for free — see
`js/geomol-assembly.js`'s `getDihedralPairs` above for exactly how the
real algorithm picks which bond to treat as the "hinge" for each
local-frame-to-local-frame handoff, and how it handles rings specially.

## Known limitations, stated honestly

- **QM9 checkpoint not converted.** Only `drugs` is shipped here — QM9's
  5-element scope (H/C/N/O/F, molecules capped at 9 heavy atoms) doesn't
  match what this app's users actually draw. `convert_geomol_checkpoint.py
  GeoMol/trained_models/qm9/ model/geomol-qm9/` would produce it the same
  way if ever needed.
- **`global_transformer: true` checkpoints unsupported.** Not a real
  limitation for the two published checkpoints (both use `false`), but a
  different future checkpoint trained with the whole-molecule
  Transformer path enabled wouldn't load — the converter refuses rather
  than silently producing something the JS forward pass can't run.
- **No training-loss / optimal-transport machinery.** This is an
  inference-only port (`generate_model_prediction`'s path through
  `embed` + `model_local_stats`/`model_pair_stats`), not the full
  `model.py` — the OT-based ensemble-diversity training loss, the
  ground-truth local-stats computation, and everything else only used
  during training was deliberately left out.
