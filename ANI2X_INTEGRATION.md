# ANI-2x neural network potential — integration notes

Adds a fully client-side path to run the real ANI-2x neural network
potential (Devereux et al., *J. Chem. Theory Comput.* 2020, 16, 7,
4192-4202; as shipped in the `torchani` Python package) -- no server, no
PyTorch. Unlike the Chemprop/NAGL models already in this app, ANI-2x
consumes 3D Cartesian coordinates directly and drives geometry
optimization in the 3D viewer, rather than predicting a 2D property.

## What's here

```
model/ani2x/
  manifest.json                    tensor shapes/offsets + AEV constants + species/self-energies
  ani2x.bin                        52 MB float32 weights (8-model ensemble x 7 elements)
scripts/
  convert_ani2x_checkpoint.py      re-run this if torchani ever ships a new ANI-2x release
js/
  ani2x-features.js                NEW -- AEV forward pass + analytic backprop to forces
  ani2x-model.js                   NEW -- ensemble forward pass, compatibility check, minimizer
model/registry.json                NEW entry: "ani2x-v1", "engine": "ani2x"
js/model-registry.js               EXTENDED -- dispatches to CC.ANI for "ani2x"-engine entries
scripts/validate_registry.py       EXTENDED -- "ani2x" engine cross-check
index.html / js/app.js             UPDATED -- "Optimize with ANI-2x" button in the 3D panel
```

## Using it

1. Draw a structure that's neutral and built only from H, C, N, O, F, S,
   Cl (ANI-2x's supported scope -- see "Compatibility check" below).
2. In the **3D view** panel, click **Generate 3D structure**, then
   **Optimize with ANI-2x…** -- reports the final energy in Hartree and
   kcal/mol once the optimization settles or times out.

Clicking **Optimize with ANI-2x…** loads the `ani2x-v1` registry model
on demand the first time it's needed (same `CC.GNN.loadRegistryModel`
path the Properties panel's model list "Load" button uses -- that list
still shows it as "Loaded" afterward, refreshed from here) rather than
requiring a separate manual "Load" click in the Properties panel first.
The weights are ~50MB, so this first click can take a few seconds on a
slow connection; the button shows "Loading ANI-2x model…" in the
progress note while that happens, then moves straight on to the
optimization itself. ANI-2x's energy/optimization result never goes
through `gnn-inference.js`'s 2D molblock/property-table merge pipeline at
all -- it's invoked directly from the 3D panel, since there's no
2D-property-shaped output to merge.

## Compatibility check

`CC.ANI.checkCompatibility(molecule)` checks two things directly against
the drawn molecule's own formal charges/elements (not against any
particular loaded checkpoint -- every real ANI-2x release has this same
scope):

- **Net formal charge must be zero.** ANI-2x was trained only on neutral
  molecules; there's no charge input to the network at all.
- **Every atom must be H, C, N, O, F, S, or Cl.** Anything else (Br, P,
  metals, ions, ...) is outside the 7-element AEV/self-energy tables the
  checkpoint defines.

The "Optimize with ANI-2x" button runs this check before doing anything
else and shows a specific, actionable message (e.g. "unsupported
element(s): Br") instead of running a nonsensical prediction or crashing
partway through.

## Architecture

**AEV (atomic environment vector).** For each atom, a 1008-dimensional
descriptor: 112 radial features (7 elements x 16 Gaussian-shell terms
over neighbor distances, cosine-cutoff at 5.1 Å) plus 896 angular
features (28 unordered element-pairs x 8 radial-shift x 4 angle-section
terms over neighbor-pair angles, cosine-cutoff at 3.5 Å). Every constant
here (cutoffs, eta/zeta, shift/section values, the species-pair -> AEV-
block index table) is read directly off the real loaded `torchani`
model's own submodule buffers by `convert_ani2x_checkpoint.py` and stored
in `manifest.json` -- none of it is hardcoded in the JS, specifically so
a different torchani version's constants can't silently drift out of
sync with what the JS actually computes.

**Per-element networks.** Each of the 8 ensemble members has its own
small MLP per element (Linear + `TightCELU` -- CELU with alpha=0.1 --
stacked, element-specific widths e.g. H: 1008→256→192→160→1, S/F/Cl:
1008→160→128→96→1), taking that atom's AEV and producing one scalar
energy contribution. Total energy = mean over the 8 members of
(sum over atoms of that member's atomic energies), plus a constant
per-element self-energy correction (also read straight from the
checkpoint).

**Forces via analytic backprop, not finite differences.** Geometry
optimization needs the gradient of energy with respect to every atom's
position. Given how much more expensive one ANI-2x evaluation is than
this app's existing from-scratch classical force field (an 8-model
ensemble plus an O(N²) AEV construction, versus a few cheap harmonic/LJ
terms), and that a minimizer calls this every iteration, finite-difference
gradients (6N extra evaluations per step) would be far slower than
computing them analytically. `ani2x-features.js`/`ani2x-model.js`
implement genuine reverse-mode backprop: standard MLP backprop through
each atom's per-element network gives dE/d(AEV); the chain rule is then
carried by hand back through the radial/angular AEV formulas (distance
and angle-gradient identities) to get dE/d(position), i.e. the force.
This is the largest new piece of from-scratch math in this codebase, so
it's validated independently below rather than just asserted.

**Optimizer.** `CC.ANI.optimizeGeometry` is a new, separate minimizer
(not a retrofit of `embed3d.js`'s classical one, which is hard-coded to
its own bond/angle/torsion/LJ term arrays and to finite-difference
gradients) -- same adaptive step-size style (grow on improvement, shrink
on rejection, honest exit reason: converged / iteration-limit / deadline
/ step-too-small) but driven by the analytic gradient above, and without
the classical optimizer's staged LJ-ramp/torsion-conformer-search
machinery.

**This does need a chemically sane starting geometry, contrary to this
doc's earlier claim otherwise.** An earlier version of this doc asserted
ANI-2x's potential energy surface "doesn't have [the classical force
field's] initial-clash blowup problem," so pre-relaxing wasn't needed.
That was wrong, and was found to be wrong directly: on
`O=C(N[C@H](C(N)=O)CO)C1=C(C)OC2=CC=C(OCC3=CN=C(C)S3)C=C21`, handing
`CC.ANI.optimizeGeometry` the raw `CC.buildInitial3D` seed -- which had
only a moderate steric clash (two O atoms ~1.1 A apart, nowhere near
overlapping) -- made it collapse an unrelated ring C-S bond to 0.3 A while
stretching the neighboring ring bond to 2.8 A. ANI-2x is a neural network
trained on realistic bond lengths/angles; fed geometry meaningfully
outside that distribution, its energy surface (and the gradient computed
from it) isn't trustworthy, and nothing in `optimizeGeometry` itself
guards against that (no bonded restraints of its own -- it fully trusts
the network). Handing it the same structure after a classical
bonds/angles/sterics relax first (see `embed3d.js`) -- even one that
hadn't fully converged within its time budget -- kept every bond in a
sane range and let ANI-2x resolve the remaining steric clashes on its own
without incident. `app.js`'s "Optimize with ANI-2x" button now runs that
classical relax automatically first whenever the on-screen geometry
hasn't already been through it (tracked via a `currentGeometryOptimized`
flag), rather than assuming any starting geometry is fine.

## Validation

Real `torchani` (the same installed package the converter reads from) was
run on 10 structurally diverse test molecules, chosen to cover all 7
supported elements and a range of hybridization/ring/functional-group
chemistry: ethanol, methylamine, dimethyl sulfide, fluorobenzene,
chlorobenzene, thiophene, benzoic acid, a combined
N/O/F/Cl benzamide, tert-butanethiol, and cyclohexane (RDKit ETKDG
conformers, not energy-minimized -- deliberately so forces are nonzero
and actually exercise the gradient code, not just the energy code).
Ground truth: total energy and per-atom forces via
`torch.autograd.grad`, computed independently of this JS port. Compared
against `CC.ANI.energyAndForces` (Node, loading the exact same
manifest.json/ani2x.bin the converter produced) on the identical
coordinates:

- **max energy error: 8.7e-5 Hartree** (~0.05 kcal/mol), on thiophene
- **max force-component error: 4.7e-7 Hartree/Å**, on fluorobenzene

Both are consistent with float32 weight-storage rounding (weights are
stored as float32; relative error is ~1e-7, matching float32 precision)
rather than a bug in either the AEV math or the analytic backprop --
same conclusion, and same order of magnitude, as this project's earlier
NAGL atom-level validation ("1.3e-8 max abs error, float32/float64
rounding, not a bug"). This confirms the forward AEV/energy pipeline
*and* the from-scratch analytic force derivation together, not just the
easier-to-get-right forward pass.

## Known limitations (honestly, not hidden)

- **No periodic boundary conditions.** This app only ever has one
  isolated molecule on the canvas, so only the non-PBC, all-pairs
  neighborlist path is implemented -- matching what's actually needed,
  not a missing feature for this app's use case.
- **Optimizer, not a full geometry/frequency toolkit.** `optimizeGeometry`
  is a plain adaptive-step gradient descent (mirroring the existing
  classical optimizer's style for consistency), not L-BFGS or a proper
  line search -- it will settle into a local minimum near the starting
  geometry, same caveat any local optimizer has, and does not attempt
  the classical optimizer's separate torsion-driven conformer search.
- **Single ANI-2x release.** `convert_ani2x_checkpoint.py` always
  converts whatever `torchani.models.ANI2x()` currently returns in the
  installed package -- there's no support (or need, yet) for loading an
  arbitrarily different ANI-family checkpoint the way the Chemprop
  converter supports arbitrary user-trained checkpoints.
