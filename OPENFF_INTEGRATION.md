# OpenFF SMIRNOFF ("Sage") force field — integration notes

## What this is

A real published small-molecule force field — [OpenFF Sage
2.1.0](https://github.com/openforcefield/openff-forcefields)
(`openff-2.1.0.offxml`) — applied to an arbitrary drawn molecule entirely
client-side: real bond/angle/torsion/vdW parameters, typed via the SMIRKS
(SMARTS-with-atom-maps) pattern each parameter carries, matched against
the molecule with RDKit.js's own substructure search. This is a genuinely
different thing from `embed3d.js`'s existing "classical" 3D generator,
which uses the *same functional form* (harmonic bonds/angles, periodic
torsions, 12-6 Lennard-Jones) but hand-tuned constants aimed at "produces
a chemically sane structure," not real force-field numbers. OpenFF Sage
is the third option in the "3D generation method" dropdown, alongside
Classical and GeoMol.

## Why SMIRNOFF is a good fit for this project

SMIRNOFF force fields (Sage, and its predecessor Parsley) parameterize
*directly on SMARTS patterns* rather than the atom-typing trees classic
force fields (AMBER, CHARMM, UFF, MMFF94) use — a parameter file is
literally a list of `(SMIRKS pattern, numeric constants)` pairs, e.g.
`<Bond smirks="[#6X4:1]-[#6X4:2]" length="1.528 * angstrom" k="..."/>`.
That maps almost directly onto machinery this project already has and
trusts: `smarts-filters.js` already runs ~1250 real SMARTS patterns
against RDKit.js's `get_qmol()`/`get_substruct_matches()` for structural
alerts. Typing a SMIRNOFF force field is the same substructure-matching
primitive, just applied to a different (much smaller, ~360-entry) pattern
list, with the results converted into force-field terms instead of
alert flags.

## What's real vs. substituted, honestly

- **Bonds, Angles, ProperTorsions, ImproperTorsions, vdW**: real Sage
  2.1.0 numbers and real SMIRKS patterns, converted mechanically (unit
  extraction only, no re-fitting or approximation) by
  `scripts/convert_openff_forcefield.py` from a real
  `openff-2.1.0.offxml` into `data/openff-sage-2.1.0.json`. Typing uses
  SMIRNOFF's own documented "last SMIRKS match in file order wins"
  hierarchy and, for improper torsions, its documented "trefoil" (apply
  the matched parameter three times, once per rotation of the three real
  substituents, each at k/3).
- **Electrostatics**: Sage's own protocol specifies AM1-BCC partial
  charges (via a semiempirical QM calculation this project has no way to
  run in a browser). This uses this project's **own NAGL-MBIS partial
  charge model** (`nagl-model.js`) instead — a real, documented
  substitution, not the official Sage protocol. NAGL-MBIS charges are a
  different (also real, also ML-derived) charge model than AM1-BCC;
  expect electrostatic energies/geometry-driving effects to be in the
  right ballpark, not numerically identical to what upstream OpenFF
  tooling would produce for the same molecule. If no NAGL-MBIS model is
  loaded, electrostatics is simply **omitted** (all charges treated as
  0) rather than faked with a placeholder.
- **Not implemented, deliberately**: SMIRNOFF Constraints (irrelevant —
  this project only minimizes to a local energy minimum, never runs
  constrained MD, so every bond including X-H ones gets a real Bonds
  harmonic term instead), LibraryCharges/ToolkitAM1BCC sections, virtual
  sites (Sage 2.1.0 doesn't define any for the small-molecule force field
  family), and fractional-bond-order interpolation (Sage 2.1.0 defines
  zero Bonds/ProperTorsions parameters that use it — checked directly
  against the real file, not assumed; a future Sage release that adds
  some would need real support added to the converter, not silent wrong
  numbers).

## SMIRKS typing, concretely

Each SMIRNOFF parameter's SMIRKS (e.g.
`[*:1]~[#6X3:2](~[*:3])~[*:4]` for a generic improper torsion) is a SMARTS
pattern with atom-map numbers (`:1`, `:2`, ...) marking which matched atom
fills which role. `js/openff-forcefield.js`'s `parseSmirksAtomMap()`
independently parses the SMIRKS text to recover, for each query-atom
position (in the same left-to-right creation order RDKit itself assigns
query atom indices — standard SMARTS/SMILES parsing behavior), which map
number that position carries — including correctly skipping over
recursive `$(...)` sub-patterns, which don't introduce new top-level
query atoms and are common in Sage's more specific SMIRKS (e.g. the
nitrogen improper torsions `i3`-`i6`, each distinguishing conjugated from
plain pyramidal amine nitrogen via a `$(...)` condition). That lets a
match's `atoms` index array be reinterpreted as "the atom at map position
1", "at map position 2", etc., regardless of how many *untagged* context
atoms the pattern also matches.

This project's own topology (which atoms are actually bonded, which
triples/quadruples of atoms form real angles/torsions) is enumerated
independently from connectivity, not from whatever SMARTS matches happen
to return — so a real bond/angle/torsion with no matching SMIRKS
parameter is a detectable, honestly-reported gap (`unmatched` counts
surfaced in the 3D panel's status note) rather than silently dropped.
Given Sage's broad elemental/valence coverage this should be rare for
ordinary drug-like molecules, but isn't guaranteed impossible for exotic
structures.

## Numeric optimization

Reuses `embed3d.js`'s already-validated implicit-hydrogen placement,
rotatable-bond detection/seeding, and generic finite-difference
minimization plumbing (exposed via `CC.Embed3DShared`) rather than
re-deriving a second copy of the same BFS side-splitting / atan2 dihedral
formula / numeric-gradient optimizer for a different energy model — only
the SMIRNOFF energy function and its parameter typing are new. The staged
minimization schedule (bonds+angles → torsions/impropers → ramp in
vdW+electrostatics together) mirrors `embed3d.js`'s `minimizeStaged` for
the same reason it exists there: introducing the stiff/expensive terms
gradually gives a much better-conditioned optimization than throwing
everything at a freshly torsion-randomized structure at once.

## Validation status

The parameter **conversion** (offxml → JSON) is mechanical, unit-checked
extraction — every attribute's unit string is checked against a fixed
expectation before the leading number is trusted, not silently assumed
(see `convert_openff_forcefield.py`). The SMIRKS **typing** logic has
been checked by hand against real Sage patterns, including nested
recursive SMARTS, but has **not** been cross-validated end-to-end against
real OpenFF toolkit output for whole molecules — no `openff-toolkit`
install was available in this environment to diff against. Treat this as
a real, from-scratch SMIRNOFF implementation using real Sage 2.1.0
numbers, not yet bit-exact-confirmed the way this project's Chemprop/NAGL
ports are (see `CHEMPROP_INTEGRATION.md`, `nagl-model.js`'s header).

## Regenerating `data/openff-sage-2.1.0.json`

```bash
# download a real Sage release .offxml, e.g.:
curl -O https://raw.githubusercontent.com/openforcefield/openff-forcefields/main/openforcefields/offxml/openff-2.1.0.offxml
python3 scripts/convert_openff_forcefield.py openff-2.1.0.offxml data/openff-sage-2.1.0.json
```
