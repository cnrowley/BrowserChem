#!/usr/bin/env python3
"""
prepare_pka_training_data.py

Reshapes the Baltruschat & Czodrowski dataset ("Machine learning meets
pKa", J. Cheminformatics 2020, github.com/czodrowskilab/Machine-learning-
meets-pKa -- real experimental/curated pKa values combined from
DataWarrior's own training set and ChEMBL25, 5,994 records covering both
acidic and basic ionizable sites) into a chemprop atom-target CSV for
this app's aqueous pKa predictor.

--- Source format ---

An SDF where each record is one (molecule, ionizable atom, pKa) triple --
a molecule with 2 ionizable sites appears as 2 separate records with the
same structure but a different `<marvin_atom>`/`<pKa>` pair. Real fields
used here: `<pKa>` (the experimental/reference value), `<marvin_atom>`
(which atom ionizes), `<marvin_pKa_type>` ('acidic' or 'basic').

--- marvin_atom indexing: confirmed empirically, not assumed ---

`marvin_atom` is 0-INDEXED (matches RDKit's own atom index directly) --
confirmed by checking, across 2000 real records, whether the element at
index `marvin_atom` vs. `marvin_atom - 1` is chemically plausible for the
stated pKa type (N for 'basic'; O/S/N for 'acidic'): 0-indexed is
plausible 99.3% (basic) / 97.7% (acidic) of the time, vs. 1.6% / 72.0%
for a 1-indexed reading. This matters a lot -- an off-by-one here would
silently mislabel nearly every training example.

--- Charge-state neutralization ---

The labeled atom itself is frequently drawn in its IONIZED form (e.g. a
carboxylic acid's acidic O as the anion `-C(=O)[O-]` rather than the
neutral `-C(=O)OH`; a basic amine N as its protonated ammonium/
pyridinium conjugate acid rather than the neutral amine) -- 45.0% of
acidic-labeled atoms and 58.8% of basic-labeled atoms, confirmed by
direct inspection of the SDF, not assumed. Since ChemCanvas always
queries this model on a neutral drawn molecule, that charge/H-count
mismatch on the exact atom being trained on is a real train/inference
distribution gap, root-caused from a live symptom (simple aliphatic
carboxylic acids over-predicted by ~4 pKa units) before being fixed
here: see the neutralization block below for the concrete before/after
and evidence.

--- One row per RECORD, not one row per molecule ---

Unlike this project's other atom-target CSVs (NMR shifts), this script
deliberately does NOT merge a molecule's multiple ionizable-site records
into one row. Each SDF record becomes its own training row, with its
own SMILES and exactly one non-NaN entry in its atom-target list, at
that record's own `marvin_atom` index. This sidesteps a real cross-
record atom-index-alignment risk entirely: merging would require
confirming that a molecule's Nth SDF record uses the exact same atom
ordering as its 1st, which is not guaranteed and not worth the risk when
chemprop's masked-atom-loss training already handles sparse/single-label
rows for the exact same molecular graph appearing multiple times without
any special handling needed.

--- Atom-order self-check: `_smilesAtomOutputOrder`, NOT `canonical=False` ---

`Chem.MolToSmiles(mol, canonical=False)` does NOT preserve the original
atom order on re-parse -- this is a real, previously-documented finding
from this project's own NMR data-prep pipeline (see NMR_INTEGRATION.md),
and re-confirmed the hard way here: an earlier version of this script
assumed `canonical=False` alone was enough (mirroring what looked like
the same trick), and silently mislabeled the target position for 2859 of
5994 records (47.7%) -- caught only by a self-check that verified the
labeled atom's own element survived at the same index after a SMILES
round-trip, not by that assumption going unquestioned. `canonical=False`
only controls whether canonical atom RANKING is used to choose a
starting atom / branch order; it says nothing about where a given
original atom index ends up in the output string. The atom that DOES
carry this information is `mol.GetProp('_smilesAtomOutputOrder')` --
after `MolToSmiles` runs, this property (set on the ORIGINAL mol, not
the output) is a string-encoded list where `order[outputPosition] =
originalAtomIndex`. This script builds its atom-target list indexed by
OUTPUT position via this exact mapping, then re-verifies (same
survives-the-round-trip element check as before) that the result is
actually correct before keeping a row.

--- Split ---

Grouped by canonical SMILES (not per-record) before splitting, so a
molecule with multiple ionizable-site records never has some of its own
records in train and others in the held-out test set (a real, avoidable
leakage risk given the one-row-per-record design above).

Usage:
    python3 prepare_pka_training_data.py \\
        [--input data/pka/combined_training_datasets_unique.sdf] \\
        [--output data/pka/pka_prepared.csv]

Needs: rdkit.
"""

import argparse
import hashlib
import sys
from pathlib import Path

from rdkit import Chem

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# TEST_FRAC is the remainder


def split_bucket(key):
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    frac = (h % 1_000_000) / 1_000_000
    if frac < TRAIN_FRAC:
        return "train"
    if frac < TRAIN_FRAC + VAL_FRAC:
        return "val"
    return "test"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/pka/combined_training_datasets_unique.sdf")
    ap.add_argument("--output", default="data/pka/pka_prepared.csv")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(
            f"{in_path} not found -- download it first:\n"
            f'  curl -sL -o {in_path} "https://raw.githubusercontent.com/czodrowskilab/'
            f'Machine-learning-meets-pKa/master/datasets/combined_training_datasets_unique.sdf"'
        )

    supplier = Chem.SDMolSupplier(str(in_path))
    rows = []
    n_total = 0
    n_dropped = 0
    for mol in supplier:
        n_total += 1
        if mol is None:
            n_dropped += 1
            continue
        if not (mol.HasProp("pKa") and mol.HasProp("marvin_atom") and mol.HasProp("marvin_pKa_type")):
            n_dropped += 1
            continue
        try:
            pka_value = float(mol.GetProp("pKa"))
            atom_idx = int(mol.GetProp("marvin_atom"))
        except ValueError:
            n_dropped += 1
            continue
        pka_type = mol.GetProp("marvin_pKa_type")
        if pka_type not in ("acidic", "basic"):
            n_dropped += 1
            continue
        if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
            n_dropped += 1
            continue

        # --- Neutralize the labeled atom's charge state (see module
        # docstring section on this) ---
        #
        # A large fraction of this SDF depicts the labeled ionizable atom
        # in its IONIZED form -- e.g. a carboxylic acid's acidic oxygen
        # drawn as the anion (-C(=O)[O-], formal charge -1, 0 H) rather
        # than the neutral acid (-C(=O)OH, charge 0, 1 H), or a basic
        # amine nitrogen drawn as its protonated ammonium/pyridinium
        # conjugate acid (+1) rather than the neutral amine. Confirmed
        # directly across the whole SDF, not assumed: 45.0% of all
        # acidic-labeled atoms and 58.8% of all basic-labeled atoms carry
        # a nonzero formal charge this way. ChemCanvas, however, always
        # queries this model on a NEUTRAL drawn molecule (predicting a
        # neutral acid's own pKa, not re-querying an already-deprotonated
        # anion) -- so roughly half of this dataset's own atom-level
        # FEATURES (formal charge, implicit-H count) for the exact atom
        # being trained on don't match what the model is asked about at
        # inference time. Root-caused directly from a real, reproducible
        # symptom, not guessed: the deployed model over-predicted simple
        # aliphatic carboxylic acid pKa by ~4 units (e.g. acetic acid:
        # predicted 8.91 vs. real 4.76) while an adjacent, never-labeled
        # atom on the SAME molecule (its carbonyl oxygen, whose
        # charge/H-count features are IDENTICAL between the anionic and
        # neutral drawings) predicted close to the correct value -- the
        # signature of a charge-state train/inference mismatch, not a
        # capacity or data-coverage problem (coverage for this exact
        # class was separately confirmed adequate: 540+291 real training
        # examples, mean pKa 4.0-4.1, nothing near 8.9).
        #
        # Fix: neutralize the labeled atom's formal charge to 0 and let
        # RDKit recompute its implicit-H count from the resulting valence
        # -- e.g. a charge=-1/0H carboxylate O becomes charge=0/1H
        # (matching a real neutral -OH), a charge=+1/1H pyridinium N
        # becomes charge=0/0H (matching neutral pyridine). Every other
        # atom, and the atom's own connectivity, is untouched.
        if mol.GetAtomWithIdx(atom_idx).GetFormalCharge() != 0:
            rw = Chem.RWMol(mol)
            target_atom = rw.GetAtomWithIdx(atom_idx)
            target_atom.SetFormalCharge(0)
            target_atom.SetNoImplicit(False)
            target_atom.SetNumExplicitHs(0)
            try:
                Chem.SanitizeMol(rw)
            except Exception:
                n_dropped += 1
                continue
            mol = rw.GetMol()

        # Real atom-order mapping (see module docstring for why
        # canonical=False alone is NOT this): output_order[outputPos] =
        # originalAtomIdx.
        try:
            smiles = Chem.MolToSmiles(mol, canonical=False)
            output_order = [int(x) for x in mol.GetProp("_smilesAtomOutputOrder").strip("[]").split(",") if x.strip() != ""]
            remol = Chem.MolFromSmiles(smiles)
        except Exception:
            n_dropped += 1
            continue
        if remol is None or remol.GetNumAtoms() != mol.GetNumAtoms() or len(output_order) != mol.GetNumAtoms():
            n_dropped += 1
            continue

        n_atoms = mol.GetNumAtoms()
        pka_list = ["'nan'"] * n_atoms
        target_output_pos = None
        for output_pos, orig_idx in enumerate(output_order):
            if orig_idx == atom_idx:
                target_output_pos = output_pos
                break
        if target_output_pos is None:
            n_dropped += 1
            continue
        pka_list[target_output_pos] = repr(pka_value)

        # Self-check: the element at the labeled OUTPUT position in the
        # re-parsed SMILES must match the element at the ORIGINAL atom
        # index in the source mol -- catches any remaining mapping bug
        # rather than trusting the arithmetic above blindly.
        if remol.GetAtomWithIdx(target_output_pos).GetSymbol() != mol.GetAtomWithIdx(atom_idx).GetSymbol():
            n_dropped += 1
            continue

        canonical_key = Chem.MolToSmiles(mol, canonical=True)
        rows.append({
            "smiles": smiles,
            "pka": "[" + ", ".join(pka_list) + "]",
            "pka_type": pka_type,
            "canonical_key": canonical_key,
        })

    print(f"{len(rows)}/{n_total} records kept ({n_dropped} dropped)", file=sys.stderr)
    if len(rows) < 500:
        sys.exit("Too few usable rows -- aborting rather than training on a tiny/broken dataset.")

    for r in rows:
        r["split"] = split_bucket(r["canonical_key"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("smiles,pka,split\n")
        for r in rows:
            f.write(f'"{r["smiles"]}","{r["pka"]}",{r["split"]}\n')
    print(f"wrote {out_path}", file=sys.stderr)

    from collections import Counter
    split_counts = Counter(r["split"] for r in rows)
    type_counts = Counter(r["pka_type"] for r in rows)
    print(f"split counts: {dict(split_counts)}", file=sys.stderr)
    print(f"type counts: {dict(type_counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
