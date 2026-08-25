#!/usr/bin/env python3
"""
add_opera_salicylate_pka_data.py

Patches a real, diagnosed gap in the aqueous-pka model's training data
(model/registry.json's aqueous-pka entry / PKA_INTEGRATION.md-style
investigation): the Baltruschat & Czodrowski dataset this model is
trained on (data/pka/combined_training_datasets_unique.sdf, prepared by
prepare_pka_training_data.py) has ZERO training rows where a phenol
ortho to a carboxylic acid (the salicylic-acid scaffold) has ITS OWN
pKa labeled -- confirmed directly by substructure search across all
5,936 prepared rows. Salicylic acid itself IS in that dataset (val
split), but only its carboxyl site is labeled; the real ~4-pKa-unit
elevation of its phenol site (from intramolecular H-bonding to the
adjacent carboxylate) is simply never taught to the model anywhere in
that corpus -- confirmed as the direct cause of a live symptom: the
shipped model predicts salicylic acid's phenol at ~9.3 (barely above
plain phenol's ~9.7) vs. the real ~13.4-13.8.

--- Source: OPERA's real experimental pKa dataset ---

EPA/NCCT's OPERA suite (github.com/kmansouri/OPERA, OPERA_Data.zip's
pKa_QR.sdf, 6503 real molecules) -- already this app's source for the
logp-v1 training set (see that registry entry). Each record has a
`pKa_a_ref` field: normally one real literature-referenced acidic pKa
value, but for many polyprotic acids it's TWO (or more) pipe-delimited
real values from different literature measurements of the SAME
molecule's different dissociation steps -- confirmed directly:
salicylic acid's own record has `pKa_a_ref = "2.99|13.42"`, both real
values (2.99 matches the ~2.97-2.98 literature consensus almost
exactly; 13.42 is a real, if slightly different, literature value for
the phenol -- other sources report ~13.6-13.82, normal literature
spread for a hard-to-measure high pKa).

--- Scope: deliberately narrow, not a wholesale dataset merge ---

Only molecules with EXACTLY one carboxylic-acid site and EXACTLY one
phenol site (by the same SMARTS js/pka-microstates.js uses) AND exactly
two real pKa_a_ref values are used -- this keeps the "which value goes
with which atom" mapping unambiguous without guessing. The two values
are sorted ascending and assigned lower->carboxyl, higher->phenol: a
carboxylic acid's pKa is essentially always well below a phenol's
(typically by 5+ units) even under the ortho-substituent effects this
patch targets, so this ordering is chemically safe, not a coin flip.
19 raw OPERA records -- 17 unique molecules after de-duplication --
match this filter; see this script's own printed output for the exact
list (salicylic acid, 5-methylsalicylic acid, 3-hydroxy-2-naphthoic
acid, several nitro-substituted salicylates, meta/para-hydroxybenzoic
acid controls, etc.) -- a small but clean, chemically diverse,
real-literature-sourced set spanning ortho/meta/para substitution
patterns, which is exactly the contrast the model needs to learn the
ortho-specific effect rather than a generic "phenol near a carbonyl"
shortcut.

--- Row format: matches prepare_pka_training_data.py exactly ---

Same one-row-per-ionizable-site convention, same
Chem.MolToSmiles(mol, canonical=False) + _smilesAtomOutputOrder
atom-position mapping (with the same round-trip element self-check),
same deterministic split_bucket(canonical_smiles) function copied
verbatim -- so any molecule already in the base dataset (e.g. salicylic
acid) lands in the identical split as before (no leakage change), and
any molecule new to this patch gets a real, reproducible split
assignment consistent with the rest of the pipeline.

--- De-duplication against the base dataset ---

For any of these 17 molecules already present in the base
pka_prepared.csv (by canonical SMILES), this script DROPS the base
dataset's existing row(s) for that molecule and replaces them with its
own -- OPERA's carboxyl values are real, more accurate literature
values for this exact class (e.g. salicylic acid: OPERA's 2.99 vs. the
base dataset's 3.35, real value ~2.97-2.98) and, critically, add the
phenol label the base dataset never had at all. Keeping both would just
give the model two conflicting labels for the same atom of the same
molecule for no benefit.

Usage:
    python3 add_opera_salicylate_pka_data.py \\
        [--opera-sdf /tmp/.../pKa_QR.sdf] \\
        [--base-csv data/pka/pka_prepared.csv] \\
        [--output data/pka/pka_prepared.csv]

Needs: rdkit.
"""

import argparse
import ast
import hashlib
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

COOH_SMARTS = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
PHENOL_SMARTS = Chem.MolFromSmarts('[OX2H1][c]')


def split_bucket(key):
    # Copied verbatim from prepare_pka_training_data.py so a molecule
    # already in the base dataset gets the IDENTICAL split assignment.
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    frac = (h % 1_000_000) / 1_000_000
    if frac < TRAIN_FRAC:
        return "train"
    if frac < TRAIN_FRAC + VAL_FRAC:
        return "val"
    return "test"


def build_row(mol, atom_idx, pka_value):
    """One (smiles, pka-list, split) row for a single labeled atom, matching
    prepare_pka_training_data.py's exact SMILES/atom-order/self-check logic."""
    try:
        smiles = Chem.MolToSmiles(mol, canonical=False)
        output_order = [int(x) for x in mol.GetProp("_smilesAtomOutputOrder").strip("[]").split(",") if x.strip() != ""]
        remol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if remol is None or remol.GetNumAtoms() != mol.GetNumAtoms() or len(output_order) != mol.GetNumAtoms():
        return None

    n_atoms = mol.GetNumAtoms()
    pka_list = ["'nan'"] * n_atoms
    target_output_pos = None
    for output_pos, orig_idx in enumerate(output_order):
        if orig_idx == atom_idx:
            target_output_pos = output_pos
            break
    if target_output_pos is None:
        return None
    pka_list[target_output_pos] = repr(pka_value)

    if remol.GetAtomWithIdx(target_output_pos).GetSymbol() != mol.GetAtomWithIdx(atom_idx).GetSymbol():
        return None

    canonical_key = Chem.MolToSmiles(mol, canonical=True)
    return {
        "smiles": smiles,
        "pka": "[" + ", ".join(pka_list) + "]",
        "split": split_bucket(canonical_key),
        "canonical_key": canonical_key,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opera-sdf", required=True)
    ap.add_argument("--base-csv", default="data/pka/pka_prepared.csv")
    ap.add_argument("--output", default="data/pka/pka_prepared.csv")
    args = ap.parse_args()

    suppl = Chem.SDMolSupplier(args.opera_sdf)
    new_rows = []
    seen_canonical = set()
    kept_molecules = []
    for mol in suppl:
        if mol is None or not mol.HasProp('pKa_a_ref'):
            continue
        ref = mol.GetProp('pKa_a_ref')
        vals = ref.split('|')
        if len(vals) != 2:
            continue
        try:
            v1, v2 = float(vals[0]), float(vals[1])
        except ValueError:
            continue
        cooh_matches = mol.GetSubstructMatches(COOH_SMARTS)
        phenol_matches = mol.GetSubstructMatches(PHENOL_SMARTS)
        if len(cooh_matches) != 1 or len(phenol_matches) != 1:
            continue

        # Sanity gap: a genuine carboxyl-vs-phenol pair is always separated
        # by several pKa units (smallest seen among the clean matches here
        # is ~4.5) -- two values only ~0.5 apart (confirmed on one real
        # case: (2E)-3-(4-hydroxyphenyl)-2-propenoic acid, "4.08|4.64")
        # are almost certainly two different literature MEASUREMENTS of
        # the SAME site, not genuinely distinct carboxyl/phenol values --
        # assigning them to different atoms would inject a wrong label.
        if abs(max(v1, v2) - min(v1, v2)) < 3.0:
            continue

        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen_canonical:
            continue  # duplicate OPERA record for the same structure
        seen_canonical.add(canonical)

        cooh_o_idx = cooh_matches[0][2]  # atomIdx=2 in [CX3](=O)[OX2H1] -- matches pka-microstates.js's carboxylic_acid site
        phenol_o_idx = phenol_matches[0][0]  # atomIdx=0 in [OX2H1][c] -- matches pka-microstates.js's phenol site

        lo, hi = sorted([v1, v2])
        row_cooh = build_row(Chem.Mol(mol), cooh_o_idx, lo)
        row_phenol = build_row(Chem.Mol(mol), phenol_o_idx, hi)
        if row_cooh is None or row_phenol is None:
            print(f"WARNING: skipping {mol.GetProp('Substance_Name') if mol.HasProp('Substance_Name') else canonical} -- row-build/self-check failed", file=sys.stderr)
            continue

        new_rows.append(row_cooh)
        new_rows.append(row_phenol)
        name = mol.GetProp('Substance_Name') if mol.HasProp('Substance_Name') else '(unnamed)'
        kept_molecules.append((name, canonical, lo, hi))

    print(f"{len(kept_molecules)} unique molecules matched (exactly 1 COOH + 1 phenol + 2 real pKa_a_ref values):", file=sys.stderr)
    for name, smi, lo, hi in kept_molecules:
        print(f"  {name:45s} {smi:40s} carboxyl={lo} phenol={hi}", file=sys.stderr)

    base_df = pd.read_csv(args.base_csv)
    new_canonicals = seen_canonical

    def base_row_canonical(smi):
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None

    base_df["_canonical"] = base_df["smiles"].apply(base_row_canonical)
    n_before = len(base_df)
    kept_base = base_df[~base_df["_canonical"].isin(new_canonicals)].drop(columns=["_canonical"])
    n_dropped = n_before - len(kept_base)
    print(f"\nBase dataset: {n_before} rows -> dropping {n_dropped} rows for molecules replaced by this patch", file=sys.stderr)

    new_df = pd.DataFrame(new_rows)[["smiles", "pka", "split"]]
    out_df = pd.concat([kept_base, new_df], ignore_index=True)
    print(f"Final dataset: {len(out_df)} rows ({len(new_df)} new)", file=sys.stderr)

    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
