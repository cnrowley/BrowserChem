#!/usr/bin/env python3
"""
join_cyp_descriptors.py

Left-joins scripts/pka-physical-baseline-harness/compute_cyp_descriptors.js's
output (data/cyp/descriptors.csv: logP, LogD(pH7), acidic/basic site pKa,
NAGL-MBIS charge min/max/mean, keyed by RDKit-canonical SMILES) onto a
CYP inhibition/substrate training CSV (smiles,label), producing a sibling
*_with_descriptors.csv chemprop can train on directly via
`--descriptors-columns logp logd pka_acidic has_acidic pka_basic
has_basic nagl_min nagl_max nagl_mean`.

Joins on RDKit-canonical SMILES (both files are already canonical --
prepare_cyp_training_data.py / prepare_cyp_substrate_training_data.py
re-parse every SMILES with RDKit before writing, and
compute_cyp_descriptors.js calls mol.get_smiles() -- but this script
re-canonicalizes both sides itself rather than trusting that, since a
silent canonicalization drift between the two independent pipelines
would otherwise fail joins silently rather than loudly). Unjoined rows
(descriptor computation failed -- see compute_cyp_descriptors.js's own
per-molecule error log, mostly real NAGL vocabulary gaps: I/Na/As/Se,
disconnected salts) are dropped and counted, never silently kept with
blank/zero descriptor values.

A second descriptor file can be merged in at the same time via --extra
(e.g. scripts/compute_cyp_sasa.py's data/cyp/sasa.csv: total_sasa,
polar_sasa) -- both files' non-smiles columns are auto-detected and
inner-joined (a row needs a match in EVERY descriptor file, not just
one, to appear in the final *_with_descriptors.csv -- SASA's own
embedding failures are a different set of molecules than the NAGL/logP/
pKa/BDE ones, so this is a real (if small) additional drop, counted).

Usage:
    python3 join_cyp_descriptors.py data/cyp/descriptors.csv data/cyp/cyp_cyp3a4.csv
    # writes data/cyp/cyp_cyp3a4_with_descriptors.csv

    python3 join_cyp_descriptors.py data/cyp/descriptors.csv --extra data/cyp/sasa.csv \\
        data/cyp/*.csv data/cyp_substrate/*.csv
"""
import argparse
import csv
import sys
from pathlib import Path

from rdkit import Chem


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_descriptors(path):
    cols = None
    table = {}
    dupes = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = [c for c in reader.fieldnames if c != "smiles"]
        for row in reader:
            key = canonical(row["smiles"])
            if key is None:
                continue
            if key in table:
                dupes += 1
                continue
            table[key] = {c: row[c] for c in cols}
    print(f"loaded {len(table)} descriptor rows ({cols}) from {path} ({dupes} duplicate canonical keys skipped)", file=sys.stderr)
    return table, cols


def merge_tables(tables):
    keys = set.intersection(*(set(t.keys()) for t in tables)) if tables else set()
    merged = {}
    for key in keys:
        row = {}
        for t in tables:
            row.update(t[key])
        merged[key] = row
    return merged


def join_one(descriptors, descriptor_cols, in_path):
    in_path = Path(in_path)
    out_path = in_path.with_name(in_path.stem + "_with_descriptors.csv")
    joined, dropped = 0, 0
    with open(in_path, newline="") as f_in, open(out_path, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames + descriptor_cols)
        writer.writeheader()
        for row in reader:
            key = canonical(row["smiles"])
            desc = descriptors.get(key) if key else None
            if desc is None:
                dropped += 1
                continue
            out_row = dict(row)
            out_row.update(desc)
            writer.writerow(out_row)
            joined += 1
    print(f"{in_path.name}: joined={joined} dropped={dropped} -> {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptors_csv")
    parser.add_argument("--extra", action="append", default=[], help="additional descriptor CSV(s) to inner-join in")
    parser.add_argument("training_csvs", nargs="+")
    args = parser.parse_args()

    tables, all_cols = [], []
    for path in [args.descriptors_csv] + args.extra:
        table, cols = load_descriptors(path)
        tables.append(table)
        all_cols += cols
    descriptors = merge_tables(tables)
    print(f"merged: {len(descriptors)} molecules with all descriptor sets present", file=sys.stderr)

    for p in args.training_csvs:
        join_one(descriptors, all_cols, p)


if __name__ == "__main__":
    main()
