#!/usr/bin/env python3
"""
prepare_b3db_training_data.py

Builds a chemprop-ready binary-classification CSV predicting blood-brain
barrier penetration from B3DB -- a much larger (7809 vs. ~2050 compounds),
more recent curated BBB permeability database than the MoleculeNet BBBP
set scripts/prepare_bbbp_training_data.py uses, compiled from 50
published sources rather than one 2012 paper's own compilation.

--- Source ---

Meng, F.; Xi, Y.; Huang, J.; Ayers, P. W. "A curated diverse molecular
database of blood-brain barrier permeability with chemical descriptors."
Sci. Data 2021, 8, 289. DOI 10.1038/s41597-021-01069-5. Fetched directly
from the paper's own GitHub data repository (github.com/theochem/B3DB,
the ONE authoritative repo for this dataset, maintained by the QC-Devs/
theochem group -- Ayers is a co-author), B3DB_classification.tsv, not a
third-party mirror.

--- Label ---

Binary, from the `BBB+/BBB-` column: BBB+ -> 1, BBB- -> 0. A compound's
row also carries a `logBB` value when available (continuous logBB isn't
used here -- this project's other CYP/hERG/Ames/BBBP panels are all
binary classifiers, kept consistent) and a `threshold`/`reference`
provenance trail this script does not need.

--- SMILES ---

Every SMILES is re-parsed and re-canonicalized with RDKit; unparseable
entries are dropped (counted, not silently ignored). Duplicate canonical
SMILES resolved by majority vote; exact ties dropped rather than
guessed, counted and reported -- same convention as
prepare_ames_training_data.py/prepare_bbbp_training_data.py.

Usage:
    python3 prepare_b3db_training_data.py [--output-dir data/b3db] [--cache-dir scripts/data_cache/b3db]

Needs: rdkit, requests.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import requests
from rdkit import Chem

SOURCE_URL = "https://raw.githubusercontent.com/theochem/B3DB/main/B3DB/B3DB_classification.tsv"
LABEL_MAP = {"BBB+": 1, "BBB-": 0}


def fetch_tsv(cache_dir: Path) -> Path:
    dest = cache_dir / "B3DB_classification.tsv"
    if not dest.exists():
        print(f"  downloading {SOURCE_URL} -> {dest}", file=sys.stderr)
        resp = requests.get(SOURCE_URL, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    else:
        print(f"  already have {dest}, skipping download", file=sys.stderr)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/b3db")
    parser.add_argument("--cache-dir", default="scripts/data_cache/b3db")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = fetch_tsv(cache_dir)
    with open(tsv_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"parsed {len(rows)} rows from {tsv_path.name}", file=sys.stderr)

    by_canonical: dict[str, list[int]] = {}
    n_unparseable = 0
    n_bad_label = 0
    for row in rows:
        label_str = row.get("BBB+/BBB-", "").strip()
        if label_str not in LABEL_MAP:
            n_bad_label += 1
            continue
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            n_unparseable += 1
            continue
        canonical = Chem.MolToSmiles(mol)
        by_canonical.setdefault(canonical, []).append(LABEL_MAP[label_str])
    print(f"{n_unparseable} SMILES failed to parse, {n_bad_label} rows had no usable BBB+/BBB- label -- excluded", file=sys.stderr)

    out_rows = []
    n_ties = 0
    for canonical, labels in by_canonical.items():
        counts = Counter(labels)
        if len(counts) > 1:
            top = counts.most_common()
            if top[0][1] == top[1][1]:
                n_ties += 1
                continue
            majority_label = top[0][0]
        else:
            majority_label = labels[0]
        out_rows.append((canonical, majority_label))
    print(f"{len(by_canonical)} unique canonical SMILES, {n_ties} exact-tie duplicates dropped -> {len(out_rows)} final rows", file=sys.stderr)

    out_path = output_dir / "b3db.csv"
    with open(out_path, "w") as f:
        f.write("smiles,label\n")
        for smiles, label in out_rows:
            f.write(f"{smiles},{label}\n")

    n_pos = sum(1 for _, label in out_rows if label == 1)
    print(f"wrote {out_path}: {len(out_rows)} rows, {n_pos} positive ({100*n_pos/len(out_rows):.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
