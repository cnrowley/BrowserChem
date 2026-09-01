#!/usr/bin/env python3
"""
prepare_bbbp_training_data.py

Builds a chemprop-ready binary-classification CSV predicting blood-brain
barrier penetration (BBBP) -- whether a compound crosses the BBB, a
standard early CNS-drug-discovery liability/relevance screen (a CNS
target needs BBB penetration; a peripherally-acting drug usually wants
to AVOID it, e.g. to limit CNS side effects).

--- Source ---

Martins, I. F.; Teixeira, A. L.; Pinheiro, L.; Falcao, A. O. "A Bayesian
Approach to in Silico Blood-Brain Barrier Penetration Modeling." J.
Chem. Inf. Model. 2012, 52, 1686-1697. DOI 10.1021/ci300124c -- ~2050
compounds compiled from the CNS/BBB pharmacology literature, distributed
as the "BBBP" dataset in Wu et al.'s MoleculeNet benchmark suite (Wu, Z.
et al. Chem. Sci. 2018, 9, 513-530, DOI 10.1039/C7SC02664A) and fetched
here directly from DeepChem's own public dataset mirror (the standard,
widely-used distribution point for every MoleculeNet dataset -- verified
live and reachable, not a third-party scrape).

--- Label ---

Binary, already provided directly in the source CSV's `p_np` column:
1 = BBB-penetrant, 0 = non-penetrant. No PubChem CID resolution needed.

--- SMILES ---

Every SMILES is re-parsed and re-canonicalized with RDKit; unparseable
entries are dropped (counted, not silently ignored) -- MoleculeNet's own
BBBP.csv has a handful of known-bad SMILES strings (this is disclosed
upstream, e.g. deepchem's own loader has documented workarounds for a
couple of specific rows), so a non-zero drop count here is expected, not
a bug in this script. Duplicate canonical SMILES resolved by majority
vote on p_np; exact ties dropped rather than guessed, counted and
reported -- same convention as prepare_ames_training_data.py.

Usage:
    python3 prepare_bbbp_training_data.py [--output-dir data/bbbp] [--cache-dir scripts/data_cache/bbbp]

Needs: rdkit, requests.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import requests
from rdkit import Chem

SOURCE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"


def fetch_csv(cache_dir: Path) -> Path:
    dest = cache_dir / "BBBP.csv"
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
    parser.add_argument("--output-dir", default="data/bbbp")
    parser.add_argument("--cache-dir", default="scripts/data_cache/bbbp")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_path = fetch_csv(cache_dir)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"parsed {len(rows)} rows from {csv_path.name}", file=sys.stderr)

    by_canonical: dict[str, list[int]] = {}
    n_unparseable = 0
    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            n_unparseable += 1
            continue
        canonical = Chem.MolToSmiles(mol)
        by_canonical.setdefault(canonical, []).append(int(row["p_np"]))
    print(f"{n_unparseable} SMILES failed to parse -- excluded", file=sys.stderr)

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

    out_path = output_dir / "bbbp.csv"
    with open(out_path, "w") as f:
        f.write("smiles,label\n")
        for smiles, label in out_rows:
            f.write(f"{smiles},{label}\n")

    n_pos = sum(1 for _, label in out_rows if label == 1)
    print(f"wrote {out_path}: {len(out_rows)} rows, {n_pos} positive ({100*n_pos/len(out_rows):.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
