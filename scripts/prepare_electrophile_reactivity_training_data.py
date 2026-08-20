#!/usr/bin/env python3
"""
prepare_electrophile_reactivity_training_data.py

Builds a chemprop-ready binary-classification CSV predicting whether a
molecule is protein-reactive (a covalent modifier / electrophilic
warhead) -- the endpoint model/registry.json's electrophile-reactivity-v1
entry already ships a checkpoint for, but whose dataset provenance was
previously undocumented (registry `dataset.name`/`citationKey`/`sourceUrl`
were all null; the shipped checkpoint's own size, 62686, doesn't match
this real source either, so this is a fresh, provenance-clean dataset for
retraining, not a byte-for-byte reconstruction of the existing checkpoint).

--- Source ---

RowleyGroup/covalent-classifier (github.com/RowleyGroup/covalent-classifier),
the real code + data release behind:

    Cano Gil, V. H.; Rowley, C. N. "Graph neural networks for identifying
    protein reactive electrophiles" Digital Discovery 2024, 3, 1776.
    https://doi.org/10.1039/D4DD00038B

Positive set: `data/SMILES_training/trainingset_covalent_smiles_larger_set.csv`
-- the LARGER positive set (per user instruction), which is the original
CovInDB/DrugBank-derived positive set plus additional electrophiles from
a 2026 Nature Cell Biology paper (per that repo's own
data/SMILES_training/README). Negative set:
`data/SMILES_training/trainingset_noncovalent_smiles.csv` (BindingDB +
DrugBank non-reactive compounds, per the same README).

Fetched directly from the repo's raw GitHub URLs (main branch) rather
than requiring a local clone, matching this project's other prepare_*
scripts (e.g. prepare_herg_training_data.py's PubChem fetch).

--- Label ---

Binary: 1 = protein-reactive (electrophile/covalent warhead), 0 = not.
Every SMILES is re-parsed and canonicalized with RDKit; unparseable rows
are dropped (counted, not silently ignored). A molecule appearing in both
the positive and negative source files (should not happen given the
source repo's own curation, but not assumed) is dropped from both rather
than guessed which label is correct.

Usage:
    python3 prepare_electrophile_reactivity_training_data.py [--output-dir data/electrophile-reactivity]

Needs: pandas, rdkit, requests.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

BASE_URL = "https://raw.githubusercontent.com/RowleyGroup/covalent-classifier/main/data/SMILES_training"
POSITIVE_URL = f"{BASE_URL}/trainingset_covalent_smiles_larger_set.csv"
NEGATIVE_URL = f"{BASE_URL}/trainingset_noncovalent_smiles.csv"


def fetch_csv(url: str, cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        print(f"  downloading {url} -> {cache_path}", file=sys.stderr)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    else:
        print(f"  already have {cache_path}, skipping download", file=sys.stderr)
    return pd.read_csv(cache_path)


def canonicalize(smiles_series: pd.Series) -> pd.Series:
    cache: dict[str, str | None] = {}

    def canon(smi: str) -> str | None:
        if smi not in cache:
            mol = Chem.MolFromSmiles(smi)
            cache[smi] = Chem.MolToSmiles(mol) if mol is not None else None
        return cache[smi]

    return smiles_series.apply(canon)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/electrophile-reactivity")
    parser.add_argument("--cache-dir", default="scripts/data_cache/electrophile-reactivity")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] fetching positive (larger) and negative sets...", file=sys.stderr)
    df_pos = fetch_csv(POSITIVE_URL, cache_dir / "trainingset_covalent_smiles_larger_set.csv")
    df_neg = fetch_csv(NEGATIVE_URL, cache_dir / "trainingset_noncovalent_smiles.csv")
    print(f"  positive (larger set): {len(df_pos)} rows, negative: {len(df_neg)} rows", file=sys.stderr)

    print("[2/4] canonicalizing SMILES with RDKit...", file=sys.stderr)
    df_pos = df_pos[["SMILES"]].copy()
    df_pos["label"] = 1
    df_neg = df_neg[["SMILES"]].copy()
    df_neg["label"] = 0

    df = pd.concat([df_pos, df_neg], ignore_index=True)
    df["smiles"] = canonicalize(df["SMILES"])
    n_bad = df["smiles"].isna().sum()
    df = df.dropna(subset=["smiles"])
    print(f"  {n_bad} rows dropped (unparseable SMILES)", file=sys.stderr)

    print("[3/4] resolving duplicates...", file=sys.stderr)
    n_before = len(df)
    label_counts = df.groupby("smiles")["label"].nunique()
    conflicting = label_counts[label_counts > 1].index
    n_conflict = len(conflicting)
    df = df[~df["smiles"].isin(conflicting)]
    df = df.drop_duplicates(subset="smiles")
    print(
        f"  {n_conflict} molecules appeared in both positive and negative sets (dropped), "
        f"{n_before - len(df) - n_conflict} exact duplicate rows collapsed",
        file=sys.stderr,
    )

    print("[4/4] writing CSV...", file=sys.stderr)
    out = df[["smiles", "label"]].reset_index(drop=True)
    dest = output_dir / "electrophile_reactivity.csv"
    out.to_csv(dest, index=False)
    n_pos = int(out["label"].sum())
    print(
        f"  electrophile-reactivity: {len(out)} molecules ({n_pos} reactive / {len(out) - n_pos} non-reactive) "
        f"-> {dest}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
