#!/usr/bin/env python3
"""
prepare_aqsoldb_training_data.py

Downloads and cleans AqSolDB (Sorkun, Khetan & Cepeda 2019, Sci Data --
"A curated aqueous solubility dataset", 9,982 unique compounds), the
dataset model/registry.json's logs-aqsoldb entry names as its likely
source. Fetched from mcsorkun/AqSolDB's own GitHub repo
(https://github.com/mcsorkun/AqSolDB), specifically
results/data_curated.csv -- the ALREADY-curated final output of that
repo's own pipeline (data-curation.py -> merge.py's curate() + this
repo's own descriptors.py), not the 9 raw per-source sub-datasets
(dataset-A.csv..dataset-I.csv) re-run from scratch here. Re-running the
raw pipeline isn't attempted: dataset-A/H's own preprocessing step
(data-preprocess.py) does live web-scraping (collect_smiles_from_web/
collect_names_from_web/collect_predictions_from_web against third-party
sites) that's slow, fragile, and already baked into this repo's
committed results/data_curated.csv -- re-deriving it would just risk
producing a DIFFERENT dataset than the one that's actually been
published and cited, not a more faithful one.

--- Is this really what the shipped model/logs-aqsoldb/ checkpoint was
trained on? Likely, not confirmed ---

Notably stronger evidence than a name match alone: this project's own
registry.json notes on logs-aqsoldb name the exact curation script
files "data-curation.py, preprocess.py, merge.py" -- and this repo
contains files named exactly data-curation.py, merge.py, and
data-preprocess.py (registry's "preprocess.py" is presumably this file,
abbreviated). That's a strong, specific match, not a generic "AqSolDB
is a well-known dataset" guess. Still, the registry's own dataset.size
field is null (never confirmed against the shipped checkpoint's actual
training row count), so this script does NOT attempt to backfill
applicability-domain data for the currently-shipped checkpoint the way
scripts/compute_applicability_domain.py does for models with a fully
confirmed exact-match source (e.g. vapor-pressure-v1) -- only for
training a fresh checkpoint from this same real, citable source.

--- Processing ---

Re-parses every SMILES with RDKit and canonicalizes; unparseable rows
dropped and counted. Keeps only smiles + Solubility (LogS, log10 mol/L)
-- the rest of AqSolDB's own columns (RDKit-computed descriptors,
reliability Group/SD/Occurrences metadata) aren't needed for chemprop
training and are dropped.

Usage:
    python3 prepare_aqsoldb_training_data.py [--output-dir data/logs-aqsoldb] [--cache-dir scripts/data_cache/aqsoldb]

Needs: pandas, rdkit, requests.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

CURATED_CSV_URL = "https://raw.githubusercontent.com/mcsorkun/AqSolDB/master/results/data_curated.csv"


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"  downloading {url} -> {dest}", file=sys.stderr)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    else:
        print(f"  already have {dest}, skipping download", file=sys.stderr)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/logs-aqsoldb")
    parser.add_argument("--cache-dir", default="scripts/data_cache/aqsoldb")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] fetching AqSolDB's curated release...", file=sys.stderr)
    raw_path = fetch(CURATED_CSV_URL, cache_dir / "data_curated.csv")
    df = pd.read_csv(raw_path)
    if "SMILES" not in df.columns or "Solubility" not in df.columns:
        sys.exit(f"expected columns SMILES/Solubility, got {list(df.columns)} -- source file layout changed?")
    n_in = len(df)

    print("[2/2] validating/canonicalizing SMILES with RDKit...", file=sys.stderr)
    valid_cache: dict[str, str | None] = {}

    def canon(smi):
        if not isinstance(smi, str) or not smi:
            return None
        if smi not in valid_cache:
            mol = Chem.MolFromSmiles(smi)
            valid_cache[smi] = Chem.MolToSmiles(mol) if mol is not None else None
        return valid_cache[smi]

    df["smiles"] = df["SMILES"].apply(canon)
    n_bad = df["smiles"].isna().sum()
    df = df.dropna(subset=["smiles"])
    out = df[["smiles", "Solubility"]].rename(columns={"Solubility": "logS"}).drop_duplicates(subset="smiles")

    print(f"  {n_in} rows in -> {n_bad} unparseable SMILES dropped -> {len(out)} molecules", file=sys.stderr)
    if len(out) < 1000:
        sys.exit("Too few usable rows -- aborting rather than training on a tiny/broken dataset.")

    dest = output_dir / "aqsoldb.csv"
    out.to_csv(dest, index=False)
    print(f"  wrote {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
