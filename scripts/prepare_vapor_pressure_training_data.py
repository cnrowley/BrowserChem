#!/usr/bin/env python3
"""
prepare_vapor_pressure_training_data.py

Downloads and cleans the exact dataset model/registry.json's
vapor-pressure-v1 entry documents: the EPA/NICEATM OPERA vapor-pressure
endpoint (Zang et al. 2017, J. Chem. Inf. Model.), as redistributed by
gkxiao/vapor-pressure's train.csv/test.csv
(https://github.com/gkxiao/vapor-pressure). Confirmed an EXACT match to
the shipped checkpoint's own documented split, not just a plausible
source: gkxiao's train.csv has 2034 data rows and test.csv has 679, and
the registry's `dataset.splitStrategy` already records "pre-defined
train/test split as published (2034/679)" -- verified by directly
downloading both files and counting, not assumed from the repo's name
alone (contrast with logs-aqsoldb / melting-point in this same
scripts/ directory, where the shipped checkpoint's exact original data
genuinely can't be confirmed this precisely).

--- Source ---

Raw columns: CAS RN, SMILES, LogVP (log10 vapor pressure, mmHg, at 25 C
per the OPERA/Zang 2017 endpoint definition). Fetched directly from
GitHub raw content (no API key/auth needed, small files).

--- Processing ---

Re-parses every SMILES with RDKit and canonicalizes; unparseable rows
are dropped and counted. Keeps the train/test files SEPARATE (not
concatenated) since that pre-defined split is exactly what
train_vapor_pressure_chemprop.sh reproduces via chemprop's
--separate-test-path, matching the registry's documented methodology
(internal RANDOM 90/10 train/val split of the train partition only,
data-seed=0 -- NOT this project's usual SCAFFOLD_BALANCED convention,
because reproducing the exact published split takes precedence here).

Usage:
    python3 prepare_vapor_pressure_training_data.py [--output-dir data/vapor-pressure] [--cache-dir scripts/data_cache/vapor_pressure]

Needs: pandas, rdkit, requests.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

TRAIN_URL = "https://raw.githubusercontent.com/gkxiao/vapor-pressure/master/train.csv"
TEST_URL = "https://raw.githubusercontent.com/gkxiao/vapor-pressure/master/test.csv"


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"  downloading {url} -> {dest}", file=sys.stderr)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    else:
        print(f"  already have {dest}, skipping download", file=sys.stderr)
    return dest


def clean(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "SMILES" not in df.columns or "LogVP" not in df.columns:
        sys.exit(f"{csv_path}: expected columns SMILES/LogVP, got {list(df.columns)} -- source file layout changed?")
    n_in = len(df)

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
    out = df[["smiles", "LogVP"]].rename(columns={"LogVP": "logVP"}).drop_duplicates(subset="smiles")
    print(f"  [{label}] {n_in} rows in -> {n_bad} unparseable SMILES dropped -> {len(out)} molecules", file=sys.stderr)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/vapor-pressure")
    parser.add_argument("--cache-dir", default="scripts/data_cache/vapor_pressure")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] fetching train.csv / test.csv...", file=sys.stderr)
    train_raw = fetch(TRAIN_URL, cache_dir / "train.csv")
    test_raw = fetch(TEST_URL, cache_dir / "test.csv")

    print("[2/2] validating/canonicalizing SMILES with RDKit...", file=sys.stderr)
    train_out = clean(train_raw, "train")
    test_out = clean(test_raw, "test")

    train_dest = output_dir / "vapor_pressure_train.csv"
    test_dest = output_dir / "vapor_pressure_test.csv"
    train_out.to_csv(train_dest, index=False)
    test_out.to_csv(test_dest, index=False)
    print(f"  wrote {train_dest} ({len(train_out)}) and {test_dest} ({len(test_out)})", file=sys.stderr)


if __name__ == "__main__":
    main()
