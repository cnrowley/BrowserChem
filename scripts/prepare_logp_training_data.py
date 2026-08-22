#!/usr/bin/env python3
"""
prepare_logp_training_data.py

Downloads and cleans EPA/NIEHS OPERA's own curated LogP training set
(LogP_2.9_update.csv, version 2.9), the dataset behind OPERA's published
LogP QSAR model (Mansouri et al.) -- confirmed live by downloading and
inspecting it directly, not assumed from the dataset's name alone: 4,191
rows, EPA CompTox DSSTox-sourced compound identifiers (CASRN/DTXCID/
DTXSID), real literature-source citations per row (pub_source_name, e.g.
"PubChem(1)", "OChem(1)"), a QC_LEVEL reliability tag per row, and
value_point_estimate in real log units ranging -6 to 12 (a sane
octanol-water LogP range). Fetched from OPERA's own GitHub repo
(github.com/kmansouri/OPERA), specifically OPERA_Data.zip's
LogP_2.9_update.csv entry -- the whole repo is a ~79MB zip (OPERA bundles
its training data, models, and MATLAB source together), so this script
downloads the zip once (cached) and extracts just that one entry.

This directly replaces model/registry.json's logp-v1 entry, whose own
notes candidly say it "was provided pre-trained without accompanying
dataset/metrics documentation" -- this is a real, citable, documented
source instead.

--- Curation choices, stated explicitly ---

1. QC_LEVEL filtering: OPERA's own QC_LEVEL column tags each row's
   reliability (DSSTox_High, Public_High_CAS, Public_High, Public_Medium,
   Public_Low, DSSTox_Low). By default this drops the two "_Low" tiers
   (~559 of 4,191 rows, ~13%) -- OPERA's own documented lower-confidence
   bucket, not a cutoff invented here. Pass --include-low-quality to keep
   everything.
2. SMILES column: uses Canonical_QSARr (OPERA's own pre-cleaned/
   canonicalized SMILES, the same column OPERA's own QSAR-ready pipeline
   trains from), re-canonicalized again through RDKit here for this
   project's own consistency with every other prepare_*_training_data.py
   script; unparseable rows dropped and counted.
3. Duplicate compounds: unlike prepare_aqsoldb_training_data.py's simpler
   drop-duplicates (AqSolDB's own curation had already deduplicated),
   OPERA's LogP set has ~129 duplicate canonical SMILES -- genuinely
   different literature measurements of the same compound (different
   pub_source_name values), not artifacts, so this AVERAGES
   value_point_estimate per canonical SMILES rather than arbitrarily
   keeping one row.

Usage:
    python3 prepare_logp_training_data.py [--output-dir data/logp-opera] [--cache-dir scripts/data_cache/opera] [--include-low-quality]

Needs: pandas, rdkit, requests.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

OPERA_ZIP_URL = "https://raw.githubusercontent.com/kmansouri/OPERA/master/OPERA_Data.zip"
LOGP_CSV_NAME = "LogP_2.9_update.csv"
LOW_QUALITY_TIERS = {"Public_Low", "DSSTox_Low"}


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"  downloading {url} -> {dest}", file=sys.stderr)
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    else:
        print(f"  already have {dest}, skipping download", file=sys.stderr)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/logp-opera")
    parser.add_argument("--cache-dir", default="scripts/data_cache/opera")
    parser.add_argument("--include-low-quality", action="store_true",
                         help="keep OPERA's own Public_Low/DSSTox_Low QC tiers instead of dropping them")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] fetching OPERA's data bundle (~79MB, cached after first run)...", file=sys.stderr)
    zip_path = fetch(OPERA_ZIP_URL, cache_dir / "OPERA_Data.zip")
    csv_path = cache_dir / LOGP_CSV_NAME
    if not csv_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extract(LOGP_CSV_NAME, cache_dir)
    df = pd.read_csv(csv_path)

    required = {"Canonical_QSARr", "value_point_estimate", "QC_LEVEL", "unit"}
    if not required.issubset(df.columns):
        sys.exit(f"expected columns {required}, got {list(df.columns)} -- source file layout changed?")
    n_in = len(df)
    bad_units = (df["unit"] != "Log units").sum()
    if bad_units:
        print(f"  WARNING: {bad_units} row(s) have a unit other than 'Log units' -- dropping them", file=sys.stderr)
        df = df[df["unit"] == "Log units"]

    print("[2/3] applying QC_LEVEL filter...", file=sys.stderr)
    if not args.include_low_quality:
        n_before = len(df)
        df = df[~df["QC_LEVEL"].isin(LOW_QUALITY_TIERS)]
        print(f"  dropped {n_before - len(df)} row(s) tagged {sorted(LOW_QUALITY_TIERS)}", file=sys.stderr)

    print("[3/3] validating/canonicalizing SMILES with RDKit and averaging duplicates...", file=sys.stderr)
    valid_cache: dict[str, str | None] = {}

    def canon(smi):
        if not isinstance(smi, str) or not smi:
            return None
        if smi not in valid_cache:
            mol = Chem.MolFromSmiles(smi)
            valid_cache[smi] = Chem.MolToSmiles(mol) if mol is not None else None
        return valid_cache[smi]

    df["smiles"] = df["Canonical_QSARr"].apply(canon)
    n_bad = df["smiles"].isna().sum()
    df = df.dropna(subset=["smiles"])

    grouped = df.groupby("smiles")["value_point_estimate"].mean().reset_index()
    out = grouped.rename(columns={"value_point_estimate": "logP"})

    n_dupes = len(df) - len(out)
    print(f"  {n_in} rows in -> {n_bad} unparseable SMILES dropped -> "
          f"{n_dupes} duplicate rows averaged -> {len(out)} molecules", file=sys.stderr)
    if len(out) < 1000:
        sys.exit("Too few usable rows -- aborting rather than training on a tiny/broken dataset.")

    dest = output_dir / "logp_opera.csv"
    out.to_csv(dest, index=False)
    print(f"  wrote {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
