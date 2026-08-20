#!/usr/bin/env python3
"""
prepare_melting_point_training_data.py

Builds a chemprop-ready molecule-level regression CSV (SMILES -> melting
point) from the Jean-Claude Bradley Double Plus Good (Highly Curated and
Validated) Melting Point Dataset (Bradley, Lang & Williams 2014,
figshare, https://doi.org/10.6084/m9.figshare.1031638.v1) -- the same
dataset model/registry.json's melting-point entry cites.

--- IMPORTANT: this does NOT reproduce the currently-shipped
melting-point checkpoint's exact original training data ---

registry.json's own notes on this model record two things this script
can't currently confirm or replicate: (1) the shipped checkpoint's
values are in KELVIN, while this dataset's own raw `mpC` column is in
Celsius, so *some* conversion/curation step was applied before the
checkpoint was trained; (2) the shipped checkpoint used "a pre-defined
train/test split provided with the dataset (train/test column)", but no
such column exists in the raw Figshare file this script downloads (its
real columns are key/name/smiles/mpC/csid/link/source/count/min/max/
range -- checked directly, not assumed). Whatever produced the
currently-shipped checkpoint's training data did more than this script
does. This script is for training a NEW melting-point checkpoint from
scratch on this same real, citable public source (with its own fresh
SCAFFOLD_BALANCED split at train time, this project's standard
convention -- see train_melting_point_chemprop.sh) -- not for
regenerating applicability-domain data for the checkpoint currently
shipped in model/melting-point/, which would misrepresent what that
checkpoint actually saw.

--- Processing ---

Converts mpC (Celsius) to mp (Kelvin, mpC + 273.15) to match this
project's registry note that the shipped checkpoint's own values are in
Kelvin. Re-parses every SMILES with RDKit and canonicalizes; unparseable
rows are dropped and counted. Duplicate SMILES (the source notes some
compounds have multiple literature measurements already averaged into
`mpC`, but a few duplicate canonical structures still occur after
RDKit re-canonicalization) are averaged rather than arbitrarily
keeping one.

Usage:
    python3 prepare_melting_point_training_data.py [--output-dir data/melting-point] [--cache-dir scripts/data_cache/melting_point]

Needs: pandas, rdkit, requests, openpyxl.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

ARTICLE_FILE_URL = "https://ndownloader.figshare.com/files/1503991"
CELSIUS_TO_KELVIN = 273.15


def fetch_xlsx(cache_dir: Path) -> Path:
    dest = cache_dir / "BradleyDoublePlusGoodMeltingPointDataset.xlsx"
    if not dest.exists():
        print(f"  downloading {ARTICLE_FILE_URL} -> {dest}", file=sys.stderr)
        resp = requests.get(ARTICLE_FILE_URL, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    else:
        print(f"  already have {dest}, skipping download", file=sys.stderr)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/melting-point")
    parser.add_argument("--cache-dir", default="scripts/data_cache/melting_point")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] fetching Bradley melting-point dataset...", file=sys.stderr)
    xlsx_path = fetch_xlsx(cache_dir)
    df = pd.read_excel(xlsx_path, sheet_name="Sheet1")
    if "smiles" not in df.columns or "mpC" not in df.columns:
        sys.exit(f"expected columns 'smiles'/'mpC', got {list(df.columns)} -- source file layout changed?")
    n_in = len(df)

    print("[2/3] validating/canonicalizing SMILES with RDKit...", file=sys.stderr)
    valid_cache: dict[str, str | None] = {}

    def canon(smi):
        if not isinstance(smi, str) or not smi:
            return None
        if smi not in valid_cache:
            mol = Chem.MolFromSmiles(smi)
            valid_cache[smi] = Chem.MolToSmiles(mol) if mol is not None else None
        return valid_cache[smi]

    df["canonical_smiles"] = df["smiles"].apply(canon)
    n_bad_smiles = df["canonical_smiles"].isna().sum()
    df = df.dropna(subset=["canonical_smiles"])
    df = df.dropna(subset=["mpC"])
    df["mp"] = (df["mpC"].astype(float) + CELSIUS_TO_KELVIN).round(2)

    print("[3/3] averaging duplicate structures and writing CSV...", file=sys.stderr)
    grouped = df.groupby("canonical_smiles")["mp"].agg(["mean", "count"]).reset_index()
    n_dup_groups = (grouped["count"] > 1).sum()
    out = grouped.rename(columns={"canonical_smiles": "smiles", "mean": "mp"})[["smiles", "mp"]]
    out["mp"] = out["mp"].round(2)

    print(
        f"  {n_in} rows in -> {n_bad_smiles} unparseable SMILES dropped, "
        f"{n_dup_groups} duplicate-structure group(s) averaged -> {len(out)} molecules",
        file=sys.stderr,
    )
    if len(out) < 1000:
        sys.exit("Too few usable rows -- aborting rather than training on a tiny/broken dataset.")

    dest = output_dir / "melting_point.csv"
    out.to_csv(dest, index=False)
    print(f"  wrote {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
