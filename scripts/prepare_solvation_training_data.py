#!/usr/bin/env python3
"""
prepare_solvation_training_data.py

Builds one chemprop-ready molecule-level regression CSV per solvent,
predicting solvation free energy (dGsolv, kcal/mol) of a solute dissolved
in that solvent -- one checkpoint per FIXED solvent (solute SMILES in,
dGsolv out), not a true two-molecule solute+solvent model. This project's
from-scratch JS D-MPNN (graph-builder.js/dmpnn.js/chemprop-model.js) and
convert_chemprop_checkpoint.py only support a single molecular graph per
checkpoint -- no multi-component/solvent-conditioned architecture exists
here (or in the upstream chemprop_solvation reference implementation this
source dataset was originally paired with, which uses a genuinely
different two-molecule architecture this project does not replicate).
A fixed panel of solvents, one single-task model each, is what
scripts/train_solvation_chemprop.sh actually trains -- sufficient for a
per-solvent table and a solvent x solvent transfer-energy matrix (both
only ever need a fixed, named solvent list), just not an arbitrary
user-typed solvent.

--- Source ---

CombiSolv-QM (Vermeire, F. H.; Green, W. H. "Transfer learning for
solvation free energies: From quantum chemistry to experiments." Chem.
Eng. J. 2021, 418, 129307. https://doi.org/10.1016/j.cej.2021.129307):
1,000,000 solute-solvent free energies computed with real COSMO-RS
(COSMOtherm), covering 11,029 solutes x 284 solvents. The paper's own
supplementary information is the canonical source; this script fetches a
byte-identical mirror hosted in su-group/SolvBERT's companion repo (a
later paper's reproduction/benchmarking code -- https://github.com/
su-group/SolvBERT/blob/master/solv-bert/data/CombiSolv-QM.csv) since
that's a stable, directly-downloadable CSV (no license file in that repo;
cached locally, gitignored, same as every other prepared dataset in this
project -- not redistributed from this repository).

The CSV's own `ssid` column is "<solvent_smiles>.<solute_smiles>" (dot-
joined, confirmed by value-counts: the first component has exactly 284
unique values, the second 11029, matching the paper's own solvent/solute
counts); `dgsolv` is the target, kcal/mol.

--- Solvent panel ---

21 solvents commonly used in organic synthesis/workup/purification,
spanning the polarity and H-bonding range (protic, aprotic polar,
aprotic nonpolar, chlorinated, aromatic) -- chosen from what's actually
well-represented in this dataset (~3200-4200 rows each). Deliberately
excludes a few solvents that ARE present in the dataset but are obsolete/
hazardous rather than in real current use (carbon tetrachloride --
ozone-depleting, phased out; nitromethane -- niche/explosive-precursor
use only; benzene -- carcinogen, toluene serves as the aromatic
representative instead).

--- Label ---

Continuous: dGsolv in kcal/mol, taken directly from the source (COSMO-RS
free energy of solvation -- more negative = more favorable/soluble).

--- SMILES ---

Every solute SMILES is re-parsed and canonicalized with RDKit;
unparseable rows are dropped (counted). Solvent matching is done on
RDKit-canonicalized SMILES (not string-identical to this script's own
SOLVENTS dict), so it's robust to the source CSV's own SMILES writing
convention. Duplicate canonical solute SMILES within one solvent's slice
(rare -- COSMO-RS is deterministic per solute/solvent pair, but a solute
occasionally appears more than once with a distinct starting geometry)
are resolved by averaging dGsolv rather than dropping, since this is a
continuous target (no majority-vote/tie concept applies).

Usage:
    python3 prepare_solvation_training_data.py [--output-dir data/solvation] [--cache-dir scripts/data_cache/solvation]

Needs: pandas, rdkit, requests.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

SOURCE_URL = "https://raw.githubusercontent.com/su-group/SolvBERT/master/solv-bert/data/CombiSolv-QM.csv"

# name -> SMILES (any valid writing; canonicalized below before matching)
SOLVENTS = {
    "water": "O",
    "methanol": "CO",
    "ethanol": "CCO",
    "2-propanol": "CC(C)O",
    "1-octanol": "CCCCCCCCO",
    "acetone": "CC(C)=O",
    "acetonitrile": "CC#N",
    "dmso": "CS(C)=O",
    "dmf": "CN(C)C=O",
    "thf": "C1CCOC1",
    "dioxane": "C1COCCO1",
    "diethyl-ether": "CCOCC",
    "ethyl-acetate": "CCOC(C)=O",
    "dcm": "ClCCl",
    "chloroform": "ClC(Cl)Cl",
    "toluene": "Cc1ccccc1",
    "hexane": "CCCCCC",
    "heptane": "CCCCCCC",
    "pyridine": "c1ccncc1",
    "acetic-acid": "CC(=O)O",
    "cyclohexane": "C1CCCCC1",
}


def canon(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def fetch_source(cache_dir: Path) -> pd.DataFrame:
    dest = cache_dir / "CombiSolv-QM.csv"
    if not dest.exists():
        print(f"  downloading {SOURCE_URL} -> {dest}", file=sys.stderr)
        resp = requests.get(SOURCE_URL, timeout=300)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    else:
        print(f"  already have {dest}, skipping download", file=sys.stderr)
    return pd.read_csv(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/solvation")
    parser.add_argument("--cache-dir", default="scripts/data_cache/solvation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] fetching CombiSolv-QM...", file=sys.stderr)
    df = fetch_source(cache_dir)
    df[["solvent_smiles", "solute_smiles"]] = df["ssid"].str.split(".", n=1, expand=True)

    print("[2/3] canonicalizing solvent panel + all unique solvent SMILES in the source...", file=sys.stderr)
    target_canon = {name: canon(smi) for name, smi in SOLVENTS.items()}
    missing = [name for name, c in target_canon.items() if c is None]
    if missing:
        raise SystemExit(f"SOLVENTS dict has unparseable SMILES: {missing}")

    source_solvent_canon = {s: canon(s) for s in df["solvent_smiles"].unique()}
    df["solvent_canon"] = df["solvent_smiles"].map(source_solvent_canon)

    valid_cache: dict[str, str | None] = {}

    def canon_cached(smi: str) -> str | None:
        if smi not in valid_cache:
            valid_cache[smi] = canon(smi)
        return valid_cache[smi]

    print("[3/3] slicing per solvent, canonicalizing solutes, writing CSVs...", file=sys.stderr)
    for name, target_smi_canon in target_canon.items():
        sub = df[df["solvent_canon"] == target_smi_canon].copy()
        if sub.empty:
            print(f"  WARNING: {name} ({target_smi_canon}) matched 0 rows -- skipping", file=sys.stderr)
            continue

        sub["canonical_smiles"] = sub["solute_smiles"].apply(canon_cached)
        n_bad = sub["canonical_smiles"].isna().sum()
        sub = sub.dropna(subset=["canonical_smiles"])

        grouped = sub.groupby("canonical_smiles")["dgsolv"].mean().reset_index()
        n_dupes = len(sub) - len(grouped)

        dest = output_dir / f"solv_{name}.csv"
        grouped.rename(columns={"canonical_smiles": "smiles"}).to_csv(dest, index=False)
        print(
            f"  {name}: {len(grouped)} molecules ({n_bad} dropped bad SMILES, "
            f"{n_dupes} duplicate solutes averaged) -> {dest}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
