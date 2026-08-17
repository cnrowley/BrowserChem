#!/usr/bin/env python3
"""
prepare_bde_training_data.py

Downloads and reshapes patonlab/BDE-db2 (S. V. Shree Sowndarya, Kim, Kim,
St. John, Paton -- "Expansion of bond dissociation prediction with machine
learning to medicinally and environmentally relevant chemical space",
Digital Discovery 2023, DOI 10.1039/D3DD00169E) into a chemprop
bond-target-ready CSV.

This is the dataset behind "Model 3" in that paper -- the final,
broadest-coverage model, extending the original ALFABET/BDE-db (St. John
et al. 2020, C/H/N/O only) to also cover halogenated (F/Cl/Br/I) and
S/P-containing organic molecules. Source file:
Datasets/training/bde_rdf_with_multi_halo_cfc_model_3.csv.gz in that repo
(~26 MB gzipped, ~834k per-BOND rows over ~65.7k unique molecules).

--- Long -> wide reshape ---

The raw file is one row per (molecule, bond): columns
`rid,molecule,bond_index,fragment1,fragment2,bond_type,bde,bdfe,bdscfe,set`.
`molecule` is a SMILES string; `bond_index` is confirmed (by direct
RDKit cross-check, not assumed) to index into `Chem.AddHs(mol)`'s own
`GetBonds()` order for that exact SMILES -- i.e. the EXPLICIT-HYDROGEN
graph, same convention this project already uses for the 1H NMR
checkpoint (chemprop-features-explicit-h.js / --add-h). Concretely: for
every molecule sampled, `bond_type` (e.g. "C-H") matches the symbols of
`Chem.AddHs(Chem.MolFromSmiles(smiles)).GetBondWithIdx(bond_index)`'s two
atoms, but NOT the plain (implicit-H) molecule's own bond order -- this
dataset was built the same way ALFABET's own BDE-db was.

chemprop's own bond-target CSV format (confirmed directly from its
installed source, chemprop/cli/utils/MAB_parsing.py) wants one row per
MOLECULE, with a `bde` column holding a Python list literal, one entry
per bond, IN THE SAME ORDER chemprop's own `--add-h`-mode featurizer will
enumerate that molecule's bonds (which is exactly `Chem.AddHs(mol)`'s own
bond order -- chemprop just calls RDKit the same way). So each molecule's
per-bond rows here get pivoted into one list of length
`Chem.AddHs(mol).GetNumBonds()`, with missing bond positions (not every
bond in every molecule has a computed BDE -- many molecules only have a
subset of bonds labeled, e.g. symmetry-redundant or chemically
uninteresting bonds are skipped) filled with the string 'nan' -- the same
placeholder convention scripts/prepare_nmr_training_data.py's own atom-
target CSVs already use, parsed by chemprop via
`np.array([...], dtype=float)`, which converts the string 'nan' to a
real NaN and everything else to its numeric value.

--- Splits ---

The source CSV already ships a `set` column (train/valid/test),
consistent per molecule (confirmed: 0/65740 molecules span more than one
split value) -- reused as-is via chemprop's `--splits-column` rather than
re-splitting, so this project's held-out test set matches the original
paper's. chemprop's splits-column reader expects exactly the lowercase
strings "train"/"val"/"test" (confirmed from chemprop/cli/train.py) --
the source's "valid" is remapped to "val" here.

--- Self-check ---

Mirrors prepare_nmr_training_data.py's own atom-order self-check: for
every molecule, re-parse the SMILES, call AddHs(), and confirm every
reported bond_index is in range and that bond_type (e.g. "C-H") matches
the actual two atoms at that index in the AddHs'd molecule. A mismatch on
any row drops that whole molecule (counted, not silently ignored) rather
than risk writing a mislabeled training row.

Usage:
    python3 prepare_bde_training_data.py [--output-dir data/bde] [--bdfe]

Needs: pandas, rdkit (present in the cov-chemprop conda env used
elsewhere in this project for chemprop training).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

SOURCE_URL = (
    "https://raw.githubusercontent.com/patonlab/BDE-db2/main/Datasets/training/"
    "bde_rdf_with_multi_halo_cfc_model_3.csv.gz"
)
EXPECTED_SIZE = 26359180  # bytes, gzipped -- confirmed via a real download; used only to skip re-downloading

SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def download(dest: Path) -> None:
    if dest.exists() and dest.stat().st_size == EXPECTED_SIZE:
        print(f"  already have {dest} ({dest.stat().st_size / 1e6:.1f} MB), skipping", file=sys.stderr)
        return
    print(f"  downloading {SOURCE_URL} -> {dest}", file=sys.stderr)
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(["curl", "-sL", "-o", str(tmp), SOURCE_URL], check=True)
    tmp.rename(dest)


def reshape(raw_csv_gz: Path, include_bdfe: bool):
    df = pd.read_csv(raw_csv_gz)
    df["set"] = df["set"].map(SPLIT_MAP)

    rows = []
    n_mols = 0
    n_dropped = 0
    n_bond_rows_dropped = 0

    for smiles, sub in df.groupby("molecule", sort=False):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_dropped += 1
            continue
        molH = Chem.AddHs(mol)
        nb = molH.GetNumBonds()

        bde_list = ["'nan'"] * nb
        bdfe_list = ["'nan'"] * nb if include_bdfe else None
        ok = True
        for _, row in sub.iterrows():
            bidx = int(row["bond_index"])
            if bidx < 0 or bidx >= nb:
                ok = False
                n_bond_rows_dropped += 1
                continue
            b = molH.GetBondWithIdx(bidx)
            sym1 = b.GetBeginAtom().GetSymbol()
            sym2 = b.GetEndAtom().GetSymbol()
            expected = {f"{sym1}-{sym2}", f"{sym2}-{sym1}"}
            if row["bond_type"] not in expected:
                ok = False
                n_bond_rows_dropped += 1
                continue
            bde_list[bidx] = repr(float(row["bde"]))
            if include_bdfe:
                bdfe_list[bidx] = repr(float(row["bdfe"]))

        if not ok:
            n_dropped += 1
            continue

        split_values = sub["set"].unique()
        split = split_values[0] if len(split_values) == 1 else "train"

        rec = {
            "smiles": smiles,
            "bde": "[" + ", ".join(bde_list) + "]",
            "split": split,
        }
        if include_bdfe:
            rec["bdfe"] = "[" + ", ".join(bdfe_list) + "]"
        rows.append(rec)
        n_mols += 1

    print(f"kept {n_mols} molecules, dropped {n_dropped} (self-check failed), "
          f"{n_bond_rows_dropped} individual bond rows failed the self-check within otherwise-dropped molecules",
          file=sys.stderr)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="data/bde")
    ap.add_argument("--bdfe", action="store_true", help="also include a bdfe target column")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "bde_rdf_with_multi_halo_cfc_model_3.csv.gz"
    download(raw_path)

    out_df = reshape(raw_path, args.bdfe)
    out_path = out_dir / "bde_bond_targets.csv"
    out_df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out_df)} molecules)", file=sys.stderr)
    print(out_df["split"].value_counts(), file=sys.stderr)


if __name__ == "__main__":
    main()
