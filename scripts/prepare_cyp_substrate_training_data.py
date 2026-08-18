#!/usr/bin/env python3
"""
prepare_cyp_substrate_training_data.py

Builds six chemprop-ready binary-classification CSVs predicting whether a
molecule is a SUBSTRATE (metabolized by) a given cytochrome P450 isoform
(CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP2E1, CYP3A4) -- the complementary
question to prepare_cyp_training_data.py's INHIBITION panel. A molecule
being a CYP substrate and being a CYP inhibitor are different biochemical
properties (turnover vs. blockade); this is deliberately a separate model
per isoform rather than a second output head bolted onto the inhibition
checkpoints, matching this project's one-checkpoint-per-registry-entry
convention.

--- Source ---

Xu et al., "Curated CYP450 Interaction Dataset: Covering the Majority of
Phase I Drug Metabolism", Scientific Data 12:513 (2025). A multi-source
compilation (DrugBank, CYP Knowledgebase, SuperCYP, plus three published
QSAR literature sets: Holmer et al., Yamashita et al., Yap et al.) of
~2,000 compounds per isoform with substrate/non-substrate labels,
published CC BY 4.0 on Figshare (article 26630515):
    https://doi.org/10.6084/m9.figshare.26630515

Each isoform ships as a pre-split {isoform}_trainingset.csv /
{isoform}_testingset.csv pair (columns: Name, SMILES, Label, Source).
Unlike prepare_cyp_training_data.py's PubChem source, this project does
NOT preserve that split -- the two files are concatenated, deduplicated,
and left unsplit, matching every other prepare_*_training_data.py script
here; scripts/train_cyp_substrate_chemprop.sh applies chemprop's own
--split SCAFFOLD_BALANCED at train time instead, for one consistent
splitting methodology across all of this project's checkpoints rather
than trusting a third party's held-out set (checked to have zero exact-
SMILES overlap with its own train file, but that's not the same
guarantee as a scaffold-disjoint split).

--- Label ---

Binary, taken directly from the source: 1 = substrate, 0 = non-substrate.
No "inconclusive" category exists in this source (unlike the inhibition
panel's PubChem assay data).

--- SMILES ---

Every SMILES is re-parsed and canonicalized with RDKit; unparseable rows
are dropped (counted, not silently ignored). If the same canonical SMILES
appears more than once for one isoform (duplicate across the train/test
split, or occasionally within one file), duplicates are resolved by
majority vote over labels; exact ties are dropped rather than guessed
-- same convention prepare_cyp_training_data.py uses for its replicate
CIDs.

Usage:
    python3 prepare_cyp_substrate_training_data.py [--output-dir data/cyp_substrate] [--cache-dir scripts/data_cache/cyp_substrate]

Needs: pandas, rdkit, requests.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

# (isoform, trainingset file id, testingset file id) on Figshare article
# 26630515 -- the plain (non "_PF" featurized) CSVs; ids confirmed live
# against the article's api.figshare.com/v2/articles/26630515 file list.
ISOFORM_FILES = {
    "cyp1a2": (53278937, 53278934),
    "cyp2c9": (53278943, 53278940),
    "cyp2c19": (53278949, 53278946),
    "cyp2d6": (53278955, 53278952),
    "cyp2e1": (53278961, 53278958),
    "cyp3a4": (53278967, 53278964),
}
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/{file_id}"


def fetch_csv(file_id: int, dest: Path) -> pd.DataFrame:
    if not dest.exists():
        url = DOWNLOAD_URL.format(file_id=file_id)
        print(f"  downloading {url} -> {dest}", file=sys.stderr)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    else:
        print(f"  already have {dest}, skipping download", file=sys.stderr)
    return pd.read_csv(dest)


def resolve_labels(group: pd.Series) -> int | None:
    counts = group.value_counts()
    if len(counts) == 1:
        return int(counts.index[0])
    if counts.iloc[0] > counts.iloc[1]:
        return int(counts.index[0])
    return None  # exact tie, drop


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/cyp_substrate")
    parser.add_argument("--cache-dir", default="scripts/data_cache/cyp_substrate")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    valid_cache: dict[str, str | None] = {}

    def canon(smi: str) -> str | None:
        if smi not in valid_cache:
            mol = Chem.MolFromSmiles(smi)
            valid_cache[smi] = Chem.MolToSmiles(mol) if mol is not None else None
        return valid_cache[smi]

    for isoform, (train_id, test_id) in ISOFORM_FILES.items():
        print(f"[{isoform}] fetching train/test CSVs...", file=sys.stderr)
        train_df = fetch_csv(train_id, cache_dir / f"{isoform}_trainingset.csv")
        test_df = fetch_csv(test_id, cache_dir / f"{isoform}_testingset.csv")
        combined = pd.concat([train_df, test_df], ignore_index=True)
        combined = combined.dropna(subset=["SMILES", "Label"])
        combined["Label"] = combined["Label"].astype(int)

        print(f"[{isoform}] canonicalizing {len(combined)} SMILES with RDKit...", file=sys.stderr)
        combined["canonical_smiles"] = combined["SMILES"].apply(canon)
        n_bad = combined["canonical_smiles"].isna().sum()
        combined = combined.dropna(subset=["canonical_smiles"])

        print(f"[{isoform}] resolving duplicate canonical SMILES (majority vote)...", file=sys.stderr)
        resolved = (
            combined.groupby("canonical_smiles")["Label"]
            .apply(resolve_labels)
            .reset_index()
            .rename(columns={"canonical_smiles": "smiles", "Label": "label"})
        )
        n_tied = resolved["label"].isna().sum()
        resolved = resolved.dropna(subset=["label"])
        resolved["label"] = resolved["label"].astype(int)

        dest = output_dir / f"cyp_substrate_{isoform.replace('cyp', '')}.csv"
        resolved.to_csv(dest, index=False)
        n_pos = int(resolved["label"].sum())
        print(
            f"[{isoform}] {len(resolved)} molecules ({n_pos} substrate / {len(resolved) - n_pos} non-substrate), "
            f"{n_bad} dropped (bad SMILES), {n_tied} dropped (exact-tie duplicates) -> {dest}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
