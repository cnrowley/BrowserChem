#!/usr/bin/env python3
"""
prepare_herg_training_data.py

Builds a chemprop-ready binary-classification CSV predicting hERG
(KCNH2) potassium channel inhibition -- the standard early cardiotoxicity
/ QT-prolongation liability screen in drug discovery (the same endpoint
TDC packages as herg_karim/hERG_Blood, just sourced directly here -- see
--- Source --- below for why).

--- Source ---

PubChem BioAssay AID 588834 ("qHTS Assay for Small Molecule Inhibitors
of the Human hERG Channel Activity", NIH Chemical Genomics Center /
NIEHS / NTP -- a 1536-well U2OS cell-based qHTS assay, CONFIRMATORY tier
(follow-up concentration-response curves, not the noisier single-point
primary screen)). Fetched directly from PubChem's own PUG-REST
concise-CSV endpoint, same approach as prepare_cyp_training_data.py used
for the CYP450 panel (AID 1851) -- one real target, so no per-isoform
GeneID mapping is needed here, just a single Target GeneID/Accession
sanity check (GeneID 3757 -> KCNH2, the hERG-encoding gene; confirmed
against the downloaded CSV's own Target Accession column, not assumed).

--- Label ---

Binary: `Activity Outcome` "Active" -> 1 (hERG inhibitor / cardiotoxicity
liability), "Inactive" -> 0. "Inconclusive" rows are dropped. A few CIDs
have multiple SIDs tested (repeat/replicate samples) with occasionally
disagreeing outcomes -- resolved by majority vote; exact ties (rare) are
dropped rather than guessed, counted and reported. Same conventions as
prepare_cyp_training_data.py throughout this file.

--- SMILES ---

The assay CSV only has PubChem CID, not structure -- CanonicalSMILES is
fetched separately from PUG-REST's compound/property endpoint, batched
(200 CIDs/request) and cached to disk so a re-run doesn't re-hit the
network for CIDs already resolved. PubChem's PUG-REST returns that
property under a column literally named "ConnectivitySMILES" now, not
"CanonicalSMILES" -- the fetch code takes whichever non-CID column comes
back rather than hardcoding either name. Every fetched SMILES is
re-parsed with RDKit; unparseable entries are dropped (counted, not
silently ignored).

Usage:
    python3 prepare_herg_training_data.py [--output-dir data/herg] [--cache-dir scripts/data_cache/herg]

Needs: pandas, rdkit, requests.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

AID = 588834
ASSAY_CSV_URL = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{AID}/concise/CSV"
PROPERTY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/property/CanonicalSMILES/CSV"
BATCH_SIZE = 200
REQUEST_DELAY_S = 0.25  # stay well under PubChem's 5 req/s PUG-REST limit

EXPECTED_GENEID = 3757  # KCNH2 (hERG)
EXPECTED_ACCESSION = "NP_001191727"

OUTCOME_TO_LABEL = {"Active": 1, "Inactive": 0}


def fetch_assay_csv(cache_dir: Path) -> pd.DataFrame:
    dest = cache_dir / f"aid{AID}_concise.csv"
    if not dest.exists():
        print(f"  downloading {ASSAY_CSV_URL} -> {dest}", file=sys.stderr)
        resp = requests.get(ASSAY_CSV_URL, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    else:
        print(f"  already have {dest}, skipping download", file=sys.stderr)
    return pd.read_csv(dest, dtype={"CID": "Int64"})


def fetch_smiles(cids: list[int], cache_dir: Path) -> dict[int, str]:
    cache_path = cache_dir / "cid_smiles.json"
    cache: dict[str, str] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    missing = [c for c in cids if str(c) not in cache]
    print(f"  {len(cids)} unique CIDs, {len(missing)} not yet cached", file=sys.stderr)

    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i : i + BATCH_SIZE]
        url = PROPERTY_URL.format(cids=",".join(str(c) for c in batch))
        for attempt in range(5):
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code == 200:
                    break
                print(f"    batch {i}: HTTP {resp.status_code}, retry {attempt+1}/5", file=sys.stderr)
            except requests.RequestException as e:
                print(f"    batch {i}: {e}, retry {attempt+1}/5", file=sys.stderr)
            time.sleep(2 ** attempt)
        else:
            print(f"    batch {i}: giving up after 5 retries, skipping {len(batch)} CIDs", file=sys.stderr)
            continue

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        # PubChem's PUG-REST now returns the CanonicalSMILES property under a
        # "ConnectivitySMILES" column header -- take whichever non-CID column
        # came back rather than hardcode the name.
        smiles_col = [c for c in df.columns if c != "CID"][0]
        for _, row in df.iterrows():
            cache[str(int(row["CID"]))] = row[smiles_col]

        if (i // BATCH_SIZE) % 10 == 0:
            cache_path.write_text(json.dumps(cache))
            print(f"    fetched {i + len(batch)}/{len(missing)}", file=sys.stderr)
        time.sleep(REQUEST_DELAY_S)

    cache_path.write_text(json.dumps(cache))
    return {int(k): v for k, v in cache.items()}


def resolve_labels(group: pd.Series) -> int | None:
    counts = group.value_counts()
    if len(counts) == 1:
        return int(counts.index[0])
    if counts.iloc[0] > counts.iloc[1]:
        return int(counts.index[0])
    return None  # exact tie, drop


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/herg")
    parser.add_argument("--cache-dir", default="scripts/data_cache/herg")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] fetching assay AID {AID}...", file=sys.stderr)
    assay = fetch_assay_csv(cache_dir)
    assay = assay.dropna(subset=["CID"])
    assay["CID"] = assay["CID"].astype(int)

    print("[2/5] verifying target GeneID/accession...", file=sys.stderr)
    seen = assay[["Target GeneID", "Target Accession"]].drop_duplicates()
    for _, row in seen.iterrows():
        gid = int(row["Target GeneID"]) if pd.notna(row["Target GeneID"]) else None
        acc = row["Target Accession"]
        if gid != EXPECTED_GENEID or acc != EXPECTED_ACCESSION:
            print(f"  WARNING: unexpected target GeneID={gid} accession={acc!r} "
                  f"(expected GeneID={EXPECTED_GENEID}, accession={EXPECTED_ACCESSION!r})", file=sys.stderr)

    assay = assay[assay["Activity Outcome"].isin(OUTCOME_TO_LABEL)]
    assay["label"] = assay["Activity Outcome"].map(OUTCOME_TO_LABEL)

    print("[3/5] resolving per-CID label (majority vote over replicate SIDs)...", file=sys.stderr)
    resolved = assay.groupby("CID")["label"].apply(resolve_labels).reset_index()
    n_tied = resolved["label"].isna().sum()
    print(f"  dropped {n_tied} exact-tie CIDs", file=sys.stderr)
    resolved = resolved.dropna(subset=["label"])
    resolved["label"] = resolved["label"].astype(int)

    print("[4/5] fetching SMILES for all unique CIDs...", file=sys.stderr)
    all_cids = sorted(resolved["CID"].unique().tolist())
    cid_smiles = fetch_smiles(all_cids, cache_dir)

    resolved["smiles"] = resolved["CID"].map(cid_smiles)
    n_missing_smiles = resolved["smiles"].isna().sum()
    print(f"  {n_missing_smiles} rows with no SMILES resolved, dropping", file=sys.stderr)
    resolved = resolved.dropna(subset=["smiles"])

    print("[5/5] validating SMILES with RDKit and writing CSV...", file=sys.stderr)
    valid_cache: dict[str, str | None] = {}

    def canon(smi: str) -> str | None:
        if smi not in valid_cache:
            mol = Chem.MolFromSmiles(smi)
            valid_cache[smi] = Chem.MolToSmiles(mol) if mol is not None else None
        return valid_cache[smi]

    resolved["canonical_smiles"] = resolved["smiles"].apply(canon)
    n_bad = resolved["canonical_smiles"].isna().sum()
    resolved = resolved.dropna(subset=["canonical_smiles"])

    out = resolved[["canonical_smiles", "label"]].rename(columns={"canonical_smiles": "smiles"})
    out = out.drop_duplicates(subset="smiles")  # a handful of CIDs canonicalize to the same graph

    dest = output_dir / "herg.csv"
    out.to_csv(dest, index=False)
    n_pos = int(out["label"].sum())
    print(
        f"  hERG: {len(out)} molecules ({n_pos} inhibitor / {len(out) - n_pos} non-inhibitor), "
        f"{n_bad} dropped (bad SMILES) -> {dest}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
