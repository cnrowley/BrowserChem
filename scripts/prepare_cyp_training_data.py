#!/usr/bin/env python3
"""
prepare_cyp_training_data.py

Builds five chemprop-ready binary-classification CSVs predicting
cytochrome P450 inhibition, one per isoform (CYP1A2, CYP2C9, CYP2C19,
CYP2D6, CYP3A4) -- the standard panel used across ADMET literature
(this is the same underlying assay TDC packages as
CYP1A2_Veith/CYP2C9_Veith/CYP2C19_Veith/CYP2D6_Veith/CYP3A4_Veith).

--- Source ---

PubChem BioAssay AID 1851 ("Cytochrome panel assay with activity
outcomes", Veith et al. 2009, Nat Biotechnol 27:1050-1055 -- qHTS of
~17k compounds across 5 luciferase-based P450-Glo isozyme assays,
NIH/NCGC). Fetched directly from PubChem's own PUG-REST concise-CSV
endpoint (TDC's usual backing store, Harvard Dataverse, was down --
HTTP 202/WAF-challenge -- when this was written; PubChem is the
underlying primary source TDC itself redistributes, so this bypasses
that outage rather than working around it with a substitute dataset).

One assay row per (SID, isoform); isoform is NOT a column directly, it's
inferred from `Target GeneID`, confirmed against known human CYP RefSeq
records:
    1544 -> NP_000752      -> CYP1A2
    1557 -> NP_000760       -> CYP2C19
    1559 -> NP_000762      -> CYP2C9
    1565 -> NP_001020332    -> CYP2D6
    1576 -> NP_059488       -> CYP3A4
(confirmed directly from the downloaded CSV's own Target Accession
column, not assumed from GeneID alone.)

--- Label ---

Binary: `Activity Outcome` "Active" -> 1 (inhibitor), "Inactive" -> 0.
"Inconclusive" rows are dropped. A few CIDs have multiple SIDs tested
against the same isoform (repeat/replicate samples) with occasionally
disagreeing outcomes -- resolved by majority vote; exact ties (rare) are
dropped rather than guessed, counted and reported.

--- SMILES ---

The assay CSV only has PubChem CID, not structure -- CanonicalSMILES is
fetched separately from PUG-REST's compound/property endpoint, batched
(200 CIDs/request, matching PubChem's documented list-size comfort zone)
and cached to disk so a re-run doesn't re-hit the network for CIDs
already resolved. (PubChem's PUG-REST now returns that property under a
column literally named "ConnectivitySMILES", not "CanonicalSMILES" --
confirmed via a live request -- so the fetch code takes whichever
non-CID column comes back rather than hardcoding either name.) Every
fetched SMILES is re-parsed with RDKit; unparseable
entries are dropped (counted, not silently ignored) rather than risk
writing a broken training row -- same convention as this project's other
prepare_*_training_data.py scripts.

No pre-baked split is available from this source (unlike BDE-db2/pKa),
so -- matching prepare_qm9_training_data.py's convention -- these CSVs
are left unsplit; scripts/train_cyp_chemprop.sh applies chemprop's own
--split SCAFFOLD_BALANCED at train time.

Usage:
    python3 prepare_cyp_training_data.py [--output-dir data/cyp] [--cache-dir scripts/data_cache/cyp]

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

AID = 1851
ASSAY_CSV_URL = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{AID}/concise/CSV"
PROPERTY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/property/CanonicalSMILES/CSV"
BATCH_SIZE = 200
REQUEST_DELAY_S = 0.25  # stay well under PubChem's 5 req/s PUG-REST limit

GENEID_TO_ISOFORM = {
    1544: "CYP1A2",
    1557: "CYP2C19",
    1559: "CYP2C9",
    1565: "CYP2D6",
    1576: "CYP3A4",
}
EXPECTED_ACCESSION = {
    1544: "NP_000752",
    1557: "NP_000760",
    1559: "NP_000762",
    1565: "NP_001020332",
    1576: "NP_059488",
}

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
        # "ConnectivitySMILES" column header (confirmed via a live request) --
        # take whichever non-CID column came back rather than hardcode the name.
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
    parser.add_argument("--output-dir", default="data/cyp")
    parser.add_argument("--cache-dir", default="scripts/data_cache/cyp")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] fetching assay AID {AID}...", file=sys.stderr)
    assay = fetch_assay_csv(cache_dir)
    assay = assay.dropna(subset=["CID"])
    assay["CID"] = assay["CID"].astype(int)

    print("[2/5] mapping Target GeneID -> isoform, verifying accessions...", file=sys.stderr)
    seen_accessions = assay[["Target GeneID", "Target Accession"]].drop_duplicates()
    for _, row in seen_accessions.iterrows():
        gid = int(row["Target GeneID"])
        acc = row["Target Accession"]
        expected = EXPECTED_ACCESSION.get(gid)
        if expected is None or acc != expected:
            print(f"  WARNING: unexpected accession {acc} for GeneID {gid} (expected {expected})", file=sys.stderr)
    assay["isoform"] = assay["Target GeneID"].map(GENEID_TO_ISOFORM)
    assay = assay.dropna(subset=["isoform"])

    assay = assay[assay["Activity Outcome"].isin(OUTCOME_TO_LABEL)]
    assay["label"] = assay["Activity Outcome"].map(OUTCOME_TO_LABEL)

    print("[3/5] resolving per-(CID, isoform) label (majority vote over replicate SIDs)...", file=sys.stderr)
    resolved = (
        assay.groupby(["isoform", "CID"])["label"]
        .apply(resolve_labels)
        .reset_index()
    )
    n_tied = resolved["label"].isna().sum()
    print(f"  dropped {n_tied} exact-tie (CID, isoform) pairs", file=sys.stderr)
    resolved = resolved.dropna(subset=["label"])
    resolved["label"] = resolved["label"].astype(int)

    print("[4/5] fetching SMILES for all unique CIDs...", file=sys.stderr)
    all_cids = sorted(resolved["CID"].unique().tolist())
    cid_smiles = fetch_smiles(all_cids, cache_dir)

    resolved["smiles"] = resolved["CID"].map(cid_smiles)
    n_missing_smiles = resolved["smiles"].isna().sum()
    print(f"  {n_missing_smiles} rows with no SMILES resolved, dropping", file=sys.stderr)
    resolved = resolved.dropna(subset=["smiles"])

    print("[5/5] validating SMILES with RDKit and writing per-isoform CSVs...", file=sys.stderr)
    valid_cache: dict[str, str | None] = {}

    def canon(smi: str) -> str | None:
        if smi not in valid_cache:
            mol = Chem.MolFromSmiles(smi)
            valid_cache[smi] = Chem.MolToSmiles(mol) if mol is not None else None
        return valid_cache[smi]

    for isoform in sorted(GENEID_TO_ISOFORM.values()):
        sub = resolved[resolved["isoform"] == isoform].copy()
        sub["canonical_smiles"] = sub["smiles"].apply(canon)
        n_bad = sub["canonical_smiles"].isna().sum()
        sub = sub.dropna(subset=["canonical_smiles"])

        out = sub[["canonical_smiles", "label"]].rename(columns={"canonical_smiles": "smiles"})
        out = out.drop_duplicates(subset="smiles")  # a handful of CIDs canonicalize to the same graph

        dest = output_dir / f"cyp_{isoform.lower()}.csv"
        out.to_csv(dest, index=False)
        n_pos = int(out["label"].sum())
        print(
            f"  {isoform}: {len(out)} molecules ({n_pos} inhibitor / {len(out) - n_pos} non-inhibitor), "
            f"{n_bad} dropped (bad SMILES) -> {dest}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
