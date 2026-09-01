#!/usr/bin/env python3
"""
prepare_ames_training_data.py

Builds a chemprop-ready binary-classification CSV predicting Ames
mutagenicity -- the standard early-discovery genotoxicity screen (a
positive Ames test is a strong stop-signal in drug discovery; the assay
itself measures reverse mutation in Salmonella typhimurium strains
exposed to the test compound).

--- Source ---

Hansen, K.; Mika, S.; Schroeter, T.; Sutter, A.; ter Laak, A.; Steger-
Hartmann, T.; Heinrich, N.; Muller, K.-R. "Benchmark Data Set for in
Silico Prediction of Ames Mutagenicity." J. Chem. Inf. Model. 2009, 49,
2077-2081. DOI 10.1021/ci900161g -- the standard public Ames benchmark
(6512 compounds, consensus-labeled from CCRIS/GeneTox/Helma/VITIC and
other literature sources), fetched directly from the paper's own ACS
Figshare supporting-information archive (figshare article 2825590, the
ONE file listed there -- confirmed via figshare's public API, not a
third-party mirror). CC BY-NC 4.0 licensed (Figshare's own recorded
license for this article) -- noncommercial; note this alongside the
citation if this project (or a fork of it) is ever used commercially.

--- Label ---

Binary, already provided directly in the source .smi file: 0 = Ames
negative (non-mutagenic), 1 = Ames positive (mutagenic). No PubChem CID
resolution step needed here (unlike prepare_cyp_training_data.py/
prepare_herg_training_data.py) -- SMILES are already in the source file.

--- SMILES ---

Every SMILES is re-parsed and re-canonicalized with RDKit; unparseable
entries are dropped (counted, not silently ignored). Duplicate canonical
SMILES (a handful exist -- the same compound entered under different
CAS numbers/identifiers) are resolved by majority vote on Activity;
exact ties are dropped rather than guessed, counted and reported --
same convention as prepare_herg_training_data.py.

Usage:
    python3 prepare_ames_training_data.py [--output-dir data/ames] [--cache-dir scripts/data_cache/ames]

Needs: rdkit, requests.
"""
import argparse
import io
import sys
import zipfile
from collections import Counter
from pathlib import Path

import requests
from rdkit import Chem

ARTICLE_ID = 2825590  # figshare article id for ci900161g's supporting info
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/4523278"
SMI_FILENAME = "smiles_cas_N6512.smi"


def fetch_zip(cache_dir: Path) -> Path:
    dest = cache_dir / "ames_hansen.zip"
    if not dest.exists():
        print(f"  downloading {DOWNLOAD_URL} -> {dest}", file=sys.stderr)
        resp = requests.get(DOWNLOAD_URL, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    else:
        print(f"  already have {dest}, skipping download", file=sys.stderr)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/ames")
    parser.add_argument("--cache-dir", default="scripts/data_cache/ames")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = fetch_zip(cache_dir)
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read(SMI_FILENAME).decode("utf-8")

    rows = []  # (smiles, cas, label)
    n_bad_lines = 0
    for line in raw.strip().splitlines():
        parts = line.split("\t")
        parts = [p.strip() for p in parts]
        if len(parts) < 3:
            n_bad_lines += 1
            continue
        smiles, cas, label = parts[0], parts[1], parts[2]
        rows.append((smiles, cas, label))
    print(f"parsed {len(rows)} rows from {SMI_FILENAME} ({n_bad_lines} malformed lines skipped)", file=sys.stderr)

    by_canonical: dict[str, list[int]] = {}
    n_unparseable = 0
    for smiles, cas, label in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_unparseable += 1
            continue
        canonical = Chem.MolToSmiles(mol)
        by_canonical.setdefault(canonical, []).append(int(label))
    print(f"{n_unparseable} SMILES failed to parse -- excluded", file=sys.stderr)

    out_rows = []
    n_ties = 0
    for canonical, labels in by_canonical.items():
        counts = Counter(labels)
        if len(counts) > 1:
            top = counts.most_common()
            if top[0][1] == top[1][1]:
                n_ties += 1
                continue
            majority_label = top[0][0]
        else:
            majority_label = labels[0]
        out_rows.append((canonical, majority_label))
    print(f"{len(by_canonical)} unique canonical SMILES, {n_ties} exact-tie duplicates dropped -> {len(out_rows)} final rows", file=sys.stderr)

    out_path = output_dir / "ames.csv"
    with open(out_path, "w") as f:
        f.write("smiles,label\n")
        for smiles, label in out_rows:
            f.write(f"{smiles},{label}\n")

    n_pos = sum(1 for _, label in out_rows if label == 1)
    print(f"wrote {out_path}: {len(out_rows)} rows, {n_pos} positive ({100*n_pos/len(out_rows):.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
