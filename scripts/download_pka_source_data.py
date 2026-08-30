#!/usr/bin/env python3
"""
download_pka_source_data.py

Idempotent downloader for the four real, public data sources
`model/registry.json`'s `pka-microstate-freeenergy` entry documents as
its training data (see that entry's own `dataset.name` field for the
full provenance/weighting story) -- fetches everything
scripts/prepare_pka_microstate_training_data.py,
scripts/prepare_pkahub_data.py, and scripts/prepare_pka_qm_pretrain_data.py
need as raw input, so a fresh clone of this repo can rebuild the training
CSVs from scratch instead of relying on data/pka/ (gitignored, and was
populated via ad hoc manual/curl fetches during development, not a
committed script -- this script exists to close that gap for anyone
rebuilding this model's data pipeline from a public checkout).

Sources (all free, no login required):

  1. IUPAC digitized pKa dataset -- Zenodo concept DOI
     10.5281/zenodo.7236452 (CC BY-NC 4.0). Resolved via the Zenodo API
     so this always fetches the LATEST version under that concept DOI
     (confirmed the version/file layout already changed once during this
     project's own development: an earlier version served
     `iupac_high-confidence_v2_3.csv` as a bare top-level file, the
     current v2.3e serves a zipped GitHub-style source snapshot instead
     -- this script handles either shape by searching the download for
     `iupac_high-confidence_v2_3.csv` wherever it ends up, rather than
     hard-coding one path).
  2. Baltruschat & Czodrowski 2020 (DataWarrior + ChEMBL25) --
     `combined_training_datasets_unique.sdf`, fetched directly from
     czodrowskilab/Machine-learning-meets-pKa on GitHub (the exact file
     scripts/prepare_pka_training_data.py already expects at
     data/pka/combined_training_datasets_unique.sdf, and this project's
     own existing `aqueous-pka` checkpoint was trained from).
  3. Nevolianis et al. 2025 JACS ("Solvation Free Energies of Anions") --
     Zenodo 10.5281/zenodo.15604045 (CC BY 4.0), `data.zip` only (the
     ~650MB of trained-model .zip files on that same record are NOT
     fetched -- this project only ever used the raw data tables inside
     data.zip, see scripts/prepare_pka_qm_pretrain_data.py's own header).
  4. pKaHub (github.com/keserulab/pkahub, Sipos-Szabo/Bajusz/Balogh/
     Keseru, J. Chem. Inf. Model. 2026) -- its two raw data tables under
     build/datafiles/data/ that scripts/prepare_pkahub_data.py reads
     directly (checked into that repo, not a build artifact you need to
     regenerate): exp_macro_pka_datapoints/combined_unified_dataset.tsv
     and microspecies/microspecies_table.tsv.

Skips re-downloading a file that's already present with a byte-for-byte
size match against what the source currently reports (same convention as
scripts/download_nmr_datasets.py), so it's safe to re-run.

Usage:
    python3 scripts/download_pka_source_data.py [--output-dir data/pka]
    python3 scripts/download_pka_source_data.py --only iupac,pkahub

Then, from --output-dir (default data/pka/):
    python3 scripts/prepare_pka_training_data.py \\
        --input data/pka/combined_training_datasets_unique.sdf
    python3 scripts/prepare_pka_microstate_training_data.py \\
        --iupac-csv data/pka/iupac_high-confidence_v2_3.csv \\
        --baltruschat-csv data/pka/pka_prepared.csv ...
    python3 scripts/prepare_pkahub_data.py \\
        --pkahub-dir data/pka/pkahub_datafiles ...
    python3 scripts/prepare_pka_qm_pretrain_data.py \\
        --nevolianis-dir data/pka/nevolianis_data/data ...
(see each script's own --help for its remaining required arguments --
this script only reproduces the raw inputs, not the prepared CSVs.)
"""

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

IUPAC_ZENODO_CONCEPT_ID = "7236452"
NEVOLIANIS_ZENODO_RECORD_ID = "15604045"
BALTRUSCHAT_URL = (
    "https://raw.githubusercontent.com/czodrowskilab/Machine-learning-meets-pKa/"
    "master/datasets/combined_training_datasets_unique.sdf"
)
PKAHUB_FILES = {
    "exp_macro_pka_datapoints/combined_unified_dataset.tsv":
        "https://raw.githubusercontent.com/keserulab/pkahub/main/build/datafiles/data/"
        "exp_macro_pka_datapoints/combined_unified_dataset.tsv",
    "microspecies/microspecies_table.tsv":
        "https://raw.githubusercontent.com/keserulab/pkahub/main/build/datafiles/data/"
        "microspecies/microspecies_table.tsv",
}


def _http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "BrowserChem-pka-data-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download_url(url, dest, expected_size=None):
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"  already have {dest} ({dest.stat().st_size / 1e6:.1f} MB), skipping", file=sys.stderr)
        return
    print(f"  downloading {url} -> {dest}", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = _http_get(url)
    dest.write_bytes(data)
    print(f"    wrote {len(data) / 1e6:.1f} MB", file=sys.stderr)


def fetch_iupac(output_dir):
    """Resolves the Zenodo concept DOI to its latest version, then finds
    iupac_high-confidence_v2_3.csv wherever that version's file(s) put it
    (bare CSV in older versions, inside a zipped repo snapshot as of
    v2.3e -- see this script's own header)."""
    print("=== IUPAC digitized pKa dataset ===", file=sys.stderr)
    dest_csv = output_dir / "iupac_high-confidence_v2_3.csv"
    if dest_csv.exists() and dest_csv.stat().st_size > 0:
        print(f"  already have {dest_csv}, skipping", file=sys.stderr)
        return
    meta = json.loads(_http_get(f"https://zenodo.org/api/records/{IUPAC_ZENODO_CONCEPT_ID}"))
    files = meta.get("files", [])
    if not files:
        sys.exit(f"Zenodo record {IUPAC_ZENODO_CONCEPT_ID} returned no files -- check https://zenodo.org/records/{IUPAC_ZENODO_CONCEPT_ID} by hand")
    for f in files:
        if f["key"].endswith("iupac_high-confidence_v2_3.csv"):
            _download_url(f["links"]["self"], dest_csv, expected_size=f["size"])
            return
    # Not a bare file in this version -- look inside any .zip for it.
    for f in files:
        if f["key"].endswith(".zip"):
            print(f"  {f['key']} doesn't match by name -- checking inside the zip", file=sys.stderr)
            data = _http_get(f["links"]["self"])
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    if name.endswith("iupac_high-confidence_v2_3.csv"):
                        dest_csv.write_bytes(zf.read(name))
                        print(f"  extracted {name} -> {dest_csv} ({dest_csv.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
                        return
    sys.exit(
        f"Could not find iupac_high-confidence_v2_3.csv in Zenodo record {IUPAC_ZENODO_CONCEPT_ID}'s "
        f"current files ({[f['key'] for f in files]}) -- the dataset's file layout has apparently "
        f"changed again; check https://zenodo.org/records/{IUPAC_ZENODO_CONCEPT_ID} by hand and update this script."
    )


def fetch_baltruschat(output_dir):
    print("=== Baltruschat & Czodrowski 2020 ===", file=sys.stderr)
    _download_url(BALTRUSCHAT_URL, output_dir / "combined_training_datasets_unique.sdf")


def fetch_nevolianis(output_dir):
    print("=== Nevolianis et al. 2025 (JACS) ===", file=sys.stderr)
    dest_dir = output_dir / "nevolianis_data"
    marker = dest_dir / "data" / "data_splits"
    if marker.exists():
        print(f"  already have {dest_dir}, skipping", file=sys.stderr)
        return
    meta = json.loads(_http_get(f"https://zenodo.org/api/records/{NEVOLIANIS_ZENODO_RECORD_ID}"))
    data_file = next((f for f in meta.get("files", []) if f["key"] == "data.zip"), None)
    if not data_file:
        sys.exit(f"Zenodo record {NEVOLIANIS_ZENODO_RECORD_ID} has no data.zip -- check https://zenodo.org/records/{NEVOLIANIS_ZENODO_RECORD_ID} by hand")
    print(f"  downloading {data_file['links']['self']} ({data_file['size'] / 1e6:.1f} MB)", file=sys.stderr)
    data = _http_get(data_file["links"]["self"], timeout=180)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)
    print(f"  extracted into {dest_dir}", file=sys.stderr)


def fetch_pkahub(output_dir):
    print("=== pKaHub ===", file=sys.stderr)
    dest_dir = output_dir / "pkahub_datafiles"
    for rel_path, url in PKAHUB_FILES.items():
        _download_url(url, dest_dir / rel_path)


FETCHERS = {
    "iupac": fetch_iupac,
    "baltruschat": fetch_baltruschat,
    "nevolianis": fetch_nevolianis,
    "pkahub": fetch_pkahub,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="data/pka")
    ap.add_argument("--only", default=None, help="comma-separated subset of: " + ",".join(FETCHERS))
    args = ap.parse_args()

    which = list(FETCHERS) if not args.only else [s.strip() for s in args.only.split(",")]
    unknown = [s for s in which if s not in FETCHERS]
    if unknown:
        sys.exit(f"unknown --only source(s): {unknown} -- choose from {list(FETCHERS)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in which:
        FETCHERS[name](output_dir)
    print("done.", file=sys.stderr)


if __name__ == "__main__":
    main()
