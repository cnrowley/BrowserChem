#!/usr/bin/env python3
"""
download_nmr_datasets.py

Idempotent downloader for the two real, open NMR chemical shift datasets
this project's NMR-shift training pipeline uses:

  - NMRShiftDB2's `nmrshiftdb2withsignals.sd` (~158 MB) -- the
    peer-reviewed open NMR database (nmrshiftdb.nmr.uni-koeln.de),
    source for 13C/1H/15N training data. Each SD record's `Spectrum 13C
    N` / `Spectrum 1H N` / `Spectrum 15N N` properties carry per-atom
    shift assignments tied to that record's own molblock atom numbering
    -- see scripts/prepare_nmr_training_data.py for the parser.

  - NMRexp (Zenodo, DOI 10.5281/zenodo.17296666) -- a much larger 2025
    literature-mined database (3.37M records, six nuclei), source for
    19F training data (NMRShiftDB2's own 19F coverage is negligible).
    Also downloads the small human-verified "checked" CSVs (per-nucleus
    QC subsets) alongside the main parquet file.

Both files are large and NOT bundled in this repo -- this script fetches
them into --output-dir (default: data/nmr/, gitignored) and skips
re-downloading a file that's already present with a byte-for-byte size
match against what the source currently reports, so it's safe to re-run.

Usage:
    python3 download_nmr_datasets.py [--output-dir data/nmr]
"""

import argparse
import subprocess
import sys
from pathlib import Path

NMRSHIFTDB2_URL = "https://sourceforge.net/projects/nmrshiftdb2/files/data/nmrshiftdb2withsignals.sd/download"

ZENODO_RECORD_API = "https://zenodo.org/api/records/17296666"
NMREXP_FILES = [
    "NMRexp_10to24_1_1004.parquet",
    "F_50_checked.csv",
    "hetero_200_checked.csv",
    "test_300_checked.csv",
]


def _download(url, dest, expected_size=None):
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"  already have {dest} ({dest.stat().st_size / 1e6:.1f} MB), skipping", file=sys.stderr)
        return
    print(f"  downloading {url} -> {dest}", file=sys.stderr)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # SourceForge/Zenodo both redirect through mirror-selection hops that
    # urllib's default redirect handling doesn't always follow cleanly --
    # curl -L (already confirmed working against both hosts) is more robust.
    # NOTE: don't set a custom User-Agent here -- both SourceForge's
    # mirror-redirect chain and Zenodo's API return 403 for a spoofed
    # "Mozilla/5.0"-style UA (confirmed directly), but work fine with
    # curl's own default UA string.
    subprocess.run(
        ["curl", "-sL", "-o", str(tmp), url],
        check=True,
    )
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"download of {url} produced an empty/missing file")
    tmp.rename(dest)
    print(f"    got {dest.stat().st_size / 1e6:.1f} MB", file=sys.stderr)


def download_nmrshiftdb2(out_dir):
    print("NMRShiftDB2 (13C/1H/15N source)", file=sys.stderr)
    _download(NMRSHIFTDB2_URL, out_dir / "nmrshiftdb2withsignals.sd")


def download_nmrexp(out_dir):
    print("NMRexp (19F source)", file=sys.stderr)
    import json
    record = json.loads(subprocess.run(
        ["curl", "-s", ZENODO_RECORD_API], check=True, capture_output=True, text=True,
    ).stdout)
    files_by_name = {f["key"]: f for f in record.get("files", [])}
    for name in NMREXP_FILES:
        meta = files_by_name.get(name)
        if meta is None:
            print(f"  WARNING: {name} not found in current Zenodo record listing -- skipping", file=sys.stderr)
            continue
        url = meta["links"]["self"]
        _download(url, out_dir / name, expected_size=meta.get("size"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/nmr", help="directory to save downloaded files into (default: data/nmr)")
    parser.add_argument("--skip-nmrshiftdb2", action="store_true")
    parser.add_argument("--skip-nmrexp", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_nmrshiftdb2:
        download_nmrshiftdb2(out_dir)
    if not args.skip_nmrexp:
        download_nmrexp(out_dir)

    print(f"done -- files in {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
