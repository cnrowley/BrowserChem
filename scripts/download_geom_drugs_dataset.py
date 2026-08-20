#!/usr/bin/env python3
"""
download_geom_drugs_dataset.py

Downloads GEOM (Axelrod & Gomez-Bombarelli 2022, Sci Data --
"GEOM, energy-annotated molecular conformations for property prediction
and molecular generation"), the dataset model/registry.json's
geomol-drugs-v1 entry names as its training source, via the exact
command GeoMol's own README documents
(github.com/PattanaikL/GeoMol -- "Download and extract the GEOM dataset
from the original source"):

    wget https://dataverse.harvard.edu/api/access/datafile/4327252
    tar -xvf 4327252

--- Why this script does NOT just run that download for you ---

Checked directly against the live Harvard Dataverse API
(api.datasets/:persistentId=doi:10.7910/DVN/JNGTDF) before writing this:
file id 4327252 (`rdkit_folder.tar.gz`, what GeoMol's README uses) is
**50.1 GB**. That's not a "fetch it into data/, gitignored" situation
like this project's other download scripts (NMR's ~158MB SD file, this
same dataset's own much smaller drugs_featurized.msgpack.tar.gz at
10.1GB is still large) -- it's genuinely impractical to fetch inside an
automated/sandboxed session, and this project's convention (see
CLAUDE.md, and scripts/qca_step1_probe.py's own header) is to document
the real command for the user to run on their own machine rather than
pretend to run something that size here.

model/geomol-drugs-v1 is a REUSED pretrained checkpoint (converted via
convert_geomol_checkpoint.py from GeoMol's own trained_models/drugs/
release), not something this project trains from scratch -- so this
script's purpose is reference/reproducibility tooling (letting someone
retrain GeoMol from scratch on the real source data GeoMol's own README
points to), not a step needed to use the currently-shipped checkpoint.

--- Available files (fetched live from Dataverse, sizes in bytes) ---

  rdkit_folder.tar.gz         50,137,579,520  (both QM9 + Drugs, what GeoMol's README uses)
  drugs_crude.msgpack.tar.gz  42,687,717,910  (Drugs only, un-featurized)
  drugs_featurized.msgpack.tar.gz 10,138,004,284  (Drugs only, featurized -- smallest Drugs-only option)
  qm9_crude.msgpack.tar.gz     1,025,881,930
  qm9_featurized.msgpack.tar.gz  131,716,778

Usage:
    python3 download_geom_drugs_dataset.py --which rdkit_folder --output-dir data/geom
        (prints the real wget/tar commands; use --run to actually execute them --
        only do this on a machine with >50GB free disk and a fast connection)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# (filename, Dataverse file id, size in bytes) -- ids/sizes confirmed live
# against https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/JNGTDF
FILES = {
    "rdkit_folder": ("rdkit_folder.tar.gz", 4327252, 50_137_579_520),
    "drugs_crude": ("drugs_crude.msgpack.tar.gz", 4360331, 42_687_717_910),
    "drugs_featurized": ("drugs_featurized.msgpack.tar.gz", 4327295, 10_138_004_284),
    "qm9_crude": ("qm9_crude.msgpack.tar.gz", 4327190, 1_025_881_930),
    "qm9_featurized": ("qm9_featurized.msgpack.tar.gz", 4327191, 131_716_778),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--which", choices=list(FILES), default="rdkit_folder",
                         help="which release to fetch (default: rdkit_folder, matching GeoMol's own README exactly)")
    parser.add_argument("--output-dir", default="data/geom")
    parser.add_argument("--run", action="store_true",
                         help="actually download+extract (default: just print the commands). "
                              "Only pass this on a machine with enough disk -- see the size table above.")
    args = parser.parse_args()

    filename, file_id, size_bytes = FILES[args.which]
    out_dir = Path(args.output_dir)
    url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"

    print(f"{args.which}: {filename} ({size_bytes / 1e9:.1f} GB)", file=sys.stderr)
    print(f"  download: wget -O {out_dir / filename} {url}", file=sys.stderr)
    print(f"  extract:  tar -xzf {out_dir / filename} -C {out_dir}", file=sys.stderr)

    if not args.run:
        print("\n(pass --run to actually execute this -- not done by default given the file size above)", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / filename
    print(f"\ndownloading to {dest} ...", file=sys.stderr)
    subprocess.run(["wget", "-O", str(dest), url], check=True)
    print(f"extracting into {out_dir} ...", file=sys.stderr)
    subprocess.run(["tar", "-xzf", str(dest), "-C", str(out_dir)], check=True)
    print("done.", file=sys.stderr)


if __name__ == "__main__":
    main()
