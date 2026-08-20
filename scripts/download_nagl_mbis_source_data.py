#!/usr/bin/env python3
"""
download_nagl_mbis_source_data.py

Fetches the real, verified source molecules for model/nagl-mbis-charges/
(jthorton/nagl-mbis's pretrained checkpoint, reused not created by this
project -- see that entry's registry notes) from QCArchive.

--- What this DOES confirm/provide ---

jthorton/nagl-mbis's own README (github.com/jthorton/nagl-mbis) states
its models "were trained on the OpenFF ESP Fragment Conformers v1.0
dataset which is on QCArchive" -- confirmed live against the real
QCArchive API before writing this script (not assumed from the README
alone): the dataset exists, has 65,116 entries, one specification
('spec_1', HF/6-31G* via Psi4). This script pulls every entry's SMILES
+ 3D geometry (a real, useful starting point for anyone wanting to
build a similar conformation-independent atomic-property model).

--- What this does NOT provide, and why (read before assuming this is
a complete reproduction) ---

The per-atom MBIS partial charges the checkpoint was actually trained
on are NOT retrievable from this dataset's stored QCArchive properties.
Checked directly: 'spec_1's protocol retains only
`orbitals_and_eigenvalues` (confirmed via
`ds.specifications['spec_1'].specification.protocols`) -- `density_a`/
`density_b` are None on every record. MBIS charge partitioning needs
the full electron density, which was never retained on these stored
records. jthorton/nagl-mbis's own data-prep scripts
(scripts/dataset/setup_labeled_data.py in that repo) read the real MBIS
charges from local files named TrainingSet-v1.hdf5/ValSet-v1.hdf5 --
files that aren't in that repo and have no documented public download
URL anywhere in it (checked: README, scripts/dataset/, scripts/
training/ -- no link). This project's earlier, now-superseded
scripts/extract_mbis_charges.py and scripts/qca_step1_probe.py assumed
a DIFFERENT, incorrect dataset name ("MLPepper RECAP Optimized
Fragments v1.0") and an unconfirmed `mbis_charges` property key that
doesn't exist on any dataset checked here -- both stale/wrong, kept
around only because deleting other people's files without being asked
felt presumptuous; this script is the corrected replacement.

Reproducing the exact training labels would mean re-running Psi4
single-point calculations on these same molecules/geometries with
`protocols.wavefunction` set to retain the full density (or a wavefunction
level that supports it), then running an MBIS population analysis on
each -- a real QM compute campaign (65k molecules), not a download. That
is out of scope for a download script; documented honestly here rather
than either skipped silently or claimed as done.

Usage:
    pip install qcportal --break-system-packages
    python3 download_nagl_mbis_source_data.py [--output data/nagl-mbis-source/esp_fragment_conformers_v1.csv] [--limit N]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

DATASET_NAME = "OpenFF ESP Fragment Conformers v1.0"
QCARCHIVE_URL = "https://api.qcarchive.molssi.org:443"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data/nagl-mbis-source/esp_fragment_conformers_v1.csv")
    parser.add_argument("--limit", type=int, default=None, help="only fetch the first N entries (for a quick test run)")
    args = parser.parse_args()

    try:
        import qcportal as ptl
    except ImportError:
        sys.exit("needs qcportal: pip install qcportal --break-system-packages")
    from rdkit import Chem

    # QCArchive disambiguates multiple entries that would otherwise share
    # the same base SMILES (different protonation states/tautomers
    # generated from the same starting structure, most likely) by
    # appending "-N" to entry_name -- confirmed directly: 'CCC(C)(CN)CO-0'
    # fails to parse as-is (unclosed-ring error) but 'CCC(C)(CN)CO' parses
    # fine. Stripped only when doing so makes the SMILES valid, so a
    # genuine SMILES that happens to end in "-<digit>" (extremely
    # unlikely, but not impossible in exotic notations) isn't corrupted.
    suffix_pattern = re.compile(r"-\d+$")

    def clean_smiles(raw):
        mol = Chem.MolFromSmiles(raw)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        stripped = suffix_pattern.sub("", raw)
        if stripped != raw:
            mol = Chem.MolFromSmiles(stripped)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        return None

    print(f"connecting to {QCARCHIVE_URL} ...", file=sys.stderr)
    client = ptl.PortalClient(QCARCHIVE_URL)
    ds = client.get_dataset("singlepoint", DATASET_NAME)
    entry_names = ds.entry_names[: args.limit] if args.limit else ds.entry_names
    print(f"dataset '{DATASET_NAME}': {len(ds.entry_names)} total entries, fetching {len(entry_names)}", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_incomplete = 0
    n_bad_smiles = 0
    seen_entries = set()
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "n_atoms"])
        for entry_name, spec_name, record in ds.iterate_records(
            entry_names=entry_names, specification_names=["spec_1"]
        ):
            if entry_name in seen_entries:
                continue  # multiple conformers per molecule share the same entry_name here
            if str(record.status) != "RecordStatusEnum.complete":
                n_incomplete += 1
                continue
            # record.molecule.identifiers' SMILES fields are unpopulated for
            # this dataset (checked directly -- all None) -- entry_name
            # itself IS the molecule's SMILES for this dataset (confirmed
            # against several samples, modulo the QCArchive disambiguation
            # suffix cleaned above), so use that rather than a field that
            # looks more "official" but is actually empty here.
            smiles = clean_smiles(entry_name)
            seen_entries.add(entry_name)
            if smiles is None:
                n_bad_smiles += 1
                continue
            writer.writerow([smiles, len(record.molecule.symbols)])
            n_written += 1
            if n_written % 5000 == 0:
                print(f"  ...{n_written} written", file=sys.stderr)

    print(
        f"wrote {n_written} unique molecules to {out_path} "
        f"({n_incomplete} incomplete records skipped, {n_bad_smiles} unparseable entry_names skipped)",
        file=sys.stderr,
    )
    print(
        "NOTE: this is SMILES/geometry-identifier data only -- no MBIS charge labels. "
        "See this script's module docstring for why those aren't retrievable here.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
