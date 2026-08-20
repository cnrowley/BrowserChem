#!/usr/bin/env python3
"""
download_ani1x_dataset.py

Downloads the ANI-1x / ANI-1ccx dataset (Smith et al. 2018 J. Chem. Phys.
148, 241733; Smith et al. 2019 Nat. Commun. 10, 2903) via its real,
confirmed public release on Figshare
(https://springernature.figshare.com/collections/The_ANI-1ccx_and_ANI-1x_data_sets_coupled-cluster_and_density_functional_theory_properties_for_molecules/4712477),
linked directly from aiqm/ANI1x_datasets (the Roitberg group's own
reader-code repo for this data).

--- IMPORTANT: this is NOT the full ANI-2x training dataset ---

model/ani2x-v1's registry entry cites Devereux et al. 2020, J. Chem.
Theory Comput. 16, 4192-4202 -- the ANI-2x paper, which extends ANI-1x
(H/C/N/O only) with additional S/F/Cl-containing molecules and a larger
sampled conformational space. This script searched aiqm's own GitHub
org (torchani, ANI1x_datasets, Scripts_for_ani2x_jtct_paper,
ani-model-zoo) and found no confirmed public download link for that
S/F/Cl-extended ANI-2x addendum specifically -- Scripts_for_ani2x_jtct_paper
only contains result-reproduction notebooks (torsion scans, MD
trajectories), not the raw training data. Per this project's own rule
against fabricating URLs, no such link is guessed or invented here.

What this script DOES give you: the real, confirmed, citable H/C/N/O
predecessor dataset ANI-2x was built on top of -- useful groundwork
(same active-learning-sampled QM9-superset design, DFT single-point
energies at wB97X/6-31G*) even though it's not the S/F/Cl-extended
superset the actual ani2x-v1 checkpoint (a REUSED pretrained torchani
release, not trained by this project -- see convert_ani2x_checkpoint.py)
was trained on.

--- Format ---

A single HDF5 file (~150MB) with a Python reader documented in
aiqm/ANI1x_datasets. Needs manual download from the Figshare collection
page (Figshare's article-level API requires resolving the specific
article id within the collection, which isn't stable/scriptable the
same way a single-article direct-file download is elsewhere in this
project) -- this script prints the real collection URL and instructions
rather than guessing an article/file id within it.

Usage:
    python3 download_ani1x_dataset.py
"""

import sys

COLLECTION_URL = (
    "https://springernature.figshare.com/collections/"
    "The_ANI-1ccx_and_ANI-1x_data_sets_coupled-cluster_and_density_functional_theory_properties_for_molecules/4712477"
)
READER_REPO = "https://github.com/aiqm/ANI1x_datasets"


def main():
    print(f"ANI-1x / ANI-1ccx dataset (real predecessor to ANI-2x, NOT the full ANI-2x S/F/Cl-extended set -- see this script's module docstring)", file=sys.stderr)
    print(f"\n1. Open the Figshare collection page and download the HDF5 data file(s) from it:", file=sys.stderr)
    print(f"   {COLLECTION_URL}", file=sys.stderr)
    print(f"\n2. Get the Python reader code + usage examples from:", file=sys.stderr)
    print(f"   {READER_REPO}", file=sys.stderr)
    print(f"\nThis has to be done manually (Figshare collection article ids aren't stable enough to hardcode a", file=sys.stderr)
    print(f"direct-download URL here without risking pointing at the wrong/stale article) -- see the docstring.", file=sys.stderr)


if __name__ == "__main__":
    main()
