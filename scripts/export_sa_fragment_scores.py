#!/usr/bin/env python3
"""
export_sa_fragment_scores.py

Converts RDKit's own Contrib/SA_Score/fpscores.pkl.gz (the fragment-ID ->
score lookup table SA_Score's "fragment score" half is built on) into a
compact binary format sascorer.js can load directly in the browser: two
parallel arrays, fragment IDs (sorted ascending, uint32) and their scores
(float32), for binary search.

Usage:
    pip install rdkit --break-system-packages
    python3 export_sa_fragment_scores.py output_dir/

Produces:
    output_dir/sa-fragment-scores.bin
    output_dir/sa-fragment-scores-manifest.json
"""

import argparse
import array
import gzip
import json
import pickle
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="directory to write the .bin + manifest.json into")
    args = parser.parse_args()

    try:
        import rdkit
    except ImportError:
        sys.exit("rdkit isn't installed -- pip install rdkit --break-system-packages")

    import os.path as op
    src = op.join(op.dirname(rdkit.__file__), "Contrib", "SA_Score", "fpscores.pkl.gz")
    if not op.exists(src):
        sys.exit(f"Couldn't find {src} -- this ships inside the rdkit pip package's "
                  "Contrib directory; if it's missing, your rdkit install may be incomplete.")

    data = pickle.load(gzip.open(src))
    table = {}
    for group in data:
        score = float(group[0])
        for frag_id in group[1:]:
            table[frag_id] = score

    ids = sorted(table.keys())
    id_arr = array.array("I", ids)                              # uint32
    score_arr = array.array("f", [table[i] for i in ids])        # float32

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "sa-fragment-scores.bin"
    manifest_path = out_dir / "sa-fragment-scores-manifest.json"

    with open(bin_path, "wb") as f:
        f.write(id_arr.tobytes())
        f.write(score_arr.tobytes())

    manifest = {
        "count": len(ids),
        "idBytes": len(id_arr) * 4,
        "scoreBytes": len(score_arr) * 4,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"wrote {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")
    print(f"wrote {manifest_path}")
    print(f"{len(ids)} fragment scores")


if __name__ == "__main__":
    main()
