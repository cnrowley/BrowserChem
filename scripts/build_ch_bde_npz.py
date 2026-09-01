#!/usr/bin/env python3
"""
build_ch_bde_npz.py

Consumes scripts/pka-physical-baseline-harness/compute_ch_bde_atom_features.js's
NDJSON output (one {"ok", "features"|"error"} per row, same order as the
input training CSV) and builds:
  - a filtered training CSV (rows where atom-feature computation failed
    are dropped, counted -- never silently kept with placeholder atoms)
  - atom_features_0.npz: arr_0..arr_{n-1}, one [n_atoms x 2] float32
    array per retained row, in the SAME order -- chemprop's own
    load_input_feats_and_descs expects exactly this (verified against
    chemprop's real source, not just the --help text, before writing
    this script).

Usage:
    python3 build_ch_bde_npz.py <training.csv> <features.ndjson> <out_csv> <out_npz>
"""
import argparse
import csv
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_csv")
    parser.add_argument("features_ndjson")
    parser.add_argument("out_csv")
    parser.add_argument("out_npz")
    args = parser.parse_args()

    with open(args.training_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    with open(args.features_ndjson) as f:
        feature_lines = [json.loads(line) for line in f]

    if len(rows) != len(feature_lines):
        raise SystemExit(f"row count mismatch: {len(rows)} csv rows vs {len(feature_lines)} feature lines")

    kept_rows = []
    arrays = []
    dropped = 0
    for row, feat in zip(rows, feature_lines):
        if not feat.get("ok"):
            dropped += 1
            continue
        arr = np.array(feat["features"], dtype=np.float32)
        kept_rows.append(row)
        arrays.append(arr)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    np.savez(args.out_npz, *arrays)
    print(f"kept {len(kept_rows)}/{len(rows)} rows ({dropped} dropped) -> {args.out_csv}, {args.out_npz}")


if __name__ == "__main__":
    main()
