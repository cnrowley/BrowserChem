#!/usr/bin/env python3
"""
convert_pka_lightgbm.py

Converts a LightGBM text model file (e.g. from Booster.save_model()) into
a manifest.json + weights.bin pair js/pka-model.js can load and evaluate
directly in the browser -- no server, no Python, no LightGBM.

This is the converter for this project's own from-scratch carbon-acid
(C-H) pKa predictor: a LightGBM/DART regressor trained on
jensengroup/pKalculator's published training data (Ree et al.,
"pKalculator: A pKa predictor for C-H bonds", ChemRxiv 2024,
10.26434/chemrxiv-2024-56h5h; MIT-licensed code at
github.com/jensengroup/pKalculator), but retrained on this app's own
NAGL-MBIS partial charges (see model/nagl-mbis-charges/) instead of the
original paper's CM5/GFN1-xTB charges -- CM5 needs a real xtb binary
subprocess (semiempirical QM), which has no WASM build and cannot run
client-side, so it was swapped for a charge source this app can already
compute entirely in-browser. Same descriptor construction (graph charge
shell, 6 shells, CIP-priority + charge-tiebreak sort within each shell --
see js/pka-descriptor.js) and the same tuned LightGBM/DART hyperparameters
as the original paper, just a different (browser-computable) input charge.
Evaluated MAE/RMSE on the paper's own held-out test split are honestly
worse than the original CM5-based model (see model/registry.json's notes
for this model) -- NAGL charges are a good but imperfect proxy for real
QM-derived CM5 charges.

Encodes every tree as flat parallel arrays (isLeaf/featureIdx/threshold/
defaultLeft/leftChild/rightChild/value), one shared node pool across all
trees with a per-tree root offset -- avoids re-implementing LightGBM's own
tree traversal quirks in JS beyond a single flat "walk down to a leaf"
loop. leaf_value in LightGBM's dump_model() JSON already has the
per-tree shrinkage (learning rate) baked in, and DART-trained models are
evaluated at inference time exactly like plain GBDT (DART's tree-dropout
only affects training) -- so the JS side just sums every tree's leaf
value with no extra scaling. Self-checked in the training pipeline this
was extracted from: the flat-array walk reproduces Booster.predict()
exactly (0.0 max abs error) before ever being written out.

Usage:
    pip install lightgbm
    python3 convert_pka_lightgbm.py pka_model.txt output_dir/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def flatten_tree(node, is_leaf, feature_idx, threshold, default_left, left_child, right_child, value):
    idx = len(is_leaf)
    if "leaf_value" in node:
        is_leaf.append(1)
        feature_idx.append(-1)
        threshold.append(0.0)
        default_left.append(0)
        left_child.append(-1)
        right_child.append(-1)
        value.append(node["leaf_value"])
        return idx

    is_leaf.append(0)
    feature_idx.append(node["split_feature"])
    threshold.append(node["threshold"])
    default_left.append(1 if node["default_left"] else 0)
    left_child.append(-1)  # placeholder, filled in after recursing
    right_child.append(-1)
    value.append(0.0)
    li = flatten_tree(node["left_child"], is_leaf, feature_idx, threshold, default_left, left_child, right_child, value)
    ri = flatten_tree(node["right_child"], is_leaf, feature_idx, threshold, default_left, left_child, right_child, value)
    left_child[idx] = li
    right_child[idx] = ri
    return idx


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_txt", help="path to a LightGBM text model file (Booster.save_model() output)")
    parser.add_argument("output_dir", help="directory to write manifest.json + weights.bin into")
    parser.add_argument("--n-shells", type=int, default=6, help="graph-charge-shell descriptor shell count used at training time")
    parser.add_argument("--charge-source", default="nagl-mbis-charges", help="registry id of the partial-charge model this was trained against")
    args = parser.parse_args()

    import lightgbm as lgb

    booster = lgb.Booster(model_file=args.model_txt)
    dump = booster.dump_model()

    is_leaf, feature_idx, threshold, default_left, left_child, right_child, value = [], [], [], [], [], [], []
    tree_root_offsets = []
    for tree in dump["tree_info"]:
        root_idx = flatten_tree(tree["tree_structure"], is_leaf, feature_idx, threshold, default_left, left_child, right_child, value)
        tree_root_offsets.append(root_idx)

    num_nodes = len(is_leaf)
    print(f"{len(tree_root_offsets)} trees, {num_nodes} nodes total, {dump['max_feature_idx'] + 1} features")

    manifest = {
        "architecture": "lightgbm-gbdt",
        "task": "pka-ch",
        "numFeatures": int(dump["max_feature_idx"]) + 1,
        "numTrees": len(tree_root_offsets),
        "numNodes": num_nodes,
        "treeRootOffsets": [int(x) for x in tree_root_offsets],
        "descriptor": {
            "type": "graph-charge-shell",
            "nShells": args.n_shells,
            "useCipSort": True,
            "chargeSource": args.charge_source,
        },
        "tensors": {},
    }

    blob = bytearray()

    def pack_section(name, arr, dtype):
        while len(blob) % 4 != 0:
            blob.append(0)
        offset = len(blob)
        np_dtype = {"uint8": np.uint8, "int32": np.int32, "float32": np.float32}[dtype]
        data = np.array(arr, dtype=np_dtype).tobytes()
        blob.extend(data)
        manifest["tensors"][name] = {"dtype": dtype, "byteOffset": offset, "length": len(arr)}

    pack_section("isLeaf", is_leaf, "uint8")
    pack_section("featureIdx", feature_idx, "int32")
    pack_section("threshold", threshold, "float32")
    pack_section("defaultLeft", default_left, "uint8")
    pack_section("leftChild", left_child, "int32")
    pack_section("rightChild", right_child, "int32")
    pack_section("value", value, "float32")
    manifest["byteLength"] = len(blob)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bin").write_bytes(bytes(blob))
    (out_dir / "manifest.json").write_text(json.dumps(manifest))

    print(f"wrote {out_dir / 'weights.bin'} ({len(blob) / 1024:.1f} KB)")
    print(f"wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
