#!/usr/bin/env python3
"""
convert_nagl_checkpoint.py

Converts a jthorton/nagl-mbis Lightning checkpoint (e.g. nagl-v1-mbis.ckpt)
into a manifest.json + weights.bin pair nagl-model.js can load directly in
the browser -- no server, no PyTorch, no DGL.

Only supports the exact architecture this project's JS actually
implements, verified against a real checkpoint's hyper_parameters:
  - convolution: SAGEConv ('mean' aggregator -- DGL's default and the
                 only one nagl-model.js's forward pass implements)
  - readout:     a single-task ('mbis-charges') per-atom FFN with
                 ReLU hidden activations and an Identity final activation
  - postprocess: "charges" (electronegativity-equalization) -- this is
                 currently the ONLY postprocess type nagl-model.js knows
                 how to run; a checkpoint using anything else (e.g. a
                 plain regression readout with no postprocess) will need
                 a new branch added there first.

If your checkpoint's atom_features differ from
[element, connectivity, ringofsize] (the set nagl-v1-mbis actually uses),
this script does NOT re-derive a new JS featurizer for you -- it embeds
the feature config it finds into the manifest for the record, but
nagl-features.js's buildAtomFeatures() is hardcoded to the nagl-v1-mbis
feature set. Verify the printed atom_features against nagl-features.js's
header comment before trusting predictions from a different checkpoint.

Usage:
    pip install torch --break-system-packages
    python3 convert_nagl_checkpoint.py nagl-v1-mbis.ckpt output_dir/
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to a nagl-mbis .ckpt file")
    parser.add_argument("output_dir", help="directory to write manifest.json + weights.bin into")
    parser.add_argument("--name", default="nagl-mbis-charges", help="base filename")
    args = parser.parse_args()

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt or "hyper_parameters" not in ckpt:
        sys.exit("This doesn't look like a Lightning checkpoint (missing state_dict/hyper_parameters).")

    sd = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]
    model_cfg = hp["config"]["model"]

    conv_cfg = model_cfg["convolution"]
    if conv_cfg["type"] != "SAGEConv":
        sys.exit(f"Unsupported convolution type {conv_cfg['type']!r} -- only SAGEConv is implemented in nagl-model.js.")

    readouts = model_cfg["readouts"]
    if len(readouts) != 1:
        sys.exit(f"Unsupported: {len(readouts)} readout heads found ({list(readouts.keys())}) -- "
                  "only a single-readout checkpoint is implemented.")
    task_name = list(readouts.keys())[0]
    readout_cfg = readouts[task_name]
    if readout_cfg["pooling"] != "atom":
        sys.exit(f"Unsupported pooling {readout_cfg['pooling']!r} -- only per-atom pooling ('atom') is implemented "
                  "(a molecule-level NAGL-MBIS readout would need a new branch added to nagl-model.js).")
    if readout_cfg.get("postprocess") != "charges":
        sys.exit(f"Unsupported postprocess {readout_cfg.get('postprocess')!r} -- only 'charges' "
                  "(electronegativity-equalization) is implemented in nagl-model.js.")

    print("atom_features (verify against nagl-features.js's header):")
    for f in model_cfg["atom_features"]:
        print(" ", f)
    if model_cfg.get("bond_features"):
        print("*** WARNING: this checkpoint has bond_features set -- nagl-features.js only implements "
              "the no-bond-features case (what nagl-v1-mbis itself uses). Predictions will be wrong "
              "if this checkpoint's convolution actually consumes bond features. ***")

    def arr(name):
        return sd[name].detach().numpy().astype("float32")

    tensors = {}
    manifest_conv_layers = []
    hidden_feats = conv_cfg["hidden_feats"]
    for i in range(len(hidden_feats)):
        prefix = f"convolution_module.{i}."
        tensors[f"conv{i}_self_weight"] = arr(prefix + "fc_self.weight")
        tensors[f"conv{i}_self_bias"] = arr(prefix + "fc_self.bias")
        tensors[f"conv{i}_neigh_weight"] = arr(prefix + "fc_neigh.weight")
        manifest_conv_layers.append({"hiddenDim": hidden_feats[i]})

    manifest_readout_layers = []
    forward_hidden = readout_cfg["forward"]["hidden_feats"]
    forward_activation = readout_cfg["forward"]["activation"]
    for i in range(len(forward_hidden)):
        idx = i * 3  # nn.Sequential slot pattern confirmed against a real checkpoint: [Linear, Activation, Dropout] per layer
        prefix = f"readout_modules.{task_name}.forward_layers.{idx}."
        tensors[f"readout{i}_weight"] = arr(prefix + "weight")
        tensors[f"readout{i}_bias"] = arr(prefix + "bias")
        manifest_readout_layers.append({"activation": forward_activation[i].lower()})

    offset = 0
    manifest_tensors = {}
    blob = bytearray()
    for name, a in tensors.items():
        flat = a.reshape(-1)
        manifest_tensors[name] = {"shape": list(a.shape), "offset": offset, "length": int(flat.size)}
        blob += flat.tobytes()
        offset += flat.size

    manifest = {
        "task": task_name,
        "architecture": "nagl-mbis-sageconv",
        "atomFeatureDim": tensors["conv0_self_weight"].shape[1],
        "convHiddenDim": hidden_feats[-1],
        "convLayers": manifest_conv_layers,
        "readoutLayers": manifest_readout_layers,
        "postprocess": "charges",
        "atomFeaturesConfig": model_cfg["atom_features"],  # informational, not read by nagl-model.js
        "tensors": manifest_tensors,
        "byteLength": len(blob),
        "dtype": "float32",
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{args.name}.bin"
    manifest_path = out_dir / f"{args.name}-manifest.json"
    bin_path.write_bytes(bytes(blob))
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {bin_path} ({len(blob) / 1024:.1f} KB)")
    print(f"wrote {manifest_path}")
    print(f"task: {task_name!r}, {len(hidden_feats)} conv layers, {len(forward_hidden)} readout layers")


if __name__ == "__main__":
    main()
