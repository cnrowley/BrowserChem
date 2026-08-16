#!/usr/bin/env python3
"""
convert_chemprop_checkpoint.py

Converts a Chemprop v2 Lightning checkpoint (a `best.pt` from `chemprop
train`) into a manifest.json + weights.bin pair that chemprop-model.js can
load directly in the browser -- no server, no PyTorch, no ONNX.

Only supports the architecture this project's JS actually implements:
  - message passing: BondMessagePassing or MABBondMessagePassing (both are
                      bond-centered D-MPNN with identical math -- the
                      latter is Chemprop's mol/atom/bond-multitask
                      variant, used whenever --atom-target-columns or
                      --bond-target-columns is set)
  - aggregation:      NormAggregation (sum / norm) for molecule-level
                      output; skipped entirely for atom-level output
  - predictor:         single-task RegressionFFN or BinaryClassificationFFN,
                        n_layers=1 (Linear -> ReLU -> Linear), with the
                        regression head's UnscaleTransform or the
                        classification head's sigmoid applied in JS.
                        Molecule-level (predictor / mol_predictor) or
                        atom-level (atom_predictor) -- not both at once,
                        and not bond-level (bond_predictor) yet.

If your checkpoint uses a different message-passing scheme
(AtomMessagePassing), a different aggregation (mean/sum/attentive), a
deeper FFN (n_layers > 1), multiclass/Dirichlet/evidential/spectral
heads, multi-task output, or a bond-level predictor, this script will
raise -- extending chemprop-model.js's forward pass to match is required
first (see chemprop-model.js's header for the shapes it expects).

Usage:
    pip install torch chemprop lightning --break-system-packages
    python3 convert_chemprop_checkpoint.py best.pt output_dir/

Produces:
    output_dir/<task>-manifest.json
    output_dir/<task>.bin
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to a Chemprop best.pt")
    parser.add_argument("output_dir", help="directory to write manifest.json + weights.bin into")
    parser.add_argument("--name", default=None, help="base filename (default: the task name from the checkpoint)")
    parser.add_argument("--graph-type", choices=["heavy", "explicit-h"], default="heavy",
                         help="'explicit-h' if this checkpoint was trained with chemprop's --add-h flag "
                              "(every hydrogen is its own graph node -- e.g. a per-atom 1H NMR shift model). "
                              "Not auto-detected: the checkpoint doesn't reliably record whether --add-h was "
                              "used, so pass this explicitly based on how you trained it. Only meaningful for "
                              "outputLevel=atom; ignored for molecule-level checkpoints.")
    parser.add_argument("--applicable-element", default=None,
                         help="restrict this atom-level checkpoint's predictions to one element, e.g. 'C' for "
                              "a 13C shift model -- chemprop-model.js will only annotate atoms of this element "
                              "and leave others blank, the same masking pattern pka-model.js uses for its own "
                              "candidate-site gating. Only meaningful for outputLevel=atom.")
    args = parser.parse_args()

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt or "hyper_parameters" not in ckpt:
        sys.exit("This doesn't look like a Chemprop Lightning checkpoint (missing state_dict/hyper_parameters).")

    sd = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]
    out_cols_raw = ckpt.get("output_columns") or ["prediction"]

    # ---- figure out molecule-level vs atom-level, and locate the right
    # message-passing / predictor sub-dicts + state_dict key prefixes for
    # each ----
    is_mol_atom_bond = "message_passing" in hp and isinstance(hp.get("agg"), (dict, type(None))) and (
        "atom_predictor" in hp or "mol_predictor" in hp
    )

    if is_mol_atom_bond:
        # MolAtomBondMPNN: hp has message_passing / agg / mol_predictor /
        # atom_predictor / bond_predictor, any of the three predictors may
        # be None. output_columns is [mol_names, atom_names, bond_names].
        mol_names, atom_names, bond_names = (out_cols_raw + [None, None, None])[:3]
        if hp.get("bond_predictor") is not None:
            sys.exit("Bond-level predictors aren't implemented yet -- only mol-level and atom-level are.")
        if hp.get("atom_predictor") is not None and hp.get("mol_predictor") is not None:
            sys.exit("A checkpoint with both a mol-level and an atom-level predictor isn't supported yet "
                      "-- chemprop-model.js's manifest format assumes exactly one output level per model.")
        if hp.get("atom_predictor") is not None:
            output_level = "atom"
            pred_hp = hp["atom_predictor"]
            task_names = atom_names
            mp_weight_key = "W_vo"  # MABBondMessagePassing's vertex-output layer
            pred_prefix = "atom_predictor"
        elif hp.get("mol_predictor") is not None:
            output_level = "molecule"
            pred_hp = hp["mol_predictor"]
            task_names = mol_names
            mp_weight_key = "W_vo"
            pred_prefix = "mol_predictor"
            if hp.get("agg") is None:
                sys.exit("mol_predictor present but agg is None -- unexpected shape, aborting rather than guessing.")
        else:
            sys.exit("Neither mol_predictor nor atom_predictor is set -- nothing to export.")
        agg_hp = hp.get("agg")
    else:
        # Plain MPNN: hp has message_passing / agg / predictor.
        output_level = "molecule"
        pred_hp = hp["predictor"]
        task_names = out_cols_raw
        mp_weight_key = "W_o"
        pred_prefix = "predictor"
        agg_hp = hp["agg"]

    mp_hp = hp["message_passing"]
    mp_cls_name = mp_hp["cls"].__name__
    if mp_cls_name not in ("BondMessagePassing", "MABBondMessagePassing"):
        sys.exit(f"Unsupported message-passing class {mp_cls_name!r} -- "
                  "only BondMessagePassing/MABBondMessagePassing are implemented in dmpnn.js.")
    if mp_hp["bias"]:
        sys.exit("Unsupported message-passing bias=True -- expected bias=False (W_i/W_h have no bias).")
    if mp_hp["d_v"] != 72 or mp_hp["d_e"] != 14:
        sys.exit(f"This checkpoint's featurizer dims (d_v={mp_hp['d_v']}, d_e={mp_hp['d_e']}) don't match "
                  "chemprop-features.js's fixed 72-dim atom / 14-dim bond featurizer (MultiHotAtomFeaturizer.v2 "
                  "+ the default MultiHotBondFeaturizer) -- it was trained with a different/custom featurizer "
                  "this project's JS port doesn't implement. Converting anyway would silently feed the JS "
                  "forward pass wrong-shaped input rather than fail loudly, so this aborts instead.")
    if args.graph_type == "explicit-h" and (hp.get("atom_predictor") if is_mol_atom_bond else None) is None:
        sys.exit("--graph-type explicit-h only makes sense for an atom-level (--atom-target-columns) checkpoint.")
    if output_level == "molecule":
        if agg_hp is None or agg_hp["cls"].__name__ != "NormAggregation":
            sys.exit("Unsupported aggregation -- expected NormAggregation for a molecule-level model.")

    pred_cls_name = pred_hp["cls"].__name__
    if pred_cls_name == "RegressionFFN":
        task_type = "regression"
    elif pred_cls_name == "BinaryClassificationFFN":
        task_type = "classification"
    else:
        sys.exit(f"Unsupported predictor {pred_cls_name!r} -- only single-task RegressionFFN "
                  "(regression) and BinaryClassificationFFN (binary classification) are "
                  "implemented in chemprop-model.js. Multiclass, Dirichlet/evidential, and "
                  "spectral heads would need a new forward-pass branch added there first.")
    if pred_hp["n_layers"] != 1:
        sys.exit(f"Unsupported predictor n_layers={pred_hp['n_layers']} -- "
                  "only a single hidden layer (n_layers=1) is implemented.")
    if pred_hp["n_tasks"] != 1:
        sys.exit(f"Multi-task checkpoints aren't supported yet (n_tasks={pred_hp['n_tasks']}) -- "
                  "chemprop-model.js assumes a single output. This applies per output level -- "
                  "a model with exactly one atom target and one mol target each would need two "
                  "separate exports, not one, and that combination isn't supported yet either "
                  "(see the mol+atom check above).")
    if not task_names or len(task_names) != 1:
        sys.exit(f"Expected exactly one target name for this output level, got {task_names!r}.")

    def arr(name):
        return sd[name].detach().numpy().astype("float32")

    tensors = {
        "W_i": arr("message_passing.W_i.weight"),
        "W_h": arr("message_passing.W_h.weight"),
        "W_o_weight": arr(f"message_passing.{mp_weight_key}.weight"),
        "W_o_bias": arr(f"message_passing.{mp_weight_key}.bias"),
        "ffn0_weight": arr(f"{pred_prefix}.ffn.0.0.weight"),
        "ffn0_bias": arr(f"{pred_prefix}.ffn.0.0.bias"),
        "ffn1_weight": arr(f"{pred_prefix}.ffn.1.2.weight"),
        "ffn1_bias": arr(f"{pred_prefix}.ffn.1.2.bias"),
    }
    if task_type == "regression":
        # Classification's output_transform is Identity() (a plain sigmoid
        # is applied in the JS forward pass instead, see chemprop-model.js)
        # -- there's nothing to export here in that case.
        tensors["out_mean"] = arr(f"{pred_prefix}.output_transform.mean").reshape(-1)
        tensors["out_scale"] = arr(f"{pred_prefix}.output_transform.scale").reshape(-1)

    offset = 0
    manifest_tensors = {}
    blob = bytearray()
    for name, a in tensors.items():
        flat = a.reshape(-1)
        manifest_tensors[name] = {"shape": list(a.shape), "offset": offset, "length": int(flat.size)}
        blob += flat.tobytes()
        offset += flat.size

    ffn_hidden_dim = pred_hp["hidden_dim"]
    if isinstance(ffn_hidden_dim, (list, tuple)):
        ffn_hidden_dim = ffn_hidden_dim[0]

    manifest = {
        "task": task_names[0],
        "taskType": task_type,
        "outputLevel": output_level,  # "molecule" (default/omitted in older manifests) or "atom"
        "graphType": args.graph_type if output_level == "atom" else "heavy",
        "applicableElement": args.applicable_element if output_level == "atom" else None,
        "architecture": "chemprop-dmpnn-v2",
        "dims": {
            "d_v": mp_hp["d_v"],
            "d_e": mp_hp["d_e"],
            "d_h": mp_hp["d_h"],
            "depth": mp_hp["depth"],
            "aggNorm": (agg_hp["norm"] if agg_hp else None),
        },
        "ffnHiddenDim": ffn_hidden_dim,
        "nTasks": 1,
        "tensors": manifest_tensors,
        "byteLength": len(blob),
        "dtype": "float32",
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.name or task_names[0]
    bin_path = out_dir / f"{base}.bin"
    manifest_path = out_dir / f"{base}-manifest.json"

    bin_path.write_bytes(bytes(blob))
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"wrote {bin_path} ({len(blob) / 1024:.1f} KB)")
    print(f"wrote {manifest_path}")
    print(f"task: {manifest['task']!r}, outputLevel: {output_level!r}, "
          f"graphType: {manifest['graphType']!r}, applicableElement: {manifest['applicableElement']!r}, "
          f"d_h={manifest['dims']['d_h']}, depth={manifest['dims']['depth']}")


if __name__ == "__main__":
    main()
