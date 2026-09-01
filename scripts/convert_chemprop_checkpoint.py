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
                      output; skipped entirely for atom-/bond-level output
  - predictor:         single-task RegressionFFN, BinaryClassificationFFN,
                        or MveFFN (regression-mve, molecule-level only --
                        see below), n_layers=1 (Linear -> ReLU -> Linear),
                        with the regression head's UnscaleTransform or the
                        classification head's sigmoid applied in JS.
                        Molecule-level (predictor / mol_predictor),
                        atom-level (atom_predictor), or bond-level
                        (bond_predictor) -- not more than one at once.
                        A bond-level checkpoint additionally needs
                        message_passing.W_eo (the edge_finalize
                        projection) -- confirmed present whenever
                        --bond-target-columns is used, since
                        MolAtomBondMPNN always attaches a bond_predictor's
                        message-passing block with return_edge_embeddings
                        implied by that predictor's existence.
  - MveFFN (Mean-Variance Estimation, `-t regression-mve -l mve` at train
    time): a real per-prediction uncertainty head, molecule-level only --
    see chemprop-model.js's applyHead() for how the browser reads back
    the (mean, variance) pair this doubles the FFN's output width to.
  - Optional extra molecule-level descriptors (chemprop's own
    `--descriptors-path`/`X_d`): if the checkpoint's state_dict has an
    `X_d_transform.mean`/`.scale` (a ScaleTransform standardizing X_d
    before it's concatenated onto the pooled graph embedding -- see
    chemprop/models/model.py's fingerprint()), those get exported too and
    ffn0_weight's own (wider) input dimension is read directly off its
    real shape, no separate flag needed. A checkpoint built this way needs
    the extra descriptor value(s) supplied at inference time via
    `CC.GNN.predictChemprop(molecule, id, extraDescriptors)`, not the
    generic no-argument prediction path (see chemprop-model.js).

If your checkpoint uses a different message-passing scheme
(AtomMessagePassing), a different aggregation (mean/sum/attentive), a
deeper FFN (n_layers > 1), multiclass/Dirichlet/evidential/spectral
heads, or multi-task output, this script will raise -- extending
chemprop-model.js's forward pass to match is required first (see
chemprop-model.js's header for the shapes it expects).

Usage:
    pip install torch chemprop lightning --break-system-packages
    python3 convert_chemprop_checkpoint.py best.pt output_dir/
    python3 convert_chemprop_checkpoint.py best.pt output_dir/ --task-key cyp2d6  # avoid a same-task collision

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
                              "(every hydrogen is its own graph node -- e.g. a per-atom 1H NMR shift model, "
                              "or any bond-level/BDE checkpoint, which always needs this). Not auto-detected: "
                              "the checkpoint doesn't reliably record whether --add-h was used, so pass this "
                              "explicitly based on how you trained it. Only meaningful for outputLevel=atom or "
                              "outputLevel=bond; ignored for molecule-level checkpoints.")
    parser.add_argument("--applicable-element", default=None,
                         help="restrict this atom-level checkpoint's predictions to one element, e.g. 'C' for "
                              "a 13C shift model -- chemprop-model.js will only annotate atoms of this element "
                              "and leave others blank, the same masking pattern pka-model.js uses for its own "
                              "candidate-site gating. Only meaningful for outputLevel=atom.")
    parser.add_argument("--task-key", default=None,
                         help="override manifest['task'] (default: the checkpoint's own target-column name, "
                              "e.g. 'label'). Purely a dictionary key -- gnn-inference.js/chemprop-model.js "
                              "merge every currently-loaded model's molecule-/atom-/bond-level output into one "
                              "object keyed by this string, so two loaded models sharing a task name silently "
                              "overwrite each other (last-loaded wins) instead of both showing up. This "
                              "project's prepare_*_training_data.py scripts conventionally name every binary-"
                              "classification target column 'label', so any two single-task classifiers "
                              "trained that way collide unless given distinct --task-key values here at "
                              "conversion time (e.g. --task-key cyp2d6). Does not affect --name/output "
                              "filenames or the model's math -- registry.json's propertyKey for this entry "
                              "should be updated to match.")
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
        n_predictors_set = sum(hp.get(k) is not None for k in ("mol_predictor", "atom_predictor", "bond_predictor"))
        if n_predictors_set > 1:
            sys.exit("A checkpoint with more than one of mol-/atom-/bond-level predictors attached isn't "
                      "supported yet -- chemprop-model.js's manifest format assumes exactly one output level "
                      "per model.")
        if hp.get("bond_predictor") is not None:
            output_level = "bond"
            pred_hp = hp["bond_predictor"]
            task_names = bond_names
            mp_weight_key = "W_eo"  # MABBondMessagePassing's edge-output layer
            pred_prefix = "bond_predictor"
        elif hp.get("atom_predictor") is not None:
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
            sys.exit("None of mol_predictor/atom_predictor/bond_predictor is set -- nothing to export.")
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
    if output_level == "bond" and args.graph_type != "explicit-h":
        sys.exit("A bond-level (--bond-target-columns) checkpoint always needs --graph-type explicit-h -- "
                  "BDE-style targets are meaningless without real per-hydrogen graph nodes.")
    if args.graph_type == "explicit-h" and output_level == "molecule":
        sys.exit("--graph-type explicit-h only makes sense for an atom-level or bond-level checkpoint.")
    if output_level == "molecule":
        agg_cls_name = agg_hp["cls"].__name__ if agg_hp else None
        if agg_cls_name not in ("NormAggregation", "MeanAggregation"):
            sys.exit(f"Unsupported aggregation {agg_cls_name!r} -- expected NormAggregation or "
                      "MeanAggregation (js/pooling.js's poolSum/aggNorm and poolMean respectively) "
                      "for a molecule-level model.")

    pred_cls_name = pred_hp["cls"].__name__
    if pred_cls_name == "RegressionFFN":
        task_type = "regression"
    elif pred_cls_name == "BinaryClassificationFFN":
        task_type = "classification"
    elif pred_cls_name == "MveFFN":
        # Mean-Variance Estimation: same RegressionFFN math plus a second
        # output channel (raw variance, softplus'd) -- see MveFFN.forward()
        # in chemprop's own nn/predictors.py. n_tasks stays 1 (still one
        # property); only the FFN's actual last-layer output width doubles
        # (n_tasks * n_targets, n_targets=2 for MVE), which the generic
        # shape-driven tensor export below already handles unchanged.
        # Scoped to molecule-level only -- chemprop-model.js's atom-/
        # bond-level code paths (and the bond-level forward/backward
        # averaging in particular) assume applyHead() returns a plain
        # number, which no longer holds for an MVE head.
        if output_level != "molecule":
            sys.exit("MveFFN (regression-mve) is only supported for molecule-level checkpoints -- "
                      "chemprop-model.js's atom-/bond-level code paths assume a plain-number output "
                      "from applyHead(), which doesn't hold for an MVE head's (mean, variance) pair.")
        task_type = "regression-mve"
    else:
        sys.exit(f"Unsupported predictor {pred_cls_name!r} -- only single-task RegressionFFN "
                  "(regression), BinaryClassificationFFN (binary classification), and MveFFN "
                  "(regression-mve, molecule-level only) are implemented in chemprop-model.js. "
                  "Multiclass, Dirichlet/evidential, and spectral heads would need a new "
                  "forward-pass branch added there first.")
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

    # Bond-level checkpoints have no atom-output layer at all (Chemprop
    # never builds message_passing.W_vo without a mol/atom predictor
    # attached) -- export message_passing.W_eo (edge_finalize) under its
    # own tensor names instead of reusing "W_o_weight"/"W_o_bias", so
    # chemprop-model.js can tell the two apart (see dmpnn.js: Wo and Weo
    # are each independently optional).
    mp_out_tensor_names = ("W_eo_weight", "W_eo_bias") if output_level == "bond" else ("W_o_weight", "W_o_bias")
    tensors = {
        "W_i": arr("message_passing.W_i.weight"),
        "W_h": arr("message_passing.W_h.weight"),
        mp_out_tensor_names[0]: arr(f"message_passing.{mp_weight_key}.weight"),
        mp_out_tensor_names[1]: arr(f"message_passing.{mp_weight_key}.bias"),
        "ffn0_weight": arr(f"{pred_prefix}.ffn.0.0.weight"),
        "ffn0_bias": arr(f"{pred_prefix}.ffn.0.0.bias"),
        "ffn1_weight": arr(f"{pred_prefix}.ffn.1.2.weight"),
        "ffn1_bias": arr(f"{pred_prefix}.ffn.1.2.bias"),
    }
    if task_type in ("regression", "regression-mve"):
        # Classification's output_transform is Identity() (a plain sigmoid
        # is applied in the JS forward pass instead, see chemprop-model.js)
        # -- there's nothing to export here in that case. MveFFN reuses
        # this SAME output_transform for its mean channel, and its
        # transform_variance() (var * scale**2, confirmed from chemprop's
        # own UnscaleTransform source) needs only this same scale tensor
        # too -- no separate variance-specific tensor to export.
        tensors["out_mean"] = arr(f"{pred_prefix}.output_transform.mean").reshape(-1)
        tensors["out_scale"] = arr(f"{pred_prefix}.output_transform.scale").reshape(-1)

    # Extra molecule-level descriptors (chemprop v2's --descriptors-path):
    # MPNN.fingerprint() does torch.cat((H, X_d_transform(X_d)), dim=1)
    # before the FFN head (confirmed directly from chemprop's real
    # nn/predictors.py + models/model.py source -- not assumed), so
    # ffn0_weight's own input dimension already reflects this (wider than
    # just the message-passing hidden size) with no separate check needed;
    # this block only needs to additionally export X_d_transform's
    # mean/scale (a plain per-descriptor (x-mean)/scale standardization,
    # ScaleTransform) so chemprop-model.js can apply the identical
    # preprocessing to whatever raw descriptor value a caller supplies at
    # inference time.
    num_extra_descriptors = 0
    if "X_d_transform.mean" in sd:
        descriptor_mean = arr("X_d_transform.mean").reshape(-1)
        descriptor_scale = arr("X_d_transform.scale").reshape(-1)
        num_extra_descriptors = int(descriptor_mean.shape[0])
        tensors["descriptor_mean"] = descriptor_mean
        tensors["descriptor_scale"] = descriptor_scale

    # Delta-learning physical-baseline recalibration (a real, plain
    # top-level pair of scalars -- not routed through the tensor blob,
    # since they're single numbers, not arrays): scripts/
    # train_pka_microstate_freeenergy.py registers `physical_scale`/
    # `physical_offset` directly as extra nn.Parameters on the model
    # itself (not part of chemprop's own architecture), needed because
    # this app's own classical force-field energy is explicitly NOT on a
    # real physical (kcal/mol-comparable) scale -- confirmed as a real bug
    # this recalibration fixes, not a theoretical concern (see that
    # script's own header for the exact before/after MAE numbers).
    physical_scale = float(sd["physical_scale"]) if "physical_scale" in sd else None
    physical_offset = float(sd["physical_offset"]) if "physical_offset" in sd else None

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
        "task": args.task_key or task_names[0],
        "taskType": task_type,
        "outputLevel": output_level,  # "molecule" (default/omitted in older manifests), "atom", or "bond"
        "graphType": args.graph_type if output_level in ("atom", "bond") else "heavy",
        "applicableElement": args.applicable_element if output_level == "atom" else None,
        "architecture": "chemprop-dmpnn-v2",
        "dims": {
            "d_v": mp_hp["d_v"],
            "d_e": mp_hp["d_e"],
            "d_h": mp_hp["d_h"],
            "depth": mp_hp["depth"],
            # "norm" (sum of atom embeddings / a fixed constant, chemprop's
            # NormAggregation) or "mean" (chemprop's MeanAggregation, i.e.
            # sum / actual atom count) -- omitted/absent means "norm" for
            # every manifest converted before this field existed, so
            # js/chemprop-model.js must default missing to "norm", not error.
            "aggregationType": "mean" if agg_hp and agg_hp["cls"].__name__ == "MeanAggregation" else "norm",
            "aggNorm": (agg_hp["norm"] if agg_hp and "norm" in agg_hp else None),
        },
        "ffnHiddenDim": ffn_hidden_dim,
        "numExtraDescriptors": num_extra_descriptors,
        "physicalScale": physical_scale,
        "physicalOffset": physical_offset,
        "nTasks": 1,
        "tensors": manifest_tensors,
        "byteLength": len(blob),
        "dtype": "float32",
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.name or args.task_key or task_names[0]
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
