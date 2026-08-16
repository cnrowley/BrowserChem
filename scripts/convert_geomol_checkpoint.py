#!/usr/bin/env python3
"""
convert_geomol_checkpoint.py

Converts a real pretrained GeoMol checkpoint (Ganea et al., "GeoMol:
Torsional Geometric Generation of Molecular 3D Conformer Ensembles",
NeurIPS 2021 -- https://github.com/PattanaikL/GeoMol, MIT license) into a
manifest.json + weights.bin pair js/geomol-model.js can load directly in
the browser -- no server, no PyTorch, no PyTorch Geometric.

This is NOT a from-memory re-derivation of GeoMol's architecture: every
tensor name/shape and every hyperparameter here is read directly off the
real trained_models/{drugs,qm9}/best_model.pt state_dict and
model_parameters.yml the official repo ships, the same "read from the
live object, never transcribed" principle convert_ani2x_checkpoint.py
already follows for ANI-2x. If a tensor is missing or an unexpected shape
turns up, this script raises rather than silently exporting something the
JS forward pass would compute wrong.

Requires the official GeoMol repo cloned locally (only its checkpoint
files and model_parameters.yml are read -- no GeoMol/torch/torch_geometric
import happens here, so none of GeoMol's own Python dependencies are
needed):

    git clone https://github.com/PattanaikL/GeoMol.git
    pip install torch pyyaml --break-system-packages
    python3 convert_geomol_checkpoint.py GeoMol/trained_models/drugs model/geomol-drugs/

Only supports the exact architecture js/geomol-model.js implements:
  - a weight-tied MetaLayer message-passing GNN (node_init/edge_init MLPs,
    then the same EdgeModel+NodeModel update applied `depth` times)
  - a single nn.TransformerEncoderLayer (post-LN, ReLU feedforward) for
    per-neighborhood local structure prediction
  - plain MLP heads (coord_pred, d_mlp, h_mol_mlp, alpha_mlp, c_mlp) --
    stacked nn.Linear + ReLU, unactivated final layer
  - global_transformer=False (the case both the published drugs and qm9
    checkpoints actually use) -- the optional whole-molecule Transformer
    path in the original code is not implemented here
"""

import argparse
import json
import sys
from pathlib import Path


def mlp_layer_dims(state_dict, prefix):
    """Reads an MLP's per-layer (in, out) dims and weight/bias tensors
    straight off state_dict keys named f'{prefix}.layers.{i}.weight' --
    the real MLP class (model/GNN.py) interleaves Linear/activation
    modules into one nn.ModuleList, so only every-other index is a
    Linear layer; this reconstructs exactly that layout rather than
    assuming a fixed depth."""
    layers = []
    i = 0
    while f"{prefix}.layers.{i}.weight" in state_dict:
        w = state_dict[f"{prefix}.layers.{i}.weight"]
        b = state_dict[f"{prefix}.layers.{i}.bias"]
        layers.append((i, w, b))
        i += 2  # ReLU activation modules sit at the odd indices, no params
    if not layers:
        sys.exit(f"No layers found under '{prefix}.layers.*' -- unexpected MLP module layout.")
    return layers


def export_mlp(tensors, state_dict, prefix, key_prefix):
    layers = mlp_layer_dims(state_dict, prefix)
    dims = [int(layers[0][1].shape[1])]
    for idx, (_, w, b) in enumerate(layers):
        tensors[f"{key_prefix}_layer{idx}_weight"] = w.detach().numpy().astype("float32")
        tensors[f"{key_prefix}_layer{idx}_bias"] = b.detach().numpy().astype("float32")
        dims.append(int(w.shape[0]))
    return dims


def export_gnn(tensors, state_dict, prefix, key_prefix):
    """Exports one GNN submodule (gnn or gnn2): node_init MLP, edge_init
    MLP, and the single weight-tied MetaLayer update (EdgeModel +
    NodeModel + the two learnable eps residual scalars)."""
    node_dims = export_mlp(tensors, state_dict, f"{prefix}.node_init", f"{key_prefix}_node_init")
    edge_dims = export_mlp(tensors, state_dict, f"{prefix}.edge_init", f"{key_prefix}_edge_init")

    tensors[f"{key_prefix}_edge_eps"] = state_dict[f"{prefix}.update.edge_eps"].detach().numpy().astype("float32")
    tensors[f"{key_prefix}_node_eps"] = state_dict[f"{prefix}.update.node_eps"].detach().numpy().astype("float32")

    for name in ("edge", "node_in", "node_out"):
        w = state_dict[f"{prefix}.update.edge_model.{name}.weight"]
        tensors[f"{key_prefix}_edgemodel_{name}_weight"] = w.detach().numpy().astype("float32")
        bias_key = f"{prefix}.update.edge_model.{name}.bias"
        if bias_key in state_dict:  # node_in/node_out are bias=False in the real EdgeModel
            tensors[f"{key_prefix}_edgemodel_{name}_bias"] = state_dict[bias_key].detach().numpy().astype("float32")

    edge_mlp_dims = export_mlp(tensors, state_dict, f"{prefix}.update.edge_model.mlp", f"{key_prefix}_edgemodel_mlp")
    node_mlp1_dims = export_mlp(tensors, state_dict, f"{prefix}.update.node_model.node_mlp_1", f"{key_prefix}_nodemodel_mlp1")
    node_mlp2_dims = export_mlp(tensors, state_dict, f"{prefix}.update.node_model.node_mlp_2", f"{key_prefix}_nodemodel_mlp2")

    return {
        "nodeInitDims": node_dims,
        "edgeInitDims": edge_dims,
        "edgeModelMlpDims": edge_mlp_dims,
        "nodeModelMlp1Dims": node_mlp1_dims,
        "nodeModelMlp2Dims": node_mlp2_dims,
    }


def export_transformer_encoder_layer(tensors, state_dict, prefix, key_prefix, n_head):
    """Standard nn.TransformerEncoderLayer (post-LN, as used with no
    norm_first override), read directly off its real parameter names."""
    in_proj_w = state_dict[f"{prefix}.self_attn.in_proj_weight"]  # (3*d_model, d_model)
    in_proj_b = state_dict[f"{prefix}.self_attn.in_proj_bias"]  # (3*d_model,)
    d_model = int(in_proj_w.shape[1])
    if in_proj_w.shape[0] != 3 * d_model:
        sys.exit(f"'{prefix}.self_attn.in_proj_weight' shape {tuple(in_proj_w.shape)} isn't (3*d_model, d_model).")
    if d_model % n_head != 0:
        sys.exit(f"d_model={d_model} not divisible by n_head={n_head}.")

    tensors[f"{key_prefix}_in_proj_weight"] = in_proj_w.detach().numpy().astype("float32")
    tensors[f"{key_prefix}_in_proj_bias"] = in_proj_b.detach().numpy().astype("float32")
    tensors[f"{key_prefix}_out_proj_weight"] = state_dict[f"{prefix}.self_attn.out_proj.weight"].detach().numpy().astype("float32")
    tensors[f"{key_prefix}_out_proj_bias"] = state_dict[f"{prefix}.self_attn.out_proj.bias"].detach().numpy().astype("float32")
    tensors[f"{key_prefix}_linear1_weight"] = state_dict[f"{prefix}.linear1.weight"].detach().numpy().astype("float32")
    tensors[f"{key_prefix}_linear1_bias"] = state_dict[f"{prefix}.linear1.bias"].detach().numpy().astype("float32")
    tensors[f"{key_prefix}_linear2_weight"] = state_dict[f"{prefix}.linear2.weight"].detach().numpy().astype("float32")
    tensors[f"{key_prefix}_linear2_bias"] = state_dict[f"{prefix}.linear2.bias"].detach().numpy().astype("float32")
    tensors[f"{key_prefix}_norm1_weight"] = state_dict[f"{prefix}.norm1.weight"].detach().numpy().astype("float32")
    tensors[f"{key_prefix}_norm1_bias"] = state_dict[f"{prefix}.norm1.bias"].detach().numpy().astype("float32")
    tensors[f"{key_prefix}_norm2_weight"] = state_dict[f"{prefix}.norm2.weight"].detach().numpy().astype("float32")
    tensors[f"{key_prefix}_norm2_bias"] = state_dict[f"{prefix}.norm2.bias"].detach().numpy().astype("float32")

    return {
        "dModel": d_model,
        "nHead": n_head,
        "dimFeedforward": int(state_dict[f"{prefix}.linear1.weight"].shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trained_model_dir", help="path to the official repo's trained_models/{drugs,qm9} directory")
    parser.add_argument("output_dir", help="directory to write manifest.json + weights.bin into")
    parser.add_argument("--name", default="geomol", help="base filename (default: geomol)")
    args = parser.parse_args()

    import torch
    import yaml

    trained_dir = Path(args.trained_model_dir)
    params_path = trained_dir / "model_parameters.yml"
    checkpoint_path = trained_dir / "best_model.pt"
    if not params_path.exists() or not checkpoint_path.exists():
        sys.exit(f"Expected both {params_path} and {checkpoint_path} to exist.")

    with open(params_path) as f:
        params = yaml.safe_load(f)
    hyperparams = params["hyperparams"]
    num_node_features = int(params["num_node_features"])
    num_edge_features = int(params["num_edge_features"])

    if hyperparams.get("global_transformer", False):
        sys.exit("This checkpoint has global_transformer=true -- js/geomol-model.js only implements the "
                  "global_transformer=false path (which both the published drugs and qm9 checkpoints use).")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    tensors = {}
    gnn_info = export_gnn(tensors, state_dict, "gnn", "gnn1")
    gnn2_info = export_gnn(tensors, state_dict, "gnn2", "gnn2")
    encoder_info = export_transformer_encoder_layer(
        tensors, state_dict, "encoder", "encoder", n_head=int(hyperparams["encoder"]["n_head"]))
    coord_pred_dims = export_mlp(tensors, state_dict, "coord_pred", "coord_pred")
    d_mlp_dims = export_mlp(tensors, state_dict, "d_mlp", "d_mlp")
    h_mol_mlp_dims = export_mlp(tensors, state_dict, "h_mol_mlp", "h_mol_mlp")
    alpha_mlp_dims = export_mlp(tensors, state_dict, "alpha_mlp", "alpha_mlp")
    c_mlp_dims = export_mlp(tensors, state_dict, "c_mlp", "c_mlp")

    if coord_pred_dims[-1] != 3:
        sys.exit(f"coord_pred's final layer outputs {coord_pred_dims[-1]} values, expected 3 (a unit direction).")
    if d_mlp_dims[-1] != 1:
        sys.exit(f"d_mlp's final layer outputs {d_mlp_dims[-1]} values, expected 1 (a bond distance).")
    if alpha_mlp_dims[-1] != 1 or c_mlp_dims[-1] != 1:
        sys.exit("alpha_mlp/c_mlp are expected to output a single scalar each.")

    offset = 0
    manifest_tensors = {}
    blob = bytearray()
    for name, arr in tensors.items():
        flat = arr.reshape(-1)
        manifest_tensors[name] = {"shape": list(arr.shape), "offset": offset, "length": int(flat.size)}
        blob += flat.tobytes()
        offset += flat.size

    manifest = {
        "architecture": "geomol",
        "task": "geomolConformer",
        "modelDim": int(hyperparams["model_dim"]),
        "randomVecDim": int(hyperparams["random_vec_dim"]),
        "randomVecStd": float(hyperparams["random_vec_std"]),
        "randomAlpha": bool(hyperparams["random_alpha"]),
        "numNodeFeatures": num_node_features,
        "numEdgeFeatures": num_edge_features,
        "gnn1Depth": int(hyperparams["gnn1"]["depth"]),
        "gnn2Depth": int(hyperparams["gnn2"]["depth"]),
        "gnn1": gnn_info,
        "gnn2": gnn2_info,
        "encoder": encoder_info,
        "coordPredDims": coord_pred_dims,
        "dMlpDims": d_mlp_dims,
        "hMolMlpDims": h_mol_mlp_dims,
        "alphaMlpDims": alpha_mlp_dims,
        "cMlpDims": c_mlp_dims,
        "tensors": manifest_tensors,
        "byteLength": len(blob),
        "dtype": "float32",
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{args.name}.bin"
    manifest_path = out_dir / "manifest.json"
    bin_path.write_bytes(bytes(blob))
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"wrote {bin_path} ({len(blob) / 1024 / 1024:.2f} MB)")
    print(f"wrote {manifest_path}")
    print(f"model_dim={manifest['modelDim']}, gnn1 depth={manifest['gnn1Depth']}, gnn2 depth={manifest['gnn2Depth']}, "
          f"encoder n_head={encoder_info['nHead']}")


if __name__ == "__main__":
    main()
