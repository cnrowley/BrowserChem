#!/usr/bin/env python3
"""
export_onnx_from_checkpoint.py

Exports a raw chemprop Lightning .pt checkpoint (the ORIGINAL training
output, e.g. <output_dir>/model_0/best.pt) directly to a self-contained
ONNX file -- unlike export_onnx.py, which reads this project's own
manifest.json/weights.bin deployed format.

Why this script exists in addition to export_onnx.py: this project's
scripts/convert_chemprop_checkpoint.py (the manifest.json/weights.bin
converter) explicitly refuses multi-task checkpoints
(`if pred_hp["n_tasks"] != 1: sys.exit(...)`), because js/chemprop-model.js's
hand-rolled JS forward pass was only ever built to assume a single output.
ONNX has no such restriction -- the message-passing/FFN math is identical
regardless of how many outputs the final linear layer produces -- so a
genuine shared-encoder multi-task model (e.g. one CHEMELEON-fine-tuned
encoder jointly predicting BBBP + hERG + 11 CYP endpoints) can be
deployed via ONNX Runtime Web even though it can never be converted to
this project's own manifest.json format.

Reuses the exact math/bugfixes already proven correct in export_onnx.py
(same H_0-must-stay-unactivated fix, same sum-then-divide mean-aggregation
workaround for the scatter_reduce(reduce="mean") ONNX-exporter bug, same
external-data merge step for onnxruntime-web compatibility) -- see that
file's docstring/comments for the full rationale. State-dict key names
(message_passing.W_i.weight, predictor.ffn.0.0.weight, etc.) are the same
ones scripts/convert_chemprop_checkpoint.py already validated against
real chemprop internals.

Usage:
    export_onnx_from_checkpoint.py <checkpoint.pt> <out.onnx> [--task-names a b c ...]

--task-names is optional metadata only (written to a sidecar .json next
to the .onnx file so the browser side knows which output column is which
property) -- it does not affect the exported graph.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn


class CheckpointModel(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" not in ckpt or "hyper_parameters" not in ckpt:
            sys.exit("This doesn't look like a Chemprop Lightning checkpoint "
                      "(missing state_dict/hyper_parameters).")
        sd = ckpt["state_dict"]
        hp = ckpt["hyper_parameters"]

        def arr(name):
            return torch.from_numpy(sd[name].detach().numpy().astype("float32").copy())

        mp_hp = hp["message_passing"]
        mp_cls_name = mp_hp["cls"].__name__
        if mp_cls_name not in ("BondMessagePassing", "MABBondMessagePassing"):
            sys.exit(f"Unsupported message-passing class {mp_cls_name!r}")
        if mp_hp["bias"]:
            sys.exit("Unsupported message-passing bias=True")

        agg_hp = hp["agg"]
        agg_cls_name = agg_hp["cls"].__name__ if agg_hp else None
        if agg_cls_name not in ("NormAggregation", "MeanAggregation"):
            sys.exit(f"Unsupported aggregation {agg_cls_name!r}")
        self.agg_type = "mean" if agg_cls_name == "MeanAggregation" else "norm"
        self.agg_norm = float(agg_hp.get("norm", 100.0)) if agg_cls_name == "NormAggregation" else None

        pred_hp = hp["predictor"]
        pred_cls_name = pred_hp["cls"].__name__
        if pred_cls_name == "RegressionFFN":
            self.task_type = "regression"
        elif pred_cls_name == "BinaryClassificationFFN":
            self.task_type = "classification"
        elif pred_cls_name == "MveFFN":
            self.task_type = "regression-mve"
        else:
            sys.exit(f"Unsupported predictor {pred_cls_name!r}")
        if pred_hp["n_layers"] != 1:
            sys.exit(f"Unsupported predictor n_layers={pred_hp['n_layers']}")

        self.n_tasks = pred_hp["n_tasks"]
        self.d_h = mp_hp["d_h"]
        self.depth = mp_hp["depth"]
        self.num_extra = 0  # no --descriptors-path fusion in this export path

        self.W_i = nn.Linear(*arr("message_passing.W_i.weight").shape[::-1], bias=False)
        self.W_i.weight.data = arr("message_passing.W_i.weight")
        self.W_h = nn.Linear(*arr("message_passing.W_h.weight").shape[::-1], bias=False)
        self.W_h.weight.data = arr("message_passing.W_h.weight")
        wo_w = arr("message_passing.W_o.weight")
        self.W_o = nn.Linear(wo_w.shape[1], wo_w.shape[0], bias=True)
        self.W_o.weight.data = wo_w
        self.W_o.bias.data = arr("message_passing.W_o.bias")

        ffn0_w = arr("predictor.ffn.0.0.weight")
        self.ffn0 = nn.Linear(ffn0_w.shape[1], ffn0_w.shape[0], bias=True)
        self.ffn0.weight.data = ffn0_w
        self.ffn0.bias.data = arr("predictor.ffn.0.0.bias")
        ffn1_w = arr("predictor.ffn.1.2.weight")
        self.ffn1 = nn.Linear(ffn1_w.shape[1], ffn1_w.shape[0], bias=True)
        self.ffn1.weight.data = ffn1_w
        self.ffn1.bias.data = arr("predictor.ffn.1.2.bias")

        if self.task_type in ("regression", "regression-mve"):
            self.register_buffer("out_mean", arr("predictor.output_transform.mean").reshape(-1))
            self.register_buffer("out_scale", arr("predictor.output_transform.scale").reshape(-1))

        target_cols = hp.get("task_names") or hp.get("target_columns")
        self.task_names = list(target_cols) if target_cols else None

    # identical math to export_onnx.py's ManifestModel.forward -- see that
    # file for the full rationale on both the H_0 and mean-aggregation
    # fixes baked in here.
    def forward(self, V, E, edge_index, rev_edge_index, batch):
        src, dst = edge_index[0], edge_index[1]
        H_0 = self.W_i(torch.cat([V[src], E], dim=1))
        H = torch.relu(H_0)
        for _ in range(1, self.depth):
            idx = dst.unsqueeze(1).repeat(1, H.shape[1])
            M_all = torch.zeros(V.shape[0], H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce(
                0, idx, H, reduce="sum", include_self=False
            )[src]
            M = M_all - H[rev_edge_index]
            H = torch.relu(H_0 + self.W_h(M))

        idx = dst.unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(V.shape[0], H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce(
            0, idx, H, reduce="sum", include_self=False
        )
        H_v = torch.relu(self.W_o(torch.cat((V, M), dim=1)))

        n_mols = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        idx2 = batch.unsqueeze(1).repeat(1, H_v.shape[1])
        Zsum = torch.zeros(n_mols, H_v.shape[1], dtype=H_v.dtype, device=H_v.device).scatter_reduce(
            0, idx2, H_v, reduce="sum", include_self=False
        )
        if self.agg_type == "mean":
            counts = torch.zeros(n_mols, dtype=H_v.dtype, device=H_v.device).scatter_reduce(
                0, batch, torch.ones_like(batch, dtype=H_v.dtype), reduce="sum", include_self=False
            )
            Z = Zsum / counts.clamp(min=1).unsqueeze(1)
        else:
            Z = Zsum / self.agg_norm

        hidden = torch.relu(self.ffn0(Z))
        raw = self.ffn1(hidden)

        if self.task_type == "classification":
            return torch.sigmoid(raw)
        if self.task_type == "regression-mve":
            mean_raw, var_raw = raw[:, : self.n_tasks], raw[:, self.n_tasks:]
            mean = mean_raw * self.out_scale + self.out_mean
            var = torch.nn.functional.softplus(var_raw) * self.out_scale * self.out_scale
            return torch.cat([mean, var], dim=1)
        return raw * self.out_scale + self.out_mean


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("out_onnx")
    p.add_argument("--task-names", nargs="+", default=None)
    args = p.parse_args()

    model = CheckpointModel(args.checkpoint)
    model.eval()
    task_names = args.task_names or model.task_names
    if not task_names:
        sys.exit("Couldn't recover task names from the checkpoint -- pass --task-names explicitly.")

    n_atoms, n_edges = 5, 8
    d_v, d_e = 72, 14
    V = torch.randn(n_atoms, d_v)
    E = torch.randn(n_edges, d_e)
    edge_index = torch.randint(0, n_atoms, (2, n_edges))
    rev_edge_index = torch.randperm(n_edges)
    batch = torch.zeros(n_atoms, dtype=torch.long)
    args_tuple = (V, E, edge_index, rev_edge_index, batch)

    with torch.no_grad():
        sanity = model(*args_tuple)
    print(f"sanity output shape: {sanity.shape} ({model.n_tasks} tasks, task_type={model.task_type})", file=sys.stderr)

    dyn_axes = {
        "V": {0: "num_atoms"}, "E": {0: "num_edges"},
        "edge_index": {1: "num_edges"}, "rev_edge_index": {0: "num_edges"},
        "batch": {0: "num_atoms"}, "output": {0: "num_mols"},
    }
    tmp_path = args.out_onnx + ".tmp"
    torch.onnx.export(
        model, args_tuple, tmp_path,
        input_names=["V", "E", "edge_index", "rev_edge_index", "batch"],
        output_names=["output"], dynamic_axes=dyn_axes,
        opset_version=18, do_constant_folding=True,
    )

    onnx_model = onnx.load(tmp_path, load_external_data=True)
    onnx.save_model(onnx_model, args.out_onnx, save_as_external_data=False)
    for f in Path(tmp_path).parent.glob(Path(tmp_path).name + "*"):
        f.unlink()
    ext_data = Path(args.out_onnx + ".data")
    if ext_data.exists():
        ext_data.unlink()

    meta_path = Path(args.out_onnx).with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "taskNames": task_names, "taskType": model.task_type, "nTasks": model.n_tasks,
        "dims": {"d_h": model.d_h, "depth": model.depth, "aggregationType": model.agg_type,
                 "aggNorm": model.agg_norm, "d_v": 72, "d_e": 14},
    }, indent=2))

    print(f"wrote {args.out_onnx} + {meta_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
