#!/usr/bin/env python3
"""
export_onnx.py

Exports a shipped molecule-level chemprop checkpoint (manifest.json +
weights.bin, this project's own canonical deployed format -- NOT the
original PyTorch .pt checkpoint, which may no longer exist on disk for
models trained in earlier sessions) to a self-contained ONNX file that
runs via onnxruntime-web in the browser.

Rebuilds an equivalent PyTorch nn.Module directly from the manifest's
tensor offsets/shapes (same read pattern as compute_applicability_domain.py's
ChempropDMPNNModel / compute_property_distributions.py's own class of
that name), so this works for ANY shipped model regardless of whether
its original training checkpoint survived -- the manifest+weights pair
IS the deployed model, and is guaranteed to exist for everything in the
registry.

Message-passing math mirrors chemprop's own
BondMessagePassing/_BondMessagePassingMixin (nn/message_passing/
{base,mixins}.py) exactly, and the aggregation is deliberately
reimplemented as sum-then-divide rather than calling a MeanAggregation-
equivalent scatter_reduce(reduce="mean") directly: PyTorch's ONNX
exporter (torch 2.x dynamo path, opset 18) was found to mistranslate
that specific reduction mode (confirmed by isolating message-passing,
which is correct, from aggregation, which wasn't, via stage-by-stage
comparison against real PyTorch output) -- sum-then-divide uses only
reduce="sum", proven correct, and is mathematically identical.

Usage:
    export_onnx.py <manifest.json> <weights.bin> <out.onnx>

Writes a single self-contained .onnx file (external-data tensors are
merged back in -- onnxruntime-web cannot load external-data files, since
it has no arbitrary filesystem access in the browser).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn


class ManifestModel(nn.Module):
    def __init__(self, manifest_path, weights_path):
        super().__init__()
        manifest = json.loads(Path(manifest_path).read_text())
        weights = np.fromfile(weights_path, dtype=np.float32)

        def tensor(name):
            t = manifest["tensors"][name]
            return torch.from_numpy(
                weights[t["offset"]: t["offset"] + t["length"]].reshape(t["shape"]).copy()
            )

        self.manifest = manifest
        self.task_type = manifest.get("taskType", "regression")
        self.output_level = manifest.get("outputLevel", "molecule")
        if self.output_level != "molecule":
            sys.exit(f"export_onnx.py only supports outputLevel=molecule checkpoints, got {self.output_level!r}")

        dims = manifest["dims"]
        self.d_h = dims["d_h"]
        self.depth = dims["depth"]
        self.agg_type = dims.get("aggregationType", "norm")
        self.agg_norm = dims.get("aggNorm")
        self.num_extra = manifest.get("numExtraDescriptors", 0)

        # message passing
        self.W_i = nn.Linear(*tensor("W_i").shape[::-1], bias=False)
        self.W_i.weight.data = tensor("W_i")
        self.W_h = nn.Linear(*tensor("W_h").shape[::-1], bias=False)
        self.W_h.weight.data = tensor("W_h")
        wo_w = tensor("W_o_weight")
        self.W_o = nn.Linear(wo_w.shape[1], wo_w.shape[0], bias=True)
        self.W_o.weight.data = wo_w
        self.W_o.bias.data = tensor("W_o_bias")

        # predictor FFN
        ffn0_w = tensor("ffn0_weight")
        self.ffn0 = nn.Linear(ffn0_w.shape[1], ffn0_w.shape[0], bias=True)
        self.ffn0.weight.data = ffn0_w
        self.ffn0.bias.data = tensor("ffn0_bias")
        ffn1_w = tensor("ffn1_weight")
        self.ffn1 = nn.Linear(ffn1_w.shape[1], ffn1_w.shape[0], bias=True)
        self.ffn1.weight.data = ffn1_w
        self.ffn1.bias.data = tensor("ffn1_bias")

        if self.num_extra:
            self.register_buffer("descriptor_mean", tensor("descriptor_mean"))
            self.register_buffer("descriptor_scale", tensor("descriptor_scale"))

        if self.task_type in ("regression", "regression-mve"):
            self.register_buffer("out_mean", tensor("out_mean"))
            self.register_buffer("out_scale", tensor("out_scale"))

    def forward(self, V, E, edge_index, rev_edge_index, batch, X_d=None):
        src, dst = edge_index[0], edge_index[1]

        # H_0 must stay UNACTIVATED -- chemprop's real update() does
        # tau(H_0 + W_h(M)) using the raw W_i(...) output, not a relu'd
        # copy of it (only the loop variable H itself gets relu'd). A
        # transcription bug here (relu baked into H_0 itself) barely
        # affects depth=1 or small/simple molecules, since it doesn't
        # break any chemprop math, but compounds across every depth step
        # for larger/more connected graphs -- confirmed by exactly this
        # symptom (near-perfect on ethanol, real divergence on
        # atorvastatin) before this was found and fixed.
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
        elif self.agg_type == "norm":
            Z = Zsum / self.agg_norm
        else:
            Z = Zsum

        if self.num_extra:
            scaled = (X_d - self.descriptor_mean) / self.descriptor_scale
            Z = torch.cat([Z, scaled], dim=1)

        hidden = torch.relu(self.ffn0(Z))
        raw = self.ffn1(hidden)

        if self.task_type == "classification":
            return torch.sigmoid(raw)
        if self.task_type == "regression-mve":
            mean = raw[:, 0:1] * self.out_scale + self.out_mean
            var = torch.nn.functional.softplus(raw[:, 1:2]) * self.out_scale * self.out_scale
            return torch.cat([mean, var], dim=1)
        return raw * self.out_scale + self.out_mean


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("weights")
    p.add_argument("out_onnx")
    args = p.parse_args()

    model = ManifestModel(args.manifest, args.weights)
    model.eval()

    n_atoms, n_edges = 5, 8
    d_v = model.manifest["dims"]["d_v"]
    d_e = model.manifest["dims"]["d_e"]
    V = torch.randn(n_atoms, d_v)
    E = torch.randn(n_edges, d_e)
    edge_index = torch.randint(0, n_atoms, (2, n_edges))
    rev_edge_index = torch.randperm(n_edges)
    batch = torch.zeros(n_atoms, dtype=torch.long)

    dyn_axes = {
        "V": {0: "num_atoms"}, "E": {0: "num_edges"},
        "edge_index": {1: "num_edges"}, "rev_edge_index": {0: "num_edges"},
        "batch": {0: "num_atoms"}, "output": {0: "num_mols"},
    }
    input_names = ["V", "E", "edge_index", "rev_edge_index", "batch"]
    args_tuple = (V, E, edge_index, rev_edge_index, batch)
    if model.num_extra:
        X_d = torch.randn(1, model.num_extra)
        dyn_axes["X_d"] = {0: "num_mols"}
        input_names.append("X_d")
        args_tuple = args_tuple + (X_d,)

    with torch.no_grad():
        sanity = model(*args_tuple)
    print(f"sanity output shape: {sanity.shape}", file=sys.stderr)

    tmp_path = args.out_onnx + ".tmp"
    torch.onnx.export(
        model, args_tuple, tmp_path,
        input_names=input_names, output_names=["output"],
        dynamic_axes=dyn_axes, opset_version=18, do_constant_folding=True,
    )

    # Merge any external-data tensors back into one self-contained file --
    # onnxruntime-web cannot load external-data references in the browser.
    onnx_model = onnx.load(tmp_path, load_external_data=True)
    onnx.save_model(onnx_model, args.out_onnx, save_as_external_data=False)
    for f in Path(tmp_path).parent.glob(Path(tmp_path).name + "*"):
        f.unlink()
    ext_data = Path(args.out_onnx + ".data")
    if ext_data.exists():
        ext_data.unlink()

    print(f"wrote {args.out_onnx}", file=sys.stderr)


if __name__ == "__main__":
    main()
