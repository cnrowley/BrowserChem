#!/usr/bin/env python3
"""
pretrain_pka_qm.py

Real transfer learning for the pka-microstate-freeenergy Chemprop D-MPNN:
warm-starts the SAME message-passing architecture
scripts/train_pka_microstate_freeenergy.py builds
(`nn.BondMessagePassing(d_h=300, depth=3)`, this project's standard) on two
real-QM auxiliary regression tasks built by
scripts/prepare_pka_qm_pretrain_data.py from the Nevolianis et al. 2025
JACS anion-solvation release, BEFORE that script's own paired delta-
learning finetuning stage runs on real experimental aqueous pKa. Saves
only the message-passing submodule's state_dict -- the per-task output
head (`RegressionFFN`) is deliberately NOT saved/reused, since the
finetuning stage builds and trains its own fresh output head on the
actual pKa-correction task; only the learned atom/bond representation
transfers, which is what "pretraining" is supposed to give the
finetuning stage a head start on, not a task-specific output mapping
that means something different for each of these three targets.

Two sequential stages, each warm-starting the next (real transfer
learning: weight-initialization transfer, not a runtime architecture
change -- see scripts/prepare_pka_qm_pretrain_data.py's own header for
why none of this QM data can be a runtime browser input at all):

  Stage A (single-molecule): predicts real COSMO-RS aqueous solvation
  free energy for a single anion structure -- ordinary per-molecule
  chemprop regression, not the paired setup the other two tasks need.
  Trains the message-passing weights specifically on CHARGED organic
  species, which is exactly where this project's own classical SMIRNOFF+
  GBSA physical baseline was independently found to be weakest (see
  scripts/train_pka_microstate_freeenergy.py's own header: Pearson
  r~0.03-0.05 standalone correlation with real pKa).

  Stage B (paired), warm-started from Stage A: predicts real gas-phase
  acidity (protonated vs. deprotonated free-energy difference, no
  solvent at all) via the identical two-forward-passes-then-subtract
  pattern the main finetuning script uses for pKa itself (just without
  that script's physical-baseline/thermodynamic-cycle terms, since gas-
  phase acidity isn't a pKa and has no physical-baseline counterpart to
  recalibrate against) -- teaches the network the solvent-independent
  intrinsic acidity/basicity signal a water-only physical baseline can
  never see on its own.

Both targets are standardized (subtract train-mean, divide by train-std)
purely for optimization stability within THIS script -- discarded after
training along with each stage's own output head, so this has no effect
on anything the finetuning stage does afterward.

Usage:
    conda run -n cov-chemprop python3 scripts/pretrain_pka_qm.py \\
        data/pka/qm_anion_cosmo_solvation.csv data/pka/qm_gas_phase_acidity.csv \\
        checkpoints/pka-qm-pretrain/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("anion_solvation_csv")
    parser.add_argument("gas_phase_acidity_csv")
    parser.add_argument("output_dir")
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--stageA-epochs", type=int, default=60, dest="stageA_epochs")
    parser.add_argument("--stageB-epochs", type=int, default=60, dest="stageB_epochs")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from chemprop import data, featurizers, nn as cp_nn
    from chemprop.nn.transforms import UnscaleTransform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", file=sys.stderr)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def new_mpnn():
        mp = cp_nn.BondMessagePassing(d_h=args.hidden_dim, depth=args.depth)
        agg = cp_nn.NormAggregation()
        output_transform = UnscaleTransform(mean=np.array([0.0], dtype=np.float32), scale=np.array([1.0], dtype=np.float32))
        ffn = cp_nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=args.hidden_dim, n_layers=1, output_transform=output_transform)
        from chemprop.models import MPNN
        model = MPNN(message_passing=mp, agg=agg, predictor=ffn)
        model.to(device)
        return model

    def train_val_split(n, val_frac):
        idx = rng.permutation(n)
        n_val = int(n * val_frac)
        return idx[n_val:], idx[:n_val]

    def forward_one(model, datapoints):
        ds = data.MoleculeDataset(datapoints, featurizer=featurizer)
        loader = data.build_dataloader(ds, batch_size=len(datapoints), shuffle=False)
        bmg, V_d, X_d, *_ = next(iter(loader))
        bmg.to(device)  # BatchMolGraph.to() mutates in place, returns None -- do NOT reassign
        return model(bmg, V_d, X_d)[:, 0]

    def batches(n, batch_size, shuffle):
        order = rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            yield order[start:start + batch_size]

    # ---- Stage A: single-molecule anion COSMO-RS solvation regression ----
    rows_a = list(csv.DictReader(open(args.anion_solvation_csv, newline="", encoding="utf-8")))
    print(f"stage A: loaded {len(rows_a)} anion COSMO-RS solvation rows", file=sys.stderr)
    values_a = np.array([float(r["dG_solv_ion_cosmo"]) for r in rows_a], dtype=np.float32)
    mean_a, std_a = float(values_a.mean()), float(values_a.std() or 1.0)
    dps_a = [data.MoleculeDatapoint.from_smi(r["anion_smiles"], np.array([0.0], dtype=np.float32)) for r in rows_a]
    train_a, val_a = train_val_split(len(rows_a), args.val_frac)

    model_a = new_mpnn()
    opt_a = torch.optim.Adam(model_a.parameters(), lr=args.lr)
    targets_a = torch.tensor((values_a - mean_a) / std_a, dtype=torch.float32, device=device)
    best_val_a, best_mp_state_a = float("inf"), None
    for epoch in range(args.stageA_epochs):
        model_a.train()
        train_loss, train_n = 0.0, 0
        for sel in batches(len(train_a), args.batch_size, shuffle=True):
            batch_idx = train_a[sel]
            pred = forward_one(model_a, [dps_a[i] for i in batch_idx])
            loss = F.mse_loss(pred, targets_a[batch_idx])
            opt_a.zero_grad(); loss.backward(); opt_a.step()
            train_loss += loss.item() * len(batch_idx); train_n += len(batch_idx)
        model_a.eval()
        with torch.no_grad():
            val_pred = forward_one(model_a, [dps_a[i] for i in val_a]) if len(val_a) else None
            val_mse = F.mse_loss(val_pred, targets_a[val_a]).item() if val_pred is not None else float("nan")
        print(f"[stage A] epoch {epoch:3d} train_mse={train_loss/max(train_n,1):.4f} val_mse={val_mse:.4f}", file=sys.stderr)
        if val_mse < best_val_a:
            best_val_a = val_mse
            best_mp_state_a = {k: v.detach().cpu().clone() for k, v in model_a.message_passing.state_dict().items()}

    print(f"stage A done: best val_mse={best_val_a:.4f} (real units: RMSE={best_val_a**0.5 * std_a:.4f} kcal/mol-ish)", file=sys.stderr)

    # ---- Stage B: paired gas-phase acidity regression, warm-started from A ----
    rows_b = list(csv.DictReader(open(args.gas_phase_acidity_csv, newline="", encoding="utf-8")))
    print(f"stage B: loaded {len(rows_b)} gas-phase acidity rows", file=sys.stderr)
    values_b = np.array([float(r["dG_gas"]) for r in rows_b], dtype=np.float32)
    mean_b, std_b = float(values_b.mean()), float(values_b.std() or 1.0)
    dps_b_p = [data.MoleculeDatapoint.from_smi(r["smiles_protonated"], np.array([0.0], dtype=np.float32)) for r in rows_b]
    dps_b_d = [data.MoleculeDatapoint.from_smi(r["smiles_deprotonated"], np.array([0.0], dtype=np.float32)) for r in rows_b]
    train_b, val_b = train_val_split(len(rows_b), args.val_frac)

    model_b = new_mpnn()
    model_b.message_passing.load_state_dict(best_mp_state_a)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=args.lr)
    targets_b = torch.tensor((values_b - mean_b) / std_b, dtype=torch.float32, device=device)
    best_val_b, best_mp_state_b = float("inf"), None
    for epoch in range(args.stageB_epochs):
        model_b.train()
        train_loss, train_n = 0.0, 0
        for sel in batches(len(train_b), args.batch_size, shuffle=True):
            batch_idx = train_b[sel]
            pred_p = forward_one(model_b, [dps_b_p[i] for i in batch_idx])
            pred_d = forward_one(model_b, [dps_b_d[i] for i in batch_idx])
            loss = F.mse_loss(pred_d - pred_p, targets_b[batch_idx])
            opt_b.zero_grad(); loss.backward(); opt_b.step()
            train_loss += loss.item() * len(batch_idx); train_n += len(batch_idx)
        model_b.eval()
        with torch.no_grad():
            if len(val_b):
                vp, vd = forward_one(model_b, [dps_b_p[i] for i in val_b]), forward_one(model_b, [dps_b_d[i] for i in val_b])
                val_mse = F.mse_loss(vd - vp, targets_b[val_b]).item()
            else:
                val_mse = float("nan")
        print(f"[stage B] epoch {epoch:3d} train_mse={train_loss/max(train_n,1):.4f} val_mse={val_mse:.4f}", file=sys.stderr)
        if val_mse < best_val_b:
            best_val_b = val_mse
            best_mp_state_b = {k: v.detach().cpu().clone() for k, v in model_b.message_passing.state_dict().items()}

    print(f"stage B done: best val_mse={best_val_b:.4f} (real units: RMSE={best_val_b**0.5 * std_b:.4f} kJ/mol-ish)", file=sys.stderr)

    mp_path = out_dir / "mp_pretrained.pt"
    torch.save(best_mp_state_b, mp_path)
    print(f"wrote {mp_path}", file=sys.stderr)
    (out_dir / "metrics.json").write_text(json.dumps({
        "stageA_n": len(rows_a), "stageA_best_val_mse_standardized": best_val_a,
        "stageB_n": len(rows_b), "stageB_best_val_mse_standardized": best_val_b,
        "hiddenDim": args.hidden_dim, "depth": args.depth, "seed": args.seed,
    }, indent=2))


if __name__ == "__main__":
    main()
