#!/usr/bin/env python3
"""
train_pka_microstate_freeenergy.py

Trains the physics+Chemprop microstate free-energy pKa model as a real
DELTA-LEARNING setup: a single weight-shared, plain-graph Chemprop D-MPNN
(BondMessagePassing + NormAggregation + RegressionFFN, this project's
standard architecture) predicts a CORRECTION scalar per MICROSTATE, added
directly to a precomputed physical baseline energy (js/pka-physical-
baseline.js's own output) on the same free-energy scale the thermodynamic-
cycle formula operates on:

    g(microstate) = physical_scale*physical_baseline(microstate) + physical_offset + chemprop_correction(microstate)

--- Real micro-ENSEMBLE, not a single pair -- see js/unipka-thermo.js's macroPKa ---

Earlier versions of this script (and of scripts/prepare_pkahub_data.py)
trained on exactly ONE protonated structure and ONE deprotonated structure
per labeled pKa transition -- the micro-pKa special case of the real
Uni-pKa thermodynamic-cycle formula. That's fine when there's truly only
one reasonable structure per charge state, but a real, confirmed bug for
molecules with competing tautomers/zwitterions at the SAME charge state
(see model/registry.json's pka-microstate-freeenergy notes for the full
diagnostic history): picking just one structure can silently pick the
WRONG one relative to what a literature pKa transition's other endpoint
implies, cross-wiring which site a training row actually represents.

The real fix: `js/unipka-thermo.js`'s `CC.UniPKAThermo.macroPKa` already
implements the general formula from the Uni-pKa paper --

    pKa(A,B) = [ logsumexp(-g_i for i in A) - logsumexp(-g_j for j in B) ] / ln(10)

-- where A and B are the FULL SETS of microstates at each charge state,
not a single pick. This script now trains against that real formula: the
input CSV's `smiles_protonated_list`/`smiles_deprotonated_list` columns
(semicolon-joined, one or more SMILES per side -- scripts/prepare_pkahub_
data.py's own header explains why a wide net here is mathematically safe:
logsumexp gives a genuinely irrelevant high-energy tautomer near-zero
weight on its own) are each run through the SAME shared-weight network
(one forward pass per microstate, batched together across a whole
training batch for efficiency), then combined per row via the formula
above before comparing to the real experimental pKa. `physical_energy_
protonated_list`/`physical_energy_deprotonated_list` (also semicolon-
joined, same order/length as their own smiles list) carry the matching
physical baseline for each microstate.

Sources with only ONE structure per side (this project's own IUPAC/
Baltruschat and i-BonD extractions, which don't have a real per-molecule
microspecies enumeration to draw an ensemble from) are simply 1-element
lists here -- the formula's own single-microstate special case is
mathematically identical to the old micro-pKa-only formula, so nothing
about how those rows are trained on actually changes.

IMPORTANT SCOPE NOTE: this is a TRAINING-DATA-ONLY improvement. At
INFERENCE time (js/pka-freeenergy-predict.js), the browser still only
ever sees ONE structure per side per detected site -- there is no
tautomer/microspecies enumerator available in this project's RDKit.js
WASM build (checked directly: no `get_tautomers`/`canonical_tautomer` on
the minimal build this project ships). The single-structure-per-side case
IS this formula's own exact special case, so runtime prediction needs no
changes at all -- this script just teaches the SAME network from
cleaner, non-cross-wired signal during training.

--- X_d feature fusion ---

Real per-microstate X_d descriptors: NAGL-MBIS charge min/max (across
that microstate's own heavy atoms) plus one logP value SHARED across
every microstate in a row (computed once, from whichever microstate in
the whole ensemble is net-charge-0 -- logp-v1 has its own real
applicability-domain gate that refuses net-charged molecules, see
scripts/pka-physical-baseline-harness/add_extra_features.js's own
header). Auto-detected from the input CSV's column names; absent ->
numExtraDescriptors=0 and this script behaves as a plain delta-learning
model with no feature fusion at all.

--- Per-row fidelity_weight: real multi-fidelity training ---

If the input CSV has a `fidelity_weight` column (scripts/prepare_pkahub_
data.py writes one; rows without it -- this project's own original/
i-BonD CSVs -- default to 1.0), each row's squared error is weighted by
it during TRAINING ONLY (never during val/test scoring, which always
uses weight=1 regardless of the source column, so "how good is this
checkpoint" never depends on which sources happened to land in test).

--- Output format ---

Saves a checkpoint in the same {"state_dict", "hyper_parameters",
"output_columns"} shape scripts/convert_chemprop_checkpoint.py already
reads. `X_d_transform.mean`/`.scale` are present in the state_dict iff
X_d feature fusion is active -- the converter already handles both cases.

Usage:
    conda run -n cov-chemprop python3 scripts/train_pka_microstate_freeenergy.py \\
        data/pka/pka_microstate_pairs_final.csv checkpoints/pka-microstate-freeenergy/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def logsumexp_np(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(x - m))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_csv")
    parser.add_argument("output_dir")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=50, help="rows (pKa transitions) per batch, not microstates")
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=15,
                         help="epochs of no val-loss improvement before ReduceLROnPlateau halves the LR")
    parser.add_argument("--early-stop-patience", type=int, default=60,
                         help="epochs of no val-loss improvement before stopping early (0 disables)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--init-checkpoint", default=None,
                         help="warm-start from a checkpoint. Two supported formats, auto-detected: "
                              "(1) a full checkpoint from this SAME script (has 'state_dict'/"
                              "'hyper_parameters'/'output_columns' keys) -- loads the WHOLE model "
                              "(message-passing + FFN + physical_scale/offset) via strict "
                              "load_state_dict, i.e. genuinely CONTINUES training; requires an "
                              "identical architecture to the checkpoint being loaded. "
                              "(2) a bare message-passing state_dict -- warm-starts ONLY the "
                              "message-passing weights, everything else stays freshly initialized.")
    args = parser.parse_args()

    import torch
    from chemprop import data, featurizers, models, nn
    from chemprop.data.splitting import make_split_indices
    from chemprop.nn.transforms import UnscaleTransform, ScaleTransform
    from rdkit import Chem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", file=sys.stderr)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    rows = []
    with open(args.data_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    print(f"loaded {len(rows)} rows", file=sys.stderr)

    has_fidelity_weight = "fidelity_weight" in fieldnames
    X_D_NAGL_COLS_P = ["naglChargeMin_protonated_list", "naglChargeMax_protonated_list"]
    X_D_NAGL_COLS_D = ["naglChargeMin_deprotonated_list", "naglChargeMax_deprotonated_list"]
    has_extra_descriptors = all(c in fieldnames for c in X_D_NAGL_COLS_P + X_D_NAGL_COLS_D + ["logP"])
    print(f"fidelity_weight column: {has_fidelity_weight}; extra descriptors (NAGL charge min/max + logP): {has_extra_descriptors}", file=sys.stderr)

    n = len(rows)
    # Scaffold-balanced split (Murcko scaffold of the DOMINANT protonated
    # structure -- lists are sorted lowest-free-energy-first by
    # scripts/prepare_pkahub_data.py, so element 0 is the dominant
    # tautomer) via chemprop's own real astartes-backed splitter.
    protonated_lists = [r["smiles_protonated_list"].split(";") for r in rows]
    mols_for_split = [Chem.MolFromSmiles(lst[0]) for lst in protonated_lists]
    bad = [lst[0] for lst, m in zip(protonated_lists, mols_for_split) if m is None]
    if bad:
        raise ValueError(f"{len(bad)} protonated SMILES failed to parse for scaffold splitting, e.g. {bad[:3]}")
    train_frac = 1.0 - args.val_frac - args.test_frac
    train_replicates, val_replicates, test_replicates = make_split_indices(
        mols_for_split, split="scaffold_balanced",
        sizes=(train_frac, args.val_frac, args.test_frac), seed=args.seed, num_replicates=1,
    )
    train_idx, val_idx, test_idx = set(train_replicates[0]), set(val_replicates[0]), set(test_replicates[0])
    splits = ["test" if i in test_idx else "val" if i in val_idx else "train" for i in range(n)]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def build_row(r):
        smiles_p = r["smiles_protonated_list"].split(";")
        smiles_d = r["smiles_deprotonated_list"].split(";")
        phys_p = [float(x) for x in r["physical_energy_protonated_list"].split(";")]
        phys_d = [float(x) for x in r["physical_energy_deprotonated_list"].split(";")]
        if len(smiles_p) != len(phys_p) or len(smiles_d) != len(phys_d):
            raise ValueError(f"smiles/physical_energy list length mismatch for inchikey={r.get('inchikey')}")
        if has_extra_descriptors:
            nagl_min_p = [float(x) for x in r[X_D_NAGL_COLS_P[0]].split(";")]
            nagl_max_p = [float(x) for x in r[X_D_NAGL_COLS_P[1]].split(";")]
            nagl_min_d = [float(x) for x in r[X_D_NAGL_COLS_D[0]].split(";")]
            nagl_max_d = [float(x) for x in r[X_D_NAGL_COLS_D[1]].split(";")]
            logp = float(r["logP"])
            x_d_p = [np.array([mn, mx, logp], dtype=np.float32) for mn, mx in zip(nagl_min_p, nagl_max_p)]
            x_d_d = [np.array([mn, mx, logp], dtype=np.float32) for mn, mx in zip(nagl_min_d, nagl_max_d)]
        else:
            x_d_p = [None] * len(smiles_p)
            x_d_d = [None] * len(smiles_d)
        dummy = np.array([0.0], dtype=np.float32)
        return {
            "dps_p": [data.MoleculeDatapoint.from_smi(s, dummy, x_d=xd) for s, xd in zip(smiles_p, x_d_p)],
            "dps_d": [data.MoleculeDatapoint.from_smi(s, dummy, x_d=xd) for s, xd in zip(smiles_d, x_d_d)],
            "phys_p": phys_p, "phys_d": phys_d,
            "pka": float(r["pka"]),
            "weight": float(r["fidelity_weight"]) if has_fidelity_weight else 1.0,
        }

    def make_dataset(which):
        return [build_row(r) for r, s in zip(rows, splits) if s == which]

    train_rows = make_dataset("train")
    val_rows = make_dataset("val")
    test_rows = make_dataset("test")

    def n_microstates(rows_):
        return sum(len(r["dps_p"]) + len(r["dps_d"]) for r in rows_)

    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} rows (transitions)", file=sys.stderr)
    print(f"train microstates={n_microstates(train_rows)} val={n_microstates(val_rows)} test={n_microstates(test_rows)} "
          f"(vs. {2*len(train_rows)}/{2*len(val_rows)}/{2*len(test_rows)} if every row were a plain 1-vs-1 pair)", file=sys.stderr)
    if has_fidelity_weight:
        train_w = np.array([r["weight"] for r in train_rows])
        print(f"train fidelity_weight: mean={train_w.mean():.3f} min={train_w.min():.3f} max={train_w.max():.3f}", file=sys.stderr)

    x_d_mean, x_d_scale = None, None
    if has_extra_descriptors:
        # Standardized from TRAIN split only. Every microstate across
        # every row's own ensemble contributes (not just one representative
        # per row) -- more real data for the same standardization stats,
        # no leakage risk either way since it's still train-only.
        all_x_d = np.stack([dp.x_d for r in train_rows for dp in r["dps_p"] + r["dps_d"]])
        x_d_mean = all_x_d.mean(axis=0)
        x_d_scale = all_x_d.std(axis=0)
        x_d_scale[x_d_scale < 1e-6] = 1.0
        print(f"extra-descriptor train stats: mean={x_d_mean} scale={x_d_scale}", file=sys.stderr)

    def physical_only_metrics(rows_):
        if not rows_:
            return float("nan"), float("nan")
        errs = []
        for r in rows_:
            lse_p = logsumexp_np([-e for e in r["phys_p"]])
            lse_d = logsumexp_np([-e for e in r["phys_d"]])
            pka_phys = (lse_p - lse_d) / np.log(10.0)
            errs.append(pka_phys - r["pka"])
        errs = np.array(errs)
        return float(np.abs(errs).mean()), float(np.sqrt((errs ** 2).mean()))

    for name, rows_ in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        mae, rmse = physical_only_metrics(rows_)
        print(f"physical-baseline-only {name}: MAE={mae:.4f} RMSE={rmse:.4f} (no ML correction at all)", file=sys.stderr)

    mp = nn.BondMessagePassing(d_h=args.hidden_dim, depth=args.depth)
    agg = nn.NormAggregation()
    # No-op output transform (mean=0, scale=1): the chemprop network's raw
    # output IS the correction term, added directly to the (recalibrated)
    # physical baseline afterward -- no target shift inside the network.
    output_transform = UnscaleTransform(mean=np.array([0.0], dtype=np.float32), scale=np.array([1.0], dtype=np.float32))
    num_extra = 3 if has_extra_descriptors else 0
    ffn_input_dim = mp.output_dim + num_extra
    ffn = nn.RegressionFFN(input_dim=ffn_input_dim, hidden_dim=args.hidden_dim, n_layers=1, output_transform=output_transform)
    x_d_transform = ScaleTransform(mean=x_d_mean, scale=x_d_scale) if has_extra_descriptors else None
    model = models.MPNN(message_passing=mp, agg=agg, predictor=ffn, X_d_transform=x_d_transform)

    # Learned affine recalibration of the physical baseline:
    # g = physical_scale*physical_baseline + physical_offset + correction.
    # NEEDED, not optional -- this app's own SMIRNOFF force field energy is
    # arbitrary hand-tuned units, not real kcal/mol. Initialized at scale=0
    # (contributes nothing initially) -- training grows it if useful.
    model.physical_scale = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    model.physical_offset = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    if args.init_checkpoint:
        loaded = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "state_dict" in loaded and "hyper_parameters" in loaded:
            model.load_state_dict(loaded["state_dict"], strict=True)
            print(f"continuing training from full checkpoint {args.init_checkpoint}", file=sys.stderr)
        else:
            mp.load_state_dict(loaded)
            print(f"warm-started message-passing weights from {args.init_checkpoint}", file=sys.stderr)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6
    )

    def correction_for(model, datapoints):
        ds = data.MoleculeDataset(datapoints, featurizer=featurizer)
        loader = data.build_dataloader(ds, batch_size=len(datapoints), shuffle=False)
        bmg, V_d, X_d, *_ = next(iter(loader))
        bmg.to(device)  # BatchMolGraph.to() mutates in place and returns None -- do NOT reassign
        if X_d is not None:
            X_d = X_d.to(device)
        out = model(bmg, V_d, X_d)
        return out[:, 0]

    def run_epoch(rows_, train_mode, return_predictions=False):
        # Fidelity weighting applies ONLY when train_mode=True (i.e. only
        # ever called on the TRAIN rows during an actual training epoch) --
        # val/test are always effectively weight=1, since this function is
        # never called with train_mode=True on non-train rows.
        model.train(train_mode)
        total_loss, total_n = 0.0, 0
        all_preds, all_trues = [], []
        order = rng.permutation(len(rows_)) if train_mode else np.arange(len(rows_))
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            for start in range(0, len(rows_), args.batch_size):
                batch_rows = [rows_[i] for i in order[start:start + args.batch_size]]
                if not batch_rows:
                    continue
                # One flat forward pass over every microstate in the whole
                # batch (both sides, every row) -- efficient, and the ONLY
                # place a real GPU forward pass happens per batch. Index
                # bookkeeping below un-flattens it back per row/side for
                # the logsumexp combination.
                flat_dps, flat_phys, slices = [], [], []
                for row in batch_rows:
                    p0 = len(flat_dps)
                    flat_dps.extend(row["dps_p"]); flat_phys.extend(row["phys_p"])
                    p1 = len(flat_dps)
                    d0 = len(flat_dps)
                    flat_dps.extend(row["dps_d"]); flat_phys.extend(row["phys_d"])
                    d1 = len(flat_dps)
                    slices.append((p0, p1, d0, d1))

                corr = correction_for(model, flat_dps)
                phys_t = torch.tensor(flat_phys, dtype=torch.float32, device=device)
                g_all = model.physical_scale * phys_t + model.physical_offset + corr

                pka_preds = []
                for (p0, p1, d0, d1) in slices:
                    lse_p = torch.logsumexp(-g_all[p0:p1], dim=0)
                    lse_d = torch.logsumexp(-g_all[d0:d1], dim=0)
                    pka_preds.append((lse_p - lse_d) / np.log(10.0))
                pka_pred_t = torch.stack(pka_preds)
                pka_true_t = torch.tensor([row["pka"] for row in batch_rows], dtype=torch.float32, device=device)
                w_t = torch.tensor(
                    [row["weight"] for row in batch_rows] if train_mode else [1.0] * len(batch_rows),
                    dtype=torch.float32, device=device,
                )
                sq_err = (pka_pred_t - pka_true_t) ** 2
                loss = (w_t * sq_err).sum() / w_t.sum()
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    # Real, observed failure mode without this: an early
                    # batch containing a physical_baseline outlier (this
                    # raw feature spans roughly -1200 to +1900 across the
                    # whole corpus) combined with the not-yet-calibrated
                    # physical_scale parameter can blow every weight to
                    # NaN by epoch 0. Standard fix: clip the gradient norm.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                total_loss += loss.item() * len(batch_rows)
                total_n += len(batch_rows)
                if return_predictions:
                    all_preds.append(pka_pred_t.detach().cpu().numpy())
                    all_trues.append(pka_true_t.detach().cpu().numpy())
        mean_loss = total_loss / max(total_n, 1)
        if return_predictions:
            preds = np.concatenate(all_preds) if all_preds else np.array([])
            trues = np.concatenate(all_trues) if all_trues else np.array([])
            return mean_loss, preds, trues
        return mean_loss

    best_val = float("inf")
    best_state = None
    epochs_since_improve = 0
    for epoch in range(args.epochs):
        train_mse = run_epoch(train_rows, train_mode=True)
        val_mse = run_epoch(val_rows, train_mode=False) if val_rows else float("nan")
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  train_mse={train_mse:.4f} train_rmse={train_mse**0.5:.4f}  "
              f"val_mse={val_mse:.4f} val_rmse={val_mse**0.5:.4f}  lr={lr:.2e}", file=sys.stderr)
        if val_rows:
            scheduler.step(val_mse)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if args.early_stop_patience and val_rows and epochs_since_improve >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch} ({epochs_since_improve} epochs with no val improvement)", file=sys.stderr)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mse, test_preds, test_trues = run_epoch(test_rows, train_mode=False, return_predictions=True) if test_rows else (float("nan"), np.array([]), np.array([]))
    print(f"FINAL best_val_rmse={best_val**0.5:.4f} test_mse={test_mse:.4f} test_rmse={test_mse**0.5:.4f}", file=sys.stderr)

    phys_mae, phys_rmse = physical_only_metrics(test_rows)
    test_err = test_preds - test_trues
    test_mae = float(np.abs(test_err).mean()) if len(test_err) else float("nan")

    # Recalibrated-physical-ONLY (scale/offset applied, but no chemprop
    # correction at all) -- isolates how much of any improvement is just
    # fixing the units vs. the actual learned graph correction.
    recal_only_errs = []
    for r in test_rows:
        g_p = [model.physical_scale.item() * e + model.physical_offset.item() for e in r["phys_p"]]
        g_d = [model.physical_scale.item() * e + model.physical_offset.item() for e in r["phys_d"]]
        pka_recal = (logsumexp_np([-x for x in g_p]) - logsumexp_np([-x for x in g_d])) / np.log(10.0)
        recal_only_errs.append(pka_recal - r["pka"])
    recal_only_errs = np.array(recal_only_errs)
    recal_only_mae = float(np.abs(recal_only_errs).mean()) if len(recal_only_errs) else float("nan")

    print(f"physical-only (RAW, no recalibration) test: MAE={phys_mae:.4f} RMSE={phys_rmse:.4f}", file=sys.stderr)
    print(f"learned physical_scale={model.physical_scale.item():.6g} physical_offset={model.physical_offset.item():.4f}", file=sys.stderr)
    print(f"physical-only (RECALIBRATED, still no chemprop correction) test: MAE={recal_only_mae:.4f}", file=sys.stderr)
    print(f"physical+correction test: MAE={test_mae:.4f} RMSE={test_mse**0.5:.4f}", file=sys.stderr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "hyper_parameters": model.hparams,
        "output_columns": ["pka_microstate_correction"],
    }
    ckpt_path = out_dir / "best.pt"
    torch.save(ckpt, ckpt_path)
    print(f"wrote {ckpt_path}", file=sys.stderr)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps({
        "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows),
        "n_train_microstates": n_microstates(train_rows), "n_val_microstates": n_microstates(val_rows), "n_test_microstates": n_microstates(test_rows),
        "best_val_rmse": best_val ** 0.5, "test_mse": test_mse, "test_rmse": test_mse ** 0.5,
        "test_mae": test_mae,
        "physical_only_raw_test_mae": phys_mae, "physical_only_raw_test_rmse": phys_rmse,
        "physical_only_recalibrated_test_mae": recal_only_mae,
        "learned_physical_scale": model.physical_scale.item(), "learned_physical_offset": model.physical_offset.item(),
        "epochs_run": epoch + 1, "max_epochs": args.epochs, "hiddenDim": args.hidden_dim, "depth": args.depth, "seed": args.seed,
        "split": "scaffold_balanced (Murcko scaffold of the dominant protonated structure, via chemprop.data.splitting.make_split_indices)",
        "lr_patience": args.lr_patience, "early_stop_patience": args.early_stop_patience,
        "init_checkpoint": args.init_checkpoint,
        "has_fidelity_weight": has_fidelity_weight, "has_extra_descriptors": has_extra_descriptors,
        "extra_descriptor_mean": x_d_mean.tolist() if has_extra_descriptors else None,
        "extra_descriptor_scale": x_d_scale.tolist() if has_extra_descriptors else None,
        "architecture": "delta-learning (learned-affine-recalibrated physical baseline + additive Chemprop correction), "
                        "trained against the real macro-pKa microstate-ENSEMBLE formula (js/unipka-thermo.js's macroPKa), not a single pair"
                        + (" + X_d feature fusion (NAGL charge min/max, logP)" if has_extra_descriptors else "")
                        + (" + fidelity-weighted loss" if has_fidelity_weight else ""),
    }, indent=2))
    print(f"wrote {metrics_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
