#!/usr/bin/env python3
"""
compute_pka_applicability_domain.py

Builds model/pka-microstate-freeenergy/applicability-domain.json --
the same two things scripts/compute_applicability_domain.py computes for
every other chemprop model in this registry (see that script's own
header for the full rationale), adapted for this model's two real
architectural differences that script can't handle:

  1. Its training CSV has paired smiles_protonated/smiles_deprotonated
     columns (one row = one acid/base transition), not a single `smiles`
     column -- so the vocabulary pass here (pure RDKit, no chemprop
     forward pass, same as the generic script) walks BOTH columns,
     treating every individual microstate as its own training example
     for vocabulary purposes (which is what actually happens at
     inference time: js/pka-freeenergy-predict.js's microstateFreeEnergy
     calls CC.GNN.predictChemprop once per side, each subject to
     CC.AD.checkVocab independently).

  2. It uses real X_d feature fusion (numExtraDescriptors=3), which the
     generic script's pure-numpy D-MPNN reimplementation doesn't support
     at all. Embedding-domain data is NOT computed here in Python --
     scripts/pka-physical-baseline-harness/compute_pka_embeddings.js
     does that first, via the REAL js/chemprop-model.js forward pass
     (CC.GNN.getPooledEmbedding, which deliberately returns the
     PRE-X_d-fusion pooled vector -- exactly what CC.AD.tierForEmbedding
     compares against at real prediction time, X_d never affects it).
     This script only consumes that JS script's NDJSON output for the
     k-means/percentile half -- same numpy Lloyd's-algorithm k-means as
     scripts/compute_applicability_domain.py, copied rather than
     imported (that script is a `__main__`-only CLI, not a library).

Usage:
    # first (from the harness dir, needs a running local server):
    CC_BASE_URL=http://localhost:8000/ node \\
      pka-physical-baseline-harness/compute_pka_embeddings.js \\
      data/pka/pka_microstate_pairs_final_v3_features.csv \\
      /tmp/pka_embeddings.ndjson --max-rows 4000

    # then:
    python3 scripts/compute_pka_applicability_domain.py \\
      data/pka/pka_microstate_pairs_final_v3_features.csv \\
      /tmp/pka_embeddings.ndjson

Writes model/pka-microstate-freeenergy/applicability-domain.json.
"""

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def kmeans(X, k, seed=0, n_restarts=3, max_iter=60):
    """Plain Lloyd's-algorithm k-means, numpy only -- identical to
    scripts/compute_applicability_domain.py's own kmeans() (copied, not
    imported: that script is a __main__-only CLI)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    best_centroids, best_inertia = None, np.inf

    for restart in range(n_restarts):
        centroids = X[rng.choice(n, size=k, replace=False)].copy()
        for _ in range(max_iter):
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)
            new_centroids = centroids.copy()
            for c in range(k):
                members = X[assignments == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        inertia = np.min(dists, axis=1).sum()
        if inertia < best_inertia:
            best_centroids, best_inertia = centroids, inertia

    return best_centroids, best_inertia


def compute_vocab(training_csv, min_net_charge_count):
    from rdkit import Chem

    elements = set()
    charges = set()
    net_charge_counts = {}
    n_total = 0
    n_failed = 0

    with open(training_csv, newline="") as f:
        reader = csv.DictReader(f)
        for col in ("smiles_protonated", "smiles_deprotonated"):
            if col not in (reader.fieldnames or []):
                sys.exit(f"{training_csv} has no column {col!r} -- columns: {reader.fieldnames}")
        for row in reader:
            for col in ("smiles_protonated", "smiles_deprotonated"):
                smi = row[col].strip()
                if not smi:
                    continue
                n_total += 1
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    n_failed += 1
                    continue
                mol_net_charge = 0
                for atom in mol.GetAtoms():
                    elements.add(atom.GetSymbol())
                    charges.add(atom.GetFormalCharge())
                    mol_net_charge += atom.GetFormalCharge()
                net_charge_counts[mol_net_charge] = net_charge_counts.get(mol_net_charge, 0) + 1

    if n_failed:
        warnings.warn(f"{n_failed}/{n_total} training microstate SMILES failed to parse -- excluded from vocab")
    print(f"parsed {n_total - n_failed} training microstates (both sides of every row)", file=sys.stderr)
    print(f"elements: {sorted(elements)}", file=sys.stderr)
    print(f"per-atom formal charges: {sorted(charges)}", file=sys.stderr)
    print(f"net molecular charge counts: {dict(sorted(net_charge_counts.items()))}", file=sys.stderr)
    net_charges = sorted(c for c, n in net_charge_counts.items() if n >= min_net_charge_count)
    dropped = {c: n for c, n in net_charge_counts.items() if n < min_net_charge_count}
    if dropped:
        print(f"net charges dropped as one-off outliers (<{min_net_charge_count} occurrences): {dropped}", file=sys.stderr)
    print(f"net molecular charges counted as genuinely supported: {net_charges}", file=sys.stderr)

    return {
        "size": n_total - n_failed,
        "elements": sorted(elements),
        "formalCharges": sorted(int(c) for c in charges),
        "netMolecularCharges": [int(c) for c in net_charges],
        "netMolecularChargeCounts": {str(int(c)): int(n) for c, n in sorted(net_charge_counts.items())},
        "minNetChargeCount": min_net_charge_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("training_csv", help="data/pka/pka_microstate_pairs_final_v3_features.csv (the real v10 training input)")
    parser.add_argument("embeddings_ndjson", help="output of compute_pka_embeddings.js")
    parser.add_argument("--registry", default="model/registry.json")
    parser.add_argument("--n-centroids", type=int, default=40)
    parser.add_argument("--min-net-charge-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="default: model/pka-microstate-freeenergy/applicability-domain.json")
    args = parser.parse_args()

    registry = json.loads(Path(args.registry).read_text())
    entry = next((m for m in registry["models"] if m["id"] == "pka-microstate-freeenergy"), None)
    if entry is None:
        sys.exit(f"no model with id 'pka-microstate-freeenergy' in {args.registry}")
    model_dir = Path(args.registry).parent / Path(entry["files"]["manifest"]).parent

    vocab = compute_vocab(args.training_csv, args.min_net_charge_count)

    embeddings, smiles_list = [], []
    with open(args.embeddings_ndjson) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            embeddings.append(rec["embedding"])
            smiles_list.append(rec["smiles"])
    X = np.array(embeddings, dtype=np.float64)
    print(f"loaded {X.shape[0]} embeddings (dim={X.shape[1]}) from {args.embeddings_ndjson}", file=sys.stderr)

    centroids, inertia = kmeans(X, args.n_centroids, seed=args.seed)
    print(f"k-means: {len(centroids)} centroids, inertia={inertia:.1f}", file=sys.stderr)

    dists_to_centroids = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    self_distances = np.min(dists_to_centroids, axis=1)
    p50, p90, p99 = np.percentile(self_distances, [50, 90, 99])
    print(f"self-distance percentiles: p50={p50:.3f} p90={p90:.3f} p99={p99:.3f}", file=sys.stderr)

    out = {
        "schemaVersion": 1,
        "modelId": "pka-microstate-freeenergy",
        "computedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "trainingSet": {
            **vocab,
            "sourceFile": str(args.training_csv),
            "note": (
                "Unlike this registry's other applicability-domain sidecars, `size` and the vocabulary "
                "counts here are over MICROSTATES (both smiles_protonated and smiles_deprotonated of "
                "every training row), not molecules -- this model's own CC.AD.checkVocab call "
                "(js/pka-freeenergy-predict.js) runs once per microstate, matching this."
            ),
        },
        "embeddingDomain": {
            "dim": int(X.shape[1]),
            "nCentroids": int(len(centroids)),
            "centroidsFitFromNMicrostates": int(X.shape[0]),
            "distanceMetric": "euclidean",
            "centroids": [[round(float(v), 5) for v in row] for row in centroids],
            "selfDistancePercentiles": {"p50": round(float(p50), 4), "p90": round(float(p90), 4), "p99": round(float(p99), 4)},
            "tierThresholds": {
                "inDomain": round(float(p90), 4),
                "borderline": round(float(p99), 4),
            },
            "notes": (
                "Distance from a query MICROSTATE's pre-X_d-fusion pooled D-MPNN embedding "
                "(CC.GNN.getPooledEmbedding, js/chemprop-model.js) to its nearest training-set "
                "centroid -- computed via scripts/pka-physical-baseline-harness/compute_pka_embeddings.js's "
                "real forward pass through the deployed checkpoint, not a numpy reimplementation. "
                "inDomain/borderline thresholds are the 90th/99th percentile of this same nearest-centroid "
                "distance computed self-consistently over a 4000-row (8000-microstate) random subsample of "
                "training. js/pka-freeenergy-predict.js reports the WORSE (higher-distance tier) of a "
                "site's own protonated/deprotonated microstate tiers as that site's confidence, since a "
                "site's pKa is only as trustworthy as its least-in-domain half. Same heuristic-not-calibrated "
                "caveat as every other applicability-domain sidecar in this registry -- see "
                "scripts/compute_applicability_domain.py's own module docstring."
            ),
        },
    }

    out_path = Path(args.output) if args.output else model_dir / "applicability-domain.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
