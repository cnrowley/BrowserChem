#!/usr/bin/env python3
"""
compute_applicability_domain.py

Computes, for one Chemprop molecule-level registry entry, the data this
app's applicability-domain gate and confidence tiering need at inference
time:

  1. TRAINING VOCABULARY -- the exact set of elements and formal charges
     actually seen in the training CSV. This is stricter than
     js/chemprop-features.js's existing checkChempropCompatibility(),
     which only checks against Chemprop's *global* one-hot featurizer
     vocabulary (~40 elements) and just warns. A molecule with an element
     the global featurizer supports but THIS model never saw in training
     (e.g. iodine, if this isoform's training set happened to contain
     none) still gets silently featurized into that vocabulary's "unknown
     element" pad bucket and produces a number with no support behind it.
     This script's output is what lets the app hard-refuse in that case
     instead of silently guessing. Per-atom formal charge alone isn't
     enough, either: a dataset built largely from zwitterions (a +1 and a
     -1 atom in the SAME, net-neutral molecule) makes both charge values
     look "seen" while a bare standalone ion (different NET molecular
     charge) has no real support at all -- confirmed to matter in
     practice on CombiSolv-QM (see solv-1-octanol-v1's registry notes),
     so net molecular charge is tracked and gated separately, with a
     minimum-occurrence-count filter (--min-net-charge-count) so a single
     one-off outlier molecule can't "unlock" a whole different class of
     query molecule.

  2. EMBEDDING-DOMAIN CONFIDENCE -- distance, in the trained D-MPNN's own
     pooled-molecule-embedding space (not raw fingerprint space), from a
     query molecule to the training distribution. This mirrors the
     Mahalanobis/k-NN latent-space OOD detection approach used in the
     broader deep-learning OOD literature (e.g. Lee et al. 2018) applied
     to what this specific network actually learned, rather than a
     generic chemical-descriptor applicability domain. Implementation:
     k-means over training-set pooled embeddings gives a small set of
     centroids (cheap to ship to the browser); nearest-centroid distance
     for every training molecule gives a self-consistency distribution,
     whose 90th/99th percentiles become the in-domain/borderline/
     out-of-domain tier thresholds.

WHAT THIS DELIBERATELY DOES NOT DO: produce a calibrated conformal-
prediction interval. Split-conformal calibration needs a calibration
set that is genuinely disjoint from what the shipped checkpoint was
fit on. For every chemprop model in this registry, `chemprop train`
performs its own internal SCAFFOLD_BALANCED split at training time and
that exact split is not saved anywhere (checkpoints/ and data/ are both
gitignored, and no split-indices file is written by chemprop train) --
so there is no way to reconstruct, after the fact, which rows were
actually held out. Re-splitting the training CSV ourselves here and
"holding out" 10% would silently evaluate the model on data it was very
likely already fit on, producing a falsely tight, overconfident interval
-- worse than having no interval at all, and a violation of this
project's documented honesty norm (see CLAUDE.md). Instead, the
already-real, already-honestly-computed held-out test-set metrics
(registry.json's `metrics.testMAE` / `metrics.testROC` / etc., produced
at training time from chemprop's own genuine internal test split) are
what the app should cite for "how big is a typical error", scoped by
the embedding-domain tier this script computes (small/typical error at
that scale if in-domain; explicitly unquantified if not).

Usage:
    python3 compute_applicability_domain.py <model-id> <training-csv> \\
        [--registry model/registry.json] [--smiles-column smiles] \\
        [--n-centroids 40] [--seed 0]

Writes model/<dir>/applicability-domain.json (dir taken from the
registry entry's files.manifest). Does NOT modify registry.json --
print the summary and hand-transcribe the notes/pointer, consistent
with registry.json being a hand-curated catalog (see CLAUDE.md).
"""

import argparse
import csv
import json
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np


class ChempropDMPNNModel:
    """Same tensor layout as compute_property_distributions.py's class of
    the same name, but exposes the pooled embedding directly instead of
    only the post-head prediction."""

    def __init__(self, manifest_path, weights_path):
        manifest = json.loads(Path(manifest_path).read_text())
        weights = np.fromfile(weights_path, dtype=np.float32)

        def tensor(name):
            t = manifest["tensors"][name]
            return weights[t["offset"]: t["offset"] + t["length"]].reshape(t["shape"])

        self.task = manifest.get("task", "prediction")
        self.task_type = manifest.get("taskType", "regression")
        self.dims = manifest["dims"]
        self.Wi = tensor("W_i")
        self.Wh = tensor("W_h")
        self.Wo = tensor("W_o_weight")
        self.Wo_bias = tensor("W_o_bias")


def pooled_embedding(mol, model, featurizer):
    """Mirrors compute_property_distributions.py's run_chemprop_molecule
    up through pooling (NormAggregation or MeanAggregation, per this
    model's own dims.aggregationType), stopping before the FFN head."""
    mg = featurizer(mol)
    num_atoms = mg.V.shape[0]
    hidden_size = model.dims["d_h"]
    depth = model.dims["depth"]

    if mg.E.shape[0] == 0:
        embeddings = [np.maximum(model.Wo @ np.concatenate([mg.V[v], np.zeros(hidden_size)]) + model.Wo_bias, 0)
                      for v in range(num_atoms)]
    else:
        edge_src = mg.edge_index[0]
        edge_dst = mg.edge_index[1]
        num_edges = len(edge_src)
        incoming_by_atom = [[] for _ in range(num_atoms)]
        for e, dst in enumerate(edge_dst):
            incoming_by_atom[dst].append(e)

        h0raw = np.stack([model.Wi @ np.concatenate([mg.V[edge_src[e]], mg.E[e]]) for e in range(num_edges)])
        h = np.maximum(h0raw, 0)

        for _ in range(1, depth):
            h_next = np.zeros_like(h)
            for e in range(num_edges):
                src_atom = edge_src[e]
                rev = mg.rev_edge_index[e]
                incoming = [e2 for e2 in incoming_by_atom[src_atom] if e2 != rev]
                message = h[incoming].sum(axis=0) if incoming else np.zeros(hidden_size)
                h_next[e] = np.maximum(h0raw[e] + model.Wh @ message, 0)
            h = h_next

        embeddings = []
        for v in range(num_atoms):
            incoming = incoming_by_atom[v]
            message = h[incoming].sum(axis=0) if incoming else np.zeros(hidden_size)
            embeddings.append(np.maximum(model.Wo @ np.concatenate([mg.V[v], message]) + model.Wo_bias, 0))

    if model.dims.get("aggregationType") == "mean":
        return np.mean(embeddings, axis=0)
    agg_norm = model.dims.get("aggNorm") or 1.0
    return np.sum(embeddings, axis=0) / agg_norm


def pairwise_dists(X, C):
    """Euclidean distance between every row of X (N,D) and every row of C
    (K,D), returned as an (N,K) array. Uses the ||x-c||^2 = ||x||^2 +
    ||c||^2 - 2*x.c^T expansion (one N*D-by-D*K matmul) rather than the
    more obvious `np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)`
    broadcast -- that broadcast briefly materializes a full (N, K, D)
    temp array, which is harmless at this project's usual d_h=300
    (N~19000, K~40 -> ~1.8GB) but a real, confirmed OOM kill (~27GB RSS,
    `dmesg`'s own oom-kill log) at CHEMELEON's d_h=2048 on the same N/K --
    caught computing AD for the cyp-herg-chemeleon-multitask models. This
    version's peak extra memory is O(N*K) for the output plus O(N*D) for
    one matmul operand, independent of how it's chunked internally."""
    X_sq = np.sum(X * X, axis=1)[:, None]
    C_sq = np.sum(C * C, axis=1)[None, :]
    sq_dists = X_sq + C_sq - 2.0 * (X @ C.T)
    np.maximum(sq_dists, 0, out=sq_dists)  # clip tiny negative values from float rounding
    return np.sqrt(sq_dists)


def kmeans(X, k, seed=0, n_restarts=3, max_iter=60):
    """Plain Lloyd's-algorithm k-means, numpy only (no sklearn dependency
    in this project's runtime scripts). Returns (centroids, best_inertia)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    best_centroids, best_inertia = None, np.inf

    for restart in range(n_restarts):
        centroids = X[rng.choice(n, size=k, replace=False)].copy()
        for _ in range(max_iter):
            dists = pairwise_dists(X, centroids)
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
        dists = pairwise_dists(X, centroids)
        inertia = np.min(dists, axis=1).sum()
        if inertia < best_inertia:
            best_centroids, best_inertia = centroids, inertia

    return best_centroids, best_inertia


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_id", help="registry.json model id, e.g. cyp1a2-substrate-v1")
    parser.add_argument("training_csv", help="the exact CSV chemprop train was pointed at for this model")
    parser.add_argument("--registry", default="model/registry.json")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--n-centroids", type=int, default=40)
    parser.add_argument("--max-embedding-molecules", type=int, default=4000,
                         help="cap on how many training molecules actually get a chemprop forward pass for "
                              "embedding-domain centroid computation (default: 4000, random subsample if the "
                              "training set is larger) -- the element/charge vocabulary is unaffected, it "
                              "always uses every molecule (that part is cheap, pure RDKit parsing)")
    parser.add_argument("--min-net-charge-count", type=int, default=5,
                         help="a net molecular charge value must occur at least this many times in training to "
                              "count as genuinely supported (default: 5) -- a single one-off outlier molecule "
                              "(e.g. one odd small charged fragment in an otherwise all-neutral dataset) "
                              "shouldn't be enough to wave through a whole different class of query molecule "
                              "(e.g. a common organic anion) that has no real representation in training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="default: model/<dir>/applicability-domain.json")
    parser.add_argument("--embeddings-csv", default=None,
                         help="skip this script's own pure-numpy D-MPNN forward pass (impractically slow "
                              "for a large encoder, e.g. the CHEMELEON foundation model's d_h=2048/depth=6 -- "
                              "39+ minutes and still not done on a 2500-molecule set, confirmed) and instead "
                              "read precomputed embeddings from this CSV (smiles,e0,e1,...,e{d_h-1}), keyed "
                              "by RDKit-canonical SMILES -- see fast_embeddings.py, which computes the exact "
                              "same math (model.fingerprint()) via chemprop's own batched GPU tensor ops in "
                              "seconds instead of minutes. Every training molecule's embedding is used (no "
                              "--max-embedding-molecules subsampling) since computing them all is already cheap.")
    args = parser.parse_args()

    from rdkit import Chem
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

    registry = json.loads(Path(args.registry).read_text())
    entry = next((m for m in registry["models"] if m["id"] == args.model_id), None)
    if entry is None:
        sys.exit(f"no model with id {args.model_id!r} in {args.registry}")
    engine = entry.get("engine", "chemprop")
    # "onnx-multitask" entries (a genuinely shared-encoder multi-task
    # checkpoint, see js/onnx-multitask-model.js) have no manifest.json/
    # weights.bin pair at all -- ChempropDMPNNModel can't load them, and
    # doesn't need to: with --embeddings-csv, `model` below is only ever
    # used for pooled_embedding()'s own from-scratch D-MPNN forward pass,
    # which is skipped entirely in that path (see fast_embeddings.py,
    # which computes the same embeddings via chemprop's own real GPU
    # tensor ops from the ORIGINAL .pt checkpoint instead). Element/charge
    # vocabulary is pure RDKit parsing of the training CSV either way,
    # completely engine-agnostic.
    if engine not in ("chemprop", "onnx-multitask"):
        sys.exit(f"{args.model_id} has engine {engine!r} -- this script only supports chemprop or "
                  "onnx-multitask molecule-level models")
    if engine == "onnx-multitask" and not args.embeddings_csv:
        sys.exit(f"{args.model_id} has engine 'onnx-multitask' -- this script can only compute its "
                  "embedding domain via --embeddings-csv (see fast_embeddings.py against the original "
                  ".pt checkpoint), since there's no manifest.json/weights.bin pair to build a "
                  "ChempropDMPNNModel from")
    if entry.get("outputLevel", "molecule") != "molecule":
        sys.exit(f"{args.model_id} is outputLevel={entry.get('outputLevel')!r} -- only molecule-level pooled embeddings are supported")

    if engine == "onnx-multitask":
        model_dir = Path(args.registry).parent / Path(entry["files"]["manifest"]).parent
        model = None
    else:
        model_dir = Path(args.registry).parent / Path(entry["files"]["manifest"]).parent
        manifest_path = Path(args.registry).parent / entry["files"]["manifest"]
        weights_path = Path(args.registry).parent / entry["files"]["weights"]
        print(f"loading {manifest_path}", file=sys.stderr)
        model = ChempropDMPNNModel(manifest_path, weights_path)
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    elements = set()
    charges = set()
    net_charge_counts = {}
    valid_mols = []
    valid_smiles = []
    n_total = 0
    n_failed = 0

    # Pass 1: pure-RDKit parsing over the FULL CSV -- cheap (no chemprop
    # forward pass), so the element/charge vocabulary reflects every
    # single training molecule, not a sample.
    with open(args.training_csv, newline="") as f:
        reader = csv.DictReader(f)
        if args.smiles_column not in (reader.fieldnames or []):
            sys.exit(f"{args.training_csv} has no column {args.smiles_column!r} -- columns: {reader.fieldnames}")
        for row in reader:
            smi = row[args.smiles_column].strip()
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
            # Per-atom formal-charge membership alone is NOT enough to gate
            # on: a training set can contain plenty of atoms with charge
            # -1 or +1 while every single molecule is still net-neutral
            # (e.g. zwitterionic amino acids -- a +1 ammonium paired with
            # a -1 carboxylate in the SAME molecule). A bare, standalone
            # anion (net charge -1) would pass a per-atom-only check even
            # though nothing resembling it was ever in training. Recording
            # net molecular charge separately is what actually catches
            # that case -- see CC.AD.checkVocab in
            # js/applicability-domain.js, which checks both. Counted (not
            # just seen), since one single one-off outlier molecule
            # shouldn't be enough to "unlock" a whole different class of
            # query molecule -- see --min-net-charge-count above.
            net_charge_counts[mol_net_charge] = net_charge_counts.get(mol_net_charge, 0) + 1
            valid_mols.append(mol)
            valid_smiles.append(Chem.MolToSmiles(mol))

    if n_failed:
        warnings.warn(f"{n_failed}/{n_total} training SMILES failed to parse -- excluded from vocab/embeddings")
    print(f"parsed {len(valid_mols)} training molecules", file=sys.stderr)
    print(f"elements: {sorted(elements)}", file=sys.stderr)
    print(f"per-atom formal charges: {sorted(charges)}", file=sys.stderr)
    print(f"net molecular charge counts: {dict(sorted(net_charge_counts.items()))}", file=sys.stderr)
    net_charges = sorted(c for c, n in net_charge_counts.items() if n >= args.min_net_charge_count)
    dropped = {c: n for c, n in net_charge_counts.items() if n < args.min_net_charge_count}
    if dropped:
        print(f"net charges dropped as one-off outliers (<{args.min_net_charge_count} occurrences): {dropped}", file=sys.stderr)
    print(f"net molecular charges counted as genuinely supported: {net_charges}", file=sys.stderr)

    # Pass 2: the actual chemprop D-MPNN forward pass, which IS expensive
    # (a real message-passing graph forward pass per molecule, in pure
    # Python) -- only run on a random subsample when the training set is
    # large. A few thousand molecules is already plenty to place 40
    # k-means centroids well; this is what keeps e.g. QM9's 130k-molecule
    # CSV or the CYP inhibition panel's 12k-molecule PubChem qHTS pull
    # tractable. The vocabulary above is NOT affected by this -- it
    # already used every molecule.
    if args.embeddings_csv:
        # Fast path -- read precomputed embeddings (fast_embeddings.py,
        # real chemprop batched GPU tensor ops) instead of this script's
        # own pure-numpy per-molecule loop. Uses every training molecule
        # (no subsampling) since computing them all this way is already
        # cheap regardless of dataset size.
        with open(args.embeddings_csv, newline="") as f:
            reader = csv.DictReader(f)
            emb_by_smiles = {}
            for row in reader:
                key = row["smiles"]
                emb_by_smiles[key] = [float(row[c]) for c in reader.fieldnames if c != "smiles"]
        sample_mols, X_rows = [], []
        n_missing = 0
        for mol, smi in zip(valid_mols, valid_smiles):
            vec = emb_by_smiles.get(smi)
            if vec is None:
                n_missing += 1
                continue
            sample_mols.append(mol)
            X_rows.append(vec)
        if n_missing:
            warnings.warn(f"{n_missing}/{len(valid_mols)} training molecules had no matching row in "
                           f"{args.embeddings_csv} (canonical-SMILES mismatch?) -- excluded from embedding domain")
        print(f"loaded {len(X_rows)} precomputed embeddings from {args.embeddings_csv}", file=sys.stderr)
        X = np.array(X_rows)
    else:
        # Slow path: the actual chemprop D-MPNN forward pass, which IS
        # expensive (a real message-passing graph forward pass per
        # molecule, in pure Python) -- only run on a random subsample
        # when the training set is large. A few thousand molecules is
        # already plenty to place 40 k-means centroids well; this is what
        # keeps e.g. QM9's 130k-molecule CSV or the CYP inhibition
        # panel's 12k-molecule PubChem qHTS pull tractable. The
        # vocabulary above is NOT affected by this -- it already used
        # every molecule. For a large encoder (e.g. a foundation-model-
        # pretrained checkpoint with d_h in the thousands), this loop is
        # impractically slow regardless of subsampling -- use
        # fast_embeddings.py + --embeddings-csv instead.
        rng = np.random.default_rng(args.seed)
        sample_mols = valid_mols
        if len(valid_mols) > args.max_embedding_molecules:
            idx = rng.choice(len(valid_mols), size=args.max_embedding_molecules, replace=False)
            sample_mols = [valid_mols[i] for i in idx]
            print(f"subsampled {len(sample_mols)}/{len(valid_mols)} molecules for embedding-domain computation "
                  f"(--max-embedding-molecules={args.max_embedding_molecules})", file=sys.stderr)
        embeddings = [pooled_embedding(mol, model, featurizer) for mol in sample_mols]
        X = np.array(embeddings)
    centroids, inertia = kmeans(X, args.n_centroids, seed=args.seed)
    print(f"k-means: {len(centroids)} centroids, inertia={inertia:.1f}", file=sys.stderr)

    dists_to_centroids = pairwise_dists(X, centroids)
    self_distances = np.min(dists_to_centroids, axis=1)
    p50, p90, p99 = np.percentile(self_distances, [50, 90, 99])
    print(f"self-distance percentiles: p50={p50:.3f} p90={p90:.3f} p99={p99:.3f}", file=sys.stderr)

    out = {
        "schemaVersion": 1,
        "modelId": args.model_id,
        "computedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "trainingSet": {
            "size": len(valid_mols),
            "sourceFile": str(args.training_csv),
            "smilesColumn": args.smiles_column,
            "elements": sorted(elements),
            "formalCharges": sorted(int(c) for c in charges),
            "netMolecularCharges": [int(c) for c in net_charges],
            "netMolecularChargeCounts": {str(int(c)): int(n) for c, n in sorted(net_charge_counts.items())},
            "minNetChargeCount": args.min_net_charge_count,
        },
        "embeddingDomain": {
            "dim": int(X.shape[1]),
            "nCentroids": int(len(centroids)),
            "centroidsFitFromNMolecules": len(sample_mols),
            "distanceMetric": "euclidean",
            "centroids": [[round(float(v), 5) for v in row] for row in centroids],
            "selfDistancePercentiles": {"p50": round(float(p50), 4), "p90": round(float(p90), 4), "p99": round(float(p99), 4)},
            "tierThresholds": {
                "inDomain": round(float(p90), 4),
                "borderline": round(float(p99), 4),
            },
            "notes": (
                "Distance from a query molecule's D-MPNN pooled embedding to its nearest "
                "training-set centroid. inDomain/borderline thresholds are the 90th/99th "
                "percentile of this same nearest-centroid distance computed self-consistently "
                "over the training set itself (so ~90% of training molecules fall inDomain by "
                "construction) -- a heuristic applicability-domain signal, not a calibrated "
                "statistical confidence interval. See compute_applicability_domain.py's module "
                "docstring for why conformal calibration isn't attempted here."
            ),
        },
    }

    out_path = Path(args.output) if args.output else model_dir / "applicability-domain.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
