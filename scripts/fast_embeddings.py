#!/usr/bin/env python3
"""
fast_embeddings.py

Computes pooled D-MPNN embeddings for every molecule in a training CSV
using the REAL PyTorch checkpoint, batched on GPU. Exists because
compute_applicability_domain.py's own pooled_embedding() -- a pure-numpy,
per-molecule, unvectorized-Python-loop reimplementation of the D-MPNN
forward pass -- is fine for this project's usual d_h=300/depth=3
checkpoints but becomes impractically slow for a large encoder (confirmed
in practice: 39+ minutes and still not finished on a 2500-molecule set
for a d_h=2048/depth=6 foundation-model-pretrained checkpoint, e.g. one
built via `--from-foundation CHEMELEON`). This script computes the exact
same math (model.fingerprint() = message_passing -> agg -> bn, identical
to what pooled_embedding()/runOneMolecule() compute) via chemprop's own
batched GPU tensor ops instead -- seconds instead of minutes, regardless
of encoder size.

Usage:
    fast_embeddings.py <checkpoint.pt> <training_csv> <out_csv>

Writes <out_csv>: smiles,e0,e1,...,e{d_h-1} -- smiles is RDKit-canonical
(matching compute_applicability_domain.py's own valid_smiles keys), one
row per molecule in <training_csv> that RDKit can parse. Feed <out_csv>
straight into compute_applicability_domain.py via its --embeddings-csv
flag to skip that script's slow forward-pass loop entirely.
"""
import argparse
import csv
import sys

import torch
from chemprop import data, models
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="a chemprop .pt checkpoint (e.g. model_0/best.pt from `chemprop train`)")
    parser.add_argument("training_csv", help="CSV with a 'smiles' column")
    parser.add_argument("out_csv")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", file=sys.stderr)
    model = models.MPNN.load_from_file(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    with open(args.training_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    smiles_list, canonical_list = [], []
    n_failed = 0
    for row in rows:
        smi = row["smiles"].strip()
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            n_failed += 1
            continue
        smiles_list.append(smi)
        canonical_list.append(Chem.MolToSmiles(mol))
    if n_failed:
        print(f"{n_failed}/{len(rows)} rows failed to parse -- excluded", file=sys.stderr)
    print(f"{len(smiles_list)} molecules", file=sys.stderr)

    datapoints = [data.MoleculeDatapoint.from_smi(smi) for smi in smiles_list]
    dataset = data.MoleculeDataset(datapoints)
    loader = data.build_dataloader(dataset, shuffle=False, batch_size=args.batch_size)

    all_emb = []
    with torch.no_grad():
        for batch in loader:
            bmg, V_d, X_d, *_ = batch
            bmg.to(device)  # BatchMolGraph.to() mutates in place, returns None
            if V_d is not None:
                V_d = V_d.to(device)
            emb = model.fingerprint(bmg, V_d, None)
            all_emb.append(emb.cpu())

    embeddings = torch.cat(all_emb, dim=0).numpy()
    print(f"embeddings shape: {embeddings.shape}", file=sys.stderr)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles"] + [f"e{i}" for i in range(embeddings.shape[1])])
        for smi, emb in zip(canonical_list, embeddings):
            w.writerow([smi] + emb.tolist())
    print(f"wrote {args.out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
