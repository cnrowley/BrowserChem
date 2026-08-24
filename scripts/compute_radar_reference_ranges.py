#!/usr/bin/env python3
"""
compute_radar_reference_ranges.py

One-off tool: derives real min/max ranges for js/radar-chart.js's
CC.RADAR_AXES from the same real FDA-approved-drug-proxy population
scripts/compute_druglikeness_distributions.py already uses (ChEMBL
max_phase==4, largest-fragment-desalted, computed from each molecule's
desalted-parent InChI -- same reasoning/caveats as that script's own
header: this is the standard field-wide proxy for "FDA-approved" absent
an FDA-specific public dataset).

Prints ready-to-paste [min, max] pairs (5th/95th percentile of the real
distribution, trimming the extreme 10% combined tails so a handful of
outliers -- e.g. a handful of very large or very small approved-drug
entries -- don't wash out the visual scale for the typical case) for
every radar axis RDKit can compute directly: mw, logP, tpsa, hbd, hba,
fsp3, rotb, qed, rings, aromaticFraction, heteroFraction, complexity,
saScore. Does NOT cover mp (melting point) or logS (aqueous solubility)
-- ChEMBL's small-molecule pull used here has no reliable per-molecule
melting-point/solubility data, and this project's own melting-point-v1/
logs-aqsoldb-v1 GNN checkpoints aren't practical to batch-run from a
plain Python script (they're this project's own hand-rolled JS D-MPNN
format, not a standard ONNX/PyTorch artifact retained after conversion).
Those two axes are left as pre-existing hand-picked heuristic ranges,
undisturbed by this script -- see radar-chart.js's own comment for that
honest caveat.

SA score uses RDKit's real Contrib/SA_Score/sascorer.py (bundled with
the rdkit pip package) -- the exact reference implementation
js/sascorer.js was itself validated against, per this project's own
CHEMPROP_INTEGRATION.md-style honesty convention.

Usage:
    python3 compute_radar_reference_ranges.py \\
        [--input properties/All_Small_Molecule_Drugs_Desalted.csv] \\
        [--low 5] [--high 95]

Needs: pandas, rdkit.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

# RDKit ships Contrib/SA_Score/sascorer.py alongside the main package but
# doesn't put it on the default import path -- add it explicitly, same as
# anyone using the real reference implementation has to.
import rdkit
_sa_score_dir = Path(rdkit.__file__).parent / "Contrib" / "SA_Score"
sys.path.insert(0, str(_sa_score_dir))
import sascorer  # noqa: E402


def compute_descriptors(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return None
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    # NumAtomStereoCenters alone (defined/assigned only) -- matches
    # RDKit.js's get_descriptors() field of the same name, which
    # js/chemistry.js's own DESCRIPTOR_FIELDS labels "Stereocenters
    # (defined)" and js/radar-chart.js's `complexity` proxy reads
    # directly, NOT the defined+unspecified total.
    stereocenters = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    return {
        # amw (RDKit.js's get_descriptors() field name) = standard average
        # molecular weight = Descriptors.MolWt, NOT CalcExactMolWt
        # (monoisotopic) -- matches both chemistry.js's own `amw` field and
        # compute_druglikeness_distributions.py's existing convention.
        "mw": Descriptors.MolWt(mol),
        # Matches CrippenClogP (chemistry.js) and compute_druglikeness_
        # distributions.py's existing "logP" field -- same Crippen model.
        "logP": Descriptors.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        # Plain NumHBD/NumHBA (RDKit.js's get_descriptors() field names),
        # NOT the Lipinski-specific variant compute_druglikeness_
        # distributions.py deliberately uses for its Rule-of-Five filter --
        # a different, real RDKit definition, matched here to what the
        # radar chart itself actually reads.
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "qed": QED.qed(mol),
        "rings": rings,
        "aromaticFraction": aromatic_atoms / heavy,
        "heteroFraction": heteroatoms / heavy,
        "complexity": (rings + stereocenters + heteroatoms) / heavy,
        "saScore": sascorer.calculateScore(mol),
    }


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="properties/All_Small_Molecule_Drugs_Desalted.csv")
    ap.add_argument("--low", type=float, default=5.0)
    ap.add_argument("--high", type=float, default=95.0)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"{in_path} not found -- run properties/drug_scale.py first (needs a ChEMBL API pull).")

    df = pd.read_csv(in_path)
    approved = df[df["max_phase"] == 4.0].copy()
    print(f"{len(approved)} max_phase==4 (approved) rows out of {len(df)} total", file=sys.stderr)

    keys = ["mw", "logP", "tpsa", "hbd", "hba", "fsp3", "rotb", "qed", "rings",
            "aromaticFraction", "heteroFraction", "complexity", "saScore"]
    records = []
    n_bad = 0
    for inchi in approved["InChI"]:
        if not isinstance(inchi, str) or inchi == "Error" or not inchi.startswith("InChI="):
            n_bad += 1
            continue
        d = compute_descriptors(inchi)
        if d is None:
            n_bad += 1
            continue
        records.append(d)

    print(f"{len(records)} molecules with usable InChI ({n_bad} dropped)", file=sys.stderr)
    if len(records) < 100:
        sys.exit("Too few usable reference molecules -- aborting.")

    print(f"\nn={len(records)}, percentile band [{args.low}, {args.high}]\n")
    for key in keys:
        vals = sorted(r[key] for r in records)
        lo = percentile(vals, args.low)
        hi = percentile(vals, args.high)
        decimals = 0 if key in ("hbd", "hba", "rotb", "rings") else 3
        print(f"{key:18s} min={round(lo, decimals):<10} max={round(hi, decimals):<10} "
              f"(true min={round(vals[0], decimals)}, true max={round(vals[-1], decimals)})")


if __name__ == "__main__":
    main()
