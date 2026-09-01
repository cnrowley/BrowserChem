#!/usr/bin/env python3
"""
compute_cyp_sasa.py

Real 3D solvent-accessible surface area (RDKit's rdFreeSASA -- the
Shrake-Rupley algorithm, same one FreeSASA implements) for every molecule
already present in data/cyp/descriptors.csv (the 2D descriptor set from
scripts/pka-physical-baseline-harness/compute_cyp_descriptors.js), as the
CYP feature-exploration plan's Step 4 -- a total- and polar-SASA pair,
the standard 3D complement to the 2D logP/LogD/pKa/NAGL-charge set
already shown to help (see model/registry.json's cyp3a4-*-v1 entries'
notes once those get updated).

One ETKDG conformer per molecule (a single ETKDGv3 embed + MMFF94
optimization, `useRandomCoords=True` for robustness on larger/unusual
structures) -- deliberately NOT this app's own full conformer search
(js/embed3d.js's torsion-driven multi-conformer sampling): this is a
bulk offline pass over ~18.5k molecules, not a per-molecule accuracy-
critical 3D structure, and SASA from one reasonable low-energy conformer
is the standard cheap proxy used elsewhere in ADMET feature engineering
(this is explicitly an experiment to see whether it helps at all before
investing in anything more expensive).

Polar SASA sums the surface area of N/O/S/P heavy atoms and any H atom
attached to one of them (matches the standard TPSA-adjacent "polar
surface" convention, not this app's own atom-level charge heatmap).

Also computes three standard 3D shape/nonplanarity descriptors from the
SAME conformer (no second embedding pass): PBF (plane of best fit --
mean atom deviation from the best-fit plane, Firth et al. 2012's
"escape from flatland" nonplanarity metric), and NPR1/NPR2 (normalized
principal-moments-of-inertia ratios -- the standard rod/disc/sphere
molecular-shape triangle; a flat aromatic-heavy molecule sits near
NPR1~0, NPR2~1, a genuinely 3D/sp3-rich one moves toward the disc/sphere
corners). Real, mechanistically plausible for CYP450: the active-site
pocket is itself a specific 3D shape, so how well a molecule's own shape
fits it is a real steric/binding-relevant signal distinct from anything
the 2D descriptors (logP/LogD/pKa/NAGL charges) or graph topology alone
capture.

Usage:
    python3 compute_cyp_sasa.py data/cyp/descriptors.csv data/cyp/sasa.csv [--limit N]

Needs: rdkit (with rdFreeSASA -- confirmed present in the cov-chemprop
conda env; run via `conda run -n cov-chemprop python3 ...`).
"""
import argparse
import csv
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA, rdMolDescriptors

POLAR_ELEMENTS = {"N", "O", "S", "P"}


def compute_sasa(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES")
    mol = Chem.AddHs(mol)
    cid = AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE, useRandomCoords=True, maxAttempts=50)
    if cid < 0:
        raise ValueError("ETKDG embedding failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # a handful of MMFF-incompatible structures (e.g. hypervalent P/S) -- keep the embedded geometry as-is
    radii = rdFreeSASA.classifyAtoms(mol)
    total_sasa = rdFreeSASA.CalcSASA(mol, radii)
    polar_sasa = 0.0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        is_polar_heavy = symbol in POLAR_ELEMENTS
        is_polar_h = symbol == "H" and atom.GetDegree() == 1 and atom.GetNeighbors()[0].GetSymbol() in POLAR_ELEMENTS
        if is_polar_heavy or is_polar_h:
            polar_sasa += atom.GetDoubleProp("SASA")
    pbf = rdMolDescriptors.CalcPBF(mol)
    npr1 = rdMolDescriptors.CalcNPR1(mol)
    npr2 = rdMolDescriptors.CalcNPR2(mol)
    return total_sasa, polar_sasa, pbf, npr1, npr2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("in_csv")
    parser.add_argument("out_csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.in_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"computing SASA for {len(rows)} molecules from {args.in_csv}", file=sys.stderr)

    ok, failed = 0, 0
    with open(args.out_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["smiles", "total_sasa", "polar_sasa", "pbf", "npr1", "npr2"])
        for i, row in enumerate(rows):
            smiles = row["smiles"]
            try:
                total_sasa, polar_sasa, pbf, npr1, npr2 = compute_sasa(smiles)
                writer.writerow([smiles, total_sasa, polar_sasa, pbf, npr1, npr2])
                ok += 1
            except Exception as err:
                failed += 1
                print(f"row {i} FAILED ({smiles}): {err}", file=sys.stderr)
            if (i + 1) % 1000 == 0 or i == len(rows) - 1:
                print(f"[{i + 1}/{len(rows)}] ok={ok} failed={failed}", file=sys.stderr)

    print(f"done. ok={ok} failed={failed}", file=sys.stderr)


if __name__ == "__main__":
    main()
