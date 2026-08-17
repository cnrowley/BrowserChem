#!/usr/bin/env python3
"""
prepare_qm9_training_data.py

Reshapes the raw QM9 dataset (Ramakrishnan et al. 2014, Scientific Data
1, 140022 -- 133,885 small organic molecules, up to 9 heavy atoms of
C/H/O/N/F, B3LYP/6-31G(2df,p)) into a chemprop-ready molecule-level CSV
with two target columns this app trains real checkpoints on:
`alpha` (isotropic polarizability) and `gap` (HOMO-LUMO gap).

--- Source ---

QM9's own SMILES+properties CSV, as redistributed by DeepChem/MoleculeNet
(https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv,
133,886 lines incl. header -- matches the paper's own molecule count).
19 columns of DFT-computed properties per molecule; only `alpha` and
`gap` are used here.

--- Units (confirmed against real chemistry, not assumed from the raw
column names alone) ---

QM9's OWN documented units are Bohr^3 for `alpha` and Hartree for
`homo`/`lumo`/`gap` -- confirmed directly here by checking two real
molecules against known literature values before trusting the columns:
  - Benzene (gdb_214, SMILES c1ccccc1): raw gap=0.2503 -> 0.2503 *
    27.211386245988 = 6.81 eV, matching the well-known ~6.8-7.0 eV
    B3LYP-level HOMO-LUMO gap reported for benzene in the literature.
    Raw alpha=57.28 -> 57.28 * 0.148185 (1 Bohr^3 in Angstrom^3,
    from CODATA's Bohr radius 0.529177210903 Angstrom, cubed) = 8.49 A^3,
    in the right range for benzene's isotropic polarizability
    (experimental ~10.3 A^3; B3LYP/6-31G(2df,p) is known to
    underestimate polarizability somewhat relative to experiment --
    that's an expected property of this level of theory, not a units
    bug here).
  - Methane (gdb_1, SMILES C): raw gap=0.5048 -> 13.74 eV, a large gap
    consistent with methane being a saturated, non-conjugated
    hydrocarbon -- chemically sensible relative to benzene's much
    smaller conjugated-system gap above.

This script converts BOTH columns into the units the app actually
displays -- alpha into Angstrom^3 (the standard cheminformatics-facing
unit; more intuitive than atomic units), gap into eV (matching HOMO-LUMO
gaps as normally quoted in chemistry, and directly usable in the browser
without needing a Hartree constant baked into the JS side) -- so the
trained checkpoint's own natural output units are already what the app
shows, no runtime physical-unit conversion needed beyond the eV -> nm
photon-energy relation the UI additionally displays alongside the gap
(E(eV) = 1239.84198 / lambda(nm) -- a real, standard physical relation,
not a QSPR fit; see js/app.js's own rendering code for the exact
constant used and its own caveat about what "gap in nm" does and doesn't
mean physically).

Usage:
    python3 prepare_qm9_training_data.py \\
        [--input data/qm9/qm9.csv] [--output data/qm9/qm9_prepared.csv]
"""

import argparse
import csv
import sys
from pathlib import Path

HARTREE_TO_EV = 27.211386245988  # CODATA 2018
BOHR_TO_ANGSTROM = 0.529177210903
BOHR3_TO_ANGSTROM3 = BOHR_TO_ANGSTROM ** 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/qm9/qm9.csv")
    ap.add_argument("--output", default="data/qm9/qm9_prepared.csv")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(
            f"{in_path} not found -- download it first:\n"
            f'  curl -sL -o {in_path} "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"'
        )

    n_in = 0
    n_out = 0
    rows_out = []
    with in_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_in += 1
            smiles = row["smiles"]
            try:
                alpha_bohr3 = float(row["alpha"])
                gap_hartree = float(row["gap"])
            except (KeyError, ValueError):
                continue
            if not smiles:
                continue
            rows_out.append({
                "smiles": smiles,
                "alpha": round(alpha_bohr3 * BOHR3_TO_ANGSTROM3, 4),
                "gap": round(gap_hartree * HARTREE_TO_EV, 4),
            })
            n_out += 1

    print(f"{n_out}/{n_in} rows kept", file=sys.stderr)
    if n_out < 1000:
        sys.exit("Too few usable rows -- aborting rather than training on a tiny/broken dataset.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "alpha", "gap"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
