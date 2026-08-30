#!/usr/bin/env python3
"""
filter_zwitterion_pairs.py

Drops microstate-pair training rows where the "protonated"/"deprotonated"
structures are NOT a clean single-site pair -- the real invariant a valid
pair must satisfy: exactly one of the two structures has ZERO formally-
charged atoms (the genuinely neutral one) and the other has EXACTLY ONE
(the real acid/base-changed atom). A pair that fails this (both sides have
1+ charged atoms, e.g. a zwitterion -- NH3+ and COO- both present -- used
as the "neutral" 0-net-charge reference instead of the real all-neutral
tautomer) means MORE than one site's protonation state differs between
the two structures, which breaks this project's own micro-pKa formula
(CC.UniPKAThermo.microPKa / this project's training loop's
`pKa_pred = (g_deprotonated - g_protonated) / ln(10)`): that formula's
whole physical meaning assumes a clean single-proton-transfer pair, not a
multi-atom change.

Confirmed real and NOT a one-off: for a molecule with both an acid site
and a base site (amino-acid-like), the lowest-free-energy charge-0
structure is often the ZWITTERION (e.g. ammonium+carboxylate) rather than
the fully-neutral tautomer -- this affects every source in this project's
corpus to some degree (measured directly on the real data: 24% of pKahub
rows, ~16% of the i-BonD/Nevolianis rows, ~4% of the original IUPAC/
Baltruschat rows), not a bug specific to one prepare_*.py script's own
site-picking logic. For pKahub specifically, this also cross-contaminates
labels: an acid-site row ("0>>-1") and a base-site row ("1>>0") on the
SAME molecule can end up sharing that same zwitterion as their charge-0
reference, so the acid row's own "deprotonated" structure ends up
describing the AMMONIUM losing its proton (a base-type change) rather
than the carboxyl losing its proton the "acid" label implies.

Same "ambiguous means don't guess" convention this project already uses
elsewhere (prepare_pka_microstate_training_data.py's SMARTS-ambiguity
drop, prepare_pkahub_data.py's own multi-site-same-transition drop) --
there is no reliable way to reconstruct which specific atom SHOULD have
changed from the data available, so these rows are dropped rather than
kept with an unverified structural pair.

Usage:
    python3 scripts/filter_zwitterion_pairs.py <in.csv> <out.csv>

Works on any of this project's microstate-pair CSVs (with or without the
physical_energy_*/fidelity_weight/extra-descriptor columns -- every
column is passed through unchanged, only rows are dropped). Needs: rdkit.
"""

import argparse
import csv
import sys

from rdkit import Chem
from rdkit.RDLogger import DisableLog

DisableLog("rdApp.*")


def n_charged_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() != 0)


def is_clean_pair(row):
    np_ = n_charged_atoms(row["smiles_protonated"])
    nd_ = n_charged_atoms(row["smiles_deprotonated"])
    if np_ is None or nd_ is None:
        return False
    return sorted([np_, nd_]) == [0, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.input_csv, newline="", encoding="utf-8")))
    kept = [r for r in rows if is_clean_pair(r)]
    dropped = len(rows) - len(kept)
    print(f"{args.input_csv}: kept {len(kept)}, dropped {dropped} (not a clean 0-vs-1-charged-atom pair) of {len(rows)}", file=sys.stderr)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(kept)
    print(f"wrote {args.output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
