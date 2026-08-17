#!/usr/bin/env python3
"""
compute_druglikeness_distributions.py

Builds data/druglikeness_reference.json: reference distributions of
drug-likeness descriptors and rule-filter pass rates over real FDA-
approved-drug-proxy molecules, for the app's "Drug likeness" panel to
compare a drawn structure against (percentile rank per descriptor, plus
what fraction of real approved drugs pass each filter).

Input: properties/All_Small_Molecule_Drugs_Desalted.csv (not committed --
a ChEMBL small-molecule pull produced by properties/drug_scale.py:
molecule_type='Small molecule', max_phase>=1, largest-fragment-desalted,
with MW/LogP/HBD/HBA/TPSA already computed on the desalted parent). This
script filters to max_phase==4 -- ChEMBL's "approved in at least one
jurisdiction" flag, the standard proxy the field uses for "FDA-approved"
in the absence of an FDA-specific public dataset (this is the same
practical convention SwissADME's own "approved drugs" comparisons and
most published drug-likeness studies use) -- then computes the
ADDITIONAL descriptors needed for the 5 rule filters below that aren't
already in that CSV (rotatable bonds, Crippen MR, heavy atom count, ring
count, carbon count, heteroatom count).

IMPORTANT: computes these from each row's InChI column, NOT its smiles
column. Spot-checked directly: 1163/3311 (35%) of the approved rows have
a MULTI-FRAGMENT raw smiles (salts/counterions, e.g. carbachol chloride's
raw smiles is "C[N+](C)(C)CCOC(N)=O.[Cl-]") that drug_scale.py's own
MW/LogP/HBD/HBA/TPSA were correctly computed on the DESALTED parent for
(via RDKit's LargestFragmentChooser) -- but the CSV's own smiles column
was never overwritten with that desalted structure, only its InChI
column was. Computing heavy-atom/ring/rotatable-bond counts from the raw
(salted) smiles instead would silently include counterion atoms for over
a third of the reference set. The InChI column is always the correct
desalted parent (confirmed: InChI=='Error' count is 0 for the approved
subset).

--- Filter thresholds ---

Pulled directly from a real SwissADME results page (not from memory) --
the exact tooltip text for each filter, as SwissADME itself states it,
citing the original paper:

  Lipinski (Pfizer): MW<=500, MLOGP<=4.15, N-or-O<=10, NH-or-OH<=5
  Ghose:             160<=MW<=480, -0.4<=WLOGP<=5.6, 40<=MR<=130, 20<=atoms<=70
  Veber (GSK):        rotatable bonds<=10, TPSA<=140
  Egan (Pharmacia):   WLOGP<=5.88, TPSA<=131.6
  Muegge (Bayer):     200<=MW<=600, -2<=XLOGP<=5, TPSA<=150, rings<=7,
                       carbons>4, heteroatoms>1, rotatable bonds<=15,
                       HBA<=10, HBD<=5

This project has no WLOGP/XLOGP/MLOGP implementation (those are
different, separately-published fragment-contribution methods, not
approximated from memory here) -- every filter above that needs one of
them uses RDKit's real Crippen LogP instead, consistently, everywhere a
substitution is needed. This is a real, disclosed approximation, not a
silent one: expect occasional pass/fail disagreement with SwissADME's
own numbers specifically from this substitution, not from the threshold
values themselves (those are exact). Lipinski's own MW/HBD/HBA
thresholds are applied with Crippen LogP<=5 (the ORIGINAL 2001 "Rule of
Five" criterion -- MW/LogP/HBD/HBA all being 500/5/5/10) rather than
SwissADME's own MLOGP<=4.15 substitution, since that's the more
literally "Rule of Five" version and doesn't need MLOGP either way.

Martin's 2005 Bioavailability Score is NOT implemented here -- its exact
decision-tree logic isn't reliably available from memory or from what
SwissADME's own page discloses (it shows the resulting number, not the
rule), and this project's own convention is to skip a metric rather than
guess at its definition.

Usage:
    python3 compute_druglikeness_distributions.py \\
        [--input properties/All_Small_Molecule_Drugs_Desalted.csv] \\
        [--output data/druglikeness_reference.json]

Needs: pandas, rdkit.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

PROPERTY_KEYS = ["mw", "logP", "tpsa", "hbd", "hba", "mr", "rotatableBonds", "heavyAtoms"]
PROPERTY_DECIMALS = {"mw": 2, "logP": 3, "tpsa": 2, "hbd": 0, "hba": 0, "mr": 2, "rotatableBonds": 0, "heavyAtoms": 0}


def compute_descriptors(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    return {
        "mw": Descriptors.MolWt(mol),
        "logP": Descriptors.MolLogP(mol),  # Crippen
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": rdMolDescriptors.CalcNumLipinskiHBD(mol),
        "hba": rdMolDescriptors.CalcNumLipinskiHBA(mol),
        "mr": Descriptors.MolMR(mol),  # Crippen MR
        "rotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "heavyAtoms": mol.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "carbonCount": sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C"),
        "heteroatomCount": rdMolDescriptors.CalcNumHeteroatoms(mol),
    }


def evaluate_filters(d):
    lipinski = sum([d["mw"] > 500, d["logP"] > 5, d["hbd"] > 5, d["hba"] > 10])
    ghose = sum([
        not (160 <= d["mw"] <= 480),
        not (-0.4 <= d["logP"] <= 5.6),
        not (40 <= d["mr"] <= 130),
        not (20 <= d["heavyAtoms"] <= 70),
    ])
    veber = sum([d["rotatableBonds"] > 10, d["tpsa"] > 140])
    egan = sum([d["logP"] > 5.88, d["tpsa"] > 131.6])
    muegge = sum([
        not (200 <= d["mw"] <= 600),
        not (-2 <= d["logP"] <= 5),
        d["tpsa"] > 150,
        d["rings"] > 7,
        not (d["carbonCount"] > 4),
        not (d["heteroatomCount"] > 1),
        d["rotatableBonds"] > 15,
        d["hba"] > 10,
        d["hbd"] > 5,
    ])
    return {"lipinski": lipinski, "ghose": ghose, "veber": veber, "egan": egan, "muegge": muegge}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="properties/All_Small_Molecule_Drugs_Desalted.csv")
    ap.add_argument("--output", default="data/druglikeness_reference.json")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"{in_path} not found -- run properties/drug_scale.py first (needs a ChEMBL API pull).")

    df = pd.read_csv(in_path)
    if "max_phase" not in df.columns:
        sys.exit("Input CSV has no 'max_phase' column -- expected drug_scale.py's own output format.")

    approved = df[df["max_phase"] == 4.0].copy()
    print(f"{len(approved)} max_phase==4 (approved) rows out of {len(df)} total", file=sys.stderr)

    records = []
    n_bad_inchi = 0
    for inchi in approved["InChI"]:
        if not isinstance(inchi, str) or inchi == "Error" or not inchi.startswith("InChI="):
            n_bad_inchi += 1
            continue
        d = compute_descriptors(inchi)
        if d is None:
            n_bad_inchi += 1
            continue
        d["violations"] = evaluate_filters(d)
        records.append(d)

    print(f"{len(records)} molecules with usable InChI ({n_bad_inchi} dropped)", file=sys.stderr)
    if len(records) < 100:
        sys.exit("Too few usable reference molecules -- aborting rather than shipping a tiny/unreliable distribution.")

    distributions = {}
    for key in PROPERTY_KEYS:
        vals = sorted(r[key] for r in records)
        decimals = PROPERTY_DECIMALS[key]
        distributions[key] = [round(v, decimals) for v in vals]

    filter_pass_rates = {}
    for name in ["lipinski", "ghose", "veber", "egan", "muegge"]:
        passing = sum(1 for r in records if r["violations"][name] == 0)
        filter_pass_rates[name] = round(passing / len(records), 4)

    output = {
        "description": (
            "Reference distributions from ChEMBL max_phase==4 (approved) small molecules, "
            "largest-fragment-desalted, descriptors computed from each molecule's desalted-parent InChI "
            "via real RDKit (Crippen LogP/MR, Lipinski-convention HBD/HBA, TPSA, rotatable bonds, heavy "
            "atom count). Crippen LogP substitutes for WLOGP/XLOGP/MLOGP in the Ghose/Egan/Muegge filters "
            "below (this project has no separate implementation of those methods) -- filter thresholds "
            "themselves are exact, pulled directly from a real SwissADME results page, not from memory."
        ),
        "n": len(records),
        "properties": distributions,
        "filterPassRates": filter_pass_rates,
        "filterDefinitions": {
            "lipinski": "MW<=500, Crippen LogP<=5, HBD<=5, HBA<=10 (original 2001 Rule-of-Five criteria)",
            "ghose": "160<=MW<=480, -0.4<=Crippen LogP<=5.6, 40<=MR<=130, 20<=heavy atoms<=70",
            "veber": "rotatable bonds<=10, TPSA<=140",
            "egan": "Crippen LogP<=5.88, TPSA<=131.6",
            "muegge": "200<=MW<=600, -2<=Crippen LogP<=5, TPSA<=150, rings<=7, carbons>4, heteroatoms>1, rotatable bonds<=15, HBA<=10, HBD<=5",
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)", file=sys.stderr)
    print("filter pass rates among approved drugs:", filter_pass_rates, file=sys.stderr)


if __name__ == "__main__":
    main()
