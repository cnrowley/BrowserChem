#!/usr/bin/env python3
"""
prepare_pka_qm_pretrain_data.py

Builds real-QM auxiliary-pretraining CSVs for the pka-microstate-freeenergy
Chemprop D-MPNN from Nevolianis, Zheng, Mueller, Baumann, Tshepelevitsh,
Kaljurand, Leito, Smirnova, Green & Leonhard, "Solvation Free Energies of
Anions: From Curated Reference Data to Predictive Models," JACS 2025 (data
release: Zenodo 10.5281/zenodo.15604045, CC BY 4.0) -- the `data.zip` file
from that release, unzipped, is this script's `--nevolianis-dir` input.

--- Why this data, and how it's used ---

The app must stay strictly browser-only (no backend, no server-side QM at
inference time -- see CLAUDE.md), so none of this real quantum-chemistry
data can become a runtime input the way js/pka-physical-baseline.js's
client-side SMIRNOFF+GBSA baseline is. It's used the way delta/transfer
learning normally works instead: as OFFLINE PRETRAINING data that warm-
starts the same Chemprop D-MPNN weights (js/chemprop-model.js's own
architecture, unchanged) before scripts/train_pka_microstate_freeenergy.py's
existing paired thermodynamic-cycle finetuning stage runs on real
experimental aqueous pKa, exactly as documented in that script's own
header. This script only prepares data; it does not train anything itself.

Two auxiliary regression targets come out of this script, both real QM
(or QM-derived) quantities the current classical-force-field physical
baseline has no access to at all:

1. Gas-phase acidity (`--out-gas-phase-csv`): from
   `data/data_splits/D2A-dGgas-{train,test}.csv`. Each row's `smiles`
   column is already a `protonated_smiles>>deprotonated_smiles` reaction
   string (confirmed directly: charge(left) == charge(right)+1 for every
   one of 3796 real aqueous-solvent D2A-pKa.csv rows checked, so this is
   a reliable, uniform convention across the whole release, not just an
   assumption) -- split directly into this project's own
   smiles_protonated/smiles_deprotonated column convention, no site
   re-detection needed (the pair is already given, unlike this project's
   own IUPAC-source prep path in prepare_pka_microstate_training_data.py,
   which has to detect which atom a single neutral SMILES + pka_type
   refers to). A solvent-independent, purely electronic-structure signal
   -- the current physical baseline only ever sees a single (aqueous)
   solvent state.

2. Anion COSMO-RS solvation free energy (`--out-anion-solvation-csv`):
   from `data/DISSOLVE2-ANIONS.csv`, filtered to solvent_smiles == "O"
   (aqueous) and keyed on `dG_solv_ion_cosmo` specifically -- confirmed
   directly against this release's own README: `dG_solv_neutral` in that
   same file is NOT COSMO-RS (the paper's own "Composite Prediction
   Method" gets neutral solvation from Chung et al.'s separate DirectML
   model instead), so only the anion/ionic side is real COSMO-RS data in
   this release -- this script deliberately does NOT emit a neutral-
   solvation pretraining CSV to avoid mislabeling a different model's
   output as COSMO-RS. This is a single-molecule (not paired) regression
   target: real implicit-solvent electrostatics for a charged organic
   species, which is exactly the physics js/pka-physical-baseline.js's
   generic GB/SASA model was independently found to capture only weakly
   (Pearson r~0.03-0.05 standalone correlation with real pKa, see
   scripts/train_pka_microstate_freeenergy.py's own header).

A third output (`--out-ibond-pka-csv`) is NOT a pretraining CSV -- it's a
real experimental-data EXPANSION candidate: `data/D2A-pKa.csv`'s aqueous
subset is entirely sourced from the curated i-BonD database (confirmed
directly: every one of 3796 solvent_smiles=="O" rows has
original_source=="ibond", zero from the IUPAC side of that same file),
which this project's own prepare_pka_microstate_training_data.py could
never use directly -- i-BonD itself has no programmatic access (see that
script's own header for the IUPAC/Baltruschat sources it uses instead).
This release already did that curation for us, in the same
protonated>>deprotonated reaction-SMILES form. IMPORTANT LIMITATION,
disclosed not hidden: this output has NO physical_energy_protonated/
physical_energy_deprotonated columns, because generating those needs this
project's own offline SMIRNOFF+GBSA batch-scoring harness (js/pka-
physical-baseline.js's header describes it as "this file's own offline
counterpart, a headless-Node batch harness reusing these same three
engines unmodified" -- that harness is not present in this repo checkout,
only its output already baked into data/pka/pka_microstate_pairs_with_
physics.csv). This CSV is therefore NOT directly finetuning-ready; folding
it into the real training set needs that harness run on these new rows
first (site_cls here is a simple formal-charge-sign heuristic, not the
full SMARTS site-name classification the IUPAC path uses, so `site_name`
is honestly left as "ibond" rather than a specific pattern name).

--- Deduplication against the existing 7,006-pair corpus ---

Every emitted row (all three outputs) is checked against the InChIKey of
BOTH the protonated and deprotonated structure of every row already in
`--existing-pairs-csv` (data/pka/pka_microstate_pairs_with_physics.csv)
and dropped if either side matches. This is deliberately conservative
(auxiliary-task rows for a molecule that's only in the existing TRAIN
split would be harmless to pretrain on, but this script doesn't know
which split a future training run's scaffold-split seed will assign it
to) -- same reasoning as this project's other dedup passes
(add_opera_salicylate_pka_data.py's own header), erring toward provably
no leakage into whatever the eventual test set turns out to be, at the
cost of discarding some pretraining rows that would likely have been
fine to keep.

Usage:
    python3 scripts/prepare_pka_qm_pretrain_data.py \\
        --nevolianis-dir /path/to/unzipped/data \\
        --existing-pairs-csv data/pka/pka_microstate_pairs_with_physics.csv \\
        --out-gas-phase-csv data/pka/qm_gas_phase_acidity.csv \\
        --out-anion-solvation-csv data/pka/qm_anion_cosmo_solvation.csv \\
        --out-ibond-pka-csv data/pka/pka_ibond_aqueous_candidate.csv

Needs: rdkit.
"""

import argparse
import csv
import sys
from collections import defaultdict


def parse_pka_vals(s):
    """`pka_vals` is a numpy-array repr string (space-separated, no commas,
    e.g. '[3.99 4.07]' or '[6.5  6.65 6.7  6.76]') -- NOT valid Python/JSON,
    confirmed directly (ast.literal_eval fails on ~470 of the aqueous i-BonD
    rows otherwise, silently discarding real data). Split on any whitespace
    after stripping brackets rather than assuming comma-separation."""
    inner = s.strip().strip("[]")
    return [float(p) for p in inner.replace(",", " ").split()]

from rdkit import Chem
from rdkit.RDLogger import DisableLog

DisableLog("rdApp.*")


def inchikey_of(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def load_exclusion_inchikeys(existing_pairs_csv):
    """InChIKeys of every microstate (both protonated and deprotonated
    sides) already present in the existing training corpus -- see file
    header for why both sides, not just the source's own 'inchikey'
    column (which refers to a third, neutral-reference structure that's
    neither microstate)."""
    keys = set()
    with open(existing_pairs_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for col in ("smiles_protonated", "smiles_deprotonated"):
                ik = inchikey_of(row[col])
                if ik:
                    keys.add(ik)
    print(f"loaded {len(keys)} exclusion InChIKeys from {existing_pairs_csv}", file=sys.stderr)
    return keys


def prepare_gas_phase_acidity(nevolianis_dir, exclude, out_path):
    rows_out = []
    dropped = defaultdict(int)
    seen_pairs = set()
    for split in ("train", "test"):
        path = f"{nevolianis_dir}/data_splits/D2A-dGgas-{split}.csv"
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rxn = row["smiles"]
                if ">>" not in rxn:
                    dropped["not_a_reaction"] += 1
                    continue
                protonated, deprotonated = rxn.split(">>", 1)
                if (protonated, deprotonated) in seen_pairs:
                    dropped["duplicate_within_source"] += 1
                    continue
                seen_pairs.add((protonated, deprotonated))
                ik_p, ik_d = inchikey_of(protonated), inchikey_of(deprotonated)
                if ik_p is None or ik_d is None:
                    dropped["smiles_unparseable"] += 1
                    continue
                if ik_p in exclude or ik_d in exclude:
                    dropped["overlaps_existing_corpus"] += 1
                    continue
                try:
                    dg = float(row["value"])
                except ValueError:
                    dropped["value_unparseable"] += 1
                    continue
                rows_out.append({"smiles_protonated": protonated, "smiles_deprotonated": deprotonated, "dG_gas": dg})

    print(f"gas-phase acidity: kept {len(rows_out)} rows", file=sys.stderr)
    for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  dropped[{k}] = {v}", file=sys.stderr)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles_protonated", "smiles_deprotonated", "dG_gas"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {out_path}", file=sys.stderr)


def prepare_anion_solvation(nevolianis_dir, exclude, out_path):
    rows_out = []
    dropped = defaultdict(int)
    seen_anions = set()
    path = f"{nevolianis_dir}/DISSOLVE2-ANIONS.csv"
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["solvent_smiles"] != "O":
                dropped["non_aqueous"] += 1
                continue
            anion = row["anion_smiles"]
            if anion in seen_anions:
                dropped["duplicate_within_source"] += 1
                continue
            seen_anions.add(anion)
            ik = inchikey_of(anion)
            if ik is None:
                dropped["smiles_unparseable"] += 1
                continue
            if ik in exclude:
                dropped["overlaps_existing_corpus"] += 1
                continue
            try:
                dg = float(row["dG_solv_ion_cosmo"])
            except (ValueError, KeyError):
                dropped["value_unparseable"] += 1
                continue
            rows_out.append({"anion_smiles": anion, "dG_solv_ion_cosmo": dg})

    print(f"anion COSMO-RS solvation: kept {len(rows_out)} rows", file=sys.stderr)
    for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  dropped[{k}] = {v}", file=sys.stderr)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["anion_smiles", "dG_solv_ion_cosmo"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {out_path}", file=sys.stderr)


def prepare_ibond_pka(nevolianis_dir, exclude, out_path):
    rows_out = []
    dropped = defaultdict(int)
    path = f"{nevolianis_dir}/D2A-pKa.csv"
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["solvent_smiles"] != "O":
                dropped["non_aqueous"] += 1
                continue
            if row["original_source"] != "ibond":
                dropped["not_ibond"] += 1  # not expected to ever fire for solvent=="O", see file header
                continue
            protonated, deprotonated = row["reaction_smiles"].split(">>", 1)
            ik_p, ik_d = inchikey_of(protonated), inchikey_of(deprotonated)
            if ik_p is None or ik_d is None:
                dropped["smiles_unparseable"] += 1
                continue
            if ik_p in exclude or ik_d in exclude:
                dropped["overlaps_existing_corpus"] += 1
                continue
            mol_p = Chem.MolFromSmiles(protonated)
            # Charge-sign heuristic only (not a SMARTS site match) -- see
            # file header's disclosed limitation for this output.
            site_cls = "acid" if Chem.GetFormalCharge(mol_p) == 0 else "base"
            try:
                pka = float(row["pKa_avg"])
                n_measurements = len(parse_pka_vals(row["pka_vals"]))
            except ValueError:
                dropped["pka_unparseable"] += 1
                continue
            rows_out.append({
                "inchikey": ik_p, "site_name": "ibond", "site_cls": site_cls,
                "smiles_protonated": protonated, "smiles_deprotonated": deprotonated,
                "pka": pka, "n_measurements": n_measurements, "sources": "ibond_nevolianis2025",
            })

    print(f"i-BonD aqueous pKa (candidate, NO physical-baseline columns): kept {len(rows_out)} rows", file=sys.stderr)
    for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  dropped[{k}] = {v}", file=sys.stderr)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "inchikey", "site_name", "site_cls", "smiles_protonated",
            "smiles_deprotonated", "pka", "n_measurements", "sources",
        ])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nevolianis-dir", required=True, help="path to the unzipped Zenodo data.zip 'data/' directory")
    parser.add_argument("--existing-pairs-csv", required=True)
    parser.add_argument("--out-gas-phase-csv", required=True)
    parser.add_argument("--out-anion-solvation-csv", required=True)
    parser.add_argument("--out-ibond-pka-csv", required=True)
    args = parser.parse_args()

    exclude = load_exclusion_inchikeys(args.existing_pairs_csv)
    prepare_gas_phase_acidity(args.nevolianis_dir, exclude, args.out_gas_phase_csv)
    prepare_anion_solvation(args.nevolianis_dir, exclude, args.out_anion_solvation_csv)
    prepare_ibond_pka(args.nevolianis_dir, exclude, args.out_ibond_pka_csv)


if __name__ == "__main__":
    main()
