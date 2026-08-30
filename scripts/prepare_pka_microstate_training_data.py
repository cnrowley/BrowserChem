#!/usr/bin/env python3
"""
prepare_pka_microstate_training_data.py

Builds the microstate-pair training CSV for the new physics+Chemprop
free-energy pKa model: one row per (protonated SMILES, deprotonated SMILES,
experimental pKa, class) triple, from two real public sources:

1. The IUPAC digitized pKa dataset (Zenodo 10.5281/zenodo.7236452,
   `iupac_high-confidence_v2_3.csv`, CC BY-NC 4.0) -- filtered to
   assessment in {Reliable, Approximate}, pka_type in {pKa1, pKaH1} (the
   monoprotic/primary-ionization case -- multiprotic pKa2+/pKaH2+ rows are
   dropped; they describe transitions between two already-ionized states,
   which this app's independent-site microstate framework doesn't yet
   model, same limitation the Uni-pKa integration had), aqueous only
   (empty `cosolvent`), near-standard temperature (15-35 C).

   The given SMILES is always the NEUTRAL/reference structure (confirmed
   directly from real examples + the dataset's own README: e.g.
   CC(=N)N/pKaH1=12.4 is the neutral amidine BASE, pKaH describing its
   conjugate acid's dissociation; CC(N)=O/pKa1=15.1 is the neutral amide
   ACID itself). `pka_type` (not `acidity_label`, which is redundant with
   it) decides direction: pKaH* rows -> the SMILES is a base (protonate it
   to build the "A" microstate, the SMILES itself is "B"); pKa* rows ->
   the SMILES is an acid (the SMILES itself is "A", deprotonate it to
   build "B").

2. This app's own already-prepared Baltruschat & Czodrowski data
   (data/pka/pka_prepared.csv, atom-target format -- already used to train
   the existing `aqueous-pka` chemprop checkpoint) -- each non-nan
   atom-target entry is one (smiles, atom_index, pka) triple; the labeled
   atom's own class (acid/base) is inferred from its element+environment
   directly via RDKit, and the microstate pair built by editing formal
   charge at that exact atom index (no re-detection needed -- the dataset
   already tells us which atom).

--- Site detection (IUPAC source only) ---

Ports js/pka-microstates.js's own validated 12-pattern SMARTS library
verbatim (same strings, same more-specific-first claim order) rather than
re-deriving a second copy -- see that file's header for the validation
history behind these patterns. A molecule is only used if the site class
implied by pka_type/acidity_label matches EXACTLY ONE detected site of
that class; zero or multiple matches means the site the literature pKa
actually refers to is ambiguous from the SMILES alone, so the row is
dropped rather than guessed at.

--- Microstate structure construction ---

Direct RDKit RWMol editing on the HEAVY-ATOM-ONLY (implicit-H) molecule --
just SetFormalCharge on the site atom, then Chem.SanitizeMol recomputes
implicit H count from that charge automatically. Deliberately never calls
Chem.AddHs() first: with explicit H atoms, a formal-charge edit alone
doesn't add/remove the actual H atom/bond (confirmed by hitting exactly
this failure mode first -- "Explicit valence ... greater than permitted"
on ~40% of rows -- before fixing it), leaving a real bonded H sitting on
a newly-charged atom with the wrong valence. This is the same
never-reimplement-RDKit's-own-charge/valence-model approach
CC.PKAMicrostates.buildMicrostateStructure already uses in JS (see that
file's own header) -- ported exactly, not reinvented.

--- Deduplication ---

Across both sources, by canonical neutral-structure InChIKey + site atom
map number (same site can appear in both the IUPAC set and Baltruschat's
own ChEMBL-derived data) -- multiple surviving measurements for the same
(molecule, site) are averaged, keeping the source-with-count and a note.

Usage:
    python3 scripts/prepare_pka_microstate_training_data.py \\
        <iupac_csv> <baltruschat_csv> <output_csv>
"""

import argparse
import csv
import sys
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.RDLogger import DisableLog

DisableLog("rdApp.*")

# Verbatim from js/pka-microstates.js's PATTERNS array (see that file's
# header for the validation history). atomIdx is the 0-based index into
# each SMARTS match tuple identifying the actual ionizable atom.
PATTERNS = [
    ("carboxylic_acid", "acid", "[CX3](=O)[OX2H1]", 2),
    ("sulfonic_acid", "acid", "[SX4](=O)(=O)[OX2H1]", 3),
    ("sulfinic_acid", "acid", "[SX3](=O)[OX2H1]", 2),
    ("phosphoric_OH", "acid", "[PX4](=O)[OX2H1]", 2),
    ("phenol", "acid", "[OX2H1][c]", 0),
    ("thiol", "acid", "[#6][SX2H1]", 1),
    ("tetrazole_NH", "acid", "[nH]1nnnc1", 0),
    ("guanidine", "base", "[NX3][CX3](=[NX2])[NX3]", 2),
    ("amidine", "base", "[NX3][CX3]=[NX2;!$(N-a)]", 2),
    ("pyridine_n", "base", "[nX2;H0;!+]", 0),
    ("aromatic_amine", "base",
     "[NX3;H2,H1,H0;!$(NC=O);!$(N[SX4](=O)(=O));!$(N[CX3]=[NX2]);$(N-a);!+]", 0),
    ("aliphatic_amine", "base",
     "[NX3;H2,H1,H0;!$(NC=O);!$(N=*);!$(N-a);!$(N[SX4](=O)(=O));!$(N[CX3]=[NX2]);!+]", 0),
]
COMPILED_PATTERNS = [(name, cls, Chem.MolFromSmarts(sm), idx) for name, cls, sm, idx in PATTERNS]


def find_sites(mol):
    """Returns [{'name','cls','atomIdx'}], more-specific-first, claimed
    atoms skipped by later/more-general patterns -- exact port of
    findIonizableSites's own dedup logic."""
    claimed = set()
    sites = []
    for name, cls, qmol, idx in COMPILED_PATTERNS:
        if qmol is None:
            continue
        for match in mol.GetSubstructMatches(qmol):
            atom_idx = match[idx]
            if atom_idx in claimed:
                continue
            claimed.add(atom_idx)
            sites.append({"name": name, "cls": cls, "atomIdx": atom_idx})
    return sites


def build_microstate_pair_by_site(mol, site_atom_idx, site_cls):
    """Returns (protonated_smiles, deprotonated_smiles) or (None, None) on
    failure. Protonated = the site atom bearing its extra H (acid's neutral
    -OH / base's cationic -NH+ form); deprotonated = the site atom missing
    that H (acid's anion / base's neutral free-base form)."""

    def edit(mol_in, protonated):
        rw = Chem.RWMol(mol_in)
        atom = rw.GetAtomWithIdx(site_atom_idx)
        if site_cls == "acid":
            atom.SetFormalCharge(0 if protonated else -1)
        else:
            atom.SetFormalCharge(1 if protonated else 0)
        try:
            m = rw.GetMol()
            Chem.SanitizeMol(m)
            return Chem.MolToSmiles(m)
        except Exception:
            return None

    return edit(mol, True), edit(mol, False)


def neutral_inchikey_and_map(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        ikey = rdMolDescriptors.CalcInchiKey(mol)
    except Exception:
        ikey = None
    return mol, ikey


def process_iupac(path):
    rows_out = []
    dropped = defaultdict(int)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["assessment"] not in ("Reliable", "Approximate"):
                dropped["assessment"] += 1
                continue
            if row["pka_type"] not in ("pKa1", "pKaH1"):
                dropped["pka_type"] += 1
                continue
            if row["cosolvent"].strip():
                dropped["cosolvent"] += 1
                continue
            try:
                t = float(row["T"])
            except (ValueError, KeyError):
                dropped["temperature_missing"] += 1
                continue
            if not (15.0 <= t <= 35.0):
                dropped["temperature_range"] += 1
                continue
            try:
                pka = float(row["pka_value"])
            except ValueError:
                dropped["pka_unparseable"] += 1
                continue

            mol, ikey = neutral_inchikey_and_map(row["SMILES"])
            if mol is None:
                dropped["smiles_unparseable"] += 1
                continue

            want_cls = "base" if row["pka_type"] == "pKaH1" else "acid"
            sites = [s for s in find_sites(mol) if s["cls"] == want_cls]
            if len(sites) != 1:
                dropped["ambiguous_or_no_site"] += 1
                continue
            site = sites[0]

            protonated, deprotonated = build_microstate_pair_by_site(mol, site["atomIdx"], site["cls"])
            if not protonated or not deprotonated:
                dropped["structure_build_failed"] += 1
                continue

            rows_out.append({
                "inchikey": ikey,
                "site_name": site["name"],
                "site_cls": site["cls"],
                "smiles_protonated": protonated,
                "smiles_deprotonated": deprotonated,
                "pka": pka,
                "source": "iupac",
            })
    return rows_out, dropped


def process_baltruschat(path):
    import ast
    rows_out = []
    dropped = defaultdict(int)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row["smiles"]
            try:
                targets = ast.literal_eval(row["pka"])
            except Exception:
                dropped["targets_unparseable"] += 1
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                dropped["smiles_unparseable"] += 1
                continue
            try:
                ikey = rdMolDescriptors.CalcInchiKey(mol)
            except Exception:
                ikey = None

            # Site class (acid/base) determined the SAME way as the IUPAC
            # path -- by matching the labeled atom against the validated
            # SMARTS site library, not by a cheap element+pKa-magnitude
            # guess. That guess was tried first and was a real bug: a
            # weakly-basic aromatic pyridine-type N (pKaH of its conjugate
            # acid legitimately anywhere in 0-7) was being misclassified
            # as an "acid" whenever its pKa happened to be < 7, then
            # charged -1 as if it had an N-H to remove -- it doesn't (it's
            # a lone-pair base), producing invalid unkekulizable anions on
            # ~1,400 rows before this fix.
            detected = {s["atomIdx"]: s["cls"] for s in find_sites(mol)}

            for atom_idx, val in enumerate(targets):
                if val == "nan" or val is None:
                    continue
                try:
                    pka = float(val)
                except (TypeError, ValueError):
                    continue
                if atom_idx >= mol.GetNumAtoms():
                    dropped["atom_index_out_of_range"] += 1
                    continue
                site_cls = detected.get(atom_idx)
                if site_cls is None:
                    dropped["site_not_covered_by_smarts_library"] += 1
                    continue

                protonated, deprotonated = build_microstate_pair_by_site(mol, atom_idx, site_cls)
                if not protonated or not deprotonated:
                    dropped["structure_build_failed"] += 1
                    continue

                rows_out.append({
                    "inchikey": ikey,
                    "site_name": "baltruschat_atom%d" % atom_idx,
                    "site_cls": site_cls,
                    "smiles_protonated": protonated,
                    "smiles_deprotonated": deprotonated,
                    "pka": pka,
                    "source": "baltruschat",
                })
    return rows_out, dropped


def dedupe(rows):
    groups = defaultdict(list)
    for r in rows:
        key = (r["inchikey"], r["smiles_protonated"], r["smiles_deprotonated"])
        groups[key].append(r)
    out = []
    for key, group in groups.items():
        pkas = [g["pka"] for g in group]
        mean_pka = sum(pkas) / len(pkas)
        sources = sorted(set(g["source"] for g in group))
        base = dict(group[0])
        del base["source"]
        base["pka"] = round(mean_pka, 3)
        base["n_measurements"] = len(group)
        base["sources"] = "+".join(sources)
        out.append(base)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("iupac_csv")
    parser.add_argument("baltruschat_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()

    iupac_rows, iupac_dropped = process_iupac(args.iupac_csv)
    print("IUPAC: kept %d rows" % len(iupac_rows), file=sys.stderr)
    for k, v in sorted(iupac_dropped.items(), key=lambda kv: -kv[1]):
        print("  dropped[%s] = %d" % (k, v), file=sys.stderr)

    balt_rows, balt_dropped = process_baltruschat(args.baltruschat_csv)
    print("Baltruschat: kept %d rows" % len(balt_rows), file=sys.stderr)
    for k, v in sorted(balt_dropped.items(), key=lambda kv: -kv[1]):
        print("  dropped[%s] = %d" % (k, v), file=sys.stderr)

    combined = dedupe(iupac_rows + balt_rows)
    print("Combined + deduped: %d unique (molecule, site) rows" % len(combined), file=sys.stderr)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "inchikey", "site_name", "site_cls", "smiles_protonated",
            "smiles_deprotonated", "pka", "n_measurements", "sources",
        ])
        writer.writeheader()
        for row in combined:
            writer.writerow(row)
    print("wrote %s" % args.output_csv, file=sys.stderr)


if __name__ == "__main__":
    main()
