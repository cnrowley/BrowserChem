#!/usr/bin/env python3
"""
compute_smarts_alert_features.py

Computes real medicinal-chemistry structural-alert features from this
app's own data/smarts_filters.json (~1250 SMARTS patterns across 8 rule
sets -- PAINS/Glaxo/Dundee/BMS/SureChEMBL/MLSMR/Inpharmatica/LINT, the
same file js/smarts-filters.js matches against for the app's own
structural-alert highlighting panel) for every molecule in a training
CSV -- offline data prep for testing whether these help Ames
mutagenicity prediction (real chemical rationale: several of these rule
sets, especially Glaxo/BMS, specifically flag reactive/unstable groups
-- the same broad chemistry class as Ames-positive mutagens, even though
none of these 8 rule sets is a purpose-built genotoxicity alert set the
way e.g. Kazius/Bursi's toxicophores are).

Pure RDKit substructure matching (`GetSubstructMatch`), the exact same
operation js/smarts-filters.js performs via RDKit.js -- both are
bindings to the same core RDKit C++ library, so this is not a parallel
reimplementation risk the way model inference would be (deterministic
substructure matching, not a trained model's forward pass).

Uses ALL ~1250 individual patterns as separate columns would be
impractical for a ~6500-row dataset (massive overfitting risk, most
patterns are drug-likeness/assay-interference flags with no plausible
Ames relevance) -- instead aggregates to one feature per rule set (does
this molecule match ANY pattern in that set?) plus a total distinct-
match count, 9 columns total (matching the 9-descriptor size of this
project's other ADMET X_d feature set for consistency, not because 9 is
special).

Usage:
    python3 compute_smarts_alert_features.py <in.csv> <out.csv>

Writes <out.csv>: smiles,alert_pains,alert_glaxo,alert_dundee,alert_bms,
alert_surechembl,alert_mlsmr,alert_inpharmatica,alert_lint,alert_total_count
(RDKit-canonical SMILES, binary 0/1 per rule-set column, integer count).
"""
import argparse
import csv
import json
import sys

from rdkit import Chem

RULE_SETS = ["PAINS", "Glaxo", "Dundee", "BMS", "SureChEMBL", "MLSMR", "Inpharmatica", "LINT"]


def load_patterns(path="data/smarts_filters.json"):
    d = json.load(open(path))
    compiled = []
    n_bad = 0
    for p in d["patterns"]:
        qmol = Chem.MolFromSmarts(p["smarts"])
        if qmol is None:
            n_bad += 1
            continue
        compiled.append((p["ruleSet"], qmol))
    print(f"compiled {len(compiled)}/{len(d['patterns'])} SMARTS patterns ({n_bad} failed to compile)", file=sys.stderr)
    return compiled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("in_csv")
    parser.add_argument("out_csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    patterns = load_patterns()

    with open(args.in_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"computing SMARTS alert features for {len(rows)} molecules from {args.in_csv}", file=sys.stderr)

    ok, failed = 0, 0
    with open(args.out_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["smiles"] + [f"alert_{rs.lower()}" for rs in RULE_SETS] + ["alert_total_count"])
        for i, row in enumerate(rows):
            smiles = row["smiles"]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError("RDKit could not parse SMILES")
                canonical = Chem.MolToSmiles(mol)
                by_ruleset = {rs: 0 for rs in RULE_SETS}
                total = 0
                for rule_set, qmol in patterns:
                    if mol.HasSubstructMatch(qmol):
                        by_ruleset[rule_set] = 1
                        total += 1
                writer.writerow([canonical] + [by_ruleset[rs] for rs in RULE_SETS] + [total])
                ok += 1
            except Exception as err:
                failed += 1
                print(f"row {i} FAILED ({smiles}): {err}", file=sys.stderr)
            if (i + 1) % 500 == 0 or i == len(rows) - 1:
                print(f"[{i + 1}/{len(rows)}] ok={ok} failed={failed}", file=sys.stderr)

    print(f"done. ok={ok} failed={failed}", file=sys.stderr)


if __name__ == "__main__":
    main()
