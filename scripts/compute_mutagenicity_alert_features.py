#!/usr/bin/env python3
"""
compute_mutagenicity_alert_features.py

Real, purpose-built genotoxicity/mutagenicity structural alerts (the
Benigni/Bossa rulebase, data/mutagenicity_alerts_benigni_bossa.json --
see that file's own `citation`/`note` for exact provenance: reproduced
from Ferrari & Gini 2010's open-access supplementary material, itself
citing the official EU JRC Benigni/Bossa rulebase, a Toxtree module) --
NOT this project's own general medchem PAINS/Glaxo/BMS/etc. filters
(data/smarts_filters.json), which a real prior experiment already showed
does NOT help Ames mutagenicity prediction (see model/registry.json's
ames-mutagenicity-v1 entry). These 9 alerts are causally-relevant DNA-
reactive/genotoxic mechanisms (epoxides, quinones, hydrazines, aliphatic
azo/azoxy, N-nitroso, acyl halides, alkyl carbamates, azide/triazene) --
a much better mechanistic match, worth testing separately rather than
folding into the already-rejected generic filter experiment.

Computes BOTH representations for comparison:
  - Molecule-level X_d: one binary column per alert (9 total) -- does
    the molecule match ANY of that alert's SMARTS variants?
  - Atom-level V_f (chemprop's real --atom-features-path, true pre-
    message-passing fusion -- see scripts/build_ch_bde_npz.py for the
    npz-format precedent this reuses): one binary channel per heavy
    atom -- is this atom part of ANY alert match at all (a single
    "flagged" channel, not one channel per alert -- keeping V_f width
    small given this dataset's size, same reasoning
    compute_ch_bde_atom_features.js used for its 2-channel BDE feature).

Usage:
    python3 compute_mutagenicity_alert_features.py <in.csv> <out_xd.csv> <out_atomfeat.ndjson>

Writes:
  <out_xd.csv>: smiles,alert_sa_1,alert_sa_6,...,alert_sa_22 (9 binary columns)
  <out_atomfeat.ndjson>: one {"ok": true, "features": [[flag], ...]} per
    input row, SAME ORDER as <in.csv> -- consumed by build_ch_bde_npz.py-
    style tooling to build chemprop's atom_features_0.npz (reuses that
    exact script, see below).
"""
import argparse
import csv
import json
import sys

from rdkit import Chem

ALERTS_PATH = "data/mutagenicity_alerts_benigni_bossa.json"


def load_alerts(path=ALERTS_PATH):
    d = json.load(open(path))
    compiled = []
    for alert in d["alerts"]:
        qmols = []
        for smarts in alert["smarts"]:
            qmol = Chem.MolFromSmarts(smarts)
            if qmol is None:
                raise SystemExit(f"failed to compile SMARTS for {alert['id']}: {smarts!r}")
            qmols.append(qmol)
        compiled.append((alert["id"], qmols))
    ring_fusion_alerts = d.get("ring_fusion_alerts", [])
    print(f"loaded {len(compiled)} SMARTS alerts + {len(ring_fusion_alerts)} ring-fusion alerts from {path}", file=sys.stderr)
    return compiled, ring_fusion_alerts


def fused_aromatic_ring_systems(mol):
    """Real ring-fusion detection for SA_18/SA_19 (polycyclic aromatic
    systems, 3+ fused rings) -- the Benigni/Bossa source gives only a
    prose definition for these two, no SMARTS, since 'three or more
    fused rings' is a global topological property SMARTS substructure
    matching can't directly express. Two SSSR rings count as fused if
    they share a bond (>=2 atoms); connected components of that
    fusion graph with >=3 aromatic rings are flagged. Returns a list of
    (atom_index_set, is_heteroaromatic) per qualifying fused system."""
    ri = mol.GetRingInfo()
    aromatic_rings = [set(r) for r in ri.AtomRings() if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)]
    n = len(aromatic_rings)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if len(aromatic_rings[i] & aromatic_rings[j]) >= 2:
                union(i, j)

    systems = {}
    for i in range(n):
        systems.setdefault(find(i), []).append(i)

    results = []
    for comp in systems.values():
        if len(comp) < 3:
            continue
        atoms = set()
        for idx in comp:
            atoms |= aromatic_rings[idx]
        is_hetero = any(mol.GetAtomWithIdx(a).GetSymbol() != "C" for a in atoms)
        results.append((atoms, is_hetero))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("in_csv")
    parser.add_argument("out_xd_csv")
    parser.add_argument("out_atomfeat_ndjson")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    alerts, ring_fusion_alerts = load_alerts()
    alert_ids = [a[0] for a in alerts] + [a["id"] for a in ring_fusion_alerts]

    with open(args.in_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"computing mutagenicity alert features for {len(rows)} molecules from {args.in_csv}", file=sys.stderr)

    ok, failed = 0, 0
    with open(args.out_xd_csv, "w", newline="") as f_xd, open(args.out_atomfeat_ndjson, "w") as f_atom:
        writer = csv.writer(f_xd)
        writer.writerow(["smiles"] + [f"alert_{aid.lower()}" for aid in alert_ids])
        for i, row in enumerate(rows):
            smiles = row["smiles"]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError("RDKit could not parse SMILES")
                canonical = Chem.MolToSmiles(mol)
                n_atoms = mol.GetNumAtoms()
                # Several of these SMARTS reference explicit hydrogens
                # ([#1]) -- confirmed live that RDKit's default substruct
                # matching against the heavy-atom-only mol misses these
                # (e.g. SA_13 hydrazine on phenylhydrazine) since #1 only
                # matches a real graph node, not an implicit H. AddHs()
                # preserves original heavy-atom indices (new H atoms are
                # appended after), so atom_idx < n_atoms below still maps
                # 1:1 onto the original heavy-atom-only mol.
                mol_h = Chem.AddHs(mol)
                atom_flagged = [0] * n_atoms
                mol_flags = []
                for aid, qmols in alerts:
                    matched = False
                    for qmol in qmols:
                        for match in mol_h.GetSubstructMatches(qmol):
                            matched = True
                            for atom_idx in match:
                                if atom_idx < n_atoms:
                                    atom_flagged[atom_idx] = 1
                    mol_flags.append(1 if matched else 0)

                ring_systems = fused_aromatic_ring_systems(mol) if ring_fusion_alerts else []
                for ring_alert in ring_fusion_alerts:
                    want_hetero = ring_alert["heteroaromatic"]
                    matched = False
                    for atoms, is_hetero in ring_systems:
                        if is_hetero == want_hetero:
                            matched = True
                            for atom_idx in atoms:
                                atom_flagged[atom_idx] = 1
                    mol_flags.append(1 if matched else 0)

                writer.writerow([canonical] + mol_flags)
                f_atom.write(json.dumps({"ok": True, "features": [[v] for v in atom_flagged]}) + "\n")
                ok += 1
            except Exception as err:
                failed += 1
                f_atom.write(json.dumps({"ok": False, "error": str(err)}) + "\n")
                print(f"row {i} FAILED ({smiles}): {err}", file=sys.stderr)
            if (i + 1) % 1000 == 0 or i == len(rows) - 1:
                print(f"[{i + 1}/{len(rows)}] ok={ok} failed={failed}", file=sys.stderr)

    print(f"done. ok={ok} failed={failed}", file=sys.stderr)


if __name__ == "__main__":
    main()
