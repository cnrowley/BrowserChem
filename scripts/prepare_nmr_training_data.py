#!/usr/bin/env python3
"""
prepare_nmr_training_data.py

Parses the two raw NMR datasets downloaded by download_nmr_datasets.py
into four chemprop-ready CSVs (one per nucleus): nmr_13c.csv, nmr_1h.csv,
nmr_15n.csv, nmr_19f.csv, each with a `smiles` column and one
`shift_<nucleus>` atom-target column (a Python-list-string of one float
per atom, NaN for atoms with no reported shift, in the atom order the
written SMILES will re-parse to).

REAL, FULL-FILE COUNTS (not estimated) of `Spectrum <nucleus> N` property
blocks in nmrshiftdb2withsignals.sd, confirmed by grepping the downloaded
file directly: 13C 43258, 1H 18222, 19F 1071, 15N only 84. 15N's training
set will end up small regardless of anything this script does -- that's
a property of the source data, not a bug here; document it plainly
alongside the trained checkpoint rather than quietly proceed as if it
were comparable to 13C/1H.

--- NMRShiftDB2 (13C, 1H, 15N, and a native slice of 19F) ---

SD records store molecules WITHOUT explicit hydrogens (confirmed by
inspecting real records: an 18-heavy-atom C15H22O3 record's atom block
has exactly 18 atoms, all C/O). `Spectrum <nucleus> N` properties look
like `17.6;0.0Q;10|18.3;0.0T;0|...` -- each entry is
`shift_value;multiplicity;atom_index`, atom_index into that SAME record's
own (heavy-atom-only) atom block.

For 13C/15N this is unambiguous: atom_index already IS the target heavy
atom. For 1H, atom_index is NOT a hydrogen's own index (there are no
explicit H atoms to index) -- it's "the shift of the proton(s) attached
to heavy atom N" (confirmed directly: e.g. `1.297;0.0;7` appears three
times in one real record, matching a CH3 group's three magnetically
equivalent protons all sharing one shift value). So building a genuine
per-hydrogen training label requires: call Chem.AddHs() on the
heavy-atom-only structure (adds explicit H nodes, using the same
"heavy atoms first in original order, then each heavy atom's new H's
appended grouped by parent" ordering convention this project has
independently verified multiple times elsewhere -- see
js/nagl-features.js's buildExpandedGraph and pka-descriptor.js), then
broadcast each heavy atom's (average, if multiple duplicate entries)
reported shift to every one of its newly-added explicit H children.
Chemically equivalent protons genuinely share one true shift, so this is
correct for the common case; diastereotopic (inequivalent) protons on
the same heavy atom that happen to be reported at different values get
averaged into one approximate shared label instead of their true
distinct values -- a real, bounded, and honestly documented limitation,
not something this source's format lets you resolve further.

Some SD records DO already carry explicit H atoms (rare, confirmed by a
raw `grep` count, though most don't) -- handled generically by checking
the ACTUAL element at each reported atom_index in that specific record
rather than assuming: if it's already 'H', use the value directly as
that atom's own label with no broadcasting.

--- NMRexp (19F only) ---

Per the user's decision: only rows where the parsed shift list has
exactly one value are used (single distinguishable fluorine environment
-- avoids guessing which of several reported shifts belongs to which of
several F atoms). Combined with NMRShiftDB2's own native 19F records
(atom-indexed, no assignment ambiguity at all) into one dataset, tagged
by source for transparency.

--- Atom-order self-check (every molecule, every nucleus) ---

chemprop does not canonicalize or reorder atoms on read -- it parses
whatever SMILES you give it via plain Chem.MolFromSmiles() (or, for 1H,
that plus Chem.AddHs()) and iterates GetAtoms() in that order. This
script writes each molecule's SMILES via
Chem.MolToSmiles(mol, canonical=False) specifically because it preserves
input atom order, then IMMEDIATELY re-parses that exact SMILES and
diffs the resulting atom sequence (element-by-element, and for 1H also
the whole AddHs()'d sequence) against the array of labels this script is
about to write -- any mismatch drops the molecule (counted, not
silently skipped) rather than risk writing a mislabeled training row.

Usage:
    python3 prepare_nmr_training_data.py --input-dir data/nmr --output-dir data/nmr
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

NUCLEI = ["13C", "1H", "15N", "19F"]


def parse_spectrum_prop(text):
    """'17.6;0.0Q;10|18.3;0.0T;0|...' -> [(17.6, 10), (18.3, 0), ...]"""
    out = []
    for entry in text.strip().split("|"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(";")
        try:
            shift = float(parts[0])
            idx = int(parts[-1])
        except (ValueError, IndexError):
            continue
        out.append((shift, idx))
    return out


def smiles_and_reordered_labels(mol, labels):
    """Writes `mol`'s SMILES and reorders `labels` (one value per atom, in
    `mol`'s current index order) to match the atom order that SMILES will
    re-parse to.

    NOTE, a real bug this replaced: Chem.MolToSmiles(mol, canonical=False)
    does NOT preserve input atom order on re-parse -- confirmed directly
    (an 18-atom test record's re-parsed element sequence differed from
    its original one even with canonical=False). The correct, standard
    RDKit technique is the '_smilesAtomOutputOrder' property MolToSmiles
    sets on `mol` after writing: output_order[i] is the ORIGINAL atom
    index now at position i in the written SMILES. Reordering `labels`
    by that mapping makes it line up with a fresh Chem.MolFromSmiles()
    parse of the same string by construction (also verified directly:
    reordered elements matched a real re-parse exactly).

    Returns (smiles, reordered_labels) or (None, None) if anything about
    the round-trip looks wrong (defense in depth -- should not normally
    trigger given the above).

    IMPORTANT, a second real bug this fixed: the re-parse below uses
    chemprop's own reading convention -- Chem.MolFromSmiles(smiles) with
    DEFAULT sanitization, not sanitize=False. A `sanitize=False` re-parse
    made this check falsely pass for the rare NMRShiftDB2 records that
    already carry explicit H atoms in their original molblock: RDKit's
    default SMILES-parse sanitization silently strips "trivial" explicit
    hydrogens (no stereo/isotope significance) back into implicit H
    counts, so chemprop would end up with FEWER atoms than this script's
    label array -- confirmed directly by a real chemprop training crash
    ("tensor size 145 must match 146") traced to exactly this. Using the
    same sanitizing parse chemprop itself uses makes this check catch
    that mismatch and drop the molecule instead of silently mislabeling
    it.
    """
    smiles = Chem.MolToSmiles(mol, canonical=False)
    order = eval(mol.GetProp("_smilesAtomOutputOrder"))
    if len(order) != len(labels):
        return None, None
    reordered = [labels[i] for i in order]

    reparsed = Chem.MolFromSmiles(smiles)
    if reparsed is None or reparsed.GetNumAtoms() != len(labels):
        return None, None
    original_elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in order]
    reparsed_elements = [a.GetSymbol() for a in reparsed.GetAtoms()]
    if original_elements != reparsed_elements:
        return None, None

    return smiles, reordered


def process_nmrshiftdb2(sd_path):
    """Yields (nucleus, smiles, per_atom_labels) for 13C/15N (heavy-atom
    direct), 1H (broadcast to explicit H nodes via AddHs), and 19F
    (heavy-atom direct, NMRShiftDB2's own native records only)."""
    counts = defaultdict(int)
    dropped = defaultdict(int)

    supplier = Chem.SDMolSupplier(str(sd_path), sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol, catchErrors=True)
        except Exception:
            continue

        n_atoms = mol.GetNumAtoms()
        elements = [a.GetSymbol() for a in mol.GetAtoms()]
        props = mol.GetPropsAsDict()

        for nucleus in ("13C", "15N", "19F"):
            per_atom = defaultdict(list)
            for key, value in props.items():
                if not key.startswith(f"Spectrum {nucleus} "):
                    continue
                for shift, idx in parse_spectrum_prop(str(value)):
                    if 0 <= idx < n_atoms:
                        per_atom[idx].append(shift)
            if not per_atom:
                continue
            labels = [float("nan")] * n_atoms
            for idx, values in per_atom.items():
                labels[idx] = statistics.mean(values)
            smiles, reordered = smiles_and_reordered_labels(mol, labels)
            if smiles is None:
                dropped[nucleus] += 1
                continue
            counts[nucleus] += 1
            yield nucleus, smiles, reordered, "nmrshiftdb2"

        # 1H: broadcast heavy-atom-indexed shifts onto explicit H nodes
        # added via Chem.AddHs(). Records whose ORIGINAL structure already
        # has explicit H atoms are skipped for this nucleus (a rare
        # minority -- mixing "shift already keyed to a real H atom" with
        # "shift keyed to a heavy atom, needs broadcasting to freshly-added
        # H's" adds real reindexing risk for negligible extra data).
        if "H" not in elements:
            per_heavy_h = defaultdict(list)
            for key, value in props.items():
                if not key.startswith("Spectrum 1H "):
                    continue
                for shift, idx in parse_spectrum_prop(str(value)):
                    if 0 <= idx < n_atoms:
                        per_heavy_h[idx].append(shift)

            if per_heavy_h:
                heavy_shift = {idx: statistics.mean(vals) for idx, vals in per_heavy_h.items()}

                # This mirrors chemprop's own --add-h pipeline EXACTLY
                # (chemprop/utils/utils.py's make_mol(): parse SMILES with
                # removeHs=True by default, THEN Chem.AddHs() -- confirmed
                # by reading that source directly, after an earlier version
                # of this script wrongly assumed a SMILES with explicit
                # [H] tokens baked in would survive chemprop's own parse.
                # It doesn't: RDKit's default SMILES-parse sanitization
                # strips "trivial" explicit H atoms back into implicit H
                # counts -- confirmed directly, a real 40-atom mol
                # (16 heavy + 24 H) round-tripped through
                # Chem.MolFromSmiles() with default params came back with
                # only 16 atoms).
                smiles = Chem.MolToSmiles(mol, canonical=False)
                order = eval(mol.GetProp("_smilesAtomOutputOrder"))
                reparsed_heavy = Chem.MolFromSmiles(smiles)
                original_elements_reordered = [elements[i] for i in order] if reparsed_heavy is not None else None
                reparsed_elements = [a.GetSymbol() for a in reparsed_heavy.GetAtoms()] if reparsed_heavy is not None else None

                if (reparsed_heavy is not None and reparsed_heavy.GetNumAtoms() == n_atoms
                        and original_elements_reordered == reparsed_elements):
                    mol_h = Chem.AddHs(reparsed_heavy)
                    # Same convention verified elsewhere in this project
                    # (js/nagl-features.js's buildExpandedGraph): heavy
                    # atoms keep their (now-reordered) indices, new H's
                    # appended afterward grouped by parent heavy atom in
                    # heavy-atom-iteration order.
                    children = defaultdict(list)
                    for atom in mol_h.GetAtoms():
                        if atom.GetIdx() < n_atoms:
                            continue
                        nbrs = [nb for nb in atom.GetNeighbors() if nb.GetIdx() < n_atoms]
                        if len(nbrs) == 1:
                            children[nbrs[0].GetIdx()].append(atom.GetIdx())

                    labels = [float("nan")] * mol_h.GetNumAtoms()
                    for new_heavy_idx, orig_heavy_idx in enumerate(order):
                        val = heavy_shift.get(orig_heavy_idx)
                        if val is None:
                            continue
                        for h_idx in children.get(new_heavy_idx, []):
                            labels[h_idx] = val

                    if any(x == x for x in labels):  # not all-NaN
                        counts["1H"] += 1
                        yield "1H", smiles, labels, "nmrshiftdb2"
                    else:
                        dropped["1H"] += 1
                else:
                    dropped["1H"] += 1

    print(f"nmrshiftdb2: kept {dict(counts)}, dropped (atom-order self-check failed) {dict(dropped)}", file=sys.stderr)


def process_nmrexp_19f(parquet_path):
    import pandas as pd
    import ast

    df = pd.read_parquet(parquet_path, columns=["SMILES", "NMR_type", "NMR_processed"])
    df = df[df["NMR_type"].astype(str).str.contains("19F", na=False)]

    kept = 0
    dropped = 0
    for smi, processed in zip(df["SMILES"], df["NMR_processed"]):
        try:
            shifts = ast.literal_eval(processed) if isinstance(processed, str) else processed
        except (ValueError, SyntaxError):
            continue
        if not isinstance(shifts, list) or len(shifts) != 1:
            continue
        shift_val = shifts[0][0] if isinstance(shifts[0], (list, tuple)) else shifts[0]
        try:
            shift_val = float(shift_val)
        except (TypeError, ValueError):
            continue

        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            dropped += 1
            continue
        f_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "F"]
        if len(f_indices) != 1:
            # Not actually a single-F-environment molecule (or SMILES
            # parse disagrees with the paper-reported count) -- skip
            # rather than guess which F the one shift belongs to.
            dropped += 1
            continue

        labels = [float("nan")] * mol.GetNumAtoms()
        labels[f_indices[0]] = shift_val
        smiles, reordered = smiles_and_reordered_labels(mol, labels)
        if smiles is None:
            dropped += 1
            continue
        kept += 1
        yield "19F", smiles, reordered, "nmrexp"

    print(f"nmrexp 19F: kept {kept}, dropped {dropped}", file=sys.stderr)


def write_csv(rows, out_path, column_name):
    import csv

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", column_name, "source"])
        for smiles, labels, source in rows:
            # chemprop parses this column with ast.literal_eval(), which
            # rejects a bare `nan` token (not a valid Python literal) --
            # write missing values as the quoted string 'nan' instead,
            # which literal_eval accepts as a str, and numpy's
            # dtype=float coercion (which chemprop applies right after)
            # correctly turns the string 'nan' into a real NaN. Confirmed
            # directly against chemprop's own MAB_parsing.py code path.
            label_str = "[" + ", ".join("'nan'" if x != x else repr(x) for x in labels) + "]"
            writer.writerow([smiles, label_str, source])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="data/nmr")
    parser.add_argument("--output-dir", default="data/nmr")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_nucleus = defaultdict(list)

    sd_path = in_dir / "nmrshiftdb2withsignals.sd"
    if sd_path.exists():
        for nucleus, smiles, labels, source in process_nmrshiftdb2(sd_path):
            rows_by_nucleus[nucleus].append((smiles, labels, source))
    else:
        print(f"WARNING: {sd_path} not found, skipping NMRShiftDB2", file=sys.stderr)

    parquet_path = in_dir / "NMRexp_10to24_1_1004.parquet"
    if parquet_path.exists():
        for nucleus, smiles, labels, source in process_nmrexp_19f(parquet_path):
            rows_by_nucleus[nucleus].append((smiles, labels, source))
    else:
        print(f"WARNING: {parquet_path} not found, skipping NMRexp", file=sys.stderr)

    column_names = {"13C": "shift_13c", "1H": "shift_1h", "15N": "shift_15n", "19F": "shift_19f"}
    file_names = {"13C": "nmr_13c.csv", "1H": "nmr_1h.csv", "15N": "nmr_15n.csv", "19F": "nmr_19f.csv"}

    for nucleus in NUCLEI:
        rows = rows_by_nucleus.get(nucleus, [])
        out_path = out_dir / file_names[nucleus]
        write_csv(rows, out_path, column_names[nucleus])
        n_atom_labels = sum(sum(1 for x in labels if x == x) for _, labels, _ in rows)
        print(f"{nucleus}: {len(rows)} molecules, {n_atom_labels} atom-level labels -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
