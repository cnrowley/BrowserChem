#!/usr/bin/env python3
"""
prepare_pkahub_data.py

Builds a microstate-ENSEMBLE training CSV -- `smiles_protonated_list`/
`smiles_deprotonated_list` (semicolon-joined SMILES, one list per side of
one detected transition), not a single SMILES pair -- from pKahub
(github.com/keserulab/pkahub, Sipos-Szabo/Bajusz/Balogh/Keseru, J. Chem.
Inf. Model. 2026, https://doi.org/10.1021/acs.jcim.6c00107; no repo LICENSE
file, cite the paper -- consistent with this project's own honesty norm
about data provenance) -- specifically its three raw data tables:
`build/datafiles/data/exp_macro_pka_datapoints/combined_unified_dataset.tsv`
(122,351 rows: one real experimental pKa measurement per row, already
merged/curated from 18 named public sources, each pre-annotated with which
charge-state transition it belongs to and an `exclude_flag`/`exclude_comment`
this project's own quality filtering doesn't need to reinvent),
`.../microspecies/microspecies_table.tsv` (every enumerated protonation
state per molecule, each with an Epik-predicted relative free energy), and
`.../molecules/molecules_table.tsv` (not actually needed by this script --
molecule identity is already carried via `molid`).

--- Why microspecies_table, not our own SMARTS site-detection ---

Every other prepare_*.py script in this project (prepare_pka_microstate_
training_data.py, prepare_pka_qm_pretrain_data.py) has to DETECT which atom
a bare (smiles, pKa) pair refers to via js/pka-microstates.js's ported
SMARTS library, with real, disclosed coverage gaps. pKahub's own data
already did that structure-assignment work (via Epik, a real cheminformatics
tool, not a guess) -- `assigned_charge_state_transition` (e.g. "1>>0")
names the exact charge-state pair, and `microspecies_table.tsv` enumerates
every real microspecies structure per molecule.

--- Real micro-ensemble, not "pick the single best tautomer" ---

An earlier version of this script picked, for each of the two charge
states in a transition, the SINGLE LOWEST-predicted_std_free_energy
microspecies -- reasonable-sounding, but a real, confirmed bug: for a
molecule with BOTH an acid site and a base site (amino-acid-like), the
lowest-energy charge-0 structure is very often the ZWITTERION (e.g.
ammonium+carboxylate -- confirmed directly on real data, and often by a
LARGE margin: 9-14 Epik free-energy units more stable than the plain
neutral tautomer in several real examples, not just a close call), not
the fully-neutral tautomer. Picking just one meant the acid-site row and
the base-site row for the SAME molecule silently shared that zwitterion
as their reference point, cross-wiring which site each row's own
structural pair actually represents (see model/registry.json's
pka-microstate-freeenergy notes for the full diagnostic history).

The real, principled fix -- not a special-cased patch for zwitterions
specifically, since the same reasoning applies to any competing tautomer
-- is to stop picking just one microspecies at all. `js/unipka-thermo.js`
already implements the real Uni-pKa paper's own macro-pKa formula for
exactly this (`CC.UniPKAThermo.macroPKa`, previously coded but never
actually exercised beyond its trivial one-vs-one special case in this
project): `pKa(A,B) = logsumexp(-g_i for i in A) - logsumexp(-g_j for j
in B), divided by ln(10)`, where A and B are the FULL sets of microstates
at each charge state, not a single pick. This script now emits every
microspecies at each charge state (capped at MAX_MICROSPECIES_PER_CHARGE
for compute boundedness -- 95% of real (molid, charge) groups have 5 or
fewer anyway, see this script's own exploratory run), and
scripts/train_pka_microstate_freeenergy.py trains against the REAL
ensemble formula instead of a single pair. This is mathematically safe
even when a wide net is cast: logsumexp naturally gives a genuinely
irrelevant high-energy tautomer near-zero weight, so including a few
extra unlikely structures costs a little compute, never distorts the
result -- unlike guessing which ONE structure was "the" right one and
sometimes guessing wrong.

--- Multiple raw measurements per (molid, transition): averaged, or dropped if genuinely ambiguous ---

`combined_unified_dataset.tsv` gives ONE ROW PER RAW LITERATURE MEASUREMENT,
not one row per site -- a heavily-studied molecule can have 30+ rows for
the exact same (molid, transition) (real, confirmed on this release: e.g.
33 separate "1>>0" measurements for one single amine, ranging 9.88-11.67,
ordinary cross-literature measurement spread for one real site). Emitting
each raw row as its own training row would (a) create dozens of near-
duplicate rows sharing the identical structure pair, silently over-
weighting whichever molecule happens to have the most citations, and (b)
for a real, separately-confirmed and rarer case -- a molecule with TWO
CHEMICALLY DIFFERENT sites that both happen to be the same nominal
charge-transition (e.g. a strongly basic aliphatic amine around pKa 8.5-9
AND a much weaker aromatic amine on the same scaffold, both "1>>0") --
silently pair whichever raw row happened to be processed first with this
script's own single representative structure, discarding the other site's
real measurement as if it were a duplicate of the first.

Real fix, not a guess: every raw measurement for one (molid, transition)
is collected first, THEN decided. `biggest_gap_split` finds the largest
gap between consecutive sorted pKa values; if that gap is >=2.5 units AND
at least 2 measurements sit on each side (MULTISITE_GAP_THRESHOLD/
MULTISITE_MIN_MINORITY below -- deliberately conservative: a single
outlier measurement is common in aggregated literature data and should
not by itself flag "two sites"), the WHOLE group is dropped as genuinely
ambiguous -- confirmed real and rare on this exact release (51 of 30,633
(molid, transition) groups with >=2 measurements). This script's own
`species_by_molid` only tracks ONE representative structure per charge
state, so there is no reliable way to tell which cluster it actually
corresponds to; same "ambiguous means don't guess" convention
prepare_pka_microstate_training_data.py already uses for SMARTS
site-detection ambiguity, applied here to structural ambiguity instead.
Everything that doesn't trip that check gets averaged into ONE row
(`pka` = mean, `n_measurements` = raw count, `sources` = every
contributing dataset name, `fidelity_weight` = the per-source tier
weight and assignment_error penalty averaged across whichever sources
actually contributed) -- both fixing the over-weighting problem and
giving a real, disclosed n_measurements/sources trail.

--- Scope: monoprotic transitions only, aqueous only, quality-filtered ---

`exclude_flag` (pKahub's own pipeline already flags non-aqueous solvents,
cosolvents, "assignment error outlier" cases, unparseable pKa values, etc.
-- see this script's own exploratory run for the real breakdown) is
respected as-is; `assigned_charge_state_transition` is restricted to
"0>>-1" (acid) and "1>>0" (base) -- this project's independent-site
microstate framework doesn't model multiprotic transitions ("-1>>-2",
"2>>1", etc.), same disclosed limitation prepare_pka_microstate_training_
data.py's own header already states for the IUPAC source.

--- Benchmark sources held out, not trained on ---

Novartis, SAMPL6, SAMPL7, SAMPL8, and euroSAMPL1 are excluded from the
training-candidate output and written to a SEPARATE benchmark CSV instead
(`--out-benchmark-csv`) -- these are the exact external test sets the
original Uni-pKa paper (and this project's own registry notes) already
compare against; training on them would make any future "compared to
Uni-pKa's own published Novartis/SAMPL6 numbers" comparison meaningless.
`ibond_AttenGpKa` (a second, differently-curated extraction of the same
underlying i-BonD database this project already has its own extraction of,
via the Nevolianis et al. 2025 release -- see prepare_pka_qm_pretrain_
data.py) is skipped entirely for this pass -- same underlying source,
unclear this adds real diversity rather than just more of the same
reliability profile, not worth the extra physical-baseline compute time
to find out in this pass.

--- fidelity_weight: a new column, real signal not a guess ---

Per-row training weight in [0, 1], NOT used by this script itself -- just
recorded for scripts/train_pka_microstate_freeenergy.py's own weighted-loss
training to consume (see that script's header for the multi-fidelity
rationale: flat-merging this project's own i-BonD extraction into the
training set was tried and made results WORSE, concentrated in exactly the
site class the added data was 100% composed of -- see model/registry.json's
pka-microstate-freeenergy notes). Two real signals feed it:
  1. A per-SOURCE base weight (TIER_WEIGHTS below) -- small, specifically-
     curated named literature compilations (Hunt, Datawarrior, Manchester,
     Caine, Jensen, AvLiLuMoVe, OOAs_and_NBs, Settimo, and this project's
     own already-used IUPAC_digitized/Baltruschat_chembl_pka) get 1.0;
     OCHEM and QSARtoolbox (large automated cross-database aggregators,
     more heterogeneous provenance per entry) get 0.5. A real, disclosed
     judgment call, not derived from a held-out validation sweep -- flagged
     as such, not presented as more rigorous than it is.
  2. `assignment_error` (pKahub's own per-row measure of how well the
     assigned charge-state transition's Epik-predicted pKa matches the
     real experimental value -- already used by pKahub's own pipeline to
     flag "outlier" rows for exclusion; here it further modulates weight
     continuously for rows that passed that filter but still have a
     nonzero error) via `max(0.3, 1 - assignment_error / 4)`, multiplied
     into the source-tier weight.

Usage:
    python3 scripts/prepare_pkahub_data.py \\
        --pkahub-dir /path/to/pkahub/build/datafiles/data \\
        --existing-pairs-csv data/pka/pka_microstate_pairs_with_physics_v2.csv \\
        --out-train-csv data/pka/pka_pkahub_candidate.csv \\
        --out-benchmark-csv data/pka/pka_pkahub_benchmark.csv

Needs: rdkit.
"""

import argparse
import csv
import sys
from collections import defaultdict

from rdkit import Chem
from rdkit.RDLogger import DisableLog

DisableLog("rdApp.*")

csv.field_size_limit(10_000_000)

BENCHMARK_SOURCES = {"Novartis", "SAMPL6", "SAMPL7", "SAMPL8", "euroSAMPL1"}
SKIP_SOURCES = {"ibond_AttenGpKa"}
TIER_WEIGHTS = defaultdict(lambda: 1.0, {
    "OCHEM": 0.5,
    "QSARtoolbox": 0.5,
})
VALID_TRANSITIONS = {"0>>-1", "1>>0"}
MAX_MICROSPECIES_PER_CHARGE = 5


def inchikey_of(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def load_exclusion_inchikeys(existing_pairs_csv):
    keys = set()
    with open(existing_pairs_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for col in ("smiles_protonated", "smiles_deprotonated"):
                ik = inchikey_of(row[col])
                if ik:
                    keys.add(ik)
    print(f"loaded {len(keys)} exclusion InChIKeys from {existing_pairs_csv}", file=sys.stderr)
    return keys


def load_all_microspecies(pkahub_dir):
    """molid -> {charge: [smiles, ...]}, ALL microspecies at each charge
    state, sorted ascending by predicted_std_free_energy and capped at
    MAX_MICROSPECIES_PER_CHARGE (keeping the lowest-energy ones -- the
    ones a real Boltzmann population would actually weight)."""
    by_key = defaultdict(list)  # (molid, charge) -> [(free_energy, smiles), ...]
    path = f"{pkahub_dir}/microspecies/microspecies_table.tsv"
    n_rows, n_unparseable = 0, 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            n_rows += 1
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                n_unparseable += 1
                continue
            charge = Chem.GetFormalCharge(mol)
            try:
                fe = float(row["predicted_std_free_energy"])
            except ValueError:
                continue
            by_key[(row["molid"], charge)].append((fe, row["smiles"]))
    print(f"microspecies: {n_rows} rows ({n_unparseable} unparseable), {len(by_key)} unique (molid, charge) groups", file=sys.stderr)
    by_molid = defaultdict(dict)
    capped, uncapped = 0, 0
    for (molid, charge), items in by_key.items():
        items.sort(key=lambda x: x[0])
        if len(items) > MAX_MICROSPECIES_PER_CHARGE:
            capped += 1
        else:
            uncapped += 1
        by_molid[molid][charge] = [s for _, s in items[:MAX_MICROSPECIES_PER_CHARGE]]
    print(f"  {capped} groups capped at {MAX_MICROSPECIES_PER_CHARGE}, {uncapped} kept in full", file=sys.stderr)
    return by_molid


def biggest_gap_split(vals):
    """Sorted vals split at their single largest consecutive gap -- returns
    (gap, low_cluster, high_cluster). See MULTISITE_GAP_THRESHOLD's own
    comment for how this is used."""
    vals = sorted(vals)
    best_gap, best_i = 0.0, None
    for i in range(1, len(vals)):
        gap = vals[i] - vals[i - 1]
        if gap > best_gap:
            best_gap, best_i = gap, i
    return best_gap, vals[:best_i], vals[best_i:]


# A real, measured phenomenon (see this script's own exploratory run, and
# the user's own explicit question this was written in response to): a
# (molid, transition) group -- e.g. "1>>0" -- can silently bundle
# measurements of TWO CHEMICALLY DIFFERENT basic sites on the same
# molecule that both happen to be a cation->neutral transition (e.g. a
# strongly basic aliphatic amine around pKa 8.5-9 AND a much weaker
# aromatic amine/amide around pKa 5.2-5.6 on the same scaffold) -- the flat
# combined_unified_dataset.tsv table has no per-microspecies foreign key
# to tell these apart, only the coarse charge-transition label. Confirmed
# real on this exact release: of 30,633 (molid, transition) groups with
# >=2 raw measurements, exactly 51 show a clean bimodal split (checked by
# eye against several real examples: two tight clusters ~3 pKa units
# apart, not just noisy scatter around one value) -- rare, but a genuine
# structural ambiguity this script cannot resolve from the data it has
# (this project's `species_by_molid` only tracks the SINGLE lowest-free-
# energy representative structure per charge state, not "the specific
# structure this one experimental value refers to"). Rather than guess
# which cluster the single representative structure actually corresponds
# to (or silently average two different real pKa's into one meaningless
# number), groups that trip this check are DROPPED entirely -- same
# "ambiguous means don't guess" convention prepare_pka_microstate_
# training_data.py already uses for SMARTS site-detection ambiguity.
MULTISITE_GAP_THRESHOLD = 2.5
MULTISITE_MIN_MINORITY = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkahub-dir", required=True)
    parser.add_argument("--existing-pairs-csv", required=True)
    parser.add_argument("--out-train-csv", required=True)
    parser.add_argument("--out-benchmark-csv", required=True)
    args = parser.parse_args()

    exclude = load_exclusion_inchikeys(args.existing_pairs_csv)
    species_by_molid = load_all_microspecies(args.pkahub_dir)

    benchmark_rows = []
    dropped = defaultdict(int)
    # (molid, transition) -> list of (pka, dataset, assignment_error) --
    # collected first, DECIDED (average vs. drop-as-ambiguous) second, so
    # every raw literature measurement of the same site is folded into
    # ONE training row instead of creating near-duplicate rows that would
    # silently over-weight whichever molecule happens to have the most
    # literature citations (some of these groups have 30+ raw rows -- see
    # this script's own exploratory run).
    train_groups = defaultdict(list)

    path = f"{args.pkahub_dir}/exp_macro_pka_datapoints/combined_unified_dataset.tsv"
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["exclude_flag"]:
                dropped["excluded_by_pkahub"] += 1
                continue
            transition = row["assigned_charge_state_transition"]
            if transition not in VALID_TRANSITIONS:
                dropped["polyprotic_or_unassigned_transition"] += 1
                continue
            ds = row["dataset"]
            if ds in SKIP_SOURCES:
                dropped["skipped_source"] += 1
                continue

            molid = row["molid"]
            species = species_by_molid.get(molid)
            if not species:
                dropped["molid_not_in_microspecies_table"] += 1
                continue
            charge_hi, charge_lo = (int(x) for x in transition.split(">>"))
            if charge_hi not in species or charge_lo not in species:
                dropped["representative_structure_missing_for_charge"] += 1
                continue

            try:
                pka = float(row["Exp_Macro_pKa_processed"])
            except ValueError:
                dropped["pka_unparseable"] += 1
                continue

            if ds in BENCHMARK_SOURCES:
                # Benchmark rows stay single-representative (lowest-energy
                # microspecies each side), NOT an ensemble -- this is an
                # external comparison surface, kept in the same simple
                # shape the model's own inference-time code actually uses
                # (which also only ever sees one structure per side, see
                # this script's own header on why the ensemble treatment
                # is training-time-only).
                protonated, deprotonated = species[charge_hi][0], species[charge_lo][0]
                ik = inchikey_of(protonated)
                benchmark_rows.append({
                    "inchikey": ik, "site_name": "pkahub", "site_cls": "acid" if transition == "0>>-1" else "base",
                    "smiles_protonated": protonated, "smiles_deprotonated": deprotonated,
                    "pka": pka, "n_measurements": 1, "sources": "pkahub_" + ds,
                })
                continue

            try:
                assignment_error = float(row["assignment_error"])
            except ValueError:
                assignment_error = 0.0
            train_groups[(molid, transition)].append((pka, ds, assignment_error))

    train_rows = []
    seen_pairs = set()  # (tuple(protonated_list), tuple(deprotonated_list)) across
    # DIFFERENT molids -- a residual duplicate-molecule-under-two-molids
    # case, not the within-molid repeat-measurement case already handled
    # by grouping above.
    for (molid, transition), measurements in train_groups.items():
        species = species_by_molid[molid]
        charge_hi, charge_lo = (int(x) for x in transition.split(">>"))
        protonated_list = species[charge_hi]
        deprotonated_list = species[charge_lo]

        if len(measurements) > 1:
            gap, lo, hi = biggest_gap_split([m[0] for m in measurements])
            if gap >= MULTISITE_GAP_THRESHOLD and min(len(lo), len(hi)) >= MULTISITE_MIN_MINORITY:
                dropped["ambiguous_multiple_sites_same_transition"] += 1
                continue

        pkas = [m[0] for m in measurements]
        datasets = sorted(set(m[1] for m in measurements))
        pka = sum(pkas) / len(pkas)
        # Per-source tier weight averaged across whichever sources actually
        # contributed to this molecule's measurements (a molecule measured
        # by both a gold-tier and a bronze-tier source lands in between,
        # not pinned to either extreme) -- assignment_error's own
        # continuous penalty is averaged the same way.
        mean_assignment_error = sum(m[2] for m in measurements) / len(measurements)
        mean_tier_weight = sum(TIER_WEIGHTS[m[1]] for m in measurements) / len(measurements)
        weight = mean_tier_weight * max(0.3, 1 - mean_assignment_error / 4)

        iks_p = [inchikey_of(s) for s in protonated_list]
        iks_d = [inchikey_of(s) for s in deprotonated_list]
        if any(ik is None for ik in iks_p + iks_d):
            dropped["smiles_unparseable"] += 1
            continue
        # Conservative by design (same convention as every other dedup
        # pass in this project): if ANY microstate in either list overlaps
        # the existing corpus, drop the WHOLE row rather than try to prune
        # just the overlapping structure out of its own ensemble.
        if any(ik in exclude for ik in iks_p + iks_d):
            dropped["overlaps_existing_corpus"] += 1
            continue
        pair_key = (tuple(protonated_list), tuple(deprotonated_list))
        if pair_key in seen_pairs:
            dropped["duplicate_molecule_under_different_molid"] += 1
            continue
        seen_pairs.add(pair_key)

        train_rows.append({
            "inchikey": iks_p[0], "site_name": "pkahub", "site_cls": "acid" if transition == "0>>-1" else "base",
            "smiles_protonated_list": ";".join(protonated_list), "smiles_deprotonated_list": ";".join(deprotonated_list),
            "pka": round(pka, 4), "n_measurements": len(measurements), "sources": "pkahub_" + "+".join(datasets),
            "fidelity_weight": round(weight, 4),
        })

    print(f"train candidates: kept {len(train_rows)}", file=sys.stderr)
    for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  dropped[{k}] = {v}", file=sys.stderr)
    print(f"benchmark (held out, not trained on): {len(benchmark_rows)}", file=sys.stderr)

    with open(args.out_train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "inchikey", "site_name", "site_cls", "smiles_protonated_list",
            "smiles_deprotonated_list", "pka", "n_measurements", "sources", "fidelity_weight",
        ])
        writer.writeheader()
        writer.writerows(train_rows)
    print(f"wrote {args.out_train_csv}", file=sys.stderr)

    with open(args.out_benchmark_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "inchikey", "site_name", "site_cls", "smiles_protonated",
            "smiles_deprotonated", "pka", "n_measurements", "sources",
        ])
        writer.writeheader()
        writer.writerows(benchmark_rows)
    print(f"wrote {args.out_benchmark_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
