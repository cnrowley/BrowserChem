"""
Extracts MBIS charges from QCArchive for the "MLPepper RECAP Optimized
Fragments v1.0" dataset and writes a chemprop atom-target training CSV.

Run in stages -- each PHASE prints what it found before moving on, so if
something's named differently than expected on the real server, you'll
see that immediately instead of getting a silently-empty or wrong CSV.

    pip install qcportal --break-system-packages   # or without that flag if not on Debian/Ubuntu system Python
    python3 extract_mbis_charges.py
"""
import json
import bz2
import csv
import qcportal as ptl

DATASET_NAME = "MLPepper RECAP Optimized Fragments v1.0"
SPEC_NAME = "wb97x-d/def2-tzvpp"  # gas-phase spec -- NOT the ddx-water one.
                                   # Change this if you specifically want
                                   # the implicit-solvent charges instead.
MBIS_PROPERTY_KEY = "mbis_charges"  # <-- CHANGE THIS once you see the real
                                     #     key name from PHASE 1's printout.
                                     #     Common alternate: "mbis charges" (space).

DATASET_JSON_PATH = "dataset.json"  # your local decompressed file
OUTPUT_CSV_PATH = "mbis_train.csv"

# ---- PHASE 1: connect and sanity-check the property key name ----
print("=== PHASE 1: connecting and checking property key name ===")
client = ptl.PortalClient("https://api.qcarchive.molssi.org:443")
ds = client.get_dataset("singlepoint", DATASET_NAME)
print(f"entries: {len(ds.entry_names)}")
print(f"specifications: {ds.specification_names}")

sample_record = None
for entry_name, spec_name, record in ds.iterate_records(
    entry_names=ds.entry_names[:5], specification_names=[SPEC_NAME], status="complete"
):
    print(f"\nentry: {entry_name}")
    keys = list(record.properties.keys()) if record.properties else []
    print(f"  property keys: {keys}")
    mbis_keys = [k for k in keys if "mbis" in k.lower()]
    print(f"  mbis-related keys found: {mbis_keys}")
    sample_record = record
    if mbis_keys:
        break

if sample_record is None:
    raise SystemExit("No completed records found for the first 5 entries -- check DATASET_NAME/SPEC_NAME.")

if MBIS_PROPERTY_KEY not in (sample_record.properties or {}):
    print(f"\n*** '{MBIS_PROPERTY_KEY}' not found in properties. ***")
    print("Update MBIS_PROPERTY_KEY above to match one of the mbis-related keys printed above, then rerun.")
    raise SystemExit(1)

sample_charges = sample_record.properties[MBIS_PROPERTY_KEY]
print(f"\nsample '{MBIS_PROPERTY_KEY}' value: {sample_charges}")
print(f"length: {len(sample_charges)}, num atoms in molecule: {len(sample_record.molecule.symbols)}")
if len(sample_charges) != len(sample_record.molecule.symbols):
    print("*** WARNING: charge array length != atom count. This property might be "
          "flattened multipoles (charge+dipole+quadrupole per atom stacked together), "
          "not a plain one-value-per-atom array. Inspect sample_charges structure by "
          "hand before proceeding -- do not assume it's directly usable as-is. ***")
    raise SystemExit(1)

print("\nPHASE 1 OK -- key name and shape look right. Continuing to full extraction...")

# ---- PHASE 2: load local dataset.json for entry_name -> atom-mapped SMILES ----
print("\n=== PHASE 2: loading local dataset.json for atom-mapped SMILES ===")
opener = bz2.open if DATASET_JSON_PATH.endswith(".bz2") else open
with opener(DATASET_JSON_PATH, "rt") as f:
    local_data = json.load(f)

entry_to_mapped_smiles = {}
for entry_name, entry in local_data["dataset"].items():
    # initial_molecules is a list (one per conformer); the mapped SMILES
    # identifier should be identical across conformers of the same
    # molecule (same connectivity/atom-map, different 3D geometry only) --
    # spot-check that assumption rather than silently trusting it.
    mapped_smiles_set = {
        mol["identifiers"]["canonical_isomeric_explicit_hydrogen_mapped_smiles"]
        for mol in entry["initial_molecules"]
    }
    if len(mapped_smiles_set) != 1:
        print(f"*** WARNING: entry {entry_name!r} has {len(mapped_smiles_set)} different "
              f"mapped SMILES across its conformers -- skipping, needs manual inspection. ***")
        continue
    entry_to_mapped_smiles[entry_name] = mapped_smiles_set.pop()

print(f"loaded {len(entry_to_mapped_smiles)} entries with a consistent mapped SMILES")

# ---- PHASE 3: query all completed records, average charges across conformers ----
print("\n=== PHASE 3: querying full dataset (this will take a while for ~69k conformers) ===")
from collections import defaultdict

charge_lists_by_entry = defaultdict(list)
n_records = 0
for entry_name, spec_name, record in ds.iterate_records(
    specification_names=[SPEC_NAME], status="complete"
):
    if entry_name not in entry_to_mapped_smiles:
        continue
    props = record.properties or {}
    if MBIS_PROPERTY_KEY not in props:
        continue
    charge_lists_by_entry[entry_name].append(props[MBIS_PROPERTY_KEY])
    n_records += 1
    if n_records % 5000 == 0:
        print(f"  ...{n_records} records processed")

print(f"total records with charges: {n_records}")
print(f"unique entries with at least one conformer's charges: {len(charge_lists_by_entry)}")

# ---- PHASE 4: average across conformers (conformation-independent charge
# is the whole design goal here -- see the dataset README) and write CSV ----
print("\n=== PHASE 4: averaging across conformers and writing CSV ===")
rows_written = 0
with open(OUTPUT_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles", "mbis_charge"])
    for entry_name, charge_lists in charge_lists_by_entry.items():
        mapped_smiles = entry_to_mapped_smiles[entry_name]
        n_atoms = len(charge_lists[0])
        if any(len(c) != n_atoms for c in charge_lists):
            print(f"*** WARNING: entry {entry_name!r} has inconsistent atom counts across "
                  f"conformers -- skipping. ***")
            continue
        avg_charges = [sum(c[i] for c in charge_lists) / len(charge_lists) for i in range(n_atoms)]
        writer.writerow([mapped_smiles, str(avg_charges)])
        rows_written += 1

print(f"\nwrote {rows_written} rows to {OUTPUT_CSV_PATH}")
