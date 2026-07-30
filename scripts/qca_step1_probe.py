"""
Step 1: run this first, on your machine (needs real network access to
QCArchive, which my sandbox doesn't have). This does NOT do the full
extraction -- it just connects, grabs ONE completed record, and prints
its actual available property keys, so we use the right string instead
of guessing.
"""
import qcportal as ptl

client = ptl.PortalClient("https://api.qcarchive.molssi.org:443")
ds = client.get_dataset("singlepoint", "MLPepper RECAP Optimized Fragments v1.0")

print("entry count:", len(ds.entry_names))
print("specifications:", ds.specification_names)

# Grab the first completed record we can find and inspect it directly --
# this tells us the real property key name(s) rather than assuming
# "mbis_charges" matches exactly what's stored.
for entry_name, spec_name, record in ds.iterate_records(status="complete"):
    print("\nentry:", entry_name)
    print("spec:", spec_name)
    print("record properties keys:", list(record.properties.keys()) if record.properties else "NONE")
    # print just the mbis-related key(s) fully, whatever they're actually called
    for k, v in (record.properties or {}).items():
        if "mbis" in k.lower():
            print(f"  {k}: {type(v)}, len={len(v) if hasattr(v,'__len__') else 'n/a'}")
            print(f"  sample: {v[:6] if hasattr(v,'__getitem__') else v}")
    print("\nrecord.molecule.symbols:", record.molecule.symbols)
    break
