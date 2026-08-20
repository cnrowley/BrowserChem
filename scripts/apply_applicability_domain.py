#!/usr/bin/env python3
"""
apply_applicability_domain.py

Patches model/registry.json to point at an already-computed
applicability-domain.json sidecar (see compute_applicability_domain.py)
for one or more models -- the same edit that pilot models
(cyp1a2-substrate-v1, solv-1-octanol-v1) got by hand, automated here so
it scales to the rest of the registry without 30+ manual JSON edits.

Deliberately does NOT round-trip the whole registry.json through
json.load()+json.dump(): that would re-serialize every single line
(re-escaping unicode, etc.) and produce a huge, unreviewable diff even
though the content is unchanged. Instead this does targeted text
surgery -- finds each model's unique "files": {...} block (anchored on
its own manifest/weights paths, which are unique per entry) and inserts
around it, leaving every other byte of the file untouched. Every entry
in the current registry.json has "files" immediately followed by
"dataset" (verified before writing this script) -- if that's no longer
true for some future entry, this script will raise rather than guess
at a wrong insertion point.

Usage:
    python3 apply_applicability_domain.py <model-id> [<model-id> ...]
    python3 apply_applicability_domain.py --all-computed   # every model/*/applicability-domain.json on disk
"""

import argparse
import json
import re
import sys
from pathlib import Path


def build_notes(sidecar):
    ts = sidecar["trainingSet"]
    ed = sidecar["embeddingDomain"]
    supported = ts.get("netMolecularCharges", [])
    counts = ts.get("netMolecularChargeCounts", {})
    min_count = ts.get("minNetChargeCount")
    dropped = {c: n for c, n in counts.items() if int(c) not in supported}

    if dropped:
        dropped_str = ", ".join(f"{c} ({n}x)" for c, n in sorted(dropped.items(), key=lambda kv: int(kv[0])))
        charge_clause = (
            f"or a net molecular charge outside {supported} "
            f"(net charge(s) {dropped_str} occurred fewer than {min_count} times out of "
            f"{ts['size']} -- treated as one-off outlier(s), not real training support), "
        )
    else:
        charge_clause = f"or a net molecular charge outside {supported}, "

    return (
        f"Computed by scripts/compute_applicability_domain.py from the exact CSV chemprop train "
        f"was pointed at ({ts['sourceFile']}). A molecule containing any element/formal-charge "
        f"combination outside this set, {charge_clause}"
        f"gets hard-blocked by the app rather than silently featurized into Chemprop's generic "
        f"'unknown element' pad bucket -- this is a per-model training-vocabulary check, stricter "
        f"than js/chemprop-features.js's checkChempropCompatibility() which only checks the shared "
        f"global featurizer vocabulary. The embedding-domain distance (nearest-centroid distance in "
        f"the trained D-MPNN's own {ed['dim']}-dim pooled-molecule-embedding space, see the sidecar "
        f"file) is a heuristic in-domain/borderline/out-of-domain confidence tier, not a calibrated "
        f"statistical guarantee -- see compute_applicability_domain.py's docstring for why conformal "
        f"calibration isn't attempted (chemprop train's internal SCAFFOLD_BALANCED split isn't "
        f"reproducible from what's on disk)."
    )


def json_str(value, indent):
    """json.dumps with a given indent, re-indented to start at column `indent`."""
    text = json.dumps(value, indent=2)
    lines = text.split("\n")
    pad = " " * indent
    return ("\n" + pad).join(lines)


def patch_one(registry_text, registry_dir, model_id, entry, sidecar_rel, sidecar):
    manifest_rel = entry["files"]["manifest"]
    weights_rel = entry["files"]["weights"]

    files_pattern = re.compile(
        r'(      "files": \{\n'
        r'        "manifest": "' + re.escape(manifest_rel) + r'",\n'
        r'        "weights": "' + re.escape(weights_rel) + r'"\n'
        r'      \},\n)'
    )
    matches = list(files_pattern.finditer(registry_text))
    if len(matches) == 0:
        raise SystemExit(f"[{model_id}] couldn't find its files{{}} block in registry.json (already patched? "
                          f"or manifest/weights paths don't match exactly)")
    if len(matches) > 1:
        raise SystemExit(f"[{model_id}] files{{}} block pattern matched {len(matches)} times -- not unique, refusing to guess")

    ts = sidecar["trainingSet"]
    ad_summary = {
        "trainingSetSize": ts["size"],
        "elements": ts["elements"],
        "formalCharges": ts["formalCharges"],
        "netMolecularCharges": ts.get("netMolecularCharges", []),
        "embeddingDomainSelfDistanceP90": sidecar["embeddingDomain"]["selfDistancePercentiles"]["p90"],
        "notes": build_notes(sidecar),
    }

    new_files_block = (
        '      "files": {\n'
        f'        "manifest": "{manifest_rel}",\n'
        f'        "weights": "{weights_rel}",\n'
        f'        "applicabilityDomain": "{sidecar_rel}"\n'
        '      },\n'
        '      "applicabilityDomain": ' + json_str(ad_summary, 6) + ',\n'
    )

    start, end = matches[0].span()
    return registry_text[:start] + new_files_block + registry_text[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_ids", nargs="*")
    parser.add_argument("--all-computed", action="store_true",
                         help="apply for every model/*/applicability-domain.json found on disk")
    parser.add_argument("--registry", default="model/registry.json")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry_dir = registry_path.parent
    registry = json.loads(registry_path.read_text())
    by_id = {m["id"]: m for m in registry["models"]}

    model_ids = list(args.model_ids)
    if args.all_computed:
        for sidecar_path in sorted(registry_dir.glob("*/applicability-domain.json")):
            sidecar = json.loads(sidecar_path.read_text())
            mid = sidecar["modelId"]
            if mid not in model_ids:
                model_ids.append(mid)

    if not model_ids:
        sys.exit("no model ids given (pass some, or --all-computed)")

    registry_text = registry_path.read_text()
    applied, skipped = [], []
    for model_id in model_ids:
        entry = by_id.get(model_id)
        if entry is None:
            print(f"[{model_id}] not in registry -- skipping", file=sys.stderr)
            skipped.append(model_id)
            continue
        if "applicabilityDomain" in entry:
            print(f"[{model_id}] already has applicabilityDomain in registry.json -- skipping", file=sys.stderr)
            skipped.append(model_id)
            continue
        model_dir = registry_dir / Path(entry["files"]["manifest"]).parent
        sidecar_path = model_dir / "applicability-domain.json"
        if not sidecar_path.exists():
            print(f"[{model_id}] no {sidecar_path} on disk -- run compute_applicability_domain.py first -- skipping", file=sys.stderr)
            skipped.append(model_id)
            continue
        sidecar = json.loads(sidecar_path.read_text())
        sidecar_rel = str(sidecar_path.relative_to(registry_dir))
        registry_text = patch_one(registry_text, registry_dir, model_id, entry, sidecar_rel, sidecar)
        applied.append(model_id)
        print(f"[{model_id}] patched", file=sys.stderr)

    if applied:
        registry_path.write_text(registry_text)
        json.loads(registry_path.read_text())  # sanity: still valid JSON
        print(f"wrote {registry_path} -- {len(applied)} applied, {len(skipped)} skipped", file=sys.stderr)
    else:
        print("nothing applied -- registry.json left unchanged", file=sys.stderr)


if __name__ == "__main__":
    main()
