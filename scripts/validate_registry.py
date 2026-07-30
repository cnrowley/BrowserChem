#!/usr/bin/env python3
"""
validate_registry.py

Sanity-checks model/registry.json before you ship it: required fields
present, no duplicate ids, referenced manifest.json/weights.bin files
actually exist on disk, and (for "chemprop"-engine entries) the declared
taskType matches what's actually baked into its technical manifest.json
(the one convert_chemprop_checkpoint.py produces) -- catching the
specific, easy-to-make mistake of hand-editing registry.json to say
"regression" for what's actually a classification checkpoint, or vice
versa. "nagl"-engine entries (convert_nagl_checkpoint.py output) get
their own, differently-shaped manifest cross-check instead.

Usage:
    python3 validate_registry.py model/registry.json
"""

import argparse
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = ["id", "displayName", "propertyKey", "files"]
VALID_TASK_TYPES = {"regression", "classification"}
VALID_ENGINES = {"chemprop", "nagl"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_path", help="path to registry.json")
    args = parser.parse_args()

    registry_path = Path(args.registry_path)
    if not registry_path.exists():
        sys.exit(f"{registry_path} doesn't exist")

    try:
        data = json.loads(registry_path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"registry.json isn't valid JSON: {e}")

    models = data.get("models", [])
    if not models:
        sys.exit("registry.json has no models listed")

    base_dir = registry_path.parent
    seen_ids = set()
    errors = []
    warnings = []

    for i, entry in enumerate(models):
        label = entry.get("id", f"entry #{i}")
        engine = entry.get("engine", "chemprop")  # entries predating this field are all chemprop

        for field in REQUIRED_FIELDS:
            if not entry.get(field):
                errors.append(f"[{label}] missing required field '{field}'")

        if engine not in VALID_ENGINES:
            errors.append(f"[{label}] engine must be one of {VALID_ENGINES}, got {engine!r}")

        entry_id = entry.get("id")
        if entry_id:
            if entry_id in seen_ids:
                errors.append(f"[{label}] duplicate id '{entry_id}'")
            seen_ids.add(entry_id)

        task_type = entry.get("taskType")
        if engine == "chemprop":
            if task_type not in VALID_TASK_TYPES:
                errors.append(f"[{label}] taskType must be one of {VALID_TASK_TYPES}, got {task_type!r}")
        elif task_type is not None:
            warnings.append(f"[{label}] taskType is ignored for engine='nagl' entries (always atom-level "
                             f"regression by construction) -- remove it, or leave it out, to avoid confusion")

        files = entry.get("files") or {}
        manifest_rel = files.get("manifest")
        weights_rel = files.get("weights")

        manifest_path = (base_dir / manifest_rel) if manifest_rel else None
        weights_path = (base_dir / weights_rel) if weights_rel else None

        if manifest_path and not manifest_path.exists():
            errors.append(f"[{label}] manifest file not found: {manifest_path}")
        if weights_path and not weights_path.exists():
            errors.append(f"[{label}] weights file not found: {weights_path}")

        if manifest_path and manifest_path.exists():
            try:
                tech_manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                errors.append(f"[{label}] {manifest_path} isn't valid JSON")
                tech_manifest = None

            if tech_manifest is not None:
                if engine == "chemprop":
                    # Cross-check the registry's declared taskType against what
                    # the technical manifest.json actually says -- catches a
                    # stale/mistyped registry entry after re-exporting a checkpoint.
                    tech_task_type = tech_manifest.get("taskType", "regression")  # manifests predating the field default to regression
                    if task_type and tech_task_type != task_type:
                        errors.append(
                            f"[{label}] registry says taskType={task_type!r} but "
                            f"{manifest_path} says taskType={tech_task_type!r}"
                        )
                    tech_task = tech_manifest.get("task")
                elif engine == "nagl":
                    # nagl-model.js currently only implements the "charges"
                    # (electronegativity-equalization) postprocess -- flag
                    # anything else early rather than let it fail silently
                    # (or loudly, but only once a user actually tries it) later.
                    postprocess = tech_manifest.get("postprocess")
                    if postprocess and postprocess != "charges":
                        errors.append(
                            f"[{label}] manifest postprocess={postprocess!r}, but nagl-model.js only "
                            f"implements 'charges' (electronegativity-equalization) -- this model "
                            f"would fail (or silently mispredict) at runtime."
                        )
                    tech_task = tech_manifest.get("task")

                registry_property_key = entry.get("propertyKey")
                if registry_property_key and tech_task and registry_property_key != tech_task:
                    warnings.append(
                        f"[{label}] registry propertyKey={registry_property_key!r} != "
                        f"manifest task={tech_task!r} (cosmetic mismatch, not fatal, "
                        f"but worth double-checking these are meant to be the same property)"
                    )

        # Softer completeness nudges -- not fatal, since a model provided
        # without full provenance is still usable, just less informative
        # in the UI.
        if not entry.get("dataset") or not entry["dataset"].get("name"):
            warnings.append(f"[{label}] no dataset name recorded")
        if not entry.get("metrics"):
            warnings.append(f"[{label}] no metrics recorded")

    print(f"{len(models)} model(s) checked.")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nOK -- no errors.")


if __name__ == "__main__":
    main()
