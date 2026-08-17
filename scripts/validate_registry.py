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
versa. "nagl"-engine entries (convert_nagl_checkpoint.py output),
"ani2x"-engine entries (convert_ani2x_checkpoint.py output),
"geomol"-engine entries (convert_geomol_checkpoint.py output), and
"pka"-engine entries (convert_pka_lightgbm.py output) each get their
own, differently-shaped manifest cross-check instead.

Usage:
    python3 validate_registry.py model/registry.json
"""

import argparse
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = ["id", "displayName", "propertyKey", "files"]
VALID_TASK_TYPES = {"regression", "classification"}
VALID_ENGINES = {"chemprop", "nagl", "ani2x", "geomol", "pka"}
VALID_CATEGORIES = {"general", "environmental-analytical", "medicinal", "structure-tools", "characterization"}


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

        # Not in REQUIRED_FIELDS (older entries could in principle ship
        # without one and still be usable), but a typo'd value is a real
        # bug -- js/app.js's renderRegistryList only has containers for
        # this exact set of categories, so an unrecognized value would
        # silently make that model vanish from the Properties panel
        # entirely rather than erroring loudly. A list, not a single
        # value, since a model is allowed to appear in more than one
        # section on purpose (e.g. melting point shows under both
        # "Environmental & analytical" and "Characterization" -- the
        # user explicitly wants that redundancy, not a single home per
        # model).
        categories = entry.get("categories")
        if categories is None:
            warnings.append(f"[{label}] no 'categories' set -- won't appear in any Properties-panel section")
        elif not isinstance(categories, list) or not categories:
            errors.append(f"[{label}] 'categories' must be a non-empty list, got {categories!r}")
        else:
            bad = [c for c in categories if c not in VALID_CATEGORIES]
            if bad:
                errors.append(f"[{label}] categories contains invalid value(s) {bad!r} -- must be from {VALID_CATEGORIES}")

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
            warnings.append(f"[{label}] taskType is ignored for engine={engine!r} entries "
                             f"(not a chemprop regression/classification head) -- remove it, or leave it "
                             f"out, to avoid confusion")

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
                    # Same cross-check for outputLevel -- a registry entry that
                    # says "atom" for what's actually a molecule-level checkpoint
                    # (or vice versa) would render into the wrong UI panel
                    # entirely (properties table vs. atom heatmap).
                    registry_output_level = entry.get("outputLevel", "molecule")
                    tech_output_level = tech_manifest.get("outputLevel", "molecule")
                    if registry_output_level != tech_output_level:
                        errors.append(
                            f"[{label}] registry says outputLevel={registry_output_level!r} but "
                            f"{manifest_path} says outputLevel={tech_output_level!r}"
                        )
                    if tech_output_level in ("atom", "bond"):
                        # graphType matters for both atom- and bond-level models --
                        # graphType='explicit-h' needs
                        # chemprop-features-explicit-h.js's graph builder
                        # (chemprop-model.js), not just the normal heavy-atom one.
                        graph_type = tech_manifest.get("graphType", "heavy")
                        if graph_type not in ("heavy", "explicit-h"):
                            errors.append(f"[{label}] manifest graphType={graph_type!r}, expected 'heavy' or 'explicit-h'")
                        if tech_output_level == "bond" and graph_type != "explicit-h":
                            errors.append(f"[{label}] bond-level manifest has graphType={graph_type!r} -- "
                                          f"a bond-level checkpoint always needs 'explicit-h' (see "
                                          f"convert_chemprop_checkpoint.py's own validation at conversion time).")
                        if tech_output_level == "atom" and not tech_manifest.get("applicableElement"):
                            warnings.append(
                                f"[{label}] atom-level chemprop model has no 'applicableElement' set -- "
                                f"it will annotate every atom regardless of element, which is only correct "
                                f"if that's genuinely intended (e.g. a per-atom property meaningful for any "
                                f"element, unlike a nucleus-specific NMR shift model)."
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
                elif engine == "ani2x":
                    # ani2x-model.js/ani2x-features.js only implement ANIRadial/
                    # ANIAngular AEV terms and the cosine cutoff function -- both
                    # are the only values convert_ani2x_checkpoint.py ever writes
                    # (it already refuses to export anything else), so this is
                    # really just confirming the manifest wasn't hand-edited into
                    # something the converter itself would never have produced.
                    species = tech_manifest.get("species") or []
                    if len(species) != 7 or set(species) != {"H", "C", "N", "O", "F", "S", "Cl"}:
                        errors.append(
                            f"[{label}] manifest species={species!r}, expected exactly "
                            f"['H','C','N','O','F','S','Cl'] -- ani2x-model.js's compatibility "
                            f"check and AEV species-pair table assume exactly these 7 elements."
                        )
                    if not tech_manifest.get("ensembleSize"):
                        errors.append(f"[{label}] manifest is missing 'ensembleSize'")
                    aev = tech_manifest.get("aev") or {}
                    if not aev.get("speciesPairIndex"):
                        errors.append(f"[{label}] manifest is missing 'aev.speciesPairIndex'")
                    tech_task = tech_manifest.get("task")
                elif engine == "geomol":
                    # geomol-model.js only implements the global_transformer=false
                    # path -- convert_geomol_checkpoint.py already refuses to export
                    # a global_transformer=true checkpoint, so this is really just
                    # confirming the manifest wasn't hand-edited afterward into
                    # something the converter itself would never have produced.
                    if tech_manifest.get("architecture") != "geomol":
                        errors.append(
                            f"[{label}] manifest architecture={tech_manifest.get('architecture')!r}, "
                            f"expected 'geomol'."
                        )
                    for required_key in ("gnn1", "gnn2", "encoder", "coordPredDims", "dMlpDims",
                                         "hMolMlpDims", "alphaMlpDims", "cMlpDims"):
                        if not tech_manifest.get(required_key):
                            errors.append(f"[{label}] manifest is missing '{required_key}'")
                    encoder = tech_manifest.get("encoder") or {}
                    if encoder.get("dModel") != 2 * (tech_manifest.get("modelDim") or -1):
                        errors.append(
                            f"[{label}] manifest encoder.dModel={encoder.get('dModel')!r} doesn't "
                            f"equal 2*modelDim={tech_manifest.get('modelDim')!r} -- geomol-model.js's "
                            f"local-structure Transformer assumes the encoder operates on "
                            f"concat(neighbor, center) pairs, i.e. exactly 2x modelDim wide."
                        )
                    tech_task = tech_manifest.get("task")
                elif engine == "pka":
                    # pka-model.js only implements the flat-tree-array
                    # "lightgbm-gbdt" format scripts/convert_pka_lightgbm.py
                    # produces, and needs its descriptor.chargeSource NAGL
                    # model to exist somewhere else in this same registry
                    # (pka-model.js checks it's actually LOADED at predict
                    # time, not just registered -- that's a runtime check
                    # this static validator can't do).
                    if tech_manifest.get("architecture") != "lightgbm-gbdt":
                        errors.append(
                            f"[{label}] manifest architecture={tech_manifest.get('architecture')!r}, "
                            f"expected 'lightgbm-gbdt'."
                        )
                    if not tech_manifest.get("numTrees"):
                        errors.append(f"[{label}] manifest is missing 'numTrees'")
                    descriptor = tech_manifest.get("descriptor") or {}
                    charge_source = descriptor.get("chargeSource")
                    if not charge_source:
                        errors.append(f"[{label}] manifest is missing 'descriptor.chargeSource'")
                    elif charge_source not in seen_ids:
                        warnings.append(
                            f"[{label}] descriptor.chargeSource={charge_source!r} doesn't match any "
                            f"registry entry id seen so far -- if it's defined later in registry.json "
                            f"this warning is a false positive, but double check the id is right."
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
