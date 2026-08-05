#!/usr/bin/env python3
"""
convert_ani2x_checkpoint.py

Converts the real pretrained ANI-2x ensemble (as shipped in the `torchani`
Python package) into a manifest.json + weights.bin pair ani2x-model.js can
load directly in the browser -- no server, no PyTorch.

Unlike convert_chemprop_checkpoint.py / convert_nagl_checkpoint.py, this
script does NOT take a user-supplied checkpoint file. ANI-2x is a single
canonical public release (8-model ensemble, trained on H/C/N/O/F/S/Cl) --
there's one of it, not a bespoke training run per user, so this script just
instantiates `torchani.models.ANI2x()` and exports exactly what's inside.

Everything numeric (per-element network layer shapes, AEV cutoffs/eta/shift
constants, species order, self energies, the species-pair index table) is
read directly off the loaded model's real submodules -- nothing is
hardcoded from memory here, specifically so this can't silently drift from
whatever torchani actually ships. If a future torchani version changes the
architecture (different activation, different ensemble size, a cutoff
function other than the cosine cutoff this project's JS implements), this
script raises rather than silently exporting something the JS forward pass
would compute wrong.

Only supports the exact architecture ani2x-model.js/ani2x-features.js
implement:
  - AEV: ANIRadial + ANIAngular terms, CutoffCosine cutoff function
         (0.5*cos(pi*r/rc)+0.5), an all-pairs (non-periodic) neighborlist
  - per-element atomic networks: torchani's AtomicNetwork (stacked Linear
    layers with a TightCELU activation -- CELU with alpha=0.1 -- between
    them, and a final unactivated Linear layer), one such network per
    element per ensemble member
  - combination: mean of the 8 ensemble members' total (summed-over-atoms)
    energy, plus a constant per-atom self-energy correction

Usage:
    pip install torch torchani --break-system-packages
    python3 convert_ani2x_checkpoint.py model/ani2x/
"""

import argparse
import json
import sys
from pathlib import Path


SUPPORTED_ELEMENTS = ("H", "C", "N", "O", "F", "S", "Cl")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="directory to write manifest.json + weights.bin into")
    parser.add_argument("--name", default="ani2x", help="base filename (default: ani2x)")
    args = parser.parse_args()

    import torch
    import torchani

    model = torchani.models.ANI2x(periodic_table_index=False)

    nnp = model.potentials["nnp"]
    aev_computer = nnp.aev_computer
    ensemble = nnp.neural_networks
    energy_shifter = model.energy_shifter
    species_converter = model.species_converter

    # ---- species order + self energies ----
    atomic_numbers = species_converter.atomic_numbers.tolist()
    element_map = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}
    species = [element_map.get(z) for z in atomic_numbers]
    if any(s is None for s in species):
        sys.exit(f"Unexpected atomic number in species_converter.atomic_numbers: {atomic_numbers} "
                  f"-- this script only knows the element map for ANI-2x's usual 7 elements.")
    if set(species) != set(SUPPORTED_ELEMENTS):
        sys.exit(f"Species set {species} doesn't match the expected ANI-2x elements {SUPPORTED_ELEMENTS} "
                  "-- ani2x-model.js's compatibility check and AEV species-pair table assume exactly these 7.")

    self_energies = energy_shifter.self_energies.detach().numpy().astype("float64").tolist()
    if len(self_energies) != len(species):
        sys.exit("self_energies length doesn't match species length -- checkpoint layout mismatch.")

    # ---- AEV constants (read straight off the real modules, not hardcoded) ----
    radial = aev_computer.radial
    angular = aev_computer.angular

    if type(radial).__name__ != "ANIRadial" or type(angular).__name__ != "ANIAngular":
        sys.exit(f"Unsupported AEV term types (radial={type(radial).__name__}, angular={type(angular).__name__}) "
                  "-- ani2x-features.js only implements ANIRadial/ANIAngular.")
    if type(radial.cutoff_fn).__name__ != "CutoffCosine" or type(angular.cutoff_fn).__name__ != "CutoffCosine":
        sys.exit("Unsupported cutoff function -- ani2x-features.js only implements the cosine cutoff "
                  "(0.5*cos(pi*r/rc)+0.5).")

    radial_eta = float(radial.eta.item())
    radial_shifts = radial.shifts.detach().numpy().astype("float64").tolist()
    radial_cutoff = float(radial.cutoff)

    angular_eta = float(angular.eta.item())
    angular_zeta = float(angular.zeta.item())
    angular_shifts = angular.shifts.detach().numpy().astype("float64").tolist()  # "ShfA" (radial shifts)
    angular_sections = angular.sections.detach().numpy().astype("float64").tolist()  # "ShfZ" (angle shifts)
    angular_cutoff = float(angular.cutoff)

    num_species = aev_computer.num_species
    num_species_pairs = aev_computer.num_species_pairs
    radial_len = aev_computer.radial_len
    angular_len = aev_computer.angular_len
    aev_len = aev_computer.out_dim

    if radial_len != num_species * len(radial_shifts):
        sys.exit("radial_len doesn't match num_species * len(radial_shifts) -- unexpected AEVComputer layout.")
    if angular_len != num_species_pairs * len(angular_shifts) * len(angular_sections):
        sys.exit("angular_len doesn't match num_species_pairs * len(shifts) * len(sections) -- unexpected AEVComputer layout.")

    # triu_index: (num_species, num_species) long tensor mapping an unordered
    # species pair to its slot in the angular AEV. Read directly rather than
    # re-derived, so the JS side never has to guess torch.triu_indices' exact
    # enumeration order.
    species_pair_index = aev_computer.triu_index.detach().numpy().astype("int64").tolist()

    # ---- per-element, per-ensemble-member atomic networks ----
    members = list(ensemble.members)
    ensemble_size = len(members)
    if ensemble_size < 1:
        sys.exit("Ensemble has no members.")

    def arr(t):
        return t.detach().numpy().astype("float32")

    tensors = {}
    network_layer_dims = {}

    for m_idx, member in enumerate(members):
        atomics = member.atomics
        member_species = list(atomics.keys())
        if set(member_species) != set(species):
            sys.exit(f"Ensemble member {m_idx} has elements {member_species}, expected {species}.")

        for element in species:
            net = atomics[element]
            if type(net.activation).__name__ != "TightCELU":
                sys.exit(f"Unsupported activation {type(net.activation).__name__!r} for element {element!r} "
                          "-- ani2x-model.js only implements TightCELU (CELU with alpha=0.1).")
            if not net.has_biases:
                sys.exit(f"Element {element!r} network has no biases -- ani2x-model.js assumes every "
                          "Linear layer has a bias, matching ANI-2x's real architecture.")

            layer_dims = [net.layers[0].in_features]
            for i, layer in enumerate(net.layers):
                w, b = arr(layer.weight), arr(layer.bias)
                tensors[f"m{m_idx}_{element}_layer{i}_weight"] = w
                tensors[f"m{m_idx}_{element}_layer{i}_bias"] = b
                layer_dims.append(w.shape[0])
            w, b = arr(net.final_layer.weight), arr(net.final_layer.bias)
            tensors[f"m{m_idx}_{element}_final_weight"] = w
            tensors[f"m{m_idx}_{element}_final_bias"] = b
            layer_dims.append(w.shape[0])

            if layer_dims[-1] != 1:
                sys.exit(f"Element {element!r} final layer outputs {layer_dims[-1]} values, expected 1 "
                          "(a single per-atom energy contribution).")

            if m_idx == 0:
                network_layer_dims[element] = layer_dims
            elif network_layer_dims[element] != layer_dims:
                sys.exit(f"Ensemble member {m_idx}'s {element!r} network has layer_dims {layer_dims}, "
                          f"but member 0's is {network_layer_dims[element]} -- expected identical architecture "
                          "across ensemble members.")

    # ---- write flat float32 blob + manifest, same offset/length-in-elements convention as the other converters ----
    offset = 0
    manifest_tensors = {}
    blob = bytearray()
    for name, a in tensors.items():
        flat = a.reshape(-1)
        manifest_tensors[name] = {"shape": list(a.shape), "offset": offset, "length": int(flat.size)}
        blob += flat.tobytes()
        offset += flat.size

    manifest = {
        "architecture": "ani2x-ensemble",
        "task": "aniEnergy",
        "species": species,
        "atomicNumbers": atomic_numbers,
        "selfEnergies": self_energies,
        "ensembleSize": ensemble_size,
        "networks": {el: {"layerDims": network_layer_dims[el]} for el in species},
        "aev": {
            "radial": {"eta": radial_eta, "shifts": radial_shifts, "cutoff": radial_cutoff},
            "angular": {
                "eta": angular_eta,
                "zeta": angular_zeta,
                "shifts": angular_shifts,
                "sections": angular_sections,
                "cutoff": angular_cutoff,
            },
            "numSpecies": num_species,
            "numSpeciesPairs": num_species_pairs,
            "radialLen": radial_len,
            "angularLen": angular_len,
            "aevLen": aev_len,
            "speciesPairIndex": species_pair_index,
        },
        "tensors": manifest_tensors,
        "byteLength": len(blob),
        "dtype": "float32",
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{args.name}.bin"
    manifest_path = out_dir / "manifest.json"
    bin_path.write_bytes(bytes(blob))
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"wrote {bin_path} ({len(blob) / 1024 / 1024:.2f} MB)")
    print(f"wrote {manifest_path}")
    print(f"species: {species}, ensemble size: {ensemble_size}, AEV length: {aev_len}")
    print(f"self energies (Hartree): {dict(zip(species, self_energies))}")


if __name__ == "__main__":
    main()
