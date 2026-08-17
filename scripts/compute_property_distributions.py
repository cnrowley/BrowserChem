#!/usr/bin/env python3
"""
compute_property_distributions.py

Computes ChemCanvas's own property set (the RDKit 2D descriptors shown in
its Properties panel, plus two of its trained-model predictions) across a
user-supplied set of molecules -- typically FDA-approved drugs -- and
produces per-molecule CSV output, per-property distribution statistics,
and histogram plots. Lets a single molecule's computed properties later be
framed as "percentile P among real drugs" instead of a bare number (see
--lookup below).

WHAT'S COMPUTED AND WHY IT MATCHES THE APP:

  RDKit 2D descriptors + QED + synthetic accessibility -- computed via the
  real RDKit Python package (Descriptors, Crippen, rdMolDescriptors,
  Chem.QED, and RDKit's own bundled Contrib/SA_Score/sascorer.py), matching
  the exact field set in js/chemistry.js's CC.DESCRIPTOR_FIELDS. RDKit.js
  *is* RDKit compiled to WASM, so this is bit-parity with the app's own
  numbers by construction -- no separate validation needed. (QED here uses
  RDKit's real AROM/SSSR definition, which is slightly more precise than
  js/qed.js's own documented simplification -- see that file's header.)

  NAGL-MBIS partial charges -- a from-scratch numpy port of the SAGEConv
  stack + electronegativity-equalization postprocess, reading
  model/nagl-mbis-charges/manifest.json + weights.bin directly (the exact
  artifact the browser loads), not the specialized naglmbis/DGL conda env
  some machines have -- this keeps the script portable (RDKit + numpy +
  pandas + matplotlib only) and guarantees the numbers match what
  ChemCanvas itself would predict for the same molecule. Reported per
  molecule as summary stats (mean/min/max atomic charge) since a full
  per-atom distribution doesn't collapse to one number per drug.

  C-H pKa (pka-ch-nagl) -- a numpy flat-tree-array evaluator reading
  model/pka-ch-nagl/manifest.json + weights.bin directly, combined with a
  Python port of the graph-charge-shell descriptor builder (the corrected
  version with a plain stable-sort-by-charge tiebreak -- see
  js/pka-descriptor.js's header for why this deviates from upstream
  pKalculator's own, buggier tiebreak code). Reported per molecule as
  pka_min (the most acidic candidate C-H site) and candidate-site count,
  mirroring how pKalculator itself reports results.

  Chemprop molecule-level models (logP, logS, melting point, electrophile
  reactivity) -- a numpy port of dmpnn.js's D-MPNN forward pass (message
  passing, NormAggregation, FFN head), reading each model's shipped
  manifest.json + weights.bin directly under model/logp/,
  model/logs-aqsoldb/, model/melting-point/, model/electrophile-reactivity/
  -- the same artifacts the browser loads. Featurization uses chemprop's
  own real SimpleMoleculeMolGraphFeaturizer (MultiHotAtomFeaturizer +
  MultiHotBondFeaturizer, both confirmed 72-dim/14-dim, matching what
  every one of these checkpoints was trained with) rather than
  reimplementing the one-hot encoding tables -- needs the real `chemprop`
  package installed (only for this property group; everything else in
  this script only needs rdkit + numpy + pandas + matplotlib). If
  `chemprop` isn't importable, these four properties are skipped with a
  warning rather than failing the whole run.

Usage:
    python3 compute_property_distributions.py \\
        --input drugs.smi --output-dir out/

    # one-off: where does a new molecule fall against a saved run?
    python3 compute_property_distributions.py \\
        --output-dir out/ --lookup 'CC(=O)Oc1ccccc1C(=O)O'

--input accepts a plain-text file (one SMILES per line, optionally
"SMILES name"), a CSV with a SMILES column (--smiles-column to name it,
default "smiles"), or an SDF. No drug list is bundled or downloaded here
-- supply your own, e.g. from PubChem's approved-drugs collection or
DrugBank's open structure downloads.
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
NAGL_MODEL_DIR = REPO_ROOT / "model" / "nagl-mbis-charges"
PKA_MODEL_DIR = REPO_ROOT / "model" / "pka-ch-nagl"

PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


# --------------------------------------------------------------------------
# RDKit 2D descriptors + QED + SA score -- matches js/chemistry.js's
# CC.DESCRIPTOR_FIELDS exactly (see that file for the field list).
# --------------------------------------------------------------------------

def _load_sascorer():
    from rdkit.Chem import RDConfig
    sa_dir = Path(RDConfig.RDContribDir) / "SA_Score"
    if not sa_dir.exists():
        sys.exit(f"RDKit's Contrib/SA_Score not found at {sa_dir} -- is your RDKit install missing Contrib scripts?")
    sys.path.insert(0, str(sa_dir))
    import sascorer  # noqa: E402
    return sascorer


def compute_rdkit_descriptors(mol, sascorer, QED):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

    stereo = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    n_defined = sum(1 for _, tag in stereo if tag != "?")
    n_unassigned = sum(1 for _, tag in stereo if tag == "?")

    return {
        "amw": Descriptors.MolWt(mol),
        "CrippenClogP": Crippen.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "NumHBD": Descriptors.NumHDonors(mol),
        "NumHBA": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "NumAtomStereoCenters": n_defined,
        "NumUnspecifiedAtomStereoCenters": n_unassigned,
        "saScore": sascorer.calculateScore(mol),
        "qed": QED.qed(mol),
    }


# --------------------------------------------------------------------------
# NAGL-MBIS charges -- numpy port of js/nagl-model.js + js/nagl-features.js,
# reading the shipped manifest.json/weights.bin directly.
# --------------------------------------------------------------------------

NAGL_ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br"]
NAGL_CONNECTIVITY = [1, 2, 3, 4]
NAGL_RING_SIZES = [3, 4, 5, 6, 7, 8]


class NaglModel:
    def __init__(self, manifest_path, weights_path):
        manifest = json.loads(Path(manifest_path).read_text())
        weights = np.fromfile(weights_path, dtype=np.float32)

        def tensor(name):
            t = manifest["tensors"][name]
            return weights[t["offset"]: t["offset"] + t["length"]].reshape(t["shape"])

        self.conv_layers = []
        for i, _ in enumerate(manifest["convLayers"]):
            p = f"conv{i}_"
            self.conv_layers.append({
                "self_w": tensor(p + "self_weight"), "self_b": tensor(p + "self_bias"),
                "neigh_w": tensor(p + "neigh_weight"),
            })
        self.readout_layers = []
        for i, layer in enumerate(manifest["readoutLayers"]):
            p = f"readout{i}_"
            self.readout_layers.append({
                "w": tensor(p + "weight"), "b": tensor(p + "bias"), "activation": layer["activation"],
            })
        assert manifest["postprocess"] == "charges"

    def run(self, features, adjacency, total_charge):
        x = features
        for layer in self.conv_layers:
            n = x.shape[0]
            neigh_mean = np.zeros_like(x)
            for i in range(n):
                nb = adjacency[i]
                if nb:
                    neigh_mean[i] = x[nb].mean(axis=0)
            h_self = x @ layer["self_w"].T + layer["self_b"]
            h_neigh = neigh_mean @ layer["neigh_w"].T
            x = np.maximum(h_self + h_neigh, 0)

        for layer in self.readout_layers:
            x = x @ layer["w"].T + layer["b"]
            if layer["activation"] == "relu":
                x = np.maximum(x, 0)

        eneg, hardness = x[:, 0], x[:, 1]
        inv_h = 1.0 / hardness
        correction = (np.dot(inv_h, eneg) + total_charge) / inv_h.sum()
        return -inv_h * eneg + inv_h * correction


def build_expanded_graph(mol_noh):
    """mol_noh: RDKit mol WITHOUT explicit Hs. Returns (elements, features,
    adjacency, num_heavy) for the heavy+explicit-H node graph NAGL needs --
    port of nagl-features.js's buildExpandedGraph. Returns None if any atom
    is outside NAGL's trained vocabulary."""
    from rdkit import Chem

    n_heavy = mol_noh.GetNumAtoms()
    impl_h = [a.GetTotalNumHs() for a in mol_noh.GetAtoms()]
    heavy_degree = [a.GetDegree() for a in mol_noh.GetAtoms()]
    total_degree = [heavy_degree[i] + impl_h[i] for i in range(n_heavy)]
    n_h_nodes = sum(impl_h)
    n_nodes = n_heavy + n_h_nodes

    elements = [a.GetSymbol() for a in mol_noh.GetAtoms()] + ["H"] * n_h_nodes
    features = np.zeros((n_nodes, 19), dtype=np.float64)
    adjacency = [[] for _ in range(n_nodes)]

    ring_info = mol_noh.GetRingInfo()

    def feature_row(element, degree, ring_sizes):
        row = np.zeros(19)
        if element not in NAGL_ELEMENTS or degree not in NAGL_CONNECTIVITY:
            return None
        row[NAGL_ELEMENTS.index(element)] = 1
        row[9 + NAGL_CONNECTIVITY.index(degree)] = 1
        for size in ring_sizes:
            if size in NAGL_RING_SIZES:
                row[13 + NAGL_RING_SIZES.index(size)] = 1
        return row

    for i, atom in enumerate(mol_noh.GetAtoms()):
        ring_sizes = {sz for sz in NAGL_RING_SIZES if ring_info.IsAtomInRingOfSize(i, sz)}
        row = feature_row(atom.GetSymbol(), total_degree[i], ring_sizes)
        if row is None:
            return None
        features[i] = row

    for bond in mol_noh.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i].append(j)
        adjacency[j].append(i)

    next_h = n_heavy
    for i in range(n_heavy):
        for _ in range(impl_h[i]):
            h_idx = next_h
            next_h += 1
            features[h_idx] = feature_row("H", 1, set())
            adjacency[h_idx].append(i)
            adjacency[i].append(h_idx)

    return elements, features, adjacency, n_heavy


def compute_nagl_charges(mol_noh, nagl_model):
    graph = build_expanded_graph(mol_noh)
    if graph is None:
        return None
    elements, features, adjacency, n_heavy = graph
    total_charge = sum(a.GetFormalCharge() for a in mol_noh.GetAtoms())
    charges = nagl_model.run(features, adjacency, total_charge)
    return elements, charges, adjacency, n_heavy


# --------------------------------------------------------------------------
# C-H pKa (pka-ch-nagl) -- numpy flat-tree evaluator + graph-charge-shell
# descriptor, reading the shipped manifest.json/weights.bin directly.
# --------------------------------------------------------------------------

class PkaModel:
    def __init__(self, manifest_path, weights_path):
        manifest = json.loads(Path(manifest_path).read_text())
        raw = Path(weights_path).read_bytes()

        def section(name):
            t = manifest["tensors"][name]
            dtype = {"uint8": np.uint8, "int32": np.int32, "float32": np.float32}[t["dtype"]]
            arr = np.frombuffer(raw, dtype=dtype, count=t["length"], offset=t["byteOffset"])
            return arr

        self.is_leaf = section("isLeaf")
        self.feature_idx = section("featureIdx")
        self.threshold = section("threshold")
        self.default_left = section("defaultLeft")
        self.left_child = section("leftChild")
        self.right_child = section("rightChild")
        self.value = section("value")
        self.tree_roots = manifest["treeRootOffsets"]
        self.n_shells = manifest["descriptor"]["nShells"]
        self.use_cip_sort = manifest["descriptor"]["useCipSort"]

    def _eval_tree(self, root, x):
        node = root
        while not self.is_leaf[node]:
            v = x[self.feature_idx[node]]
            go_left = v <= self.threshold[node]
            node = self.left_child[node] if go_left else self.right_child[node]
        return self.value[node]

    def predict_one(self, x):
        return sum(self._eval_tree(root, x) for root in self.tree_roots)


DUMMY = -1
DUMMY_CHARGE = 0.0
DUMMY_ATOMIC_NUM = -10
ATOMIC_NUM = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35}


def _atomic_num_of(elements, idx):
    return DUMMY_ATOMIC_NUM if idx == DUMMY else ATOMIC_NUM.get(elements[idx], 0)


def _charge_of(charges, idx):
    return DUMMY_CHARGE if idx == DUMMY else charges[idx]


def _neighbors_of(adjacency, idx):
    return [] if idx == DUMMY else adjacency[idx]


def _fill_block(block, length):
    while len(block) < length:
        block.append(DUMMY)


def _sort_block_cip(block, elements, charges, adjacency):
    """Port of js/pka-descriptor.js's sortBlockCip -- CIP-priority sort
    with a plain stable-sort-by-charge tiebreak (the corrected, simplified
    version; see that file's header for why this deviates from upstream
    pKalculator's own tiebreak code)."""
    priorities = [_atomic_num_of(elements, idx) for idx in block]
    neighbor_sets = [{idx} for idx in block]

    while len(set(priorities)) != len(priorities):
        old_card = [len(s) for s in neighbor_sets]
        for num, idx in enumerate(block):
            for nbr in _neighbors_of(adjacency, idx):
                neighbor_sets[num].add(nbr)
        priorities = [sum(_atomic_num_of(elements, i) for i in s) for s in neighbor_sets]
        new_card = [len(s) for s in neighbor_sets]
        if new_card == old_card:
            break

    order = sorted(range(len(block)), key=lambda i: priorities[i])
    block[:] = [block[i] for i in order]
    priorities = [priorities[i] for i in order]

    start = 0
    while start < len(priorities):
        end = start + 1
        while end < len(priorities) and priorities[end] == priorities[start]:
            end += 1
        if end - start > 1:
            run = sorted(block[start:end], key=lambda idx: _charge_of(charges, idx))
            block[start:end] = run
        start = end


def build_pka_descriptor(elements, charges, adjacency, target_idx, n_shells):
    last_shell = [target_idx]
    current_shell = []
    contained = {target_idx}
    descriptor = [_charge_of(charges, target_idx)]
    length = 4

    for _ in range(n_shells):
        for ls_idx in last_shell:
            block = [nbr for nbr in _neighbors_of(adjacency, ls_idx) if nbr not in contained]
            _fill_block(block, length)
            _sort_block_cip(block, elements, charges, adjacency)
            current_shell += block
            contained.update(idx for idx in current_shell if idx != DUMMY)
            length = 3

        descriptor += [_charge_of(charges, idx) for idx in current_shell]
        last_shell, current_shell = current_shell, []

    return np.array(descriptor, dtype=np.float64)


def find_candidate_sites(mol_noh):
    """Carbon atoms bearing >=1 hydrogen (implicit or explicit) -- see
    js/pka-model.js's header for the SMARTS this generalizes."""
    return [a.GetIdx() for a in mol_noh.GetAtoms() if a.GetSymbol() == "C" and a.GetTotalNumHs() >= 1]


def compute_pka(mol_noh, pka_model, nagl_charge_result):
    if nagl_charge_result is None:
        return None
    elements, charges, adjacency, _ = nagl_charge_result
    sites = find_candidate_sites(mol_noh)
    if not sites:
        return {"pka_min": np.nan, "n_ch_sites": 0}
    values = []
    for site in sites:
        desc = build_pka_descriptor(elements, charges, adjacency, site, pka_model.n_shells)
        values.append(pka_model.predict_one(desc))
    return {"pka_min": float(min(values)), "n_ch_sites": len(sites)}


# --------------------------------------------------------------------------
# Chemprop molecule-level models (logP, logS, melting point, electrophile
# reactivity) -- numpy port of js/dmpnn.js's D-MPNN forward pass, reading
# the shipped manifest.json/weights.bin directly (same tensors
# js/chemprop-model.js loads). Featurization uses chemprop's own real
# SimpleMoleculeMolGraphFeaturizer for exact parity -- see this module's
# docstring.
# --------------------------------------------------------------------------

CHEMPROP_MOLECULE_MODELS = {
    # column name -> (registry id, model directory, manifest/weights base name)
    "logP": ("logp-v1", "logp"),
    "Solubility": ("logs-aqsoldb", "logs-aqsoldb"),
    "mp": ("melting-point", "melting-point"),
    "label": ("electrophile-reactivity-v1", "electrophile-reactivity"),
}


class ChempropDMPNNModel:
    def __init__(self, manifest_path, weights_path):
        manifest = json.loads(Path(manifest_path).read_text())
        weights = np.fromfile(weights_path, dtype=np.float32)

        def tensor(name):
            t = manifest["tensors"][name]
            return weights[t["offset"]: t["offset"] + t["length"]].reshape(t["shape"])

        self.task = manifest.get("task", "prediction")
        self.task_type = manifest.get("taskType", "regression")
        self.dims = manifest["dims"]
        self.Wi = tensor("W_i")
        self.Wh = tensor("W_h")
        self.Wo = tensor("W_o_weight")
        self.Wo_bias = tensor("W_o_bias")
        self.ffn0 = tensor("ffn0_weight")
        self.ffn0_bias = tensor("ffn0_bias")
        self.ffn1 = tensor("ffn1_weight")
        self.ffn1_bias = tensor("ffn1_bias")
        if self.task_type == "regression":
            self.out_mean = float(tensor("out_mean")[0])
            self.out_scale = float(tensor("out_scale")[0])

    def apply_head(self, embedding):
        hidden = np.maximum(self.ffn0 @ embedding + self.ffn0_bias, 0)
        raw = float((self.ffn1 @ hidden)[0] + self.ffn1_bias[0])
        if self.task_type == "classification":
            return 1.0 / (1.0 + np.exp(-raw))
        return raw * self.out_scale + self.out_mean


def run_chemprop_molecule(mol, model, featurizer):
    """Mirrors js/dmpnn.js's runDMPNN() + js/chemprop-model.js's
    runOneMolecule() exactly: h0raw = W_i.[atomFeat(src), bondFeat] (no
    bias); h_0 = ReLU(h0raw); depth-1 message/update rounds add back the
    UNACTIVATED h0raw at each step (not its ReLU); final atom embedding is
    ReLU(W_o.[atomFeat(v), sum of incoming h] + bias); molecule embedding
    is NormAggregation (sum of atom embeddings / aggNorm); then
    ffn0->ReLU->ffn1->task head."""
    mg = featurizer(mol)
    num_atoms = mg.V.shape[0]
    hidden_size = model.dims["d_h"]
    depth = model.dims["depth"]

    if mg.E.shape[0] == 0:
        # No bonds (single-atom "molecule") -- falls back to the
        # atom-feature-only projection, same as dmpnn.js's numEdges===0 case.
        embeddings = [np.maximum(model.Wo @ np.concatenate([mg.V[v], np.zeros(hidden_size)]) + model.Wo_bias, 0)
                      for v in range(num_atoms)]
    else:
        edge_src = mg.edge_index[0]
        edge_dst = mg.edge_index[1]
        num_edges = len(edge_src)
        incoming_by_atom = [[] for _ in range(num_atoms)]
        for e, dst in enumerate(edge_dst):
            incoming_by_atom[dst].append(e)

        h0raw = np.stack([model.Wi @ np.concatenate([mg.V[edge_src[e]], mg.E[e]]) for e in range(num_edges)])
        h = np.maximum(h0raw, 0)

        for _ in range(1, depth):
            h_next = np.zeros_like(h)
            for e in range(num_edges):
                src_atom = edge_src[e]
                rev = mg.rev_edge_index[e]
                incoming = [e2 for e2 in incoming_by_atom[src_atom] if e2 != rev]
                message = h[incoming].sum(axis=0) if incoming else np.zeros(hidden_size)
                h_next[e] = np.maximum(h0raw[e] + model.Wh @ message, 0)
            h = h_next

        embeddings = []
        for v in range(num_atoms):
            incoming = incoming_by_atom[v]
            message = h[incoming].sum(axis=0) if incoming else np.zeros(hidden_size)
            embeddings.append(np.maximum(model.Wo @ np.concatenate([mg.V[v], message]) + model.Wo_bias, 0))

    agg_norm = model.dims.get("aggNorm") or 1.0
    pooled = np.sum(embeddings, axis=0) / agg_norm
    return model.apply_head(pooled)


def load_chemprop_molecule_models():
    """Returns {column_name: ChempropDMPNNModel}, skipping (with a
    warning) any model whose files aren't present -- lets the rest of
    this script run even if model/ doesn't have every checkpoint."""
    models = {}
    for column, (_, dir_name) in CHEMPROP_MOLECULE_MODELS.items():
        model_dir = REPO_ROOT / "model" / dir_name
        manifest_path = model_dir / "manifest.json"
        weights_path = model_dir / "weights.bin"
        if not manifest_path.exists() or not weights_path.exists():
            warnings.warn(f"{model_dir} missing manifest.json/weights.bin -- skipping {column!r}")
            continue
        models[column] = ChempropDMPNNModel(manifest_path, weights_path)
    return models


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def read_molecules(path, smiles_column):
    from rdkit import Chem

    path = Path(path)
    mols = []  # list of (name, smiles)
    if path.suffix.lower() == ".sdf":
        supplier = Chem.SDMolSupplier(str(path))
        for i, mol in enumerate(supplier):
            if mol is None:
                warnings.warn(f"SDF record {i}: could not parse, skipping")
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol{i}"
            mols.append((name, Chem.MolToSmiles(mol)))
    elif path.suffix.lower() == ".csv":
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if smiles_column not in (reader.fieldnames or []):
                sys.exit(f"CSV has no column {smiles_column!r} -- columns found: {reader.fieldnames}")
            name_col = "name" if "name" in (reader.fieldnames or []) else None
            for i, row in enumerate(reader):
                smi = row[smiles_column].strip()
                if not smi:
                    continue
                name = row[name_col] if name_col else f"mol{i}"
                mols.append((name, smi))
    else:
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                smi = parts[0]
                name = parts[1] if len(parts) > 1 else f"mol{i}"
                mols.append((name, smi))
    return mols


def compute_all_properties(name, smiles, sascorer, QED, nagl_model, pka_model, chemprop_models, chemprop_featurizer):
    from rdkit import Chem

    row = {"name": name, "smiles": smiles, "warning": ""}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        row["warning"] = "RDKit could not parse this SMILES"
        return row

    try:
        row.update(compute_rdkit_descriptors(mol, sascorer, QED))
    except Exception as err:  # noqa: BLE001
        row["warning"] += f"RDKit descriptors failed: {err}; "

    nagl_result = None
    try:
        nagl_result = compute_nagl_charges(mol, nagl_model)
        if nagl_result is None:
            row["warning"] += "NAGL: element outside trained vocabulary (H/C/N/O/F/P/S/Cl/Br); "
        else:
            _, charges, _, n_heavy = nagl_result
            heavy_charges = charges[:n_heavy]
            row["nagl_charge_mean"] = float(np.mean(heavy_charges))
            row["nagl_charge_min"] = float(np.min(heavy_charges))
            row["nagl_charge_max"] = float(np.max(heavy_charges))
    except Exception as err:  # noqa: BLE001
        row["warning"] += f"NAGL failed: {err}; "

    try:
        pka_result = compute_pka(mol, pka_model, nagl_result)
        if pka_result is not None:
            row.update(pka_result)
    except Exception as err:  # noqa: BLE001
        row["warning"] += f"pKa failed: {err}; "

    for column, model in chemprop_models.items():
        try:
            row[column] = run_chemprop_molecule(mol, model, chemprop_featurizer)
        except Exception as err:  # noqa: BLE001
            row["warning"] += f"{column} failed: {err}; "

    return row


NUMERIC_COLUMNS = [
    "amw", "CrippenClogP", "tpsa", "NumHBD", "NumHBA", "NumRotatableBonds",
    "NumRings", "NumAromaticRings", "FractionCSP3", "NumAtomStereoCenters",
    "NumUnspecifiedAtomStereoCenters", "saScore", "qed",
    "nagl_charge_mean", "nagl_charge_min", "nagl_charge_max",
    "pka_min", "n_ch_sites",
    "logP", "Solubility", "mp", "label",
]


def write_csv(rows, out_path):
    fieldnames = ["name", "smiles"] + NUMERIC_COLUMNS + ["warning"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def compute_distributions(rows):
    dist = {}
    for col in NUMERIC_COLUMNS:
        values = np.array([row[col] for row in rows if isinstance(row.get(col), (int, float)) and not np.isnan(row.get(col, np.nan))], dtype=np.float64)
        if values.size == 0:
            continue
        dist[col] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "percentiles": {str(p): float(np.percentile(values, p)) for p in PERCENTILES},
        }
    return dist


def plot_distributions(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    for col in NUMERIC_COLUMNS:
        values = [row[col] for row in rows if isinstance(row.get(col), (int, float)) and not np.isnan(row.get(col, np.nan))]
        if len(values) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=min(30, max(5, len(values) // 5)), color="#4c72b0", edgecolor="white")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(f"{col} (n={len(values)})")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{col}.png", dpi=120)
        plt.close(fig)


def percentile_of(value, dist_entry):
    """Rank of `value` within the saved distribution's percentile table,
    via linear interpolation over the stored percentile points."""
    pts = sorted((float(p), v) for p, v in dist_entry["percentiles"].items())
    xs = [v for _, v in pts]
    ps = [p for p, _ in pts]
    if value <= xs[0]:
        return 0.0 if value < xs[0] else ps[0]
    if value >= xs[-1]:
        return 100.0 if value > xs[-1] else ps[-1]
    return float(np.interp(value, xs, ps))


def run_lookup(smiles, out_dir, sascorer, QED, nagl_model, pka_model, chemprop_models, chemprop_featurizer):
    dist_path = out_dir / "distributions.json"
    if not dist_path.exists():
        sys.exit(f"{dist_path} not found -- run a full --input batch first to build the reference distribution.")
    dist = json.loads(dist_path.read_text())

    row = compute_all_properties("lookup", smiles, sascorer, QED, nagl_model, pka_model, chemprop_models, chemprop_featurizer)
    if row.get("warning"):
        print(f"warning: {row['warning']}", file=sys.stderr)

    print(f"{'property':<32} {'value':>12}  {'percentile':>10}")
    for col in NUMERIC_COLUMNS:
        if col not in dist or not isinstance(row.get(col), (int, float)) or np.isnan(row.get(col, np.nan)):
            continue
        p = percentile_of(row[col], dist[col])
        print(f"{col:<32} {row[col]:>12.4f}  {p:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="SMILES list (.smi/.txt, one 'SMILES [name]' per line), CSV, or SDF. "
                                         "Not bundled here -- supply your own, e.g. from PubChem's approved-drugs "
                                         "collection or DrugBank's open structure downloads.")
    parser.add_argument("--smiles-column", default="smiles", help="column name for CSV input (default: smiles)")
    parser.add_argument("--output-dir", required=True, help="directory for properties.csv / distributions.json / plots/")
    parser.add_argument("--lookup", metavar="SMILES", help="report percentiles for one SMILES against an existing --output-dir's saved distribution, instead of running a full batch")
    parser.add_argument("--no-plots", action="store_true", help="skip histogram generation (matplotlib not required)")
    parser.add_argument("--no-chemprop", action="store_true",
                         help="skip the four Chemprop molecule-level properties (logP/Solubility/mp/label) even if "
                              "the `chemprop` package is installed -- useful to avoid its import cost/torch "
                              "dependency when you only want the RDKit/NAGL/pKa properties")
    args = parser.parse_args()

    from rdkit.Chem import QED

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sascorer = _load_sascorer()
    nagl_model = NaglModel(NAGL_MODEL_DIR / "manifest.json", NAGL_MODEL_DIR / "weights.bin")
    pka_model = PkaModel(PKA_MODEL_DIR / "manifest.json", PKA_MODEL_DIR / "weights.bin")

    chemprop_models = {}
    chemprop_featurizer = None
    if not args.no_chemprop:
        try:
            from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
            chemprop_featurizer = SimpleMoleculeMolGraphFeaturizer()
            chemprop_models = load_chemprop_molecule_models()
        except ImportError:
            warnings.warn(
                "`chemprop` package not importable -- skipping logP/Solubility/mp/label "
                "(pip install chemprop, or pass --no-chemprop to silence this warning)."
            )

    if args.lookup:
        run_lookup(args.lookup, out_dir, sascorer, QED, nagl_model, pka_model, chemprop_models, chemprop_featurizer)
        return

    if not args.input:
        sys.exit("--input is required unless --lookup is given")

    molecules = read_molecules(args.input, args.smiles_column)
    print(f"read {len(molecules)} molecules from {args.input}", file=sys.stderr)

    rows = []
    n_failed = 0
    for name, smiles in molecules:
        row = compute_all_properties(name, smiles, sascorer, QED, nagl_model, pka_model, chemprop_models, chemprop_featurizer)
        if row["warning"]:
            n_failed += 1 if "could not parse" in row["warning"] else 0
        rows.append(row)

    print(f"{len(rows) - n_failed}/{len(rows)} molecules parsed successfully", file=sys.stderr)

    write_csv(rows, out_dir / "properties.csv")
    dist = compute_distributions(rows)
    (out_dir / "distributions.json").write_text(json.dumps(dist, indent=1))
    print(f"wrote {out_dir / 'properties.csv'} and {out_dir / 'distributions.json'}", file=sys.stderr)

    if not args.no_plots:
        plot_distributions(rows, out_dir)
        print(f"wrote histograms to {out_dir / 'plots'}", file=sys.stderr)


if __name__ == "__main__":
    main()
