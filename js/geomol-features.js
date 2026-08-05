/**
 * geomol-features.js
 *
 * Featurization exactly matching GeoMol's real
 * `model/featurization.py:featurize_mol_from_smiles` (Ganea et al.,
 * NeurIPS 2021 -- github.com/PattanaikL/GeoMol) for the "drugs" element
 * table, since that's the checkpoint this integration ships (see
 * GEOMOL_INTEGRATION.md). This is deliberately NOT reusing
 * atom-features.js/chemprop-features.js's feature *layout* -- GeoMol's
 * own published architecture expects its own specific 74-dim node /
 * 4-dim edge vectors in its own specific field order, verified against a
 * live run of the real PyTorch model (see GEOMOL_INTEGRATION.md's
 * validation section), not something to approximate independently.
 *
 * Unlike this app's own Molecule model (implicit hydrogens, no explicit
 * H atoms/nodes), GeoMol's original Python featurizer runs on an
 * RDKit `Chem.AddHs()`'d molecule -- every hydrogen is its own graph
 * node. RDKit.js's `add_hs_in_place()` does the same thing (confirmed
 * directly: heavy atoms keep their original index order, hydrogens are
 * appended afterward in neighbor order, identical to real RDKit's own
 * AddHs), so that's what this file runs its own RDKit annotation pass
 * against -- CC.GNN.getRDKitAnnotations() isn't reused here since it's
 * keyed to this app's own implicit-H atom ordering, not the
 * explicit-H-included ordering GeoMol needs.
 *
 * Known gaps against the real RDKit-Python featurizer, honestly:
 *   - Hybridization: RDKit.js's get_json() has no per-atom hybridization
 *     field at all (same gap chemprop-features.js already documented and
 *     worked around) -- reuses that file's validated guessHybridization()
 *     rather than a second, independent guess. One correction needed here
 *     specifically: guessHybridization() was only ever exercised on heavy
 *     atoms (this app never draws explicit H); a bare hydrogen falls
 *     through its logic to 'SP3', but real RDKit reports hydrogen's
 *     hybridization as something outside GeoMol's five-way
 *     SP/SP2/SP3/SP3D/SP3D2 choice list, landing in the "other" bucket
 *     one_k_encoding reserves for exactly that -- confirmed directly
 *     against a live GeoMol ground-truth run, not assumed. Special-cased
 *     below (every H atom is forced into the "other" bucket) rather than
 *     silently misclassifying every hydrogen as sp3.
 *   - Implicit valence: GeoMol's "implicit valence" one-hot block is
 *     computed on the *already-AddHs'd* molecule, where by definition no
 *     atom has any implicit valence left -- confirmed directly against
 *     ground truth (this block is the constant one-hot "0" for every
 *     atom of every test molecule tried). Hardcoded as that constant
 *     rather than computed, since there is nothing left to compute.
 */

window.CC = window.CC || {};
CC.GNN = window.CC.GNN || {};

(function () {
  // Exact order from model/featurization.py's drugs_types dict --
  // position is the one-hot index, nothing here is alphabetical or
  // otherwise reorderable.
  const DRUGS_TYPES = [
    'H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si',
    'P', 'S', 'Cl', 'K', 'Ca', 'V', 'Cr', 'Mn', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Ag', 'In', 'Sb', 'I', 'Gd',
    'Pt', 'Au', 'Hg', 'Bi',
  ];

  const DEGREE_CHOICES = [0, 1, 2, 3, 4, 5, 6];
  const HYBRIDIZATION_CHOICES = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'];
  const FORMAL_CHARGE_CHOICES = [-1, 0, 1];
  const RING_SIZE_CHOICES = [3, 4, 5, 6, 7, 8];
  const RING_COUNT_CHOICES = [0, 1, 2, 3];
  const BOND_TYPE_ORDER = ['single', 'double', 'triple', 'aromatic']; // index = GeoMol's edge_type

  // one_k_encoding: exact one-hot + trailing "unmatched" slot, matching
  // GeoMol's own one_k_encoding() (see model/featurization.py) --
  // anything not in `choices` lands in that extra slot rather than being
  // silently dropped or clamped to the nearest choice.
  function oneK(value, choices) {
    const vec = new Array(choices.length + 1).fill(0);
    const idx = choices.indexOf(value);
    vec[idx === -1 ? choices.length : idx] = 1;
    return vec;
  }

  // Reverse of CC.ELEMENT_TO_ATOMIC_NUMBER -- covers every element this
  // app can actually draw (H, C, N, O, F, S, Cl, Br) plus everything else
  // that table knows; GeoMol's own drugs_types list goes further (Li, Na,
  // Pt, ...) but this app has no way to produce those atoms in the first
  // place, so there's nothing to look up for them.
  const ATOMIC_NUMBER_TO_ELEMENT = {};
  Object.keys(CC.ELEMENT_TO_ATOMIC_NUMBER).forEach(function (sym) {
    ATOMIC_NUMBER_TO_ELEMENT[CC.ELEMENT_TO_ATOMIC_NUMBER[sym]] = sym;
  });

  function parseGeomolMol(molecule) {
    const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
    if (!RDKit) throw new Error('RDKit.js is not ready yet');

    const molblock = CC.moleculeToMolblock(molecule);
    const mol = RDKit.get_mol(molblock);
    if (!mol || !mol.is_valid()) {
      if (mol) mol.delete();
      throw new Error('RDKit could not parse this structure');
    }
    try {
      mol.add_hs_in_place();
      const json = JSON.parse(mol.get_json());
      const molData = json.molecules[0];
      const defaults = (json.defaults && json.defaults.atom) || { z: 6, chg: 0, impHs: 0, stereo: 'unspecified' };
      const bondDefaults = (json.defaults && json.defaults.bond) || { bo: 1 };
      const ext = (molData.extensions || []).find(function (e) { return e.name === 'rdkitRepresentation'; });

      const aromaticAtomIndices = new Set((ext && ext.aromaticAtoms) || []);
      const aromaticBondIndices = new Set((ext && ext.aromaticBonds) || []);
      const ringSizesByAtomIndex = new Map();
      const ringCountByAtomIndex = new Map();
      ((ext && ext.atomRings) || []).forEach(function (ring) {
        ring.forEach(function (idx) {
          if (!ringSizesByAtomIndex.has(idx)) ringSizesByAtomIndex.set(idx, new Set());
          ringSizesByAtomIndex.get(idx).add(ring.length);
          ringCountByAtomIndex.set(idx, (ringCountByAtomIndex.get(idx) || 0) + 1);
        });
      });

      return {
        atoms: molData.atoms || [],
        bonds: molData.bonds || [],
        atomDefaults: defaults,
        bondDefaults: bondDefaults,
        aromaticAtomIndices: aromaticAtomIndices,
        aromaticBondIndices: aromaticBondIndices,
        ringSizesByAtomIndex: ringSizesByAtomIndex,
        ringCountByAtomIndex: ringCountByAtomIndex,
        rings: (ext && ext.atomRings) || [], // raw per-ring atom-index lists, needed for dihedral-pair traversal ordering
      };
    } finally {
      mol.delete();
    }
  }

  /**
   * Builds everything CC.GeoMol's forward pass needs from a CC.Molecule:
   * node features (n_atoms x 74), edge features (n_edges x 4, directed --
   * every bond emitted both ways like the real featurizer's row/col
   * construction), edge_index (2 x n_edges), a neighbors map (atom index
   * -> array of neighbor indices, only for atoms with >1 neighbor,
   * matching get_neighbor_ids' "degree > 1" filter), and chiralTag
   * (-1 CW / +1 CCW / 0 unspecified per atom, GeoMol's own convention).
   *
   * Does NOT need this app's own atom-id system past this point --
   * everything downstream (the GNN, local-structure prediction, dihedral
   * assembly) works purely in these dense 0..n-1 indices, same as the
   * real PyTorch Geometric Data object would.
   */
  CC.GNN.buildGeomolInput = function (molecule) {
    const parsed = parseGeomolMol(molecule);
    const atoms = parsed.atoms;
    const n = atoms.length;

    // ---- node features ----
    const x = new Array(n);
    const elementByIndex = new Array(n);
    for (let i = 0; i < n; i++) {
      const a = atoms[i];
      const z = a.z !== undefined ? a.z : parsed.atomDefaults.z;
      elementByIndex[i] = ATOMIC_NUMBER_TO_ELEMENT[z] || null;
    }

    // Degree (post-AddHs, so H-X bonds count too) and per-atom neighbor
    // list, derived straight from the bonds list -- exactly what real
    // RDKit's GetDegree()/GetNeighbors() report on the same AddHs'd mol.
    const neighborsFull = Array.from({ length: n }, function () { return []; });
    parsed.bonds.forEach(function (b) {
      const a1 = b.atoms[0], a2 = b.atoms[1];
      neighborsFull[a1].push(a2);
      neighborsFull[a2].push(a1);
    });

    for (let i = 0; i < n; i++) {
      const a = atoms[i];
      const z = a.z !== undefined ? a.z : parsed.atomDefaults.z;
      const element = elementByIndex[i];
      const charge = a.chg !== undefined ? a.chg : parsed.atomDefaults.chg;
      const isAromatic = parsed.aromaticAtomIndices.has(i);
      const degree = neighborsFull[i].length;
      const ringSizes = parsed.ringSizesByAtomIndex.get(i) || new Set();
      const ringCount = parsed.ringCountByAtomIndex.get(i) || 0;

      const typeOneHot = new Array(DRUGS_TYPES.length).fill(0);
      const typeIdx = DRUGS_TYPES.indexOf(element);
      if (typeIdx === -1) {
        throw new Error('Element "' + element + '" is outside GeoMol\'s supported (drugs) element table');
      }
      typeOneHot[typeIdx] = 1;

      // Hybridization needs this app's own heavy-atom bond graph
      // (guessHybridization walks molecule.getBondsForAtom), which isn't
      // available from just this AddHs-index loop -- filled in by the
      // dedicated pass below instead of computed here.

      const row = [].concat(
        typeOneHot,
        [z, isAromatic ? 1 : 0],
        oneK(Math.min(degree, 6), DEGREE_CHOICES),
        [0, 0, 0, 0, 0, 0], // hybridization placeholder, filled in below
        [1, 0, 0, 0, 0, 0, 0, 0], // implicit valence: always "0" post-AddHs (see file header)
        oneK(Math.max(-1, Math.min(1, charge)), FORMAL_CHARGE_CHOICES),
        RING_SIZE_CHOICES.map(function (size) { return ringSizes.has(size) ? 1 : 0; }),
        oneK(Math.min(ringCount, 3), RING_COUNT_CHOICES)
      );
      x[i] = row;
    }

    // ---- hybridization pass (needs this app's own heavy-atom bond graph) ----
    // Build a heavy-atom-id -> AddHs-index map by relying on the fact
    // add_hs_in_place() preserves heavy-atom order (confirmed directly --
    // see file header) -- the first heavyAtoms.length AddHs-indices are
    // exactly this app's own atoms, in molecule.atoms iteration order.
    const heavyAtoms = Array.from(molecule.atoms.values());
    const aromaticSetForGuess = new Set();
    heavyAtoms.forEach(function (a, i) { if (parsed.aromaticAtomIndices.has(i)) aromaticSetForGuess.add(i); });
    const atomIdToIndex = new Map();
    heavyAtoms.forEach(function (a, i) { atomIdToIndex.set(a.id, i); });

    for (let i = 0; i < n; i++) {
      const element = elementByIndex[i];
      let hybChoice;
      if (element === 'H') {
        hybChoice = null;
      } else {
        const heavyAtom = heavyAtoms[i]; // valid since heavy atoms occupy indices [0, heavyAtoms.length)
        const degree = neighborsFull[i].length;
        hybChoice = CC.GNN.guessHybridization(
          molecule, heavyAtom.id, parsed.aromaticAtomIndices.has(i), degree, element, atomIdToIndex, aromaticSetForGuess
        );
      }
      const hybOneHot = oneK(hybChoice, HYBRIDIZATION_CHOICES);
      for (let k = 0; k < hybOneHot.length; k++) x[i][35 + 2 + 8 + k] = hybOneHot[k];
    }

    // ---- edge features + directed edge_index (both directions, like the real featurizer) ----
    const edgeSrc = [];
    const edgeDst = [];
    const edgeAttr = [];
    parsed.bonds.forEach(function (b, bondIdx) {
      const a1 = b.atoms[0], a2 = b.atoms[1];
      const isAromatic = parsed.aromaticBondIndices.has(bondIdx);
      const bo = b.bo !== undefined ? b.bo : parsed.bondDefaults.bo;
      const bondType = isAromatic ? 'aromatic' : (bo === 2 ? 'double' : bo === 3 ? 'triple' : 'single');
      const typeIdx = BOND_TYPE_ORDER.indexOf(bondType);
      const oneHot = [0, 0, 0, 0];
      oneHot[typeIdx] = 1;

      edgeSrc.push(a1); edgeDst.push(a2); edgeAttr.push(oneHot);
      edgeSrc.push(a2); edgeDst.push(a1); edgeAttr.push(oneHot);
    });

    // ---- neighbors map (degree > 1 only, matching get_neighbor_ids) ----
    const neighbors = {};
    for (let i = 0; i < n; i++) {
      if (neighborsFull[i].length > 1) neighbors[i] = neighborsFull[i].slice();
    }

    // ---- chirality (-1 CW / +1 CCW / 0 unspecified, GeoMol's own sign convention) ----
    const chiralTag = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const stereo = atoms[i].stereo;
      if (stereo === 'cw') chiralTag[i] = -1;
      else if (stereo === 'ccw') chiralTag[i] = 1;
    }

    return {
      numAtoms: n,
      x: x,
      edgeSrc: edgeSrc,
      edgeDst: edgeDst,
      edgeAttr: edgeAttr,
      neighbors: neighbors,
      chiralTag: chiralTag,
      elementByIndex: elementByIndex,
      rings: parsed.rings, // array of arrays of atom indices (RDKit's SSSR) -- needed for dihedral-pair traversal ordering
    };
  };
})();
