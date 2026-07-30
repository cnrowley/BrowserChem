/**
 * graph-builder.js
 *
 * Two jobs:
 *   1. getRDKitAnnotations() — ask RDKit.js for aromaticity and ring
 *      membership via its CommonChem-style get_json() output, since our
 *      own Molecule model doesn't perceive rings or aromaticity itself.
 *   2. buildMolGraph() — turn a Molecule into the directed-edge tensors a
 *      D-MPNN needs: every undirected bond becomes two directed edges
 *      (i->j and j->i), each edge knows its reverse edge's index (needed
 *      to exclude reverse-edge messages during message passing), and each
 *      atom knows which directed edges point into it (for aggregation).
 *
 * RDKit.js's JSMol doesn't expose atom-by-atom aromaticity queries
 * directly, but get_json() returns the full CommonChem representation
 * including an "rdkitRepresentation" extension with aromaticAtoms,
 * aromaticBonds, and atomRings arrays — that's what we parse here.
 */

window.CC = window.CC || {};
CC.GNN = window.CC.GNN || {};

CC.GNN.getRDKitAnnotations = function (molblock) {
  const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
  const empty = {
    aromaticAtomIndices: new Set(),
    ringAtomIndices: new Set(),
    ringSizesByAtomIndex: new Map(),
    aromaticBondPairs: new Set(),
    ringBondPairs: new Set(),
    chiralTagByAtomIndex: new Map(),
    bondStereoByPair: new Map(),
    numHByAtomIndex: new Map(),
    atomicNumByAtomIndex: new Map(),
  };
  if (!RDKit) return empty;

  let mol = null;
  try {
    mol = RDKit.get_mol(molblock);
    if (!mol || !mol.is_valid()) {
      if (mol) mol.delete();
      return empty;
    }

    const json = JSON.parse(mol.get_json());
    mol.delete();

    const molData = json.molecules && json.molecules[0];
    if (!molData) return empty;

    // Defaults per the CommonChem spec (fields are omitted when they equal
    // these, e.g. a plain aliphatic carbon has no "z" key at all).
    const defaults = (json.defaults && json.defaults.atom) || { z: 6, impHs: 0 };

    // Per-atom chiral tag ("cw"/"ccw", from RDKit's stereo perception off
    // the molblock's wedge/hash bonds + 2D coordinates) and implicit H
    // count — both read straight from the parsed atom list, which is
    // index-aligned with our own atom ordering since RDKit preserves
    // molblock atom order.
    const chiralTagByAtomIndex = new Map();
    const numHByAtomIndex = new Map();
    const atomicNumByAtomIndex = new Map();
    (molData.atoms || []).forEach(function (atom, idx) {
      const stereo = atom.stereo; // 'cw' | 'ccw' | undefined (unspecified)
      if (stereo === 'cw') chiralTagByAtomIndex.set(idx, 'cw');
      else if (stereo === 'ccw') chiralTagByAtomIndex.set(idx, 'ccw');
      numHByAtomIndex.set(idx, atom.impHs !== undefined ? atom.impHs : defaults.impHs);
      atomicNumByAtomIndex.set(idx, atom.z !== undefined ? atom.z : defaults.z);
    });

    const ext = (molData.extensions || []).find(function (e) { return e.name === 'rdkitRepresentation'; });

    const aromaticAtomIndices = new Set((ext && ext.aromaticAtoms) || []);
    const ringAtomIndices = new Set();
    const ringBondPairs = new Set();
    // Per-atom set of ring SIZES it belongs to (a fused-ring atom can be
    // in more than one) -- distinct from ringAtomIndices above (which
    // only tracks "in some ring", not which size). Used by
    // nagl-features.js's ringofsize feature; built from this same
    // atomRings pass rather than a second RDKit query.
    const ringSizesByAtomIndex = new Map();
    ((ext && ext.atomRings) || []).forEach(function (ring) {
      ring.forEach(function (idx) {
        ringAtomIndices.add(idx);
        if (!ringSizesByAtomIndex.has(idx)) ringSizesByAtomIndex.set(idx, new Set());
        ringSizesByAtomIndex.get(idx).add(ring.length);
      });
      for (let k = 0; k < ring.length; k++) {
        const a = ring[k];
        const b = ring[(k + 1) % ring.length];
        ringBondPairs.add(a < b ? a + '_' + b : b + '_' + a);
      }
    });

    // Bonds don't carry their endpoint indices directly in the aromaticBonds
    // list (that's a list of bond indices, not atom pairs) — resolve via
    // molData.bonds[bondIdx].atoms. Same pass also picks up each bond's
    // "stereo" field ('cis'/'trans'/'either', when RDKit could determine
    // it), keyed the same way as the other bond-pair sets.
    const aromaticBondPairs = new Set();
    const aromaticBondIdx = new Set((ext && ext.aromaticBonds) || []);
    const bondStereoByPair = new Map();
    (molData.bonds || []).forEach(function (bond, idx) {
      if (!bond.atoms || bond.atoms.length !== 2) return;
      const a = bond.atoms[0], b = bond.atoms[1];
      const key = a < b ? a + '_' + b : b + '_' + a;
      if (aromaticBondIdx.has(idx)) aromaticBondPairs.add(key);
      if (bond.stereo) bondStereoByPair.set(key, bond.stereo);
    });

    return {
      aromaticAtomIndices: aromaticAtomIndices,
      ringAtomIndices: ringAtomIndices,
      ringSizesByAtomIndex: ringSizesByAtomIndex,
      aromaticBondPairs: aromaticBondPairs,
      ringBondPairs: ringBondPairs,
      chiralTagByAtomIndex: chiralTagByAtomIndex,
      bondStereoByPair: bondStereoByPair,
      numHByAtomIndex: numHByAtomIndex,
      atomicNumByAtomIndex: atomicNumByAtomIndex,
    };
  } catch (err) {
    if (mol) mol.delete();
    return empty;
  }
};

// Shared by buildMolGraph (demo featurizer) and buildMolGraphChemprop
// (exact Chemprop featurizer): turns per-atom/per-bond feature rows into
// the directed-edge tensors a D-MPNN needs.
function assembleMolGraph(atoms, atomFeat, bondFeat) {
  const edgeSrc = [];
  const edgeDst = [];
  const edgeFeatures = [];
  const revEdge = [];
  const incomingEdgesByAtom = atoms.map(function () { return []; });

  bondFeat.rows.forEach(function (b) {
    const eForward = edgeSrc.length;
    edgeSrc.push(b.i); edgeDst.push(b.j); edgeFeatures.push(b.features);

    const eBackward = edgeSrc.length;
    edgeSrc.push(b.j); edgeDst.push(b.i); edgeFeatures.push(b.features);

    revEdge[eForward] = eBackward;
    revEdge[eBackward] = eForward;

    incomingEdgesByAtom[b.j].push(eForward);
    incomingEdgesByAtom[b.i].push(eBackward);
  });

  return {
    numAtoms: atoms.length,
    atomFeatures: atomFeat.rows,
    atomFeatureDim: atomFeat.dim,
    bondFeatureDim: bondFeat.dim,
    edgeSrc: edgeSrc,
    edgeDst: edgeDst,
    edgeFeatures: edgeFeatures,
    revEdge: revEdge,
    incomingEdgesByAtom: incomingEdgesByAtom,
    atomIds: atomFeat.atomIds,
  };
}

CC.GNN.buildMolGraph = function (molecule, annotations) {
  const atoms = Array.from(molecule.atoms.values());
  const atomIdToIndex = new Map();
  atoms.forEach(function (a, i) { atomIdToIndex.set(a.id, i); });

  const atomFeat = CC.GNN.buildAtomFeatures(molecule, annotations);
  const bondFeat = CC.GNN.buildBondFeatures(molecule, atomIdToIndex, annotations);
  return assembleMolGraph(atoms, atomFeat, bondFeat);
};

/**
 * Same shape as buildMolGraph, but using the bit-exact Chemprop v2
 * featurizers (chemprop-features.js) instead of this app's own simplified
 * ones. Use this path when running a real Chemprop checkpoint
 * (chemprop-model.js) — the demo D-MPNN can keep using buildMolGraph.
 */
CC.GNN.buildMolGraphChemprop = function (molecule, annotations) {
  const atoms = Array.from(molecule.atoms.values());
  const atomIdToIndex = new Map();
  atoms.forEach(function (a, i) { atomIdToIndex.set(a.id, i); });

  const atomFeat = CC.GNN.buildChempropAtomFeatures(molecule, annotations);
  const bondFeat = CC.GNN.buildChempropBondFeatures(molecule, atomIdToIndex, annotations, atomFeat.sp2ByAtomId);
  return assembleMolGraph(atoms, atomFeat, bondFeat);
};
