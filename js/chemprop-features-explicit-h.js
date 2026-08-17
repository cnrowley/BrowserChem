/**
 * chemprop-features-explicit-h.js
 *
 * Builds the D-MPNN graph a Chemprop checkpoint trained with `--add-h`
 * needs (every hydrogen gets its own graph node, instead of an implicit
 * "num Hs" feature on its parent heavy atom) -- required for the 1H NMR
 * shift checkpoint (checkpoints/nmr-1h/, see scripts/train_nmr_chemprop.sh),
 * since a per-atom 1H shift genuinely needs a per-hydrogen-atom output,
 * and js/graph-builder.js's normal buildMolGraphChemprop() only ever
 * builds heavy-atom nodes (H folded into the same "num Hs" feature block
 * every other checkpoint in this project uses).
 *
 * Confirmed directly against real RDKit (not assumed) exactly what
 * chemprop's own MultiHotAtomFeaturizer.v2() features look like after
 * Chem.AddHs(): every heavy atom's TotalDegree increases by its H count
 * (explicit neighbors now, same as before numerically) while
 * TotalNumHs() drops to 0 (no more *implicit* H's to count) --
 * so heavy atoms here get the SAME degree bucket as
 * chemprop-features.js's normal builder already computes (heavyDegree +
 * numH), just with the num-Hs feature block forced to 0. Each new H
 * node gets TotalDegree=1, TotalNumHs=0, and RDKit's own
 * GetHybridization() for an explicit H atom is HybridizationType.
 * UNSPECIFIED -- not a value in chemprop-features.js's
 * CHEMPROP_HYBRIDIZATION_CHOICES list at all, so it lands in that
 * feature's pad/"unknown" bucket exactly the way real Chemprop's own
 * `self.hybridizations.get(feat, len(choices))` does for the same
 * value -- not guessed, this is what real RDKit reports.
 *
 * Node ordering matches the same convention already established
 * elsewhere in this project for exactly this heavy-then-explicit-H
 * layout (js/nagl-features.js's buildExpandedGraph, js/pka-descriptor.js):
 * heavy atoms keep their molecule.atoms order, synthetic H nodes are
 * appended afterward grouped by parent heavy atom in heavy-atom-iteration
 * order -- the same order Chem.AddHs() itself produces, which is what
 * scripts/prepare_nmr_training_data.py's training labels were built
 * against (see that script's own header for the equivalent Python-side
 * reasoning).
 */

window.CC = window.CC || {};
CC.GNN = window.CC.GNN || {};

(function () {
  function oneHot(value, choices) { return CC.GNN.chempropOneHotBlock(value, choices); }

  function heavyAtomRow(molecule, atom, index, annotations, numHByIdx, aromaticSet, chiralByIdx, atomIdToIndex) {
    const z = CC.GNN.CHEMPROP_ELEMENT_TO_Z[atom.element] || 0;
    const isAromatic = aromaticSet.has(index);
    const heavyDegree = molecule.getDegree(atom.id);
    const numH = numHByIdx.has(index) ? numHByIdx.get(index) : 0;
    const totalDegree = heavyDegree + numH; // same total as before -- H's are now explicit neighbors instead of implicit
    const hybridization = CC.GNN.guessHybridization(
      molecule, atom.id, isAromatic, totalDegree, atom.element, atomIdToIndex, aromaticSet
    );
    const mass = (CC.GNN.CHEMPROP_ATOMIC_WEIGHT_BY_Z[z] || 12.011) * 0.01;
    const chiralStr = chiralByIdx.get(index);
    const chiralTag = chiralStr === 'cw' ? 1 : chiralStr === 'ccw' ? 2 : 0;

    return {
      row: [].concat(
        oneHot(z, CC.GNN.CHEMPROP_ATOMIC_NUM_CHOICES),
        oneHot(Math.min(totalDegree, 6), CC.GNN.CHEMPROP_DEGREE_CHOICES),
        oneHot(atom.charge, CC.GNN.CHEMPROP_FORMAL_CHARGE_CHOICES),
        oneHot(chiralTag, CC.GNN.CHEMPROP_CHIRAL_TAG_CHOICES),
        oneHot(0, CC.GNN.CHEMPROP_NUM_H_CHOICES), // 0 -- all H's are explicit nodes now, matching real GetTotalNumHs() on an AddHs'd atom
        oneHot(hybridization, CC.GNN.CHEMPROP_HYBRIDIZATION_CHOICES),
        [isAromatic ? 1 : 0],
        [mass]
      ),
      isSp2ish: hybridization === 'SP' || hybridization === 'SP2' || hybridization === 'SP2D',
    };
  }

  function hydrogenRow() {
    const z = CC.GNN.CHEMPROP_ELEMENT_TO_Z.H;
    const mass = CC.GNN.CHEMPROP_ATOMIC_WEIGHT_BY_Z[z] * 0.01;
    return [].concat(
      oneHot(z, CC.GNN.CHEMPROP_ATOMIC_NUM_CHOICES),
      oneHot(1, CC.GNN.CHEMPROP_DEGREE_CHOICES), // a synthetic H node always has exactly one bond, to its parent heavy atom
      oneHot(0, CC.GNN.CHEMPROP_FORMAL_CHARGE_CHOICES),
      oneHot(0, CC.GNN.CHEMPROP_CHIRAL_TAG_CHOICES),
      oneHot(0, CC.GNN.CHEMPROP_NUM_H_CHOICES),
      oneHot('UNSPECIFIED', CC.GNN.CHEMPROP_HYBRIDIZATION_CHOICES), // real RDKit's GetHybridization() for an explicit H -- not in the choice list, lands in the pad bucket exactly like real Chemprop's own encoder
      [0], // never aromatic
      [mass]
    );
  }

  function singleBondFeatureRow() {
    // A heavy-H bond: single, not aromatic, not conjugated, not in a
    // ring, no stereo -- same 14-dim layout buildChempropBondFeatures
    // uses for a real bond.
    return [].concat(
      [0],
      CC.GNN.chempropOneHotBlock('single', CC.GNN.CHEMPROP_BOND_TYPE_CHOICES).slice(0, CC.GNN.CHEMPROP_BOND_TYPE_CHOICES.length),
      [0], [0],
      oneHot(0, CC.GNN.CHEMPROP_STEREO_CHOICES)
    );
  }

  /**
   * Same conceptual output as graph-builder.js's buildMolGraphChemprop(),
   * but over the expanded heavy+explicit-H node set. Distinct extra
   * fields callers need: `numHeavyAtoms` (how many leading rows are real
   * drawn atoms) and `hNodesByHeavyIndex` (heavy atom index -> array of
   * that atom's own explicit-H node indices, for aggregating a per-H
   * prediction back down to one displayable value per heavy atom --
   * see chemprop-model.js's runOneAtomLevelExplicitH()).
   */
  CC.GNN.buildMolGraphChempropExplicitH = function (molecule, annotations) {
    const heavyAtoms = Array.from(molecule.atoms.values());
    const atomIdToIndex = new Map();
    heavyAtoms.forEach(function (a, i) { atomIdToIndex.set(a.id, i); });

    const aromaticSet = (annotations && annotations.aromaticAtomIndices) || new Set();
    const chiralByIdx = (annotations && annotations.chiralTagByAtomIndex) || new Map();
    const numHByIdx = (annotations && annotations.numHByAtomIndex) || new Map();

    const numHeavyAtoms = heavyAtoms.length;
    const implicitHCounts = heavyAtoms.map(function (atom, i) {
      return numHByIdx.has(i) ? numHByIdx.get(i) : 0;
    });
    const totalHNodes = implicitHCounts.reduce(function (a, b) { return a + b; }, 0);
    const numNodes = numHeavyAtoms + totalHNodes;

    const atomFeatures = new Array(numNodes);
    const sp2ByAtomId = new Map();
    heavyAtoms.forEach(function (atom, i) {
      const built = heavyAtomRow(molecule, atom, i, annotations, numHByIdx, aromaticSet, chiralByIdx, atomIdToIndex);
      atomFeatures[i] = built.row;
      sp2ByAtomId.set(atom.id, built.isSp2ish);
    });

    const hNodesByHeavyIndex = heavyAtoms.map(function () { return []; });
    let nextH = numHeavyAtoms;
    heavyAtoms.forEach(function (atom, i) {
      for (let k = 0; k < implicitHCounts[i]; k++) {
        const hIdx = nextH++;
        atomFeatures[hIdx] = hydrogenRow();
        hNodesByHeavyIndex[i].push(hIdx);
      }
    });

    // Bonds: real heavy-heavy bonds (full featurizer, same as
    // buildChempropBondFeatures) plus one single-bond edge per synthetic
    // H node to its parent.
    const bondFeat = CC.GNN.buildChempropBondFeatures(molecule, atomIdToIndex, annotations, sp2ByAtomId);
    const bondRows = bondFeat.rows.slice();
    const numRealBonds = bondRows.length;
    // Canonical bond index (position within bondRows) of each heavy
    // atom's own attached-H bonds -- for a bond-level (BDE) checkpoint's
    // per-atom "weakest attached C-H" aggregation, mirroring
    // hNodesByHeavyIndex above but indexing bonds instead of H nodes.
    const hBondCanonicalIndexByHeavyIndex = heavyAtoms.map(function () { return []; });
    heavyAtoms.forEach(function (atom, i) {
      hNodesByHeavyIndex[i].forEach(function (hIdx) {
        hBondCanonicalIndexByHeavyIndex[i].push(bondRows.length);
        bondRows.push({ i: i, j: hIdx, features: singleBondFeatureRow() });
      });
    });

    const edgeSrc = [];
    const edgeDst = [];
    const edgeFeatures = [];
    const revEdge = [];
    const incomingEdgesByAtom = new Array(numNodes);
    for (let i = 0; i < numNodes; i++) incomingEdgesByAtom[i] = [];

    bondRows.forEach(function (b) {
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
      numAtoms: numNodes,
      atomFeatures: atomFeatures,
      atomFeatureDim: atomFeatures.length > 0 ? atomFeatures[0].length : 72,
      bondFeatureDim: bondRows.length > 0 ? bondRows[0].features.length : 14,
      edgeSrc: edgeSrc,
      edgeDst: edgeDst,
      edgeFeatures: edgeFeatures,
      revEdge: revEdge,
      incomingEdgesByAtom: incomingEdgesByAtom,
      atomIds: heavyAtoms.map(function (a) { return a.id; }),
      numHeavyAtoms: numHeavyAtoms,
      hNodesByHeavyIndex: hNodesByHeavyIndex,
      numRealBonds: numRealBonds,
      bondIds: Array.from(molecule.bonds.values()).map(function (b) { return b.id; }),
      hBondCanonicalIndexByHeavyIndex: hBondCanonicalIndexByHeavyIndex,
    };
  };
})();
