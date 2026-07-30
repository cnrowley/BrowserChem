/**
 * nagl-features.js
 *
 * Bit-exact atom featurizer for jthorton/nagl-mbis's charge models,
 * reverse-engineered from real source (SimonBoothroyd/nagl's
 * nagl/features.py for element/connectivity, naglmbis's own
 * naglmbis/features/atom.py for the ring-of-size feature it registers
 * on top of core NAGL) -- not guessed at.
 *
 * This is a genuinely different, simpler featurization than
 * chemprop-features.js's Chemprop-compatible one: 19 dims total (vs.
 * Chemprop's 72), no bond features at all (the SAGEConv graph
 * convolution this feeds only uses connectivity, not bond order/type),
 * and no padding/"unknown" bucket -- every one-hot here is built via
 * nagl's own `one_hot_encode(item, elements) = elements.index(item)`,
 * which throws if the value isn't in the list. That's a real modeling
 * limitation of the original model, not something to silently paper
 * over here: an element outside the trained set, or a degree outside
 * 1-4, genuinely has no defined answer from this checkpoint.
 *
 * EXPLICIT HYDROGENS, READ THIS FIRST: nagl-mbis's own
 * `compute_properties()` does NOT add hydrogens for you -- it builds a
 * DGL graph directly from whatever RDKit Mol it's handed, and its own
 * usage examples feed it an OpenFF `Molecule.to_rdkit()` result, which
 * (OpenFF Molecules always store explicit Hs internally) always has
 * every hydrogen as a real graph node. "connectivity" being one-hot
 * over [1,2,3,4] with no 0 confirms this too -- a carbon with only
 * heavy-atom bonds counted (this app's molecules don't require drawing
 * H atoms explicitly) would often show degree 0-2, outside that range.
 * So: this featurizer expands each heavy atom's implicit hydrogens into
 * real synthetic H graph nodes before computing anything, the same way
 * AddHs()/to_rdkit() would -- get this wrong and every "connectivity"
 * feature is silently off by however many implicit H's an atom has.
 *
 * Feature layout (order matters -- concatenated via torch.hstack in
 * this exact order in nagl's real config):
 *   element (9):      one-hot over [H, C, N, O, F, P, S, Cl, Br]
 *   connectivity (4): one-hot over degree [1, 2, 3, 4]
 *                      -- nagl's own definition: len(atom.GetBonds()),
 *                         i.e. total degree AFTER hydrogens are made
 *                         explicit (see above), not heavy-atom-only
 *   ringofsize (6):   multi-hot (NOT mutually exclusive -- a fused-ring
 *                      atom can set more than one bit) over ring sizes
 *                      [3, 4, 5, 6, 7, 8], each independently checked
 *                      via RDKit's IsAtomInRingOfSize (always all-zero
 *                      for a hydrogen -- H atoms are never ring members)
 *
 * This element/connectivity/ring-size triple is what THIS SPECIFIC
 * checkpoint (nagl-v1-mbis) was trained with -- confirmed directly from
 * its own saved hyper_parameters, not assumed from the nagl-mbis
 * codebase in general (the feature set is configurable per checkpoint).
 * If you load a different NAGL-MBIS checkpoint, re-verify its
 * hyper_parameters.config.model.atom_features against this list before
 * trusting predictions from it.
 */

window.CC = window.CC || {};
CC.NAGL = window.CC.NAGL || {};

(function () {
  const ELEMENT_CHOICES = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'];
  const CONNECTIVITY_CHOICES = [1, 2, 3, 4];
  const RING_SIZE_CHOICES = [3, 4, 5, 6, 7, 8];

  CC.NAGL.ATOM_FEATURE_DIM = ELEMENT_CHOICES.length + CONNECTIVITY_CHOICES.length + RING_SIZE_CHOICES.length; // 19

  // Strict one-hot matching nagl's own one_hot_encode(): no pad/unknown
  // slot. Returns null (rather than throwing) so callers can surface a
  // clear "this molecule is outside what this model supports" message
  // instead of a raw exception from deep inside a forward pass.
  function oneHotStrict(value, choices) {
    const idx = choices.indexOf(value);
    if (idx === -1) return null;
    const vec = new Array(choices.length).fill(0);
    vec[idx] = 1;
    return vec;
  }

  function featureRowFor(element, degree, ringSizes, unsupportedList, refIndex, refAtomId) {
    const elementVec = oneHotStrict(element, ELEMENT_CHOICES);
    if (!elementVec) {
      unsupportedList.push({ index: refIndex, atomId: refAtomId, reason: 'element "' + element + '" is outside this model\'s trained set ' + JSON.stringify(ELEMENT_CHOICES) });
    }
    const connectivityVec = oneHotStrict(degree, CONNECTIVITY_CHOICES);
    if (!connectivityVec) {
      unsupportedList.push({ index: refIndex, atomId: refAtomId, reason: 'degree ' + degree + ' is outside this model\'s trained range ' + JSON.stringify(CONNECTIVITY_CHOICES) });
    }
    const ringVec = RING_SIZE_CHOICES.map(function (size) { return ringSizes.has(size) ? 1 : 0; });
    return [].concat(
      elementVec || new Array(ELEMENT_CHOICES.length).fill(0),
      connectivityVec || new Array(CONNECTIVITY_CHOICES.length).fill(0),
      ringVec
    );
  }

  /**
   * Build the expanded graph (heavy atoms + explicit synthetic H nodes)
   * this model actually needs. `annotations` is graph-builder.js's
   * CC.GNN.getRDKitAnnotations() output, reused for implicit-H counts
   * (numHByAtomIndex) and ring-size membership.
   *
   * Returns:
   *   rows              -- feature row per graph node, heavy atoms first
   *                         (in molecule.atoms order), synthetic H nodes
   *                         appended after
   *   adjacency         -- array (per node) of neighbor node indices,
   *                         covering both original heavy-heavy bonds and
   *                         the new heavy-H bonds
   *   numHeavyAtoms      -- how many of the leading rows are real
   *                          (displayable) atoms vs. synthetic H nodes
   *   atomIds            -- ids for the real heavy atoms only, length
   *                          numHeavyAtoms (synthetic H nodes have no
   *                          id -- there's nothing in the drawn
   *                          structure to attach a result to)
   *   unsupportedAtoms    -- as before, refers to heavy-atom indices only
   */
  CC.NAGL.buildExpandedGraph = function (molecule, annotations) {
    const heavyAtoms = Array.from(molecule.atoms.values());
    const heavyAtomIdToIndex = new Map();
    heavyAtoms.forEach(function (a, i) { heavyAtomIdToIndex.set(a.id, i); });

    const numHByIndex = annotations.numHByAtomIndex || new Map();
    const ringSizesByIndex = annotations.ringSizesByAtomIndex || new Map();
    const unsupportedAtoms = [];

    // First pass: how many synthetic H nodes does each heavy atom need,
    // and what's each heavy atom's TOTAL degree (heavy neighbors + those
    // new explicit H's) -- this total is nagl's "connectivity" feature.
    const implicitHCounts = heavyAtoms.map(function (atom, i) {
      return numHByIndex.has(i) ? numHByIndex.get(i) : 0;
    });
    const heavyDegrees = heavyAtoms.map(function (atom) { return molecule.getDegree(atom.id); });
    const totalDegrees = heavyDegrees.map(function (d, i) { return d + implicitHCounts[i]; });

    const numHeavyAtoms = heavyAtoms.length;
    const totalHNodes = implicitHCounts.reduce(function (a, b) { return a + b; }, 0);
    const numNodes = numHeavyAtoms + totalHNodes;

    const rows = new Array(numNodes);
    const adjacency = Array.from({ length: numNodes }, function () { return []; });

    heavyAtoms.forEach(function (atom, i) {
      rows[i] = featureRowFor(atom.element, totalDegrees[i], ringSizesByIndex.get(i) || new Set(), unsupportedAtoms, i, atom.id);
    });

    // Original heavy-heavy bonds.
    Array.from(molecule.bonds.values()).forEach(function (bond) {
      const i = heavyAtomIdToIndex.get(bond.a1), j = heavyAtomIdToIndex.get(bond.a2);
      adjacency[i].push(j);
      adjacency[j].push(i);
    });

    // Synthetic explicit-H nodes, one bond each to their parent heavy atom.
    let nextHNode = numHeavyAtoms;
    heavyAtoms.forEach(function (atom, i) {
      for (let k = 0; k < implicitHCounts[i]; k++) {
        const hIdx = nextHNode++;
        rows[hIdx] = featureRowFor('H', 1, new Set(), unsupportedAtoms, hIdx, null);
        adjacency[hIdx].push(i);
        adjacency[i].push(hIdx);
      }
    });

    return {
      rows: rows,
      adjacency: adjacency,
      numHeavyAtoms: numHeavyAtoms,
      atomIds: heavyAtoms.map(function (a) { return a.id; }),
      unsupportedAtoms: unsupportedAtoms,
      dim: CC.NAGL.ATOM_FEATURE_DIM,
    };
  };
})();
