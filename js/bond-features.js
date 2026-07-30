/**
 * bond-features.js
 *
 * Per-bond feature vector: bond type one-hot, a conjugation approximation,
 * ring membership (from RDKit annotations), wedge/hash flags, and a
 * geometric cis/trans classification for double bonds (see stereo2d.js
 * for exactly what that does and doesn't mean).
 *
 * Honest scope note: real conjugation perception needs RDKit's bond-level
 * analysis, which isn't exposed granularly enough through RDKit.js's JS
 * bindings to pull per-bond without more plumbing than this stage covers.
 * Conjugation here is approximated as "non-single or aromatic" — a
 * reasonable proxy, not the real algorithm.
 */

window.CC = window.CC || {};
CC.GNN = window.CC.GNN || {};

(function () {
  const BOND_TYPE_CHOICES = ['single', 'double', 'triple', 'aromatic'];
  const DOUBLE_BOND_STEREO_CHOICES = ['none', 'cis', 'trans', 'unknown'];

  function bondTypeName(order, isAromatic) {
    if (isAromatic) return 'aromatic';
    if (order === 2) return 'double';
    if (order === 3) return 'triple';
    return 'single';
  }

  function oneHot(value, choices) {
    const vec = new Array(choices.length).fill(0);
    const idx = choices.indexOf(value);
    if (idx !== -1) vec[idx] = 1;
    return vec;
  }

  /**
   * annotations: same RDKit-derived object as atom-features.js expects,
   * plus ringBondPairs (Set of "i_j" atom-index pairs that are in a ring).
   */
  CC.GNN.buildBondFeatures = function (molecule, atomIdToIndex, annotations) {
    const bonds = Array.from(molecule.bonds.values());
    const aromaticBondPairs = (annotations && annotations.aromaticBondPairs) || new Set();
    const ringBondPairs = (annotations && annotations.ringBondPairs) || new Set();
    const doubleBondStereo = CC.getDoubleBondStereo(molecule);

    const rows = bonds.map(function (b) {
      const i = atomIdToIndex.get(b.a1);
      const j = atomIdToIndex.get(b.a2);
      const key = i < j ? i + '_' + j : j + '_' + i;
      const isAromatic = aromaticBondPairs.has(key);
      const inRing = ringBondPairs.has(key);
      const conjugated = isAromatic || b.order === 2;
      const stereoLabel = b.order === 2 ? (doubleBondStereo.get(b.id) || 'unknown') : 'none';

      return {
        i: i,
        j: j,
        features: [].concat(
          oneHot(bondTypeName(b.order, isAromatic), BOND_TYPE_CHOICES),
          [conjugated ? 1 : 0],
          [inRing ? 1 : 0],
          [b.stereo === 'wedge' ? 1 : 0],
          [b.stereo === 'hash' ? 1 : 0],
          oneHot(stereoLabel, DOUBLE_BOND_STEREO_CHOICES)
        ),
      };
    });

    return {
      rows: rows,
      dim: rows.length > 0 ? rows[0].features.length : BOND_TYPE_CHOICES.length + 4 + DOUBLE_BOND_STEREO_CHOICES.length,
    };
  };
})();
