/**
 * atom-features.js
 *
 * Builds a per-atom feature vector in the style of Chemprop's atom
 * featurization: concatenated one-hot blocks for categorical properties
 * plus a couple of scalars.
 *
 * Honest scope note: the original Chemprop atom feature vector is ~133
 * dimensions wide, largely because it one-hot-encodes atomic number across
 * the full periodic table. This editor only supports a handful of elements
 * (see elements.js), so the element block here is deliberately smaller.
 *
 * Chirality: this used to always report "unspecified" — this app didn't
 * track wedge/hash bonds at all. Now that it does (stereo2d.js), this
 * block reports whether the atom is the stereocenter end of a wedge or
 * hash bond. That's real, unambiguous 2D drawing data — it is NOT a
 * computed CIP R/S label, which requires ranking substituents by priority
 * (an algorithm this app doesn't implement). Don't read 'wedge'/'hash'
 * here as "this atom is R" or "this atom is S".
 */

window.CC = window.CC || {};
CC.GNN = window.CC.GNN || {};

(function () {
  const ELEMENT_CHOICES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si'];
  const DEGREE_CHOICES = [0, 1, 2, 3, 4, 5];
  const CHARGE_CHOICES = [-2, -1, 0, 1, 2];
  const NUMH_CHOICES = [0, 1, 2, 3, 4];
  const HYBRID_CHOICES = ['sp', 'sp2', 'sp3', 'other'];
  const CHIRALITY_CHOICES = ['none', 'wedge', 'hash'];

  // Rough atomic masses (amu), scaled by /100 as a feature the same way
  // Chemprop scales mass — just a normalization convenience, not chemistry.
  const ATOMIC_MASS = {
    H: 1.008, C: 12.011, N: 14.007, O: 15.999, F: 18.998,
    P: 30.974, S: 32.06, Cl: 35.45, Br: 79.904, I: 126.90,
    B: 10.81, Si: 28.085,
  };

  function oneHot(value, choices) {
    const vec = new Array(choices.length + 1).fill(0); // last slot = "other"
    const idx = choices.indexOf(value);
    vec[idx === -1 ? choices.length : idx] = 1;
    return vec;
  }

  function usedValence(molecule, atomId) {
    return molecule.getBondsForAtom(atomId).reduce(function (sum, b) { return sum + b.order; }, 0);
  }

  function implicitHCount(molecule, atomId, element) {
    const data = CC.elementData(element);
    return Math.max(0, data.valence - usedValence(molecule, atomId));
  }

  function guessHybridization(molecule, atomId, isAromatic) {
    if (isAromatic) return 'sp2';
    const orders = molecule.getBondsForAtom(atomId).map(function (b) { return b.order; });
    if (orders.indexOf(3) !== -1) return 'sp';
    if (orders.indexOf(2) !== -1) return 'sp2';
    return 'sp3';
  }

  /**
   * Build the full atom feature matrix for a molecule.
   * `annotations` (optional) is RDKit-derived aromaticity/ring info from
   * graph-builder.js's getRDKitAnnotations(); if omitted (RDKit not ready,
   * or structure invalid), aromaticity/ring flags default to false rather
   * than throwing — the feature vector is still well-formed, just missing
   * that signal.
   */
  CC.GNN.buildAtomFeatures = function (molecule, annotations) {
    const atoms = Array.from(molecule.atoms.values());
    const aromaticSet = (annotations && annotations.aromaticAtomIndices) || new Set();
    const ringSet = (annotations && annotations.ringAtomIndices) || new Set();
    const wedgeHashByAtom = CC.getWedgeHashByAtom(molecule);

    const rows = atoms.map(function (atom, index) {
      const isAromatic = aromaticSet.has(index);
      const inRing = ringSet.has(index);
      const numH = implicitHCount(molecule, atom.id, atom.element);
      const hybridization = guessHybridization(molecule, atom.id, isAromatic);
      const mass = (ATOMIC_MASS[atom.element] || 12.011) / 100;
      const chirality = wedgeHashByAtom.get(atom.id) || 'none';

      return [].concat(
        oneHot(atom.element, ELEMENT_CHOICES),
        oneHot(Math.min(molecule.getDegree(atom.id), 5), DEGREE_CHOICES),
        oneHot(Math.max(-2, Math.min(2, atom.charge)), CHARGE_CHOICES),
        oneHot(chirality, CHIRALITY_CHOICES),
        oneHot(Math.min(numH, 4), NUMH_CHOICES),
        oneHot(hybridization, HYBRID_CHOICES),
        [isAromatic ? 1 : 0],
        [inRing ? 1 : 0],
        [mass]
      );
    });

    return {
      rows: rows,
      dim: rows.length > 0 ? rows[0].length : 0,
      atomIds: atoms.map(function (a) { return a.id; }),
    };
  };
})();
