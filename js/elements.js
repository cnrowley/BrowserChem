/**
 * elements.js
 *
 * Small reference tables the 3D embedder needs: approximate covalent radii
 * (for ideal bond lengths) and default valence (for how many implicit
 * hydrogens to add). These are simplified, neutral-atom values — good
 * enough for a "simple" force field, not a substitute for RDKit's proper
 * valence model.
 */

window.CC = window.CC || {};

CC.ELEMENT_DATA = {
  H: { radius: 0.31, valence: 1 },
  C: { radius: 0.76, valence: 4 },
  N: { radius: 0.71, valence: 3 },
  O: { radius: 0.66, valence: 2 },
  F: { radius: 0.57, valence: 1 },
  P: { radius: 1.07, valence: 3 },
  S: { radius: 1.05, valence: 2 },
  Cl: { radius: 1.02, valence: 1 },
  Br: { radius: 1.20, valence: 1 },
  I: { radius: 1.39, valence: 1 },
};

// Bondi van der Waals radii (Å) -- NOT the same thing as the covalent
// radii above, and shouldn't be substituted for them. Covalent radii sum
// to a *bonded* distance (~1.5 Å for two carbons); vdW radii sum to the
// distance two *non-bonded* atoms settle at when just touching (~3.4 Å
// for two carbons) -- meaningfully larger. embed3d.js's nonbonded/
// steric term needs this one; idealBondLength() below needs the other.
// Values: Bondi, J. Phys. Chem. 1964, 68, 3, 441-451.
CC.VDW_RADIUS = {
  H: 1.20, C: 1.70, N: 1.55, O: 1.52, F: 1.47,
  P: 1.80, S: 1.80, Cl: 1.75, Br: 1.85, I: 1.98,
};

// Real double/triple bonds are shorter than the sum of single-bond covalent
// radii would suggest; these are rough correction factors, not derived from
// a real parameter set.
CC.BOND_ORDER_FACTOR = { 1: 1.0, 2: 0.87, 3: 0.78 };

// Element symbol -> atomic number, H through Kr plus I -- shared by
// anything that needs a real atomic number rather than just this file's
// bond-length valence table (chemprop-features.js's v2 featurizer,
// sascorer.js's Morgan connectivity invariants).
CC.ELEMENT_TO_ATOMIC_NUMBER = {
  H: 1, He: 2, Li: 3, Be: 4, B: 5, C: 6, N: 7, O: 8, F: 9, Ne: 10,
  Na: 11, Mg: 12, Al: 13, Si: 14, P: 15, S: 16, Cl: 17, Ar: 18, K: 19,
  Ca: 20, Sc: 21, Ti: 22, V: 23, Cr: 24, Mn: 25, Fe: 26, Co: 27, Ni: 28,
  Cu: 29, Zn: 30, Ga: 31, Ge: 32, As: 33, Se: 34, Br: 35, Kr: 36, I: 53,
};

CC.elementData = function (symbol) {
  return CC.ELEMENT_DATA[symbol] || CC.ELEMENT_DATA.C;
};

CC.idealBondLength = function (el1, el2, order) {
  const d1 = CC.elementData(el1);
  const d2 = CC.elementData(el2);
  const factor = CC.BOND_ORDER_FACTOR[order] || 1.0;
  return (d1.radius + d2.radius) * factor;
};
