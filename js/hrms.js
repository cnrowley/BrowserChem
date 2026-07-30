/**
 * hrms.js
 *
 * Exact (monoisotopic) mass and common HRMS adduct m/z values, for
 * matching against high-resolution mass spectrometry data.
 *
 * The neutral exact mass comes straight from RDKit.js's own exactmw
 * (already validated, real isotope masses -- not recomputed here).
 * Adduct deltas use CODATA physical constants (proton/electron rest
 * mass) and NIST monoisotopic atomic masses, not rounded textbook
 * approximations -- getting the 4th decimal place right matters for
 * this specific feature, since that's the precision HRMS is actually
 * reported and matched at.
 */

window.CC = window.CC || {};

(function () {
  // CODATA / NIST values (Da).
  const PROTON_MASS = 1.00727646688;
  const ELECTRON_MASS = 0.00054857990907;
  const MONO = {
    H: 1.00782503207, C: 12.0, N: 14.0030740048, O: 15.99491461956,
    Na: 22.9897692809, K: 38.9637064864, Cl: 34.96885268,
  };
  const NH4_MASS = MONO.N + 4 * MONO.H - ELECTRON_MASS; // NH4+ cation
  const H2O_MASS = 2 * MONO.H + MONO.O;

  // { label, chargeSign, delta } -- m/z = neutral exact mass + delta,
  // delta already includes the missing/extra electron for the resulting
  // ion (so this is a direct m/z, not "mass of a neutral H atom added").
  CC.HRMS_ADDUCTS = [
    { label: '[M+H]+', delta: PROTON_MASS, mode: '+' },
    { label: '[M+Na]+', delta: MONO.Na - ELECTRON_MASS, mode: '+' },
    { label: '[M+K]+', delta: MONO.K - ELECTRON_MASS, mode: '+' },
    { label: '[M+NH4]+', delta: NH4_MASS, mode: '+' },
    { label: '[M+H-H2O]+', delta: PROTON_MASS - H2O_MASS, mode: '+' },
    { label: '[M-H]-', delta: -PROTON_MASS, mode: '-' },
    { label: '[M+Cl]-', delta: MONO.Cl + ELECTRON_MASS, mode: '-' },
    { label: '[M-H2O+H]+', delta: PROTON_MASS - H2O_MASS, mode: '+', duplicate: true },
  ].filter(function (a) { return !a.duplicate; });

  /**
   * Molecular formula in Hill order (C first, H second, everything else
   * alphabetically) from the app's own Molecule model directly -- not
   * from RDKit, which doesn't expose a formula method in this build.
   * Implicit H count per atom uses the same valence-minus-bond-order-sum
   * logic already validated for atom labels (render.js).
   */
  CC.computeMolecularFormula = function (molecule) {
    const counts = {};
    let totalH = 0;
    molecule.atoms.forEach(function (atom) {
      counts[atom.element] = (counts[atom.element] || 0) + 1;
      const data = CC.elementData(atom.element);
      const used = molecule.getBondsForAtom(atom.id).reduce(function (s, b) { return s + b.order; }, 0);
      totalH += Math.max(0, data.valence - used);
    });
    if (totalH > 0) counts.H = (counts.H || 0) + totalH;

    const elements = Object.keys(counts).filter(function (e) { return e !== 'C' && e !== 'H'; }).sort();
    const order = (counts.C ? ['C'] : []).concat(counts.H ? ['H'] : []).concat(elements);
    return order.map(function (el) {
      const n = counts[el];
      return n === 1 ? el : el + n;
    }).join('');
  };

  /**
   * descriptors: analyzeMolblock's descriptors object (needs exactmw).
   * Returns { formula, exactMass, adducts: [{label, mz}] }, or null if
   * exactmw isn't available.
   */
  CC.computeHRMS = function (molecule, descriptors) {
    if (!descriptors || typeof descriptors.exactmw !== 'number') return null;
    const exactMass = descriptors.exactmw;
    return {
      formula: CC.computeMolecularFormula(molecule),
      exactMass: exactMass,
      adducts: CC.HRMS_ADDUCTS.map(function (a) {
        return { label: a.label, mode: a.mode, mz: exactMass + a.delta };
      }),
    };
  };
})();
