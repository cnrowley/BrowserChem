/**
 * implicit-solvent.js
 *
 * A real (not trained/fit in this project) implicit-solvent free energy
 * calculation: Generalized Born (GB) electrostatics via the
 * Hawkins-Cramer-Truhlar (HCT) pairwise descreening approximation --
 * AMBER's "igb=1", the original/simplest real GB variant -- plus a
 * SASA-based nonpolar (cavity/dispersion) term (see
 * nonpolarSolvationEnergy below: CC.DSASA.compute() when available, an
 * exact-analytical SASA construction with a real gradient, dsasa.js).
 * Needs an already-built 3D conformer (see embed3d.js/geomol-model.js)
 * and per-atom partial charges (this project's NAGL-MBIS model, see
 * nagl-model.js) as inputs; computes nothing about geometry or charges
 * itself.
 *
 * Deliberately NOT the later OBC refinement (AMBER igb=2/5, GB-Neck,
 * GB-Neck2) -- those add a fitted tanh(alpha*sum - beta*sum^2 +
 * gamma*sum^3) correction (and, for the Neck variants, a whole separate
 * "neck" integral) on top of this same HCT pairwise sum, each with its own
 * set of fitted constants. Plain HCT is a smaller, fully auditable
 * from-scratch implementation with fewer numbers to get right and keep
 * honest about, at the cost of being the least accurate GB variant still
 * in common use -- a real accuracy/complexity tradeoff, not hidden.
 *
 * The HCT pairwise descreening loop (computeBornRadii below) is ported
 * directly from OpenMM's own reference CPU implementation
 * (ReferenceObc.cpp::computeBornRadii -- OBC only diverges from HCT AFTER
 * this same per-atom `sum` is computed), not re-derived from the original
 * Hawkins/Cramer/Truhlar 1995/1996 papers -- OpenMM's is the executable,
 * widely-used ground truth this was checked against. Verified against the
 * analytic single-ion limit this formula must reduce to (a lone charged
 * atom's Born radius equals its own intrinsic radius, giving the textbook
 * Born solvation energy dG = -166*(1-1/eps)*q^2/R) before being trusted
 * for real molecules.
 *
 * NOT yet cross-validated against a real GB implementation's own output
 * numbers for a real molecule (only the analytic single-ion limit above),
 * unlike qed.js/sascorer.js's bit-exact validation against RDKit --
 * treat absolute values as approximate; relative comparisons between
 * solvents/molecules are the more trustworthy use of this feature.
 */

window.CC = window.CC || {};
CC.Solvent = window.CC.Solvent || {};

(function () {
  // Standard HCT/OBC convention (Hawkins/Cramer/Truhlar; confirmed against
  // OpenMM's own ReferenceObc.cpp, which uses the same 0.09 A constant).
  const DIELECTRIC_OFFSET = 0.09; // Å

  // Standard Coulomb's-law unit-conversion constant (kcal*Å/(mol*e^2)) --
  // a physical constant, not a fitted parameter.
  const COULOMB_CONST = 332.0636;

  // Still, Tempczyk, Hawley, Hendrickson, J. Am. Chem. Soc. 1990, 112,
  // 6127 -- the standard AMBER GB/SA default surface tension for the
  // SASA-based nonpolar term (widely reused as-is, not re-fit here).
  const SURFACE_TENSION = 0.0072; // kcal/(mol*Å^2)

  // Solute interior dielectric -- standard GB practice (this project
  // doesn't separately model intramolecular polarizability).
  const EPS_IN = 1.0;

  // "mbondi" GB radii (Å) -- Hawkins/Cramer/Truhlar's own radius set for
  // HCT, distinct from (though close to) the plain Bondi vdW radii
  // CC.VDW_RADIUS already uses for SASA (elements.js). The real mbondi
  // set gives hydrogen a bonded-atom-dependent radius (1.3 Å if bonded to
  // carbon, 0.8 Å if bonded to nitrogen, 1.2 Å otherwise); simplified here
  // to a single default (1.2 Å, matching mbondi's own "otherwise" case)
  // rather than tracking bonding context -- a real simplification, not a
  // hidden one. Elements outside HCT's original six (Br, I) fall back to
  // plain Bondi radii, since HCT was never parametrized for them.
  const GB_RADIUS = {
    H: 1.20, C: 1.70, N: 1.55, O: 1.50, F: 1.50, P: 1.85, S: 1.80, Cl: 1.70,
  };

  // HCT pairwise-descreening scale factors S, as tabulated in OpenMM's own
  // customgbforces.py (_SCREEN_PARAMETERS, first tuple element of each
  // entry). 0.80 is OpenMM's own fallback for elements outside this
  // six-element set (used here for Br, I) -- not itself literature-
  // validated for those elements, just the standard "unknown element"
  // default.
  const SCALE_FACTOR = {
    H: 0.85, C: 0.72, N: 0.79, O: 0.85, F: 0.88, P: 0.86, S: 0.96,
  };
  const DEFAULT_SCALE_FACTOR = 0.80;

  function gbRadius(element) {
    return GB_RADIUS[element] || (CC.VDW_RADIUS && CC.VDW_RADIUS[element]) || 1.70;
  }
  function scaleFactor(element) {
    return SCALE_FACTOR[element] !== undefined ? SCALE_FACTOR[element] : DEFAULT_SCALE_FACTOR;
  }

  /**
   * Effective Born radii via HCT pairwise descreening. atoms: array of
   * {element, x, y, z} in real Angstroms (heavy atoms + implicit
   * hydrogens, any order -- no ordering assumed here). O(n^2), fine at
   * the atom counts this app draws.
   */
  CC.Solvent.computeBornRadii = function (atoms) {
    const n = atoms.length;
    const rho = atoms.map(function (a) { return gbRadius(a.element) - DIELECTRIC_OFFSET; });
    const bornRadii = new Array(n);

    for (let i = 0; i < n; i++) {
      const rhoI = rho[i];
      let sum = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = atoms[i].x - atoms[j].x, dy = atoms[i].y - atoms[j].y, dz = atoms[i].z - atoms[j].z;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1e-6) continue; // coincident atoms -- shouldn't happen in a real conformer

        const sRhoJ = rho[j] * scaleFactor(atoms[j].element);
        const rPlusS = r + sRhoJ;
        if (rhoI >= rPlusS) continue; // j's descreening sphere doesn't reach atom i at all

        const lIj = 1 / Math.max(rhoI, Math.abs(r - sRhoJ));
        const uIj = 1 / rPlusS;
        const lIj2 = lIj * lIj, uIj2 = uIj * uIj;
        const ratio = Math.log(uIj / lIj);
        let term = lIj - uIj + 0.25 * r * (uIj2 - lIj2) + (0.5 / r) * ratio +
          (0.25 * sRhoJ * sRhoJ / r) * (lIj2 - uIj2);

        // Atom i entirely engulfed inside atom j's descreening sphere --
        // not in the original HCT paper; the extra term (also present in
        // OpenMM's own reference code) is Tinker's fix to keep the
        // integral well-behaved in that case.
        if (rhoI < sRhoJ - r) term += 2 * (1 / rhoI - lIj);

        sum += term;
      }
      const descreening = 0.5 * rhoI * sum;
      bornRadii[i] = 1 / (1 / rhoI - descreening);
    }
    return bornRadii;
  };

  /**
   * Still-formula polar solvation free energy (kcal/mol). atoms/charges
   * must be the same length and in the same order (charges[i] is the
   * partial charge, in elementary charge units e, of atoms[i]).
   */
  CC.Solvent.polarSolvationEnergy = function (atoms, charges, epsSolvent) {
    const bornRadii = CC.Solvent.computeBornRadii(atoms);
    const n = atoms.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        let r2 = 0;
        if (i !== j) {
          const dx = atoms[i].x - atoms[j].x, dy = atoms[i].y - atoms[j].y, dz = atoms[i].z - atoms[j].z;
          r2 = dx * dx + dy * dy + dz * dz;
        }
        const RiRj = bornRadii[i] * bornRadii[j];
        const fGB = Math.sqrt(r2 + RiRj * Math.exp(-r2 / (4 * RiRj)));
        const weight = i === j ? 0.5 : 1.0; // each i<j pair once; self term halved
        sum += weight * charges[i] * charges[j] / fGB;
      }
    }
    return -COULOMB_CONST * (1 / EPS_IN - 1 / epsSolvent) * sum;
  };

  /**
   * SASA-based nonpolar (cavity + dispersion) term. Uses CC.DSASA.compute()
   * (dsasa.js) -- an exact-analytical-SASA construction (weighted-Delaunay
   * alpha complex + inclusion-exclusion, Cao/Hummel/Wang/Simmerling/
   * Coutsias JCTC 2024 / Hummel PhD thesis) validated within a fraction of
   * a percent to ~3.5% of Shrake-Rupley on real molecules, WITH a real,
   * gradient-consistent derivative (0.00% error against independent
   * numerical differentiation of dsasa.js's own energy value -- see that
   * file's header for the full validation) -- unlike CC.SASA.compute()
   * (steric-accessibility.js's Shrake-Rupley), which has no meaningful
   * gradient at all (a step function of atom position: points on a
   * sampled sphere flip discretely between buried/exposed). Falls back to
   * CC.SASA.compute() (value only, gradient null) if CC.DSASA isn't
   * loaded for some reason -- shouldn't happen in the shipped app (see
   * index.html's script order) but keeps this file from hard-crashing if
   * it does.
   *
   * Returns { energy, totalSasa, gradient }. `gradient` is a
   * Float64Array(3*atoms.length) of d(energy)/dposition (already scaled
   * by SURFACE_TENSION, matching `energy`'s own units) when CC.DSASA
   * computed it, or null when the Shrake-Rupley fallback was used.
   */
  // wantGradient (default true): threaded straight through to
  // CC.DSASA.compute -- see that function's own comment. Pass false when
  // only `.energy`/`.totalSasa` are needed (e.g. CC.Solvent.predict below,
  // which never reads `.gradient` at all) to skip real, measured
  // per-call cost for no benefit.
  //
  // sasaModel ('dsasa' default, or 'shrake-rupley'): user-selectable via
  // the Implicit Solvent panel's "Nonpolar (SASA) model" dropdown (see
  // app.js's setupSolventPanel) -- an explicit choice now, not silent
  // auto-detection. 'shrake-rupley' forces the original sampled-sphere
  // method this app used before dsasa.js existed, even if CC.DSASA is
  // loaded; it has no analytical gradient (`gradient` comes back null),
  // so embed3d.js's numericResidualGradient falls back to finite-
  // differencing this term too when it's selected (see that function's
  // own comment) -- slower per iteration than dSASA's energy-only fast
  // path, but the same numerical behavior this app originally shipped
  // with, kept available as a real fallback/comparison rather than
  // removed outright.
  CC.Solvent.nonpolarSolvationEnergy = function (atoms, wantGradient, sasaModel) {
    if (wantGradient === undefined) wantGradient = true;
    const useDsasa = sasaModel !== 'shrake-rupley' && window.CC && CC.DSASA && CC.DSASA.compute;
    if (useDsasa) {
      const res = CC.DSASA.compute(atoms, wantGradient);
      let gradient = null;
      if (wantGradient) {
        gradient = new Float64Array(res.gradient.length);
        for (let i = 0; i < gradient.length; i++) gradient[i] = SURFACE_TENSION * res.gradient[i];
      }
      return { energy: SURFACE_TENSION * res.totalSASA, totalSasa: res.totalSASA, gradient: gradient };
    }
    const perAtom = CC.SASA.compute(atoms, {});
    const totalSasa = perAtom.reduce(function (s, a) { return s + a.sasa; }, 0);
    return { energy: SURFACE_TENSION * totalSasa, totalSasa: totalSasa, gradient: null };
  };

  // Dielectric constants: standard literature values at/near room
  // temperature, not measured/validated by this project.
  CC.Solvent.SOLVENTS = [
    { id: 'water', name: 'Water', eps: 78.5 },
    { id: 'dmso', name: 'DMSO', eps: 46.7 },
    { id: 'methanol', name: 'Methanol', eps: 32.7 },
    { id: 'ethanol', name: 'Ethanol', eps: 24.3 },
    { id: 'acetonitrile', name: 'Acetonitrile', eps: 37.5 },
    { id: 'dmf', name: 'DMF', eps: 36.7 },
    { id: 'acetone', name: 'Acetone', eps: 20.7 },
    { id: 'dichloromethane', name: 'Dichloromethane (DCM)', eps: 8.93 },
    { id: 'thf', name: 'THF', eps: 7.52 },
    { id: 'chloroform', name: 'Chloroform', eps: 4.81 },
    { id: 'ethyl_acetate', name: 'Ethyl acetate', eps: 6.02 },
    { id: 'diethyl_ether', name: 'Diethyl ether', eps: 4.33 },
    { id: 'toluene', name: 'Toluene', eps: 2.38 },
    { id: 'hexane', name: 'Hexane', eps: 1.88 },
  ];

  /**
   * Full result for one (molecule, solvent) pair. atoms: 3D atom array
   * (heavy + implicit H, real Angstroms). charges: same length/order,
   * partial charge per atom (e.g. CC.NAGL.predictAll().charges).
   */
  CC.Solvent.predict = function (atoms, charges, epsSolvent, sasaModel) {
    if (!atoms || atoms.length === 0) throw new Error('No 3D structure to compute solvation energy from');
    if (atoms.length !== charges.length) {
      throw new Error('Atom count (' + atoms.length + ') does not match charge count (' + charges.length + ')');
    }
    if (!(epsSolvent >= 1)) throw new Error('Solvent dielectric constant must be a number ≥ 1');

    const polar = CC.Solvent.polarSolvationEnergy(atoms, charges, epsSolvent);
    // false: this function's own return value below never exposes
    // nonpolar's `.gradient` (only `.energy`/`.totalSasa`), and every
    // caller of predict() only ever reads polar/nonpolar/total/totalSasa
    // -- see nonpolarSolvationEnergy's wantGradient comment.
    const nonpolar = CC.Solvent.nonpolarSolvationEnergy(atoms, false, sasaModel);
    return {
      polar: polar,
      nonpolar: nonpolar.energy,
      total: polar + nonpolar.energy,
      totalSasa: nonpolar.totalSasa,
      epsSolvent: epsSolvent,
    };
  };
})();
