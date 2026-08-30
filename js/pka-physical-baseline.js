/**
 * pka-physical-baseline.js
 *
 * Computes the physical free-energy baseline js/pka-freeenergy-predict.js
 * adds a Chemprop-predicted correction on top of (delta learning -- see
 * that file and scripts/train_pka_microstate_freeenergy.py): NAGL-MBIS
 * partial charges -> SMIRNOFF Sage conformer optimization WITH implicit
 * GB/SA solvation included as a real energy/gradient term throughout the
 * optimization itself (not appended afterward as a single-point
 * correction on an already-settled vacuum geometry). Reuses
 * js/embed3d.js/js/openff-forcefield.js/js/implicit-solvent.js exactly as
 * already validated elsewhere in this app -- no new physics, just wiring
 * three already-real engines together correctly for this specific use.
 *
 * --- Why solvent has to be part of the optimization, not appended after ---
 *
 * A real bug in this file's first version (caught by the user, not found
 * independently): it ran SMIRNOFF optimization in pure vacuum, then
 * computed GB/SA solvation as a single-point energy on that already-
 * settled gas-phase geometry. For any CHARGED microstate (every
 * deprotonated acid, every protonated base -- i.e. every microstate this
 * function is ever actually called on, see js/pka-freeenergy-predict.js),
 * that lets the conformer search see full, UNSCREENED vacuum-strength
 * Coulomb interactions between charged/polar groups while it's choosing
 * the geometry -- water screens those interactions by ~78x (eps=78.5),
 * so a vacuum-optimized geometry can be measurably, artificially folded
 * or distorted relative to what the molecule would actually look like in
 * solution, before solvation is even applied. This matters most for
 * molecules with multiple polar/ionizable groups, exactly where getting
 * the geometry right also matters most for
 * js/pka-electrostatic-correction.js's own charge-charge coupling term to
 * mean anything.
 *
 * The fix needs no new capability: CC.OpenFF.optimize3D already accepts
 * an `opts.solvent` object and threads it into the real energy/gradient
 * functions the optimizer itself calls
 * (solvationEnergySMIRNOFF/gradientSMIRNOFF in js/openff-forcefield.js)
 * -- this app's own 3D panel "Include implicit solvent" checkbox
 * (js/app.js's getSolventSettings()) already exercises this exact path.
 * Passing `naglModelId` + `solvent: {enabled, epsSolvent, sasaModel}` is
 * enough -- CC.OpenFF.optimize3D computes charges once internally (via
 * naglModelId) and injects them into the solvent object itself
 * (confirmed directly in source: `optimizeGivenSeedSMIRNOFF`'s own
 * `chargesResult.charges` assignment), matching getSolventSettings()'s
 * own shape exactly rather than inventing a new one.
 *
 * Same computation, same code, this file's own offline counterpart (a
 * headless-Node batch harness reusing these same three engines
 * unmodified) built the training set's own physical_energy_protonated/
 * deprotonated columns from -- the model was trained on exactly this
 * function's own output distribution, not a different one, by
 * construction.
 */

window.CC = window.CC || {};
CC.PKAPhysicalBaseline = window.CC.PKAPhysicalBaseline || {};

(function () {
  /**
   * Returns a single scalar (kcal/mol-ish; vacuum force-field energy +
   * GB/SA solvation, both from the SAME solvent-included optimization)
   * for one molecule, or throws if a usable 3D structure/optimization
   * couldn't be produced (callers should treat this as "cannot score this
   * microstate", not silently substitute a placeholder value -- a missing
   * physical feature is not the same as a physical feature of zero).
   * `opts.naglModelId`: which loaded NAGL model to use for electrostatics
   * (falls back to zero charges -- and correspondingly zero solvation
   * term, since solvationEnergySMIRNOFF requires real charges -- not an
   * error, if omitted/not loaded, same graceful-degradation convention
   * getChargesForAtoms3D already has). `opts.timeBudgetMs` (default
   * 3000): SMIRNOFF optimization budget -- deliberately short
   * (single-attempt, not this app's usual multi-attempt conformer
   * search) to keep per-microstate cost bounded; matches the
   * training-set computation's own budget.
   */
  CC.PKAPhysicalBaseline.compute = async function (molecule, opts) {
    opts = opts || {};
    const timeBudgetMs = opts.timeBudgetMs || 3000;
    // GB/SASA solvation is a real O(N^2)-ish per-iteration cost on top of
    // SMIRNOFF's own terms (confirmed directly: 60 iterations alone took
    // ~10s on a ~30-heavy-atom molecule -- not a deadline-enforcement bug,
    // genuinely that expensive per step with solvent on). `iterations`
    // caps the optimizer's own iteration count directly rather than
    // relying on timeBudgetMs alone (which still has its own effect, but
    // a per-iteration cost this large can overshoot a short deadline by a
    // lot before the next check). 40 is a deliberately modest cap -- most
    // of the real geometric relaxation happens in the earliest iterations
    // for a rough single-conformer estimate like this; not claimed to be
    // fully gradient-converged.
    const iterations = opts.iterations || 40;
    const solventOpts = { enabled: true, epsSolvent: 78.5, sasaModel: 'default' };

    const initial = CC.buildInitial3D(molecule);
    if (!initial.atoms.length) throw new Error('no 3D structure could be built for this molecule');

    let optimized;
    try {
      if (!CC.OpenFF.isForceFieldLoaded()) await CC.OpenFF.loadForceField();
      optimized = await CC.OpenFF.optimize3D(initial, {
        timeBudgetMs: timeBudgetMs, attempts: 1, iterations: iterations, naglModelId: opts.naglModelId, solvent: solventOpts,
      });
    } catch (err) {
      // Fallback (SMIRNOFF SMARTS typing failed -- an exotic element):
      // this app's own from-scratch force field (embed3d.js) has no
      // naglModelId-driven internal charge computation the way
      // CC.OpenFF.optimize3D does, so the solvent object needs real
      // charges supplied directly rather than just a naglModelId.
      const chargesResult = CC.OpenFF.getChargesForAtoms3D(molecule, initial.atoms, initial.bonds, opts.naglModelId);
      const solvent = Object.assign({}, solventOpts, { charges: chargesResult.charges });
      optimized = await CC.optimize3D(initial, { timeBudgetMs: timeBudgetMs, attempts: 1, iterations: iterations, solvent: solvent });
    }
    if (!optimized || !optimized.atoms || !optimized.atoms.length || optimized.energy == null || !isFinite(optimized.energy)) {
      // `== null` alone lets a NaN/Infinity energy through silently (NaN
      // == null is false) -- a real, confirmed gap: some exotic-element
      // combinations (e.g. an arsenic- or selenium-bearing nitro compound)
      // converge to a NaN energy rather than throwing, which the offline
      // training-data harness caught only by explicitly checking its own
      // output CSV for NaN values, not because this function raised an
      // error the way its own contract says it should.
      throw new Error('SMIRNOFF/force-field optimization produced no usable structure');
    }

    return optimized.energy;
  };
})();
