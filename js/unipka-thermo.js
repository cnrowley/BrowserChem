/**
 * unipka-thermo.js
 *
 * Combines per-microstate free energies into an actual macro-/micro-pKa
 * value via the real thermodynamic-cycle formula (Uni-pKa paper, "Bridging
 * Machine Learning and Thermodynamics for Accurate pKa Prediction", JACS
 * Au 2024, eqs 1/2) -- NOT an independent-site approximation. Deliberately
 * source-agnostic: this file doesn't care WHERE a "free energy" scalar per
 * microstate comes from, only how to combine them correctly. Two real
 * sources have used it so far:
 *   - A from-scratch Uni-Mol transformer port (this project's first pass
 *     at this feature, since removed as disproportionately heavy for what
 *     this app needs -- see git history / project memory if relevant),
 *     which is what originally validated `macroPKa` against dptech-corp/
 *     Uni-pKa's own unimol/losses/reg_loss.py (FinetuneMSELoss.compute_loss)
 *     and reproduced real experimental pKa for acetic acid (predicted
 *     4.59, exp. 4.76) and phenol (predicted 9.94, exp. 10.0) end to end.
 *   - The current js/pka-freeenergy-predict.js, a real Chemprop D-MPNN
 *     (scripts/train_pka_microstate_freeenergy.py) predicting `g` per
 *     microstate from its graph plus a physical SMIRNOFF+solvation
 *     baseline feature (js/pka-physical-baseline.js) -- same formula
 *     below, unchanged, `targetMean=0, targetStd=1` since that model was
 *     trained fresh to directly support this formula with no shift.
 *
 * Confirmed formula and sign convention: A is the higher-proton-count
 * macrostate (the acid side, e.g. HA), B is the lower-proton-count
 * macrostate (the conjugate base, e.g. A-):
 *
 *   pKa(A,B) = targetMean + targetStd *
 *              (logsumexp(-g_i for i in A) - logsumexp(-g_j for j in B))
 *              / ln(10)
 *
 * `targetMean`/`targetStd` are whatever target-normalization convention
 * the specific free-energy source was trained with (a checkpoint-specific
 * constant, not something this file hardcodes).
 *
 * `CC.UniPKAThermo.populationFractions` (paper eqs 3/4, the full
 * multi-macrostate pH-dependent Boltzmann population over an entire
 * protonation ensemble) is a direct, literal port of the paper's stated
 * formulas, NOT validated against a real reference implementation the way
 * macroPKa is -- no free-energy source used in this project so far
 * actually exercises this combination. Treat it as a documented,
 * lower-confidence extension -- the Titration tab integration
 * (js/app.js's setupTitrationPanel) only needs `macroPKa`'s (via
 * `microPKa`'s single-microstate special case) well-validated per-site
 * pKa values, which feed the existing, independently-validated
 * js/pka-titration.js Henderson-Hasselbalch curve machinery exactly the
 * way `aqueous-pka` already does; it does not need this function at all.
 */

window.CC = window.CC || {};
CC.UniPKAThermo = window.CC.UniPKAThermo || {};

(function () {
  const LN10 = Math.log(10);

  // log(sum(exp(values))), numerically stable via max-subtraction.
  function logSumExp(values) {
    let max = -Infinity;
    for (let i = 0; i < values.length; i++) if (values[i] > max) max = values[i];
    if (!isFinite(max)) return max; // all -Infinity (empty/degenerate input)
    let sum = 0;
    for (let i = 0; i < values.length; i++) sum += Math.exp(values[i] - max);
    return max + Math.log(sum);
  }

  /**
   * `freeEnergiesA`/`freeEnergiesB`: arrays of raw per-microstate free
   * energies, one per microstate in each macrostate -- usually length 1
   * each (this project's own callers, both past and current, only ever
   * build the micro-pKa special case -- see js/pka-freeenergy-predict.js),
   * longer if a caller enumerates real tautomeric/degenerate microstates.
   * `targetMean`/`targetStd`: whatever target-normalization convention the
   * free-energy source was trained with.
   */
  CC.UniPKAThermo.macroPKa = function (freeEnergiesA, freeEnergiesB, targetMean, targetStd) {
    if (!freeEnergiesA.length || !freeEnergiesB.length) {
      throw new Error('macroPKa needs at least one microstate free energy in each macrostate');
    }
    const negA = freeEnergiesA.map(function (g) { return -g; });
    const negB = freeEnergiesB.map(function (g) { return -g; });
    const lseA = logSumExp(negA);
    const lseB = logSumExp(negB);
    return targetMean + targetStd * (lseA - lseB) / LN10;
  };

  /**
   * micro-pKa (paper eq 1) is the macro-pKa formula's special case of
   * exactly one microstate on each side -- a single acid/base structure
   * pair, no macrostate-level ensembling.
   */
  CC.UniPKAThermo.microPKa = function (freeEnergyA, freeEnergyB, targetMean, targetStd) {
    return CC.UniPKAThermo.macroPKa([freeEnergyA], [freeEnergyB], targetMean, targetStd);
  };

  /**
   * pH-dependent population fraction across a FULL protonation ensemble
   * (paper eqs 3/4) -- see file header's validation-status caveat before
   * relying on this for anything beyond exploratory use.
   *
   * `microstates`: array of { freeEnergy, protonCount }, spanning every
   * microstate in every macrostate of the ensemble (not just two adjacent
   * macrostates) -- `protonCount` is each microstate's proton count
   * relative to a single consistent reference across the whole ensemble
   * (e.g. js/pka-microstates.js's own net-charge/proton-count bookkeeping
   * from `enumerateMicrostates()`, reused as-is here rather than
   * recomputed). Returns a parallel array of population fractions
   * (summing to 1 across the whole input array) at the given pH.
   *
   * Note this deliberately does NOT apply targetMean/targetStd -- that
   * normalization is only meaningful for the pairwise macro-pKa
   * combination above (see file header); applying it per-microstate here
   * would not reproduce eq 4 as stated in the paper.
   */
  CC.UniPKAThermo.populationFractions = function (microstates, pH) {
    const logWeights = microstates.map(function (m) {
      return -m.freeEnergy - m.protonCount * LN10 * pH;
    });
    const lse = logSumExp(logWeights);
    return logWeights.map(function (lw) { return Math.exp(lw - lse); });
  };
})();
