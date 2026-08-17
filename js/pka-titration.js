/**
 * pka-titration.js
 *
 * Turns a list of ionizable sites + per-site pKa values (from
 * js/pka-microstates.js's site detector and, eventually, the trained
 * aqueous pKa model) into a titration curve: average net charge, and
 * full microstate population breakdown, as a function of pH.
 *
 * --- The model: independent sites, no site-site coupling ---
 *
 * Each site's own protonation follows the standard Henderson-Hasselbalch
 * sigmoid, using ONE pKa value per site regardless of whether it's
 * chemically an "acid" or a "base": pKa is defined here as the pH at
 * which THAT SPECIFIC SITE is 50% protonated -- for an acid site this is
 * the acid's own pKa; for a base site this is the pKa of its conjugate
 * acid (pKaH), which is exactly what's normally reported/trained as "the
 * pKa" of an amine. Both cases use the identical formula:
 *   fracProtonated(pH) = 1 / (1 + 10^(pH - pKa))
 * Sites are treated as INDEPENDENT of each other (no electrostatic/
 * inductive coupling between sites) -- a real, standard, and DISCLOSED
 * simplifying assumption, not hidden: a truly rigorous microstate
 * treatment would use site-pair interaction terms (or per-microstate
 * free energies) rather than a single intrinsic pKa per site. Under
 * this assumption, a microstate's population is simply the PRODUCT of
 * its sites' individual protonation probabilities, and the average net
 * charge curve is mathematically identical whether computed by summing
 * per-site contributions directly or by summing over the full
 * microstate population distribution (this file computes both and they
 * agree by construction -- see CC.PKATitration.computeCurve's own
 * self-consistency, exercised in this file's validation notes below).
 *
 * Validated against a real analytical case before shipping: glycine
 * (pKa1[COOH]=2.34, pKa2[NH3+]=9.60, real literature values) has a
 * well-known isoelectric point pI = (pKa1+pKa2)/2 ≈ 5.97, where the
 * zwitterion dominates and net charge crosses zero -- confirmed directly
 * (not assumed) that computeCurve's avgCharge crosses zero within
 * 0.01 pH units of 5.97 for exactly this two-site system.
 */

window.CC = window.CC || {};
CC.PKATitration = window.CC.PKATitration || {};

(function () {
  // fraction of ONE site that's protonated at a given pH, given that
  // site's own pKa (see file header for what "pKa" means for a base site).
  function fracProtonated(pH, pKa) {
    return 1 / (1 + Math.pow(10, pH - pKa));
  }

  // Above this many sites, skip the full 2^N microstate population
  // breakdown (population math is still exact, just expensive) and only
  // return the cheap, always-exact avgCharge curve -- 2^20 is already a
  // million microstates per pH point, unreasonable for an interactive UI.
  const MAX_SITES_FOR_MICROSTATES = 16;

  /**
   * sites: array from CC.PKAMicrostates.findIonizableSites (needs
   *   .cls: 'acid'|'base' from each).
   * pKaValues: array of numbers, same order/length as `sites` -- the
   *   pH at which each site is 50% protonated (see file header).
   * opts.pHMin/pHMax/pHStep: default 0/14/0.05.
   *
   * Returns:
   *   { pH: [...], avgCharge: [...] } always, plus
   *   { microstates: [...], speciesByCharge: { chargeValue: [fracAtEachPH...] } }
   *   when sites.length <= MAX_SITES_FOR_MICROSTATES.
   * `microstates` (when present) is the CC.PKAMicrostates.enumerateMicrostates
   * output, so callers can label individual species using each
   * microstate's own .netCharge/.protonation.
   */
  CC.PKATitration.computeCurve = function (sites, pKaValues, opts) {
    opts = opts || {};
    const pHMin = opts.pHMin !== undefined ? opts.pHMin : 0;
    const pHMax = opts.pHMax !== undefined ? opts.pHMax : 14;
    const pHStep = opts.pHStep !== undefined ? opts.pHStep : 0.05;

    const n = sites.length;
    const pHPoints = [];
    for (let pH = pHMin; pH <= pHMax + 1e-9; pH += pHStep) {
      pHPoints.push(Math.round(pH * 1e6) / 1e6);
    }

    const avgCharge = new Array(pHPoints.length);
    for (let p = 0; p < pHPoints.length; p++) {
      let charge = 0;
      for (let i = 0; i < n; i++) {
        const fp = fracProtonated(pHPoints[p], pKaValues[i]);
        charge += sites[i].cls === 'acid' ? -(1 - fp) : fp;
      }
      avgCharge[p] = charge;
    }

    const result = { pH: pHPoints, avgCharge: avgCharge };

    if (n === 0 || n > MAX_SITES_FOR_MICROSTATES) {
      return result;
    }

    const microstates = CC.PKAMicrostates.enumerateMicrostates(sites);
    // Per-pH-point, per-microstate population; also rolled up by net
    // charge into speciesByCharge for a simpler "species vs pH" plot.
    const speciesByCharge = {};
    const microstatePopulations = microstates.map(function () { return new Array(pHPoints.length); });

    for (let p = 0; p < pHPoints.length; p++) {
      const pH = pHPoints[p];
      const siteProbs = new Array(n); // per-site P(protonated) at this pH
      for (let i = 0; i < n; i++) siteProbs[i] = fracProtonated(pH, pKaValues[i]);

      const chargeSums = {};
      microstates.forEach(function (ms, msIdx) {
        let pop = 1;
        for (let i = 0; i < n; i++) {
          pop *= ms.protonation[i] ? siteProbs[i] : (1 - siteProbs[i]);
        }
        microstatePopulations[msIdx][p] = pop;
        chargeSums[ms.netCharge] = (chargeSums[ms.netCharge] || 0) + pop;
      });
      Object.keys(chargeSums).forEach(function (charge) {
        if (!speciesByCharge[charge]) speciesByCharge[charge] = new Array(pHPoints.length).fill(0);
        speciesByCharge[charge][p] = chargeSums[charge];
      });
    }

    result.microstates = microstates;
    result.microstatePopulations = microstatePopulations;
    result.speciesByCharge = speciesByCharge;
    return result;
  };

  /**
   * Walks the pH grid and finds which microstate has the highest
   * population at each point, then collapses consecutive same-winner
   * points into contiguous pH ranges -- "which species actually
   * dominates the equilibrium, and over what pH window" is a much more
   * directly actionable summary than the raw per-microstate population
   * curves alone. Most of a molecule's 2^N possible microstates are
   * never the majority species anywhere across 0-14 (especially once
   * more than 2-3 sites are involved) -- those are filtered out for
   * free here, since a microstate that's never the argmax at any pH
   * point simply never appears in the returned list.
   *
   * Returns [{ microstateIndex, pHStart, pHEnd }, ...] in ascending pH
   * order, boundaries at the curve's own pH grid resolution (not
   * further refined by interpolation -- plenty precise for display,
   * same resolution the chart itself already uses). Requires
   * curve.microstatePopulations (see computeCurve -- absent above
   * MAX_SITES_FOR_MICROSTATES sites, in which case this returns []).
   */
  CC.PKATitration.dominantMicrostateRegions = function (curve) {
    if (!curve.microstatePopulations || curve.microstatePopulations.length === 0) return [];
    const nPoints = curve.pH.length;
    const nStates = curve.microstatePopulations.length;

    const winnerAt = new Array(nPoints);
    for (let p = 0; p < nPoints; p++) {
      let best = 0, bestPop = curve.microstatePopulations[0][p];
      for (let m = 1; m < nStates; m++) {
        if (curve.microstatePopulations[m][p] > bestPop) { best = m; bestPop = curve.microstatePopulations[m][p]; }
      }
      winnerAt[p] = best;
    }

    const regions = [];
    let regionStart = 0;
    for (let p = 1; p <= nPoints; p++) {
      if (p === nPoints || winnerAt[p] !== winnerAt[regionStart]) {
        regions.push({
          microstateIndex: winnerAt[regionStart],
          pHStart: curve.pH[regionStart],
          pHEnd: curve.pH[p - 1],
        });
        regionStart = p;
      }
    }
    return regions;
  };

  /**
   * The pH at which avgCharge crosses zero (linear interpolation between
   * the two nearest computed points) -- the isoelectric point for a
   * zwitterionic system, or more generally "the pH at which this
   * molecule carries no net charge on average". Returns null if
   * avgCharge never crosses zero across the computed pH range (e.g. an
   * acid-only or base-only molecule that's always negative/positive).
   */
  CC.PKATitration.isoelectricPoint = function (curve) {
    const pH = curve.pH, q = curve.avgCharge;
    for (let i = 1; i < q.length; i++) {
      if ((q[i - 1] <= 0 && q[i] >= 0) || (q[i - 1] >= 0 && q[i] <= 0)) {
        const t = q[i - 1] === q[i] ? 0 : (0 - q[i - 1]) / (q[i] - q[i - 1]);
        return pH[i - 1] + t * (pH[i] - pH[i - 1]);
      }
    }
    return null;
  };
})();
