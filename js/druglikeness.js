/**
 * druglikeness.js
 *
 * Five real, published rule-based "drug-likeness" filters (Lipinski,
 * Ghose, Veber, Egan, Muegge) evaluated on RDKit's own already-computed
 * 2D descriptors (chemistry.js's `descriptors` object — see
 * CC.DESCRIPTOR_FIELDS's header for where that comes from), plus
 * percentile-rank comparison against a real FDA-approved-drug-proxy
 * reference distribution (data/druglikeness_reference.json, built by
 * scripts/compute_druglikeness_distributions.py from ChEMBL max_phase==4
 * molecules — see that script's own header for the full provenance and
 * exact filter thresholds, pulled directly from a real SwissADME results
 * page, not from memory).
 *
 * IMPORTANT DISCLOSED APPROXIMATION: this project has no separate
 * implementation of WLOGP/XLOGP3/MLOGP (the specific fragment-
 * contribution LogP methods Ghose/Egan/Muegge's original published
 * thresholds actually use) — every filter below substitutes RDKit's own
 * Crippen LogP (`CrippenClogP`) wherever one of those is called for.
 * Threshold NUMBERS themselves are exact and unmodified; only the LogP
 * method feeding them differs, and consistently so (same substitution
 * the reference-distribution script itself makes, so the app's live
 * numbers and the reference distribution they're compared against are
 * at least internally consistent with each other, even where they may
 * disagree with SwissADME's own reported answer for the same molecule).
 *
 * Martin's 2005 Bioavailability Score is deliberately NOT implemented —
 * its exact decision-tree logic isn't reliably known here, and this
 * project's convention is to skip a metric rather than guess at its
 * definition.
 */

window.CC = window.CC || {};
CC.DrugLikeness = window.CC.DrugLikeness || {};

(function () {
  let referenceData = null; // data/druglikeness_reference.json, fetched once and cached
  let referenceLoadPromise = null;

  CC.DrugLikeness.loadReference = function () {
    if (referenceData) return Promise.resolve(referenceData);
    if (referenceLoadPromise) return referenceLoadPromise;
    referenceLoadPromise = fetch('data/druglikeness_reference.json')
      .then(function (r) {
        if (!r.ok) throw new Error('failed to fetch druglikeness_reference.json: ' + r.status);
        return r.json();
      })
      .then(function (data) {
        referenceData = data;
        return data;
      });
    return referenceLoadPromise;
  };

  CC.DrugLikeness.hasReference = function () { return !!referenceData; };
  CC.DrugLikeness.getReferenceMeta = function () {
    return referenceData ? { n: referenceData.n, description: referenceData.description, filterPassRates: referenceData.filterPassRates } : null;
  };

  // Binary search into the reference's sorted array for `key` -- returns
  // a 0-100 percentile (what fraction of the ~3300 approved-drug
  // reference set has a value <= this one), or null if no reference is
  // loaded / the key isn't one of the ones exported.
  CC.DrugLikeness.percentileRank = function (key, value) {
    if (!referenceData || typeof value !== 'number' || isNaN(value)) return null;
    const arr = referenceData.properties[key];
    if (!arr || arr.length === 0) return null;
    let lo = 0, hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (arr[mid] <= value) lo = mid + 1; else hi = mid;
    }
    return (lo / arr.length) * 100;
  };

  /**
   * Pulls the specific descriptor values these filters need out of
   * chemistry.js's full RDKit descriptor blob (`descriptors`) plus a
   * carbon-atom count from the molecule itself (Muegge's own criterion,
   * not something RDKit's get_descriptors() exposes directly).
   */
  function extractInputs(descriptors, molecule) {
    let carbonCount = 0;
    if (molecule) {
      molecule.atoms.forEach(function (a) { if (a.element === 'C') carbonCount++; });
    }
    return {
      mw: descriptors.amw,
      logP: descriptors.CrippenClogP,
      tpsa: descriptors.tpsa,
      hbd: descriptors.lipinskiHBD,
      hba: descriptors.lipinskiHBA,
      mr: descriptors.CrippenMR,
      rotatableBonds: descriptors.NumRotatableBonds,
      heavyAtoms: descriptors.NumHeavyAtoms,
      rings: descriptors.NumRings,
      carbonCount: carbonCount,
      heteroatomCount: descriptors.NumHeteroatoms,
    };
  }

  // Same five rules as scripts/compute_druglikeness_distributions.py's
  // evaluate_filters() -- kept deliberately identical so the live app's
  // pass/fail and the reference distribution's pass RATES describe the
  // exact same rule. Each criterion carries its own label/value/units
  // alongside the pass/fail boolean (not just a bare boolean) so the
  // per-rule detail modal can show exactly which specific criteria this
  // molecule matches or violates, not just an aggregate count.
  function evaluateFilters(d) {
    const rules = {
      lipinski: [
        { label: 'Molecular weight ≤ 500', value: d.mw, unit: 'g/mol', violated: d.mw > 500 },
        { label: 'LogP (Crippen) ≤ 5', value: d.logP, unit: '', violated: d.logP > 5 },
        { label: 'H-bond donors ≤ 5', value: d.hbd, unit: '', violated: d.hbd > 5 },
        { label: 'H-bond acceptors ≤ 10', value: d.hba, unit: '', violated: d.hba > 10 },
      ],
      ghose: [
        { label: 'Molecular weight 160–480', value: d.mw, unit: 'g/mol', violated: !(d.mw >= 160 && d.mw <= 480) },
        { label: 'LogP (Crippen) -0.4–5.6', value: d.logP, unit: '', violated: !(d.logP >= -0.4 && d.logP <= 5.6) },
        { label: 'Molar refractivity 40–130', value: d.mr, unit: '', violated: !(d.mr >= 40 && d.mr <= 130) },
        { label: 'Heavy atoms 20–70', value: d.heavyAtoms, unit: '', violated: !(d.heavyAtoms >= 20 && d.heavyAtoms <= 70) },
      ],
      veber: [
        { label: 'Rotatable bonds ≤ 10', value: d.rotatableBonds, unit: '', violated: d.rotatableBonds > 10 },
        { label: 'Topological PSA ≤ 140', value: d.tpsa, unit: 'Å²', violated: d.tpsa > 140 },
      ],
      egan: [
        { label: 'LogP (Crippen) ≤ 5.88', value: d.logP, unit: '', violated: d.logP > 5.88 },
        { label: 'Topological PSA ≤ 131.6', value: d.tpsa, unit: 'Å²', violated: d.tpsa > 131.6 },
      ],
      muegge: [
        { label: 'Molecular weight 200–600', value: d.mw, unit: 'g/mol', violated: !(d.mw >= 200 && d.mw <= 600) },
        { label: 'LogP (Crippen) -2–5', value: d.logP, unit: '', violated: !(d.logP >= -2 && d.logP <= 5) },
        { label: 'Topological PSA ≤ 150', value: d.tpsa, unit: 'Å²', violated: d.tpsa > 150 },
        { label: 'Rings ≤ 7', value: d.rings, unit: '', violated: d.rings > 7 },
        { label: 'Carbon count > 4', value: d.carbonCount, unit: '', violated: !(d.carbonCount > 4) },
        { label: 'Heteroatom count > 1', value: d.heteroatomCount, unit: '', violated: !(d.heteroatomCount > 1) },
        { label: 'Rotatable bonds ≤ 15', value: d.rotatableBonds, unit: '', violated: d.rotatableBonds > 15 },
        { label: 'H-bond acceptors ≤ 10', value: d.hba, unit: '', violated: d.hba > 10 },
        { label: 'H-bond donors ≤ 5', value: d.hbd, unit: '', violated: d.hbd > 5 },
      ],
    };
    const violations = {};
    Object.keys(rules).forEach(function (name) {
      violations[name] = rules[name].filter(function (c) { return c.violated; }).length;
    });
    return { violations: violations, rules: rules };
  }

  const FILTER_LABELS = {
    lipinski: 'Lipinski (Rule of Five)',
    ghose: 'Ghose',
    veber: 'Veber',
    egan: 'Egan',
    muegge: 'Muegge',
  };
  const PROPERTY_LABELS = {
    mw: 'Molecular weight',
    logP: 'LogP (Crippen)',
    tpsa: 'Topological PSA',
    hbd: 'H-bond donors',
    hba: 'H-bond acceptors',
    mr: 'Molar refractivity',
    rotatableBonds: 'Rotatable bonds',
    heavyAtoms: 'Heavy atoms',
  };
  CC.DrugLikeness.FILTER_LABELS = FILTER_LABELS;
  CC.DrugLikeness.PROPERTY_LABELS = PROPERTY_LABELS;

  /**
   * Full evaluation for the current structure: filter pass/fail +
   * violation counts (each with the reference set's own pass rate, if
   * loaded) and per-property percentile ranks against the approved-drug
   * reference distribution.
   *
   * Returns { inputs, filters: { <name>: {violations, pass, referencePassRate, criteria} },
   *           percentiles: { <propKey>: percentileOrNull } }.
   * `criteria` is the per-criterion breakdown (label/value/unit/violated)
   * for that rule -- what the "why" infobox shows, since "2 violations"
   * alone doesn't say which two.
   * Works even if the reference JSON hasn't loaded yet (filters/percentiles
   * just come back without referencePassRate / as null) -- callers should
   * still call CC.DrugLikeness.loadReference() themselves so it's ready
   * for next time.
   */
  CC.DrugLikeness.evaluate = function (descriptors, molecule) {
    const inputs = extractInputs(descriptors, molecule);
    const evaluated = evaluateFilters(inputs);
    const filters = {};
    Object.keys(evaluated.violations).forEach(function (name) {
      filters[name] = {
        violations: evaluated.violations[name],
        pass: evaluated.violations[name] === 0,
        referencePassRate: referenceData ? referenceData.filterPassRates[name] : null,
        criteria: evaluated.rules[name],
      };
    });
    const percentiles = {};
    Object.keys(PROPERTY_LABELS).forEach(function (key) {
      percentiles[key] = CC.DrugLikeness.percentileRank(key, inputs[key]);
    });
    return { inputs: inputs, filters: filters, percentiles: percentiles };
  };
})();
