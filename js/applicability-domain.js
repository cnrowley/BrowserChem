/**
 * applicability-domain.js
 *
 * Per-model (not per-engine) applicability-domain data, produced offline
 * by scripts/compute_applicability_domain.py from a model's REAL training
 * CSV and shipped as model/<dir>/applicability-domain.json (pointed to by
 * registry.json's files.applicabilityDomain). Two things this gives the
 * app that js/chemprop-features.js's checkChempropCompatibility() can't:
 *
 *   1. HARD per-model element/formal-charge gate. checkChempropCompatibility
 *      only checks against Chemprop's shared GLOBAL one-hot featurizer
 *      vocabulary (~40 elements) and treats a miss as a soft warning,
 *      since the featurizer itself pads gracefully to an "unknown"
 *      bucket rather than erroring. That's a real answer to "will this
 *      run at all" but not to "was this model ever trained on anything
 *      like this" -- a model can support (be featurizable for) an
 *      element it never once saw during training. CC.AD.checkVocab
 *      checks against the exact element/charge set this SPECIFIC
 *      checkpoint's training data actually contained, and is wired into
 *      structure-validation.js's hard-compatibility gate (tier
 *      'blocked') and directly into chemprop-model.js's prediction loop
 *      (a blocked model is skipped, not run, even if already loaded from
 *      a previous, compatible molecule).
 *
 *   2. Embedding-domain confidence tier. Distance from a query molecule's
 *      D-MPNN pooled embedding (the same 300-dim vector the model's own
 *      FFN head reads, already computed for free during prediction) to
 *      that same training set's embedding distribution, summarized as a
 *      small set of k-means centroids. Nearest-centroid distance below
 *      the training set's own 90th-percentile self-distance is
 *      'in-domain'; below the 99th percentile is 'borderline'; beyond
 *      that is 'out-of-domain'. This is a heuristic signal (see
 *      compute_applicability_domain.py's docstring for why it's not a
 *      calibrated statistical interval), not a guarantee -- surfaced to
 *      the user as a tier label, never as a false-precision number.
 *
 * A model with no applicability-domain.json (most of the registry, for
 * now -- this is a new, incrementally-rolled-out pipeline, see
 * CLAUDE.md/registry.json notes) degrades gracefully: checkVocab returns
 * compatible:true (falls back to the existing global-vocabulary check
 * elsewhere), and tierForEmbedding returns tier:'unknown'. Absence of
 * data is never treated as "outside the domain" -- only real recorded
 * data blocks a prediction.
 */

window.CC = window.CC || {};
CC.AD = window.CC.AD || {};

(function () {
  const sidecars = new Map(); // modelId -> parsed applicability-domain.json, or 'unavailable'

  function resolveUrl(base, relative) {
    try {
      return new URL(relative, base).href;
    } catch (err) {
      return new URL(relative, window.location.href).href;
    }
  }

  /**
   * Best-effort background fetch of every registry entry's
   * applicability-domain.json (if it declares one). Never rejects --
   * a fetch failure just leaves that model's sidecar 'unavailable',
   * same as never having one. Call once after the registry itself has
   * loaded (see model-registry.js).
   */
  CC.AD.prefetch = function (entries, registryBaseUrl) {
    (entries || []).forEach(function (entry) {
      const rel = entry.files && entry.files.applicabilityDomain;
      if (!rel || sidecars.has(entry.id)) return;
      const url = resolveUrl(registryBaseUrl, rel);
      fetch(url)
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (data) { sidecars.set(entry.id, data); })
        .catch(function () { sidecars.set(entry.id, 'unavailable'); });
    });
  };

  CC.AD.getSidecar = function (modelId) {
    const s = sidecars.get(modelId);
    return s && s !== 'unavailable' ? s : null;
  };

  /**
   * { compatible: boolean, issues: string[] }. No sidecar loaded (yet,
   * or ever) => compatible:true -- absence of data is not evidence of
   * incompatibility, see file header.
   */
  CC.AD.checkVocab = function (molecule, modelId) {
    const sidecar = CC.AD.getSidecar(modelId);
    if (!sidecar || !molecule) return { compatible: true, issues: [] };
    const trainingSet = sidecar.trainingSet;
    if (!trainingSet) return { compatible: true, issues: [] };
    const elements = new Set(trainingSet.elements || []);
    const charges = new Set(trainingSet.formalCharges || []);
    const netCharges = new Set(trainingSet.netMolecularCharges || []);
    const issues = [];
    const atoms = Array.from(molecule.atoms.values());
    let netCharge = 0;
    atoms.forEach(function (atom) {
      netCharge += atom.charge;
      if (elements.size > 0 && !elements.has(atom.element)) {
        issues.push('element "' + atom.element + '" never appeared in this model’s ' +
          trainingSet.size + '-molecule training set (trained on: ' + Array.from(elements).sort().join(', ') + ')');
      }
      if (charges.size > 0 && !charges.has(atom.charge)) {
        issues.push('formal charge ' + atom.charge + ' on a ' + atom.element + ' atom never appeared in training ' +
          '(trained on charges: ' + Array.from(charges).sort(function (a, b) { return a - b; }).join(', ') + ')');
      }
    });
    // Per-atom charge membership alone misses the common case of a
    // training set built entirely from net-neutral zwitterions (e.g. a
    // +1 ammonium paired with a -1 carboxylate in the SAME molecule) --
    // every individual charge value would look "seen", but a bare
    // standalone anion/cation (different net molecular charge) is still
    // a real extrapolation. See compute_applicability_domain.py's
    // net_charges computation for why this is tracked separately.
    if (atoms.length > 0 && netCharges.size > 0 && !netCharges.has(netCharge)) {
      issues.push('net molecular charge ' + netCharge + ' never appeared in training ' +
        '(trained on net charges: ' + Array.from(netCharges).sort(function (a, b) { return a - b; }).join(', ') + ')');
    }
    return { compatible: issues.length === 0, issues: issues };
  };

  /**
   * { tier: 'in-domain'|'borderline'|'out-of-domain'|'unknown', distance, thresholds }.
   * `embedding` is the model's own pooled molecule-embedding array
   * (same dimensionality as sidecar.embeddingDomain.dim).
   */
  CC.AD.tierForEmbedding = function (modelId, embedding) {
    const sidecar = CC.AD.getSidecar(modelId);
    const domain = sidecar && sidecar.embeddingDomain;
    if (!domain || !domain.centroids || domain.centroids.length === 0 || !embedding) {
      return { tier: 'unknown' };
    }
    let minDist = Infinity;
    for (let c = 0; c < domain.centroids.length; c++) {
      const centroid = domain.centroids[c];
      let sumSq = 0;
      for (let i = 0; i < embedding.length; i++) {
        const d = embedding[i] - centroid[i];
        sumSq += d * d;
      }
      const dist = Math.sqrt(sumSq);
      if (dist < minDist) minDist = dist;
    }
    const thresholds = domain.tierThresholds || {};
    let tier = 'out-of-domain';
    if (thresholds.inDomain !== undefined && minDist <= thresholds.inDomain) tier = 'in-domain';
    else if (thresholds.borderline !== undefined && minDist <= thresholds.borderline) tier = 'borderline';
    return { tier: tier, distance: minDist, thresholds: thresholds };
  };
})();
