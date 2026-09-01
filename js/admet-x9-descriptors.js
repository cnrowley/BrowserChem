/**
 * admet-x9-descriptors.js
 *
 * Real X_d feature-fusion descriptors ([logP, LogD(pH7), most-acidic
 * site pKa, has-acidic-site flag, most-basic site pKa, has-basic-site
 * flag, NAGL-MBIS charge min/max/mean]) shared by every checkpoint in
 * MODEL_IDS below -- originally built for the CYP450 substrate/
 * metabolism panel (renamed from cyp-descriptors.js once the same
 * recipe proved worth reusing), now also used by bbbp-v1 (blood-brain
 * barrier penetration). Each consumer was retrained with these fused in
 * only after a real offline experiment
 * (scripts/pka-physical-baseline-harness/compute_admet_x9_descriptors.js +
 * scripts/join_admet_x9_descriptors.py, 3 fixed seeds per variant)
 * showed a real, consistent test-set win -- see each model's own
 * `metrics.note` in model/registry.json for the exact before/after
 * numbers. One endpoint this same experiment was ALSO tried on did NOT
 * get retrained with it, because the numbers said no: CYP450 INHIBITION
 * (cyp{isoform}-inhibition-v1 -- its real gain needed 3D SASA instead,
 * judged not worth the cost). Ames mutagenicity (ames-mutagenicity-v1)
 * was ALSO tried and initially rejected on this basis (these 9
 * descriptors alone made it slightly worse than its own mutagenicity-
 * alert features) -- but it does now use this same recipe, fused
 * alongside its alert features rather than in place of them; it is
 * deliberately NOT in MODEL_IDS below because it needs a combined
 * 34-descriptor vector (22 alert flags + 2 ring-fusion flags + these 9
 * ADMET descriptors + electrophile-reactivity-v1's score) that this
 * generic per-model recipe can't express -- see js/ames-descriptors.js,
 * which calls CC.ADMETDescriptors.compute() directly as one ingredient
 * of its own single registered provider instead. That negative-result
 * asymmetry (SASA needed instead, not in addition) is deliberate, not an
 * oversight -- don't add a model to MODEL_IDS below without the same
 * kind of real before/after evidence backing it.
 *
 * This is the browser-runtime SIBLING of compute_admet_x9_descriptors.js's
 * offline data-prep logic -- same formulas, same real deployed models
 * (logp-v1, aqueous-pka, a loaded NAGL-MBIS charge model), same fixed
 * column order the training CSVs used (--descriptors-columns logp logd
 * pka_acidic has_acidic pka_basic has_basic nagl_min nagl_max
 * nagl_mean). Not literally the same code (that script runs offline in
 * Node against a static-server fetch, this runs synchronously inside
 * the live app against already-loaded in-memory models), but every
 * individual step reuses this app's own real inference code unmodified:
 * CC.GNN.predictChemprop, CC.PKAMicrostates.findIonizableSites,
 * CC.PKATitration.fractionNeutral, CC.NAGL.predict -- the exact same
 * calls js/app.js's own LogD panel (renderLogD) and Titration tab
 * already make, not a parallel reimplementation of any of THEIR logic.
 *
 * Wired into the generic prediction path itself, not called directly by
 * any specific UI panel: registers with model-registry.js's
 * CC.GNN.registerPrerequisiteModels (so loading any of MODEL_IDS also
 * transparently loads logp-v1/aqueous-pka/nagl-mbis-charges alongside
 * it, from ANY load path -- autoLoadApplicableModels, a per-model UI
 * load button, category activation) and chemprop-model.js's
 * CC.GNN.registerExtraDescriptorsProvider (so CC.GNN.predictMolecule /
 * predictAllChempropModels compute and fuse these descriptors
 * automatically for ANY caller, not just one that specifically knows to
 * call CC.GNN.predictChemprop(molecule, id, extraDescriptors) directly
 * the way js/pka-freeenergy-predict.js does for pka-microstate-
 * freeenergy). This file is the ONLY place that needs to know these
 * checkpoints exist and what they need -- js/app.js has no
 * descriptor-fusion-specific glue at all.
 */
window.CC = window.CC || {};
CC.ADMETDescriptors = window.CC.ADMETDescriptors || {};

(function () {
  var LOGP_MODEL_ID = 'logp-v1';
  var PKA_MODEL_ID = 'aqueous-pka';
  var NAGL_MODEL_ID = 'nagl-mbis-charges';

  // Checkpoints retrained with this feature set -- kept as an explicit
  // list (not inferred from registry metadata) matching this project's
  // existing convention of hardcoding a small, specific model-id list
  // for one-off special-cased inference paths (js/pka-freeenergy-
  // predict.js's own LOGP_MODEL_ID constant, js/app.js's Titration tab
  // checking `pkaSource === 'pka-microstate-freeenergy'` by literal id).
  // cyp2e1-substrate-v1 and cyp2c9-substrate-v1 are deliberately NOT here
  // as of the 2026-09-01 rework(s) -- cyp2e1-substrate-v1 first moved to
  // a SMILES-only checkpoint fine-tuned from a multitask-pretrained
  // encoder (this recipe's own real ablation showed ADMET-9 helping
  // CYP2C9 the most of any isoform but never helping CYP2E1 as much as
  // multitask representation transfer did); cyp2c9-substrate-v1 then
  // ALSO moved off this recipe the same day, to a SMILES-only checkpoint
  // fine-tuned from the CHEMELEON foundation model (--from-foundation
  // CHEMELEON, a D-MPNN pretrained on 1M molecules to predict Mordred
  // descriptors -- Burns 2025) -- a real, decisive win over ADMET-9+
  // class-balance on every metric (see model/registry.json's
  // cyp2c9-substrate-v1 entry for the full comparison, including a real
  // head-to-head loss against a from-scratch-fine-tuned ChemBERTa-2
  // transformer baseline). Both are numExtraDescriptors=0 now.
  CC.ADMETDescriptors.MODEL_IDS = [
    'cyp1a2-substrate-v1',
    'cyp2c19-substrate-v1',
    'cyp2d6-substrate-v1',
    'cyp3a4-substrate-v1',
    'bbbp-v1',
  ];

  /**
   * Computes the fixed-order 9-element X_d descriptor array for
   * `molecule`. Requires logp-v1, aqueous-pka, and `naglModelId` (a
   * NAGL-MBIS charge model) already loaded -- throws a clear error
   * naming whichever is missing rather than silently auto-loading
   * (loading models is the caller's job, same convention as every other
   * engine in this project -- see js/pka-model.js's CC.PKA.predict).
   */
  CC.ADMETDescriptors.compute = function (molecule, naglModelId) {
    if (!CC.GNN.hasChempropModel(LOGP_MODEL_ID)) throw new Error('ADMET descriptor features need "' + LOGP_MODEL_ID + '" loaded');
    if (!CC.GNN.hasChempropModel(PKA_MODEL_ID)) throw new Error('ADMET descriptor features need "' + PKA_MODEL_ID + '" loaded');
    if (!naglModelId || !CC.NAGL.hasModel(naglModelId)) throw new Error('ADMET descriptor features need a NAGL-MBIS charge model loaded');

    const logpResult = CC.GNN.predictChemprop(molecule, LOGP_MODEL_ID);
    const logP = logpResult.molecularProperties.logP;
    if (typeof logP !== 'number') throw new Error('logp-v1 produced no usable prediction for this molecule');

    const sites = CC.PKAMicrostates.findIonizableSites(molecule);
    let pkaAcidic = 7, hasAcidic = 0, pkaBasic = 7, hasBasic = 0, fractionNeutral = 1;
    if (sites.length > 0) {
      const pkaResult = CC.GNN.predictChemprop(molecule, PKA_MODEL_ID);
      const pkaByAtomId = {};
      pkaResult.atomIds.forEach(function (atomId, idx) {
        const props = pkaResult.atomProperties[idx];
        if (props && typeof props.pka === 'number') pkaByAtomId[atomId] = props.pka;
      });
      const validSites = [], validPKa = [];
      sites.forEach(function (site) {
        if (typeof pkaByAtomId[site.atomId] === 'number') {
          validSites.push(site);
          validPKa.push(pkaByAtomId[site.atomId]);
        }
      });
      const acidPkas = [], basePkas = [];
      validSites.forEach(function (s, idx) {
        (s.cls === 'acid' ? acidPkas : basePkas).push(validPKa[idx]);
      });
      if (acidPkas.length) { pkaAcidic = Math.min.apply(null, acidPkas); hasAcidic = 1; }
      if (basePkas.length) { pkaBasic = Math.max.apply(null, basePkas); hasBasic = 1; }
      if (validSites.length) fractionNeutral = CC.PKATitration.fractionNeutral(validSites, validPKa, 7.0);
    }
    const logD = logP + Math.log10(fractionNeutral);

    const naglResult = CC.NAGL.predict(molecule, naglModelId);
    const charges = naglResult.atomProperties.map(function (p) { return Object.values(p)[0]; }).filter(function (v) { return typeof v === 'number'; });
    if (!charges.length) throw new Error('no NAGL charges available for this molecule');
    const naglMin = Math.min.apply(null, charges);
    const naglMax = Math.max.apply(null, charges);
    const naglMean = charges.reduce(function (a, b) { return a + b; }, 0) / charges.length;

    return [logP, logD, pkaAcidic, hasAcidic, pkaBasic, hasBasic, naglMin, naglMax, naglMean];
  };

  // See model-registry.js's CC.GNN.registerPrerequisiteModels header for
  // why this lives here instead of in js/app.js: loadRegistryModel is
  // the one real choke point every load path in the app goes through, so
  // registering here guarantees any of MODEL_IDS getting loaded --
  // however that happens -- pulls its prerequisites in alongside it.
  CC.GNN.registerPrerequisiteModels(function (entry) {
    if (CC.ADMETDescriptors.MODEL_IDS.indexOf(entry.id) === -1) return [];
    return [LOGP_MODEL_ID, PKA_MODEL_ID, NAGL_MODEL_ID];
  });

  // See chemprop-model.js's CC.GNN.registerExtraDescriptorsProvider
  // header. Only claims models in MODEL_IDS -- returning undefined (not
  // throwing) for anything else lets other registered providers, or the
  // generic "skip with a clear reason" fallback, still apply normally.
  // If the prerequisites above somehow aren't loaded yet (a caller that
  // bypassed loadRegistryModel entirely), compute() throws and this
  // provider's caller (computeExtraDescriptors in chemprop-model.js)
  // catches it and tries the next provider / falls through to the
  // generic skip -- same graceful degradation as before, not a crash.
  CC.GNN.registerExtraDescriptorsProvider(function (model, molecule) {
    if (CC.ADMETDescriptors.MODEL_IDS.indexOf(model.id) === -1) return undefined;
    return CC.ADMETDescriptors.compute(molecule, NAGL_MODEL_ID);
  });
})();
