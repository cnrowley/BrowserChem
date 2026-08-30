/**
 * pka-freeenergy-predict.js
 *
 * Ties together this app's own site detection (js/pka-microstates.js),
 * physical-baseline energy (js/pka-physical-baseline.js), the trained
 * pka-microstate-freeenergy Chemprop checkpoint (a normal `engine:
 * "chemprop"` registry entry -- no bespoke engine needed, unlike the
 * earlier Uni-Mol port: this model IS just a plain graph-only Chemprop
 * D-MPNN, loaded and run through the same js/chemprop-model.js every
 * other model here uses), the charge-charge electrostatic correction
 * (js/pka-electrostatic-correction.js, reused from the `aqueous-pka` path
 * rather than reimplemented), and the validated thermodynamic-cycle
 * formula (js/unipka-thermo.js) into one per-site pKa computation for the
 * Titration tab.
 *
 * --- Delta learning PLUS real X_d feature fusion (not the physical baseline itself) ---
 *
 * Each microstate's free energy is g = physical_baseline + chemprop's
 * OWN raw output -- an explicit additive correction on the physical
 * baseline's own scale, computed here (not inside chemprop-model.js). The
 * "no feature fusion" convention from this file's earlier versions still
 * holds for the physical_baseline scalar specifically (it stays an
 * additive delta-learning term, never fed in as an X_d descriptor) -- but
 * the currently-deployed checkpoint DOES use real X_d descriptors for two
 * OTHER signals the physical baseline can't see at all: this microstate's
 * own NAGL-MBIS charge extremes (min/max across its heavy atoms -- same
 * two numbers scripts/pka-physical-baseline-harness/add_extra_features.js
 * computed for every training row) and this app's own logp-v1 model's
 * prediction for it. `microstateFreeEnergy` below computes both fresh for
 * whichever molecule it's given, auto-loading `logp-v1` if it isn't
 * already loaded (same "just make it work" convenience this file's own
 * NAGL-charge dependency already has via `opts.naglModelId`) -- checked
 * against `info.numExtraDescriptors` so a future checkpoint trained
 * without feature fusion (numExtraDescriptors===0) skips this work
 * entirely rather than passing descriptors a graph-only model doesn't
 * expect.
 *
 * --- Same independent-site-per-microstate-pair scope as before ---
 *
 * Exactly ONE microstate pair per detected site (this site flipped, every
 * other site held at its neutral reference protonation state) -- the
 * micro-pKa special case (real, not an approximation of it), for the same
 * reasons js/pka-titration.js's own header already discloses for
 * `aqueous-pka`. `targetMean=0, targetStd=1` is passed to
 * CC.UniPKAThermo.microPKa because THIS model (trained fresh, not
 * finetuning a pretrained checkpoint with its own pre-existing target
 * normalization) was trained to directly support the formula with no
 * shift -- see scripts/train_pka_microstate_freeenergy.py's own header.
 * What's NOT independent-site any more: `predictAllSites` below now
 * applies js/pka-electrostatic-correction.js's real Generalized-Born
 * charge-charge coupling term across the detected site list afterward,
 * same as `aqueous-pka` already does -- a real (if bounded, see that
 * file's own header) accounting for site-site interaction this model's
 * own per-site computation doesn't see on its own.
 */

window.CC = window.CC || {};
CC.PKAFreeEnergy = window.CC.PKAFreeEnergy || {};

(function () {
  const LOGP_MODEL_ID = 'logp-v1';

  function netFormalCharge(molecule) {
    let net = 0;
    molecule.atoms.forEach(function (a) { net += a.charge; });
    return net;
  }

  /**
   * logp-v1 has its own real applicability-domain vocabulary gate (see
   * js/applicability-domain.js's CC.AD.checkVocab, enforced inside
   * CC.GNN.predictChemprop) that refuses to run on a net-charged molecule
   * -- confirmed directly: it was trained on net charges {0, 1} only, so
   * every deprotonated ANION this app ever builds would be refused.
   * scripts/train_pka_microstate_freeenergy.py's own feature-fusion
   * training already works around this at the DATA level (one shared
   * logP value per row, computed from whichever microstate is net-
   * charge-0 -- see that script's own header) -- this mirrors it exactly
   * at inference time, computed once per site rather than once per
   * microstate. Auto-loads logp-v1 if it isn't already loaded, same
   * "just make it work" convenience `opts.naglModelId` already gets from
   * its own callers.
   */
  async function sharedLogPFor(acidMolecule, baseMolecule) {
    const neutral = netFormalCharge(acidMolecule) === 0 ? acidMolecule
      : netFormalCharge(baseMolecule) === 0 ? baseMolecule : null;
    if (!neutral) throw new Error('neither microstate of this site is net-charge-0 -- cannot compute a logp-v1-compatible logP');
    if (!CC.GNN.isRegistryModelLoaded(LOGP_MODEL_ID)) {
      await CC.GNN.loadRegistryModel(LOGP_MODEL_ID);
    }
    const logpResult = CC.GNN.predictChemprop(neutral, LOGP_MODEL_ID);
    const logP = logpResult.molecularProperties.logP;
    if (typeof logP !== 'number') throw new Error('logp-v1 produced no usable prediction for this site');
    return logP;
  }

  /**
   * [naglChargeMin, naglChargeMax, logP] for one microstate, in the exact
   * order scripts/pka-physical-baseline-harness/add_extra_features.js
   * wrote its own naglChargeMin/naglChargeMax/logP training columns --
   * order matters here (X_d is a plain positional array, not a keyed
   * object). `sharedLogP` is the SAME value for both microstates of one
   * site (see sharedLogPFor above), not recomputed per microstate.
   */
  function extraDescriptorsFor(molecule, naglModelId, sharedLogP) {
    if (!naglModelId || !CC.NAGL.hasModel(naglModelId)) {
      throw new Error('extra descriptors need a loaded NAGL-MBIS model (opts.naglModelId)');
    }
    const naglResult = CC.NAGL.predict(molecule, naglModelId);
    const charges = naglResult.atomProperties.map(function (p) { return Object.values(p)[0]; });
    if (!charges.length) throw new Error('no NAGL charges available for this molecule');
    const naglChargeMin = Math.min.apply(null, charges);
    const naglChargeMax = Math.max.apply(null, charges);
    return [naglChargeMin, naglChargeMax, sharedLogP];
  }

  async function microstateFreeEnergy(chempropModelId, molecule, opts, sharedLogP) {
    const physicalBaseline = await CC.PKAPhysicalBaseline.compute(molecule, opts);
    const info = CC.GNN.getChempropModelInfo(chempropModelId);
    if (!info) throw new Error('Chemprop model "' + chempropModelId + '" is not loaded');
    const extraDescriptors = info.numExtraDescriptors
      ? extraDescriptorsFor(molecule, opts && opts.naglModelId, sharedLogP)
      : undefined;
    const result = CC.GNN.predictChemprop(molecule, chempropModelId, extraDescriptors);
    const correction = result.molecularProperties[info.task];
    // g = physical_scale*physical_baseline + physical_offset + correction
    // -- the learned affine recalibration is NOT optional: this app's own
    // SMIRNOFF force-field energy is explicitly documented elsewhere
    // (js/app.js's own 3D panel code) as arbitrary hand-tuned units, not
    // real kcal/mol, so it cannot be added directly into a formula that
    // assumes real physical energy units -- confirmed as a real bug
    // during training (MAE 9.2 with scale assumed =1, MAE 0.82 once
    // scale/offset were learned jointly with the correction -- see
    // scripts/train_pka_microstate_freeenergy.py's own header).
    const scale = info.physicalScale != null ? info.physicalScale : 1;
    const offset = info.physicalOffset != null ? info.physicalOffset : 0;
    return scale * physicalBaseline + offset + correction;
  }

  // Same reference-protonation convention as js/pka-microstates.js's own
  // enumerateMicrostates (acid sites default protonated=true, base sites
  // default protonated=false), site `siteIndex` flipped to build the two
  // microstate structures its own protonation/deprotonation connects.
  function buildSiteMacrostates(molecule, sites, siteIndex) {
    const referenceProtonation = sites.map(function (s) { return s.cls === 'acid'; });
    const acidProtonation = referenceProtonation.slice();
    acidProtonation[siteIndex] = true;
    const baseProtonation = referenceProtonation.slice();
    baseProtonation[siteIndex] = false;

    const acidResult = CC.PKAMicrostates.buildMicrostateStructure(molecule, sites, { protonation: acidProtonation });
    const baseResult = CC.PKAMicrostates.buildMicrostateStructure(molecule, sites, { protonation: baseProtonation });
    return { acidMolecule: acidResult && acidResult.molecule, baseMolecule: baseResult && baseResult.molecule };
  }

  /**
   * Computes the pKa for one detected site (from
   * CC.PKAMicrostates.findIonizableSites). `opts` (all optional):
   * `naglModelId`, `timeBudgetMs` -- forwarded to
   * CC.PKAPhysicalBaseline.compute.
   */
  CC.PKAFreeEnergy.computeSitePka = async function (chempropModelId, molecule, sites, siteIndex, opts) {
    const macro = buildSiteMacrostates(molecule, sites, siteIndex);
    if (!macro.acidMolecule || !macro.baseMolecule) {
      throw new Error('could not build a valid structure for site ' + siteIndex);
    }
    const info = CC.GNN.getChempropModelInfo(chempropModelId);
    const sharedLogP = (info && info.numExtraDescriptors)
      ? await sharedLogPFor(macro.acidMolecule, macro.baseMolecule)
      : undefined;
    const gA = await microstateFreeEnergy(chempropModelId, macro.acidMolecule, opts, sharedLogP);
    const gB = await microstateFreeEnergy(chempropModelId, macro.baseMolecule, opts, sharedLogP);
    const pKa = CC.UniPKAThermo.microPKa(gA, gB, 0, 1);
    return { pKa: pKa, freeEnergyA: gA, freeEnergyB: gB };
  };

  /**
   * Builds one solvent-included-optimized 3D structure of the whole
   * molecule AS DRAWN (every site at its neutral reference state, formal
   * charges from the molecule itself -- NOT a specific protonated/
   * deprotonated microstate) for js/pka-electrostatic-correction.js's own
   * charge-charge distance calculation, which needs real 3D geometry with
   * every site simultaneously present, unlike the per-site microstate
   * PAIRS the rest of this file builds. Same solvent-included optimizer
   * call as js/pka-physical-baseline.js (consistency: the correction's
   * geometry shouldn't be built under different physics than the free
   * energies it's correcting), just not wrapped in that file's own
   * function since this needs the optimized atoms array itself, not a
   * collapsed energy scalar.
   */
  async function buildReferenceGeometry(molecule, opts) {
    const initial = CC.buildInitial3D(molecule);
    if (!initial.atoms.length) throw new Error('no 3D structure could be built for this molecule');
    if (!CC.OpenFF.isForceFieldLoaded()) await CC.OpenFF.loadForceField();
    const optimized = await CC.OpenFF.optimize3D(initial, {
      timeBudgetMs: (opts && opts.timeBudgetMs) || 3000, attempts: 1, naglModelId: opts && opts.naglModelId,
      solvent: { enabled: true, epsSolvent: 78.5, sasaModel: 'default' },
    });
    if (!optimized || !optimized.atoms || !optimized.atoms.length) throw new Error('could not build reference 3D geometry for electrostatic correction');
    return optimized.atoms;
  }

  /**
   * Computes pKa at every detected site, in the same
   * { atomProperties, atomIds, backend, modelId } shape every other
   * atom-level engine in this app returns -- NOT registered as a
   * CC.ModelAdapters engine (like the earlier Uni-Mol integration's own
   * predictNotSupported note explains: this needs CC.embed3D/SMIRNOFF
   * internally, so it's inherently async and can't run through the
   * generic synchronous property-merge path). The Titration tab
   * (js/app.js's setupTitrationPanel) is the real, intended caller.
   *
   * For more than one detected site, applies
   * CC.PKAElectrostaticCorrection.compute on top of the independently-
   * computed per-site values (same real, bounded Generalized-Born
   * charge-charge coupling correction `aqueous-pka` already uses) -- a
   * no-op for every molecule this app currently ships a test for (all
   * single-site), real for the multi-site case this integration didn't
   * previously account for at all. A failed correction (e.g. the
   * reference geometry doesn't converge) falls back to the uncorrected
   * per-site values rather than failing the whole prediction.
   */
  CC.PKAFreeEnergy.predictAllSites = async function (chempropModelId, molecule, opts) {
    const heavyAtoms = Array.from(molecule.atoms.values());
    const atomIds = heavyAtoms.map(function (a) { return a.id; });
    const atomProperties = heavyAtoms.map(function () { return {}; });

    if (heavyAtoms.length === 0) {
      return { atomProperties: atomProperties, atomIds: atomIds, backend: 'pka-freeenergy', modelId: chempropModelId };
    }

    const sites = CC.PKAMicrostates.findIonizableSites(molecule);
    const scoredSites = [];
    const basePKa = [];
    for (let i = 0; i < sites.length; i++) {
      try {
        const pKa = (await CC.PKAFreeEnergy.computeSitePka(chempropModelId, molecule, sites, i, opts)).pKa;
        scoredSites.push(sites[i]);
        basePKa.push(pKa);
      } catch (err) {
        continue; // one unscoreable site (e.g. an element NAGL can't charge) doesn't sink the others
      }
    }

    let finalPKa = basePKa;
    if (scoredSites.length > 1) {
      try {
        const atoms3D = await buildReferenceGeometry(molecule, opts);
        finalPKa = CC.PKAElectrostaticCorrection.compute(molecule, scoredSites, basePKa, atoms3D).correctedPKa;
      } catch (err) {
        finalPKa = basePKa; // disclosed fallback, not a silent one -- see doc comment above
      }
    }

    scoredSites.forEach(function (site, i) {
      const atomIdx = heavyAtoms.findIndex(function (a) { return a.id === site.atomId; });
      if (atomIdx >= 0) atomProperties[atomIdx]['pka-microstate-freeenergy'] = finalPKa[i];
    });

    return { atomProperties: atomProperties, atomIds: atomIds, backend: 'pka-freeenergy', modelId: chempropModelId };
  };
})();
