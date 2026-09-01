/**
 * ames-descriptors.js
 *
 * Real X_d feature fusion for ames-mutagenicity-v1: 9 purpose-built
 * genotoxicity/mutagenicity structural alerts from the Benigni/Bossa
 * rulebase (a Toxtree module) -- see data/mutagenicity_alerts_benigni_bossa.json
 * for full citation/provenance (Ferrari & Gini 2010, BMC Chemistry,
 * open access, Additional File 4, itself citing the official EU JRC
 * Benigni/Bossa rulebase technical report). These 9 exact SMARTS
 * strings, hardcoded below in the SAME order compute_mutagenicity_
 * alert_features.py used to build the training data, are what
 * ames-mutagenicity-v1 was actually trained with.
 *
 * A real offline experiment (3 fixed seeds per variant) found this is
 * the single best feature group tried for Ames mutagenicity -- better
 * than electrophile-reactivity-v1's own predicted probability (mean
 * test ROC-AUC 0.850 vs 0.844), and MUCH better than this project's
 * general medchem PAINS/Glaxo/BMS/etc. filters, which actively HURT
 * (0.822). Combining these alerts with electrophile-reactivity's score
 * added essentially nothing on top (0.851, within seed noise of 0.850
 * alone) -- so electrophile-reactivity-v1 was deliberately dropped as a
 * dependency: same accuracy, one fewer model to load at inference time,
 * since these alerts are pure deterministic RDKit SMARTS matching, not
 * a second trained model. See model/registry.json's ames-mutagenicity-v1
 * entry for the complete before/after comparison across everything
 * tried (including what did NOT help: generic filters, PAH ring-fusion
 * detection, atom-level one-hot encoding of these same alerts, and
 * transfer learning from electrophile-reactivity-v1's own weights).
 *
 * Same self-registering architecture as js/admet-x9-descriptors.js
 * (model-registry.js's CC.GNN.registerPrerequisiteModels + chemprop-
 * model.js's CC.GNN.registerExtraDescriptorsProvider) -- but with an
 * EMPTY prerequisite list, since no other model needs to be loaded:
 * the only "prerequisite" here is RDKit itself, already loaded by the
 * time any prediction can run at all.
 *
 * Several of these SMARTS reference explicit hydrogens ([#1]) -- e.g.
 * SA_13 (hydrazine) only matches phenylhydrazine correctly against an
 * explicit-H graph, confirmed by hand during development (RDKit's
 * default substructure matching only sees the heavy-atom graph, so #1
 * never matches an implicit H). `rdmol.add_hs_in_place()` below is not
 * optional -- omitting it silently under-counts several real alerts,
 * the same bug scripts/compute_mutagenicity_alert_features.py's own
 * Python implementation had before it was caught and fixed there too.
 */
window.CC = window.CC || {};
CC.AmesDescriptors = window.CC.AmesDescriptors || {};

(function () {
  // [id, [smarts variants -- ANY match counts]], same order/definitions
  // as data/mutagenicity_alerts_benigni_bossa.json's "alerts" array.
  var MUTAGENICITY_ALERTS = [
    ['sa_1', ['[!$([OH1,SH1])]C(=O)[Br,Cl,F,I]']],
    ['sa_6', ['[O,S]=C1[O,S]CC1', 'O=S1(=O)(CCCO1)']],
    ['sa_7', ['C1[O,N]C1']],
    ['sa_12', ['O=[#6]1[#6]=,:[#6][#6](=O)[#6]=,:[#6]1', 'O=[#6]1[#6]=,:[#6][#6]=,:[#6][#6]1(=O)']],
    ['sa_13', ['[N+0]!@;-[N+0](=[!O;!N])', '[N+0]([#1,*])!@;-[N+0]([#1,*])']],
    ['sa_14', ['[C,#1]N=[NX2][C,#1]', '[$(C=[N+]=[N-]);!$(C=[N+]=[N-]=N);!$(C=[N+]=[N-]N)]', 'C=[$(N=N);!$(N=N=N);!$(N=NN)]', 'CN=NO']],
    ['sa_16', ['[NX3]([CX4,#1])([CX4,#1])C(=[O,S])[O,S][CX4]']],
    ['sa_21', ['[C,c]N[NX2;v3]=O']],
    ['sa_22', ['[N]=[N]-[N]', '[N]=[N]=[N]']],
  ];

  CC.AmesDescriptors.MODEL_IDS = ['ames-mutagenicity-v1'];

  // Compiled lazily (needs RDKit loaded) and cached -- recompiling ~15
  // SMARTS query mols per prediction would be wasteful.
  var compiledPatterns = null;
  function compilePatterns(RDKit) {
    if (compiledPatterns) return compiledPatterns;
    compiledPatterns = MUTAGENICITY_ALERTS.map(function (entry) {
      return entry[1].map(function (smarts) { return RDKit.get_qmol(smarts); });
    });
    return compiledPatterns;
  }

  /**
   * Computes the fixed-order 9-element X_d descriptor array for
   * `molecule`: one binary flag per alert (does the molecule match ANY
   * of that alert's SMARTS variants?).
   */
  CC.AmesDescriptors.compute = function (molecule) {
    const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
    if (!RDKit) throw new Error('RDKit not loaded');

    const molblock = CC.moleculeToMolblock(molecule);
    const rdmol = RDKit.get_mol(molblock);
    if (!rdmol || !rdmol.is_valid()) throw new Error('RDKit could not parse this molecule for mutagenicity alert matching');
    try {
      rdmol.add_hs_in_place();
      const patterns = compilePatterns(RDKit);
      return patterns.map(function (qmols) {
        return qmols.some(function (qmol) {
          const matches = JSON.parse(rdmol.get_substruct_matches(qmol));
          // get_substruct_matches returns the literal string "{}" (not
          // "[]") for zero matches -- see js/pka-microstates.js's own
          // header for this same RDKit.js quirk.
          return Array.isArray(matches) && matches.length > 0;
        }) ? 1 : 0;
      });
    } finally {
      rdmol.delete();
    }
  };

  CC.GNN.registerPrerequisiteModels(function (entry) {
    if (CC.AmesDescriptors.MODEL_IDS.indexOf(entry.id) === -1) return [];
    return []; // no other model needed -- pure RDKit SMARTS matching
  });

  CC.GNN.registerExtraDescriptorsProvider(function (model, molecule) {
    if (CC.AmesDescriptors.MODEL_IDS.indexOf(model.id) === -1) return undefined;
    return CC.AmesDescriptors.compute(molecule);
  });
})();
