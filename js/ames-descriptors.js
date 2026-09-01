/**
 * ames-descriptors.js
 *
 * Real X_d feature fusion for ames-mutagenicity-v1, UPDATED 2026-09-01
 * after a real, self-discovered bug fix in the alert-extraction tooling
 * (see data/mutagenicity_alerts_benigni_bossa.json's own `note` for the
 * full story: earlier passes over the open-source Toxtree Java rule
 * files were matching SMARTS text sitting inside comment blocks as if
 * it were the live rule -- fixed, and every alert below has since been
 * independently confirmed by hand against the comment-stripped source).
 *
 * 34 X_d descriptors total, in this exact fixed order (must match
 * data/ames/ames_mutagen24_admet9_electro.csv's column order and the
 * --descriptors-columns argument used to train the shipped checkpoint):
 *   1-22. Twenty-two purpose-built genotoxicity/mutagenicity structural
 *         alerts from the Benigni/Bossa rulebase (a Toxtree module) --
 *         nine from Ferrari & Gini 2010's open-access supplementary
 *         material (SA_1, SA_6, SA_7, SA_12, SA_13, SA_14, SA_16, SA_21,
 *         SA_22), thirteen more hand-extracted from Toxtree's own Java
 *         source and independently verified (SA_4, SA_10, SA_11, SA_15,
 *         SA_17, SA_30, SA_31b, SA_31c, SA_42, SA_44, SA_47, SA_49,
 *         SA_50) -- see data/mutagenicity_alerts_benigni_bossa.json for
 *         full provenance/citation and exactly what was excluded and why.
 *   23-24. Two ring-fusion alerts (SA_18: non-heteroaromatic polycyclic
 *         aromatic system, 3+ fused rings; SA_19: heteroaromatic version)
 *         -- these have no SMARTS in the source (the rulebase gives only
 *         a prose definition, since "3+ fused rings" is a global
 *         topological property substructure matching can't express), so
 *         they're custom ring-fusion detection code, a JS port of
 *         scripts/compute_mutagenicity_alert_features.py's
 *         fused_aromatic_ring_systems() using RDKit.js's get_json()
 *         "rdkitRepresentation" extension (atomRings + aromaticAtoms) --
 *         the same access pattern js/graph-builder.js and js/sascorer.js
 *         already use for ring/aromaticity data RDKit.js doesn't expose
 *         through a dedicated API.
 *   25-33. The same 9 ADMET descriptors (logP, LogD(pH7), acidic/basic
 *         site pKa + flags, NAGL-MBIS charge min/max/mean) as
 *         js/admet-x9-descriptors.js computes for the CYP450/BBBP
 *         panels -- reused via CC.ADMETDescriptors.compute() directly,
 *         not reimplemented.
 *   34.   electrophile-reactivity-v1's own predicted probability
 *         (protein-reactivity / covalent-warhead classifier).
 *
 * A real 5-seed offline comparison (scripts/compute_mutagenicity_alert_
 * features.py + scripts/join_admet_x9_descriptors.py, SAME 5 fixed
 * data-seeds/splits for both configurations so the comparison is paired,
 * not just an average-of-different-samples comparison) found this
 * combined 34-descriptor set gives a real, if modest, improvement over
 * the previous 9-alert-alone model: mean test ROC-AUC 0.862 vs 0.852 for
 * the 9-alert baseline retrained on the identical 5 splits (per-seed:
 * this configuration wins 4 of 5 splits; the one loss is the old
 * checkpoint's own unusually favorable seed-0 split, which is why a
 * single-seed number would have been misleading here -- see
 * model/registry.json's metrics.note for the complete before/after
 * numbers). Earlier single-feature-group experiments this file's
 * previous version also ran are still valid and are folded into that
 * note: general medchem PAINS/Glaxo/BMS/etc. filters actively hurt
 * (0.822), and the 9-alerts-alone / 9-alerts-plus-electrophile
 * configurations were statistically tied (0.850 vs 0.851, both on the
 * OLD 3-seed protocol) -- what changed the outcome here wasn't just
 * "more features", it was fixing which alerts were actually correct and
 * combining them with the already-validated ADMET-9 recipe.
 *
 * Same self-registering architecture as js/admet-x9-descriptors.js
 * (model-registry.js's CC.GNN.registerPrerequisiteModels + chemprop-
 * model.js's CC.GNN.registerExtraDescriptorsProvider) -- now WITH
 * prerequisites (logp-v1, aqueous-pka, nagl-mbis-charges,
 * electrophile-reactivity-v1), unlike the old 9-alert-only version.
 *
 * Several SMARTS below reference explicit hydrogens ([#1]) -- e.g. SA_13
 * (hydrazine) only matches phenylhydrazine correctly against an
 * explicit-H graph, confirmed by hand during development (RDKit's
 * default substructure matching only sees the heavy-atom graph, so #1
 * never matches an implicit H). `rdmol.add_hs_in_place()` below is not
 * optional -- omitting it silently under-counts several real alerts,
 * the same bug scripts/compute_mutagenicity_alert_features.py's own
 * Python implementation had before it was caught and fixed there too.
 * Ring-fusion detection runs on the pre-AddHs mol (atom indices for
 * atomRings/aromaticAtoms are heavy-atom indices, unaffected either way,
 * but there is no reason to pay AddHs's cost twice).
 */
window.CC = window.CC || {};
CC.AmesDescriptors = window.CC.AmesDescriptors || {};

(function () {
  var LOGP_MODEL_ID = 'logp-v1';
  var PKA_MODEL_ID = 'aqueous-pka';
  var NAGL_MODEL_ID = 'nagl-mbis-charges';
  var ELECTROPHILE_MODEL_ID = 'electrophile-reactivity-v1';

  // [id, [smarts variants -- ANY match counts]], in the EXACT order
  // data/ames/ames_mutagen24_admet9_electro.csv's alert_sa_* columns
  // were trained with (data/mutagenicity_alerts_benigni_bossa.json's
  // "alerts" array order: original 9 first, then the 13 Toxtree-sourced
  // additions).
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
    ['sa_4', ['[CX3]([!Cl;!Br;!F;!I;!$(C=O)])(!@[#1,CX4])=[CX3]([Cl,F,Br,I])([#1,CX4])']],
    ['sa_10', ['[!a,#1;!$(C1(=O)C=CC(=O)C=C1)][#6]([!a,#1;!$(C1(=O)C=CC(=O)C=C1)])!:;=[#6][#6](=O)[!O;!$([#6]1:,=[#6][#6](=O)[#6]:,=[#6][#6](=O)1)]']],
    ['sa_11', ['[#6][$([CX3H1]);!$(CC=C)](=O)']],
    ['sa_15', ['[NX2]=C=[O,S]']],
    ['sa_17', ['[#7X3][#6](=[SX1])[!$([O,S][CX4])!$([OH,SH])!$([O-,S-])]']],
    ['sa_30', ['O=c1ccc2ccccc2(o1)', 'O=C1C=Cc2ccccc2O1']],
    ['sa_31b', ['[Cl,Br,F,I]c1ccc2ccccc2(c1)', '[Cl,Br,F,I]c1ccc(cc1)!@c2ccc(cc2)[Cl,Br,F,I]', 'c1cc(ccc1[!R]c2ccc(cc2)[Cl,Br,F,I])[Cl,Br,F,I]']],
    ['sa_31c', ['c1ccc2Oc3cc(ccc3(Oc2(c1)))[Cl,Br,F,I]']],
    ['sa_42', ['O=C(O)c1ccccc1C(=O)O', 'O=C(O)[CX4;!R][CX4;!R][CX4;!R][CX4;!R]C(=O)O']],
    ['sa_44', ['[Cl,F][C;!$(Cc)]=C([Cl,F])[Cl,F]', '[Cl,F]C#C[Cl,F]', 'Cl[C;!$(Cc)]=C(Cl)Cl']],
    ['sa_47', ['Oc2ccccc2c1ccccc1', 'Oc1c(c2ccccc2)cccc1']],
    ['sa_49', ['n1c[nH]cc1', 'n2c1ccccc1nc2']],
    ['sa_50', ['[#6]1[#6](=O)[#7][#6](=O)[#6]1']],
  ];

  CC.AmesDescriptors.MODEL_IDS = ['ames-mutagenicity-v1'];

  // Compiled lazily (needs RDKit loaded) and cached -- recompiling ~35
  // SMARTS query mols per prediction would be wasteful.
  var compiledPatterns = null;
  function compilePatterns(RDKit) {
    if (compiledPatterns) return compiledPatterns;
    compiledPatterns = MUTAGENICITY_ALERTS.map(function (entry) {
      return entry[1].map(function (smarts) { return RDKit.get_qmol(smarts); });
    });
    return compiledPatterns;
  }

  function smartsAlertFlags(RDKit, molblock) {
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
  }

  /**
   * JS port of scripts/compute_mutagenicity_alert_features.py's
   * fused_aromatic_ring_systems(): connected components of aromatic
   * SSSR rings sharing >=2 atoms (a fused bond), keeping only components
   * with 3+ rings. Returns [{atoms: Set<atomIndex>, isHetero: bool}, ...].
   */
  function fusedAromaticRingSystems(RDKit, molblock) {
    const mol = RDKit.get_mol(molblock);
    if (!mol || !mol.is_valid()) { if (mol) mol.delete(); return []; }
    try {
      const json = JSON.parse(mol.get_json());
      const molData = json.molecules && json.molecules[0];
      if (!molData) return [];
      const ext = (molData.extensions || []).find(function (e) { return e.name === 'rdkitRepresentation'; });
      const atomRingsRaw = (ext && ext.atomRings) || [];
      const aromaticAtoms = new Set((ext && ext.aromaticAtoms) || []);
      const defaultZ = (json.defaults && json.defaults.atom && json.defaults.atom.z) || 6;

      const aromaticRings = atomRingsRaw
        .filter(function (ring) { return ring.every(function (a) { return aromaticAtoms.has(a); }); })
        .map(function (r) { return new Set(r); });

      const n = aromaticRings.length;
      const parent = [];
      for (let i = 0; i < n; i++) parent.push(i);
      function find(x) { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
      function union(a, b) { const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb; }
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          let shared = 0;
          aromaticRings[i].forEach(function (a) { if (aromaticRings[j].has(a)) shared++; });
          if (shared >= 2) union(i, j);
        }
      }

      const groups = new Map();
      for (let i = 0; i < n; i++) {
        const root = find(i);
        if (!groups.has(root)) groups.set(root, []);
        groups.get(root).push(i);
      }

      const results = [];
      groups.forEach(function (comp) {
        if (comp.length < 3) return;
        const atoms = new Set();
        comp.forEach(function (idx) { aromaticRings[idx].forEach(function (a) { atoms.add(a); }); });
        let isHetero = false;
        (molData.atoms || []).forEach(function (atom, idx) {
          if (!atoms.has(idx)) return;
          const z = atom.z !== undefined ? atom.z : defaultZ;
          if (z !== 6) isHetero = true;
        });
        results.push({ atoms: atoms, isHetero: isHetero });
      });
      return results;
    } finally {
      mol.delete();
    }
  }

  function ringFusionFlags(RDKit, molblock) {
    const systems = fusedAromaticRingSystems(RDKit, molblock);
    const sa18 = systems.some(function (s) { return !s.isHetero; }) ? 1 : 0; // non-heteroaromatic PAH
    const sa19 = systems.some(function (s) { return s.isHetero; }) ? 1 : 0; // heteroaromatic PAH
    return [sa18, sa19];
  }

  /**
   * Computes the fixed-order 34-element X_d descriptor array for
   * `molecule`: 22 SMARTS alert flags + 2 ring-fusion flags + 9 ADMET
   * descriptors + electrophile-reactivity-v1's predicted probability.
   * Requires logp-v1, aqueous-pka, nagl-mbis-charges, and
   * electrophile-reactivity-v1 already loaded (registered as
   * prerequisites below -- loading is the caller's job, same convention
   * as every other engine in this project).
   */
  CC.AmesDescriptors.compute = function (molecule) {
    const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
    if (!RDKit) throw new Error('RDKit not loaded');
    if (!CC.GNN.hasChempropModel(ELECTROPHILE_MODEL_ID)) throw new Error('Ames descriptor features need "' + ELECTROPHILE_MODEL_ID + '" loaded');

    const molblock = CC.moleculeToMolblock(molecule);
    const alertFlags = smartsAlertFlags(RDKit, molblock);
    const ringFlags = ringFusionFlags(RDKit, molblock);
    const admet9 = CC.ADMETDescriptors.compute(molecule, NAGL_MODEL_ID);

    const electroResult = CC.GNN.predictChemprop(molecule, ELECTROPHILE_MODEL_ID);
    const electroScore = electroResult.molecularProperties.label;
    if (typeof electroScore !== 'number') throw new Error('electrophile-reactivity-v1 produced no usable prediction for this molecule');

    return alertFlags.concat(ringFlags, admet9, [electroScore]);
  };

  CC.GNN.registerPrerequisiteModels(function (entry) {
    if (CC.AmesDescriptors.MODEL_IDS.indexOf(entry.id) === -1) return [];
    return [LOGP_MODEL_ID, PKA_MODEL_ID, NAGL_MODEL_ID, ELECTROPHILE_MODEL_ID];
  });

  CC.GNN.registerExtraDescriptorsProvider(function (model, molecule) {
    if (CC.AmesDescriptors.MODEL_IDS.indexOf(model.id) === -1) return undefined;
    return CC.AmesDescriptors.compute(molecule);
  });
})();
