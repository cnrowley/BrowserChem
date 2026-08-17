/**
 * pka-model.js
 *
 * Runs this project's from-scratch carbon-acid (C-H) pKa predictor
 * fully client-side: a LightGBM/DART gradient-boosted-tree regressor
 * (model/pka-ch-nagl/) evaluated over pka-descriptor.js's graph-charge-
 * shell descriptor, which itself is built from this app's own NAGL-MBIS
 * partial charges (model/nagl-mbis-charges/, nagl-model.js) rather than
 * the CM5/GFN1-xTB charges the original method (jensengroup/pKalculator,
 * Ree et al., ChemRxiv 2024, 10.26434/chemrxiv-2024-56h5h) uses -- CM5
 * needs a real xtb binary subprocess with no WASM build, so it can't run
 * in a browser at all. See PKA_INTEGRATION.md for the full retraining
 * writeup and honest accuracy comparison against the original CM5-based
 * model (this one is measurably less accurate -- NAGL charges are a
 * good but imperfect approximation of real semiempirical-QM charges).
 *
 * Candidate sites: any carbon atom bearing at least one hydrogen
 * (implicit or explicit), matching the union of pKalculator's own
 * site-enumeration SMARTS ([CX4;H1-4], [CX3;H1-2], [CX2;H1] --
 * qm_pkalculator/modify_smiles.py's `rm_proton` conditions) -- in
 * practice just "any real carbon with a nonzero total H count", since
 * degree-1 carbons essentially don't occur in ordinary drawn structures.
 *
 * Tree format: every tree is flattened into one shared node pool
 * (parallel typed arrays: isLeaf/featureIdx/threshold/defaultLeft/
 * leftChild/rightChild/value), with a per-tree root offset --
 * scripts/convert_pka_lightgbm.py's header has the full self-check that
 * this flat-array walk reproduces LightGBM's own Booster.predict()
 * exactly. A DART-trained model is evaluated at inference time exactly
 * like plain GBDT (DART's tree-dropout only affects training) -- so
 * prediction is just "sum every tree's leaf value", no extra scaling.
 */

window.CC = window.CC || {};
CC.PKA = window.CC.PKA || {};

(function () {
  const models = new Map(); // id -> parsed model

  CC.PKA.loadModel = function (id, manifestUrl, binUrl) {
    return Promise.all([
      fetch(manifestUrl).then(function (r) {
        if (!r.ok) throw new Error('failed to fetch manifest: ' + r.status);
        return r.json();
      }),
      fetch(binUrl).then(function (r) {
        if (!r.ok) throw new Error('failed to fetch weights: ' + r.status);
        return r.arrayBuffer();
      }),
    ]).then(function (results) {
      return CC.PKA.loadModelFromBuffers(id, results[0], results[1]);
    });
  };

  CC.PKA.loadModelFromBuffers = function (id, manifest, binArrayBuffer) {
    if (!id) throw new Error('loadModelFromBuffers needs an id');

    function section(name) {
      const t = manifest.tensors[name];
      if (!t) throw new Error('manifest is missing tensor "' + name + '"');
      if (t.dtype === 'uint8') return new Uint8Array(binArrayBuffer, t.byteOffset, t.length);
      if (t.dtype === 'int32') return new Int32Array(binArrayBuffer, t.byteOffset, t.length);
      if (t.dtype === 'float32') return new Float32Array(binArrayBuffer, t.byteOffset, t.length);
      throw new Error('unknown dtype "' + t.dtype + '" for tensor "' + name + '"');
    }

    const model = {
      id: id,
      task: manifest.task || 'pka-ch',
      numFeatures: manifest.numFeatures,
      numTrees: manifest.numTrees,
      treeRootOffsets: manifest.treeRootOffsets,
      descriptor: manifest.descriptor || { type: 'graph-charge-shell', nShells: 6, useCipSort: true, chargeSource: 'nagl-mbis-charges' },
      isLeaf: section('isLeaf'),
      featureIdx: section('featureIdx'),
      threshold: section('threshold'),
      defaultLeft: section('defaultLeft'),
      leftChild: section('leftChild'),
      rightChild: section('rightChild'),
      value: section('value'),
    };

    models.set(id, model);
    return { id: id, task: model.task };
  };

  CC.PKA.hasModel = function (id) { return id ? models.has(id) : models.size > 0; };
  CC.PKA.getLoadedModelIds = function () { return Array.from(models.keys()); };
  CC.PKA.clearModel = function (id) { if (id) models.delete(id); else models.clear(); };

  function evalTree(model, rootNode, x) {
    let node = rootNode;
    while (!model.isLeaf[node]) {
      const v = x[model.featureIdx[node]];
      const goLeft = (v === null || v === undefined || Number.isNaN(v)) ? !!model.defaultLeft[node] : v <= model.threshold[node];
      node = goLeft ? model.leftChild[node] : model.rightChild[node];
    }
    return model.value[node];
  }

  function evalModel(model, x) {
    let sum = 0;
    for (let t = 0; t < model.numTrees; t++) sum += evalTree(model, model.treeRootOffsets[t], x);
    return sum;
  }

  /**
   * Carbon atoms bearing at least one hydrogen (implicit or explicit) --
   * see file header for the SMARTS this generalizes from. Returns heavy
   * atom indices (into molecule.atoms.values() order).
   */
  CC.PKA.findCandidateSites = function (molecule) {
    const molblock = CC.moleculeToMolblock(molecule);
    const annotations = CC.GNN.getRDKitAnnotations(molblock);
    const heavyAtoms = Array.from(molecule.atoms.values());
    const numHByIndex = annotations.numHByAtomIndex || new Map();
    const sites = [];
    heavyAtoms.forEach(function (atom, i) {
      if (atom.element !== 'C') return;
      const numH = numHByIndex.has(i) ? numHByIndex.get(i) : 0;
      if (numH >= 1) sites.push(i);
    });
    return sites;
  };

  /**
   * Predicts C-H pKa for every candidate carbon site in `molecule`.
   * Requires the NAGL-MBIS charge model this pKa model was trained
   * against (model.descriptor.chargeSource) to already be loaded --
   * throws a clear error naming it if not, rather than silently loading
   * it (loading models is the caller's/UI's job, same convention as
   * every other engine in this project).
   *
   * Returns { atomProperties, atomIds, backend: 'pka', modelId } in the
   * same shape chemprop-model.js/nagl-model.js's atom-level output uses
   * (see gnn-inference.js's merge logic): atomProperties has one entry
   * per HEAVY atom in molecule.atoms.values() order, empty ({}) for
   * non-candidate atoms so array indices still line up for merging.
   */
  CC.PKA.predict = function (molecule, id) {
    const model = models.get(id);
    if (!model) throw new Error('No pKa model loaded under id "' + id + '"');

    const chargeSourceId = model.descriptor.chargeSource;
    if (!window.CC.NAGL || !CC.NAGL.hasModel || !CC.NAGL.hasModel(chargeSourceId)) {
      throw new Error('This pKa model needs the "' + chargeSourceId + '" partial-charge model loaded first.');
    }

    const heavyAtoms = Array.from(molecule.atoms.values());
    const atomIds = heavyAtoms.map(function (a) { return a.id; });
    const atomProperties = heavyAtoms.map(function () { return {}; });

    if (heavyAtoms.length === 0) {
      return { atomProperties: atomProperties, atomIds: atomIds, backend: 'pka', modelId: id };
    }

    const sites = CC.PKA.findCandidateSites(molecule);
    if (sites.length === 0) {
      return { atomProperties: atomProperties, atomIds: atomIds, backend: 'pka', modelId: id };
    }

    const full = CC.NAGL.predictAll(molecule, chargeSourceId);
    const nShells = model.descriptor.nShells;
    const useCipSort = model.descriptor.useCipSort;

    sites.forEach(function (siteIdx) {
      const desc = CC.PKA.buildDescriptor(full.elements, full.charges, full.adjacency, siteIdx, nShells, useCipSort);
      const pka = evalModel(model, desc);
      atomProperties[siteIdx][model.task] = pka;
    });

    return { atomProperties: atomProperties, atomIds: atomIds, backend: 'pka', modelId: id };
  };

  /**
   * Real requirement per predict() above: a pKa checkpoint needs its own
   * specific NAGL-MBIS charge model (model.descriptor.chargeSource)
   * already loaded. This family-level check (same "engine in general,
   * not whichever specific checkpoint happens to be loaded" convention
   * every other checkCompatibility in this project already uses -- see
   * CC.GNN.checkChempropCompatibility's own header) can't know exactly
   * which chargeSource id a not-yet-loaded pKa checkpoint will need, so
   * it checks the weaker but still real fact: is ANY NAGL model loaded
   * at all. predict()'s own check above still catches a mismatched
   * chargeSource id precisely, with a name, at call time.
   *
   * Registered as this engine's `validate` -- previously there was no
   * checkCompatibility for 'pka' at all, so structure-validation.js
   * silently fell through to the generic chemprop vocabulary check
   * instead of this real one.
   */
  CC.PKA.checkCompatibility = function (molecule) {
    if (!molecule || molecule.atoms.size === 0) return { compatible: true, issues: [] };
    if (!window.CC.NAGL || !CC.NAGL.hasModel || !CC.NAGL.hasModel()) {
      return { compatible: false, issues: ['requires a NAGL-MBIS partial-charge model to be loaded first'] };
    }
    return { compatible: true, issues: [] };
  };

  // See model-adapters.js's header.
  CC.ModelAdapters.register('pka', {
    kind: 'property',
    load: CC.PKA.loadModel,
    unload: CC.PKA.clearModel,
    hasModel: CC.PKA.hasModel,
    getLoadedModelIds: CC.PKA.getLoadedModelIds,
    validate: CC.PKA.checkCompatibility,
    predict: CC.PKA.predict,
  });
})();
