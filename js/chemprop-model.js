/**
 * chemprop-model.js
 *
 * Loads real trained Chemprop D-MPNNs (each exported from a `best.pt`
 * Lightning checkpoint into a small manifest.json + weights.bin pair —
 * see convert_chemprop_checkpoint.py) and runs them fully client-side:
 * no server, no ONNX Runtime, no PyTorch. Reuses dmpnn.js's
 * BondMessagePassing forward pass and chemprop-features.js's bit-exact
 * featurizers; adds the readout (norm aggregation) and the final FFN
 * predictor head.
 *
 * Multiple models can be loaded at once, each keyed by an id (see
 * model-registry.js, which is the normal way models get loaded — this
 * file also works standalone for the manual "load from files" path).
 * A molecule prediction runs every currently-loaded model and merges
 * their outputs — a chemist can have logP, hERG, and SA-adjacent
 * properties loaded simultaneously without them competing for a single
 * "the loaded model" slot the way earlier versions of this file worked.
 *
 * Supports both output heads Chemprop's RegressionFFN/BinaryClassificationFFN
 * produce (manifest.taskType tells this file which): regression applies
 * the checkpoint's UnscaleTransform (raw * scale + mean); classification
 * applies a plain sigmoid, matching BinaryClassificationFFN.forward()'s
 * `Y.sigmoid()` exactly (see chemprop's nn/predictors.py) — the
 * probability of the positive class, not a raw logit.
 *
 * Also supports atom-level models (manifest.outputLevel === "atom", e.g.
 * per-atom partial charges) alongside the usual molecule-level ones
 * (outputLevel === "molecule", or omitted — older manifests predate this
 * field and always meant molecule-level). The underlying D-MPNN math is
 * identical either way — Chemprop's MABBondMessagePassing (used whenever
 * --atom-target-columns or --bond-target-columns is set) is the exact
 * same bond-message-passing formula as plain BondMessagePassing, just
 * under a different class name, confirmed directly against its own
 * docstring. An atom-level model only changes what happens *after*
 * message passing: no pooling at all, and the FFN head runs once per
 * atom on that atom's own embedding, instead of once on the
 * NormAggregation-pooled molecule vector.
 *
 * Why not ONNX? See gnn-inference.js's header: exporting a D-MPNN's
 * dynamic scatter/gather ops to ONNX and running them in onnxruntime-web
 * is genuinely fragile. This app's D-MPNN is small (hidden~300, depth~3)
 * and molecules drawn by hand are tiny (tens of atoms), so a plain JS
 * forward pass is both simpler and fast enough — this is the same
 * strategy dmpnn.js's "demo" backend already uses, just with real weights.
 *
 * Expected checkpoint shape (what the conversion script must produce):
 *   manifest.taskType = "regression" | "classification"
 *   manifest.outputLevel = "molecule" | "atom" (default "molecule")
 *   manifest.graphType = "heavy" | "explicit-h" (default "heavy",
 *                    atom-level only) — "explicit-h" means the checkpoint
 *                    was trained with chemprop's --add-h (every hydrogen
 *                    is its own graph node, e.g. a per-atom 1H NMR shift
 *                    checkpoint) and needs
 *                    chemprop-features-explicit-h.js's graph builder
 *                    instead of the normal heavy-atom-only one.
 *   manifest.applicableElement = an element symbol, e.g. "C" (atom-level
 *                    only, optional) — if set, only atoms of this element
 *                    get a value in the returned atomProperties (others
 *                    are left as {}), the same pattern pka-model.js uses
 *                    for its own candidate-site gating. For an
 *                    "explicit-h" model this also selects which heavy
 *                    atoms' attached-H predictions get aggregated and
 *                    surfaced (see runOneAtomLevelExplicitH()).
 *   manifest.dims = { d_v: 72, d_e: 14, d_h, depth, aggNorm }
 *                    — aggNorm is null/unused for an atom-level model,
 *                      since there's no pooling step to normalize.
 *   manifest.tensors.{W_i, W_h, W_o_weight, W_o_bias,
 *                      ffn0_weight, ffn0_bias, ffn1_weight, ffn1_bias} = { shape, offset, length }
 *   manifest.tensors.{out_mean, out_scale} = { shape, offset, length }
 *                      — regression only; absent for classification,
 *                        since BinaryClassificationFFN's output_transform
 *                        is Identity() (nothing to unscale).
 *   weights.bin = all of those tensors concatenated as float32, in that
 *                 same order, row-major (matches nn.Linear.weight layout).
 */

window.CC = window.CC || {};
CC.GNN = window.CC.GNN || {};

(function () {
  const models = new Map(); // id -> { dims, task, taskType, Wi, WiBias, Wh, WhBias, Wo, WoBias, ffn0, ffn0Bias, ffn1, ffn1Bias, outMean, outScale }

  // Slice a flat Float32Array into an array of row views (Float32Array
  // subarrays) — the shape dmpnn.js's matVecBias expects for a weight
  // matrix, and cheap since subarray() doesn't copy.
  function toRows(flat, shape) {
    const rows = shape[0];
    const cols = shape.length > 1 ? shape[1] : 1;
    const out = new Array(rows);
    for (let r = 0; r < rows; r++) out[r] = flat.subarray(r * cols, (r + 1) * cols);
    return out;
  }

  function relu(v) { return v.map(function (x) { return Math.max(0, x); }); }
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  /**
   * Fetch and parse a manifest+bin pair for one model, keyed by `id`
   * (caller's choice — model-registry.js uses the registry entry's id).
   * Returns a Promise resolving to model info ({ id, task, taskType, dims }).
   */
  CC.GNN.loadChempropModel = function (id, manifestUrl, binUrl) {
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
      return CC.GNN.loadChempropModelFromBuffers(id, results[0], results[1]);
    });
  };

  /**
   * Same as loadChempropModel, but from an already-parsed manifest object
   * and an ArrayBuffer of weights (useful for a <input type=file> flow
   * where the browser already has both in memory).
   */
  CC.GNN.loadChempropModelFromBuffers = function (id, manifest, binArrayBuffer) {
    if (!id) throw new Error('loadChempropModelFromBuffers needs an id to key this model under');
    const bytes = new Float32Array(binArrayBuffer);
    function tensor(name) {
      const t = manifest.tensors[name];
      if (!t) throw new Error('manifest is missing tensor "' + name + '"');
      return bytes.subarray(t.offset, t.offset + t.length);
    }
    function tensorShape(name) { return manifest.tensors[name].shape; }

    const model = {
      id: id,
      task: manifest.task || 'prediction',
      taskType: manifest.taskType || 'regression', // older manifests predate this field
      outputLevel: manifest.outputLevel || 'molecule', // older manifests predate this field too, always meant molecule-level
      graphType: manifest.graphType || 'heavy',
      applicableElement: manifest.applicableElement || null,
      dims: manifest.dims,
      Wi: toRows(tensor('W_i'), tensorShape('W_i')), WiBias: null,
      Wh: toRows(tensor('W_h'), tensorShape('W_h')), WhBias: null,
      Wo: toRows(tensor('W_o_weight'), tensorShape('W_o_weight')),
      WoBias: Array.from(tensor('W_o_bias')),
      ffn0: toRows(tensor('ffn0_weight'), tensorShape('ffn0_weight')),
      ffn0Bias: Array.from(tensor('ffn0_bias')),
      ffn1: toRows(tensor('ffn1_weight'), tensorShape('ffn1_weight')),
      ffn1Bias: Array.from(tensor('ffn1_bias')),
    };
    if (model.taskType === 'regression') {
      model.outMean = tensor('out_mean')[0];
      model.outScale = tensor('out_scale')[0];
    }

    models.set(id, model);
    return { id: id, task: model.task, taskType: model.taskType, dims: model.dims };
  };

  // With an id: is that specific model loaded. Without: is anything loaded.
  CC.GNN.hasChempropModel = function (id) {
    return id ? models.has(id) : models.size > 0;
  };
  CC.GNN.getChempropModelInfo = function (id) {
    const m = models.get(id);
    return m ? { id: m.id, task: m.task, taskType: m.taskType, outputLevel: m.outputLevel, graphType: m.graphType, applicableElement: m.applicableElement, dims: m.dims } : null;
  };
  CC.GNN.getLoadedChempropModelIds = function () { return Array.from(models.keys()); };
  CC.GNN.getLoadedChempropModels = function () {
    return Array.from(models.values()).map(function (m) {
      return { id: m.id, task: m.task, taskType: m.taskType, outputLevel: m.outputLevel, graphType: m.graphType, applicableElement: m.applicableElement, dims: m.dims };
    });
  };
  // With an id: unload just that model. Without: unload everything.
  CC.GNN.clearChempropModel = function (id) {
    if (id) models.delete(id); else models.clear();
  };

  function matVecBias(W, bias, x) {
    const y = new Array(W.length);
    for (let r = 0; r < W.length; r++) {
      let sum = bias ? bias[r] : 0;
      const row = W[r];
      for (let c = 0; c < x.length; c++) sum += row[c] * x[c];
      y[r] = sum;
    }
    return y;
  }

  // Applies ffn0 -> ReLU -> ffn1 -> task-appropriate output head to a
  // single embedding vector (either the pooled molecule vector, or one
  // atom's own embedding — this part of the math doesn't care which).
  function applyHead(model, embedding) {
    const hidden = relu(matVecBias(model.ffn0, model.ffn0Bias, embedding));
    const raw = matVecBias(model.ffn1, model.ffn1Bias, hidden)[0];
    return model.taskType === 'classification'
      ? sigmoid(raw) // BinaryClassificationFFN.forward(): Y.sigmoid() -- a probability, not a logit
      : raw * model.outScale + model.outMean;
  }

  function runDMPNNFor(model, graph) {
    const dmpnnWeights = {
      hiddenSize: model.dims.d_h,
      Wi: model.Wi, WiBias: model.WiBias,
      Wh: model.Wh, WhBias: model.WhBias,
      Wo: model.Wo, WoBias: model.WoBias,
    };
    return CC.GNN.runDMPNN(graph, dmpnnWeights, { depth: model.dims.depth });
  }

  // Molecule-level: NormAggregation (sum of atom embeddings / aggNorm),
  // then one FFN application on the pooled vector.
  function runOneMolecule(model, graph) {
    const out = runDMPNNFor(model, graph);
    const pooled = CC.GNN.poolSum(out.atomEmbeddings, model.dims.d_h)
      .map(function (x) { return x / model.dims.aggNorm; });
    return applyHead(model, pooled);
  }

  // Atom-level, "heavy" graph: no pooling at all -- the FFN runs once per
  // atom, directly on that atom's own embedding (see header comment:
  // MABBondMessagePassing's per-atom output is the same W_o/W_vo
  // computation plain BondMessagePassing already does, Chemprop just
  // skips the aggregation step for this case). Returns an array of
  // values, one per atom, in the same order as graph.atomIds.
  function runOneAtomLevel(model, graph) {
    const out = runDMPNNFor(model, graph);
    return out.atomEmbeddings.map(function (embedding) { return applyHead(model, embedding); });
  }

  // Atom-level, "explicit-h" graph (chemprop-features-explicit-h.js):
  // runs the FFN on EVERY node (heavy + synthetic H), then aggregates
  // each heavy atom's own attached-H predictions down to one number
  // (mean) -- this app's atom-heatmap UI colors real drawn (heavy) atoms
  // only, the same reason nagl-model.js's public predict() doesn't
  // surface its own synthetic H nodes' results individually. A real,
  // deliberate loss of resolution (e.g. a CH2's two diastereotopic
  // protons collapse to their average), not a bug -- documented in
  // js/pka-model.js's sibling pattern and worth the same honesty here.
  // Returns an array of length graph.numHeavyAtoms, NaN for any heavy
  // atom with no attached H's (nothing to aggregate).
  function runOneAtomLevelExplicitH(model, graph) {
    const out = runDMPNNFor(model, graph);
    const values = out.atomEmbeddings.map(function (embedding) { return applyHead(model, embedding); });
    return graph.hNodesByHeavyIndex.map(function (hIndices) {
      if (hIndices.length === 0) return NaN;
      const sum = hIndices.reduce(function (s, hIdx) { return s + values[hIdx]; }, 0);
      return sum / hIndices.length;
    });
  }

  // Builds (and lazily caches) whichever graph(s) the currently-loaded
  // models actually need -- most predictions only ever need the "heavy"
  // graph, so the (pricier) explicit-H graph is only built if some
  // loaded model's graphType asks for it.
  function buildGraphsForMolecule(molecule) {
    const molblock = CC.moleculeToMolblock(molecule);
    const annotations = CC.GNN.getRDKitAnnotations(molblock);
    let heavy = null;
    let explicitH = null;
    return {
      forGraphType: function (graphType) {
        if (graphType === 'explicit-h') {
          if (!explicitH) explicitH = CC.GNN.buildMolGraphChempropExplicitH(molecule, annotations);
          return explicitH;
        }
        if (!heavy) heavy = CC.GNN.buildMolGraphChemprop(molecule, annotations);
        return heavy;
      },
    };
  }

  // Zeroes out (leaves absent from) any atomProperties entry whose heavy
  // atom doesn't match model.applicableElement, e.g. a 13C shift model
  // should only ever annotate carbon atoms -- same pattern
  // pka-model.js's candidate-site gating uses. No-op if the model didn't
  // declare an applicableElement.
  //
  // Doesn't apply to an "explicit-h" model: there applicableElement='H'
  // documents which underlying atoms the model targets, not a literal
  // heavy-atom element filter -- hydrogen is never a heavy atom in this
  // app's molecule representation, so that comparison would always be
  // false and silently drop every prediction. runOneAtomLevelExplicitH()
  // already returns NaN for any heavy atom with no attached H's, and the
  // NaN check in the callers below does that filtering instead.
  function elementMask(molecule, model) {
    if (!model.applicableElement || model.graphType === 'explicit-h') return null;
    const heavyAtoms = Array.from(molecule.atoms.values());
    return heavyAtoms.map(function (a) { return a.element === model.applicableElement; });
  }

  function runAtomLevelModel(molecule, model, graphs) {
    const graph = graphs.forGraphType(model.graphType);
    const values = model.graphType === 'explicit-h' ? runOneAtomLevelExplicitH(model, graph) : runOneAtomLevel(model, graph);
    const mask = elementMask(molecule, model);
    return { values: values, atomIds: graph.atomIds, mask: mask };
  }

  /**
   * Run one specific loaded model on a molecule.
   * Molecule-level: { molecularProperties: { [task]: number }, propertyMeta, atomIds, backend: 'chemprop', modelId }.
   * Atom-level:      { atomProperties: [{ [task]: number }, ...] (one per atom, matching atomIds), atomIds, backend: 'chemprop', modelId }.
   */
  CC.GNN.predictChemprop = function (molecule, id) {
    const model = models.get(id);
    if (!model) throw new Error('No Chemprop model loaded under id "' + id + '"');

    const graphs = buildGraphsForMolecule(molecule);

    if (molecule.atoms.size === 0) {
      return model.outputLevel === 'atom'
        ? { atomProperties: [], atomIds: [], backend: 'chemprop', modelId: id }
        : { molecularProperties: {}, propertyMeta: {}, atomIds: [], backend: 'chemprop', modelId: id };
    }

    if (model.outputLevel === 'atom') {
      const result = runAtomLevelModel(molecule, model, graphs);
      const atomProperties = result.values.map(function (v, i) {
        const obj = {};
        if ((!result.mask || result.mask[i]) && v === v) obj[model.task] = v; // v === v excludes NaN (no attached H's)
        return obj;
      });
      return { atomProperties: atomProperties, atomIds: result.atomIds, backend: 'chemprop', modelId: id };
    }

    const graph = graphs.forGraphType('heavy');
    const molecularProperties = {};
    const propertyMeta = {};
    molecularProperties[model.task] = runOneMolecule(model, graph);
    propertyMeta[model.task] = { taskType: model.taskType, modelId: model.id };
    return { molecularProperties: molecularProperties, propertyMeta: propertyMeta, atomIds: graph.atomIds, backend: 'chemprop', modelId: id };
  };

  /**
   * Run every currently-loaded model on a molecule and merge their
   * outputs into one result. The (only somewhat expensive) featurization
   * + graph build happens once per graph TYPE and is shared across every
   * model needing that type (most models share the "heavy" graph; only
   * an explicit-H-trained model like the 1H NMR checkpoint needs the
   * pricier explicit-H graph, built lazily and only if actually needed).
   *
   * If two loaded models happen to share the same task/property name,
   * the later one (in load order) wins that key — give models distinct
   * task names to avoid this in practice.
   *
   * Returns { molecularProperties, propertyMeta, atomProperties, atomIds, backend: 'chemprop' } --
   * molecularProperties/propertyMeta cover any molecule-level models loaded,
   * atomProperties (one object per atom, matching atomIds) covers any
   * atom-level models loaded, merged together if both kinds are loaded
   * at once (all atom-level models here share the SAME atomIds -- the
   * molecule's own heavy atoms -- regardless of which graph type each
   * individual model's own forward pass used internally). propertyMeta
   * lets callers (see app.js's renderGNNOutput) render a classification
   * model's output as Positive/Negative + score rather than a plain
   * regression number.
   */
  CC.GNN.predictAllChempropModels = function (molecule) {
    if (models.size === 0) throw new Error('No Chemprop models loaded');

    if (molecule.atoms.size === 0) {
      return { molecularProperties: {}, propertyMeta: {}, atomProperties: [], atomIds: [], backend: 'chemprop' };
    }

    const graphs = buildGraphsForMolecule(molecule);
    const heavyAtomIds = Array.from(molecule.atoms.values()).map(function (a) { return a.id; });

    const molecularProperties = {};
    const propertyMeta = {};
    let atomProperties = null; // built lazily -- most predictions won't have any atom-level models loaded

    models.forEach(function (model) {
      if (model.outputLevel === 'atom') {
        const result = runAtomLevelModel(molecule, model, graphs);
        if (!atomProperties) atomProperties = heavyAtomIds.map(function () { return {}; });
        result.values.forEach(function (v, i) {
          if ((!result.mask || result.mask[i]) && v === v) atomProperties[i][model.task] = v;
        });
      } else {
        const graph = graphs.forGraphType('heavy');
        molecularProperties[model.task] = runOneMolecule(model, graph);
        propertyMeta[model.task] = { taskType: model.taskType, modelId: model.id };
      }
    });

    return {
      molecularProperties: molecularProperties,
      propertyMeta: propertyMeta,
      atomProperties: atomProperties || [],
      atomIds: heavyAtomIds,
      backend: 'chemprop',
    };
  };
})();
