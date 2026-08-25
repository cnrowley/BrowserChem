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
 * Supports the output heads Chemprop's RegressionFFN/BinaryClassificationFFN/
 * MveFFN produce (manifest.taskType tells this file which): regression
 * applies the checkpoint's UnscaleTransform (raw * scale + mean);
 * classification applies a plain sigmoid, matching
 * BinaryClassificationFFN.forward()'s `Y.sigmoid()` exactly (see chemprop's
 * nn/predictors.py) — the probability of the positive class, not a raw
 * logit; "regression-mve" (Mean-Variance Estimation, molecule-level only —
 * see convert_chemprop_checkpoint.py) applies the same UnscaleTransform to
 * a mean channel plus a softplus'd, scale^2-scaled variance channel
 * (MveFFN.forward()'s exact math), returning real per-prediction
 * (aleatoric) uncertainty alongside the value rather than a bare number —
 * see applyHead() and runOneMolecule().
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
 * A third level, outputLevel === "bond" (e.g. bond dissociation
 * enthalpy — see BDE_INTEGRATION.md), reads the per-directed-edge
 * embedding dmpnn.js's edge_finalize step produces (needs the
 * checkpoint's extra W_eo/W_eo_bias tensors, absent from every
 * molecule-/atom-level manifest). Confirmed directly from chemprop's own
 * installed source (chemprop/models/mol_atom_bond.py): the bond
 * predictor's input is concat([H_e[edge], H_e[reverse_edge]]), and the
 * final per-bond value averages the FFN's output over both directions,
 * `(pred[forward] + pred[backward]) / 2` — see runOneBondLevel().
 *
 * Why not ONNX? See gnn-inference.js's header: exporting a D-MPNN's
 * dynamic scatter/gather ops to ONNX and running them in onnxruntime-web
 * is genuinely fragile. This app's D-MPNN is small (hidden~300, depth~3)
 * and molecules drawn by hand are tiny (tens of atoms), so a plain JS
 * forward pass is both simpler and fast enough — this is the same
 * strategy dmpnn.js's "demo" backend already uses, just with real weights.
 *
 * Expected checkpoint shape (what the conversion script must produce):
 *   manifest.taskType = "regression" | "classification" | "regression-mve"
 *                    ("regression-mve" is molecule-level only)
 *   manifest.outputLevel = "molecule" | "atom" | "bond" (default "molecule")
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
 *   manifest.tensors.{W_eo_weight, W_eo_bias} = { shape, offset, length }
 *                      — bond-level (outputLevel === "bond") only; this
 *                        is the extra edge_finalize projection plain
 *                        molecule-/atom-level checkpoints never needed.
 *                        ffn0_weight's input dim is 2*d_h for a
 *                        bond-level checkpoint (concat of an edge's own
 *                        embedding with its reverse edge's), vs d_h for
 *                        atom/molecule.
 *   manifest.tensors.{out_mean, out_scale} = { shape, offset, length }
 *                      — regression and regression-mve only; absent for
 *                        classification, since BinaryClassificationFFN's
 *                        output_transform is Identity() (nothing to
 *                        unscale). regression-mve's variance channel
 *                        reuses this same out_scale (squared, no mean
 *                        shift) rather than needing its own tensor.
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
      // A bond-level (BDE) checkpoint has no atom-output layer at all --
      // Chemprop never builds message_passing.W_vo without a mol/atom
      // predictor attached -- so this manifest simply has no W_o_weight
      // tensor, and dmpnn.js treats Wo/WoBias as optional accordingly.
      Wo: manifest.outputLevel === 'bond' ? null : toRows(tensor('W_o_weight'), tensorShape('W_o_weight')),
      WoBias: manifest.outputLevel === 'bond' ? null : Array.from(tensor('W_o_bias')),
      ffn0: toRows(tensor('ffn0_weight'), tensorShape('ffn0_weight')),
      ffn0Bias: Array.from(tensor('ffn0_bias')),
      ffn1: toRows(tensor('ffn1_weight'), tensorShape('ffn1_weight')),
      ffn1Bias: Array.from(tensor('ffn1_bias')),
    };
    if (model.taskType === 'regression' || model.taskType === 'regression-mve') {
      model.outMean = tensor('out_mean')[0];
      model.outScale = tensor('out_scale')[0];
    }
    if (model.outputLevel === 'bond') {
      model.Weo = toRows(tensor('W_eo_weight'), tensorShape('W_eo_weight'));
      model.WeoBias = Array.from(tensor('W_eo_bias'));
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

  // Numerically-stable softplus: log(1+exp(x)), but computed as
  // x + log1p(exp(-x)) for large x to avoid overflowing exp(x) — standard
  // trick, same one PyTorch's own F.softplus uses internally.
  function softplus(x) {
    return x > 20 ? x : Math.log1p(Math.exp(x));
  }

  // Applies ffn0 -> ReLU -> ffn1 -> task-appropriate output head to a
  // single embedding vector (either the pooled molecule vector, or one
  // atom's own embedding — this part of the math doesn't care which).
  // Returns a plain number for 'classification'/'regression', or
  // { value, uncertainty } for 'regression-mve' — callers that can
  // receive an MVE model (currently only runOneMolecule) must check
  // model.taskType before unwrapping; the atom-/bond-level call sites
  // never see 'regression-mve' (convert_chemprop_checkpoint.py scopes
  // MveFFN export to molecule-level checkpoints only).
  function applyHead(model, embedding) {
    const hidden = relu(matVecBias(model.ffn0, model.ffn0Bias, embedding));
    const raw = matVecBias(model.ffn1, model.ffn1Bias, hidden);
    if (model.taskType === 'classification') {
      return sigmoid(raw[0]); // BinaryClassificationFFN.forward(): Y.sigmoid() -- a probability, not a logit
    }
    if (model.taskType === 'regression-mve') {
      // MveFFN.forward() (chemprop's nn/predictors.py): Y = ffn(Z), split
      // into (mean, var); var = softplus(var); mean = output_transform
      // (mean); var = output_transform.transform_variance(var), which for
      // the linear UnscaleTransform this project uses is var * scale^2
      // (confirmed from chemprop's own UnscaleTransform.transform_variance
      // source — a pure squared-scale, no mean shift, since variance has
      // no additive offset). Returns a standard-deviation-style number
      // (sqrt of the unscaled variance) rather than raw variance — more
      // directly interpretable next to the predicted value in the UI.
      const mean = raw[0] * model.outScale + model.outMean;
      const variance = softplus(raw[1]) * model.outScale * model.outScale;
      return { value: mean, uncertainty: Math.sqrt(variance) };
    }
    return raw[0] * model.outScale + model.outMean;
  }

  function runDMPNNFor(model, graph) {
    const dmpnnWeights = {
      hiddenSize: model.dims.d_h,
      Wi: model.Wi, WiBias: model.WiBias,
      Wh: model.Wh, WhBias: model.WhBias,
      Wo: model.Wo, WoBias: model.WoBias,
      Weo: model.Weo, WeoBias: model.WeoBias,
    };
    return CC.GNN.runDMPNN(graph, dmpnnWeights, { depth: model.dims.depth });
  }

  // Molecule-level: NormAggregation (sum of atom embeddings / aggNorm),
  // then one FFN application on the pooled vector. Returns the pooled
  // embedding alongside the value -- it's the same 300-dim vector
  // CC.AD.tierForEmbedding compares against a model's training-set
  // centroids (see applicability-domain.js), already computed for free
  // here, not a second forward pass.
  function runOneMolecule(model, graph) {
    const out = runDMPNNFor(model, graph);
    const pooled = CC.GNN.poolSum(out.atomEmbeddings, model.dims.d_h)
      .map(function (x) { return x / model.dims.aggNorm; });
    const head = applyHead(model, pooled);
    // regression-mve's applyHead returns { value, uncertainty }; every
    // other taskType returns a plain number -- unwrap here so `value`
    // stays a plain number everywhere else, and `uncertainty` (undefined
    // for non-MVE models) rides alongside it for the caller to fold into
    // propertyMeta next to the existing applicability-domain confidence.
    if (model.taskType === 'regression-mve') {
      return { value: head.value, uncertainty: head.uncertainty, pooled: pooled };
    }
    return { value: head, pooled: pooled };
  }

  // Per-model (not just per-engine) training-vocabulary gate -- see
  // applicability-domain.js's header. Checked here, at actual prediction
  // time, not just at auto-load time (app.js's autoLoadApplicableModels):
  // a model loaded while a PREVIOUS, compatible molecule was on the
  // canvas stays loaded, so the load-time gate alone can't stop it from
  // silently running again on a newly-drawn, out-of-vocabulary molecule.
  // Returns null if fine, or a reason string if this model should be
  // skipped entirely for this molecule.
  function blockedReason(model, molecule) {
    if (!window.CC.AD || !CC.AD.checkVocab) return null;
    const check = CC.AD.checkVocab(molecule, model.id);
    if (check.compatible) return null;
    return 'outside its training vocabulary (' + check.issues.join('; ') + ')';
  }

  // QM9's B3LYP/6-31G(2df,p)-computed targets (isotropic polarizability,
  // HOMO-LUMO gap): QM9's training distribution is <=9-heavy-atom C/H/O/N/F
  // molecules, so virtually every real-world drug-like molecule reads as
  // 'out-of-domain' -- a true but uninformative badge on every single
  // prediction. Suppressed per explicit user request, consistent with this
  // app's existing policy of only surfacing applicability-domain confidence
  // for models trained on experimental (not DFT/QM-computed) data -- see
  // CLAUDE.md and model/registry.json's per-model notes.
  const CONFIDENCE_BADGE_EXCLUDED_MODELS = new Set(['qm9-polarizability', 'qm9-homo-lumo-gap']);

  function confidenceMeta(model, pooled) {
    if (!window.CC.AD || !CC.AD.tierForEmbedding) return undefined;
    if (CONFIDENCE_BADGE_EXCLUDED_MODELS.has(model.id)) return undefined;
    const tier = CC.AD.tierForEmbedding(model.id, pooled);
    return tier.tier === 'unknown' ? undefined : tier;
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

  // Bond-level (e.g. BDE), "explicit-h" graph only (a bond-level
  // checkpoint always needs the real per-hydrogen graph -- C-H bonds are
  // most of what's chemically interesting here, e.g. finding the weakest
  // C-H in a molecule). graph.edgeSrc/edgeFeatures store each canonical
  // (undirected) bond as a consecutive forward/backward pair -- edges 2k
  // and 2k+1 -- exactly matching Chemprop's own BatchMolGraph edge
  // ordering, so this replicates its bond predictor precisely (confirmed
  // from chemprop/models/mol_atom_bond.py): each direction's fingerprint
  // is concat([own edge embedding, reverse edge's embedding]), run
  // through the FFN head separately, then the two directions' outputs
  // are averaged. Returns one value per canonical bond (heavy-heavy AND
  // heavy-H), in graph.bondIds/hBondCanonicalIndexByHeavyIndex order --
  // callers slice the first graph.numRealBonds off for the visible
  // (heavy-heavy) bonds and use the rest for the per-atom aggregate.
  function runOneBondLevel(model, graph) {
    const out = runDMPNNFor(model, graph);
    const He = out.edgeEmbeddings;
    const numCanonicalBonds = He.length / 2;
    const values = new Array(numCanonicalBonds);
    for (let k = 0; k < numCanonicalBonds; k++) {
      const eFwd = 2 * k, eBack = 2 * k + 1;
      const predFwd = applyHead(model, He[eFwd].concat(He[eBack]));
      const predBack = applyHead(model, He[eBack].concat(He[eFwd]));
      values[k] = (predFwd + predBack) / 2;
    }
    return values;
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

  // Bond-level (e.g. BDE): always needs the explicit-H graph (a bond-
  // level checkpoint's whole point is per-bond values including C-H).
  // Returns the visible (heavy-heavy, drawn-on-canvas) bond values in
  // molecule.bonds order, plus a per-heavy-atom "weakest attached-H bond"
  // aggregate (min, not mean -- unlike the 1H shift aggregate, where
  // several equivalent protons genuinely share one true value, several
  // C-H's on the same atom generally do NOT share one true BDE, and the
  // chemically useful number here is "how easy is it to break the
  // weakest bond on this atom", not their average).
  // Display key for the per-atom "weakest attached-H bond" aggregate,
  // e.g. task "bde" -> "BDE-XH" (X-H, since it covers whichever element
  // that heavy atom's own attached hydrogens are bonded through -- C-H,
  // N-H, O-H, ... not just carbon).
  function weakestHKey(model) { return model.task.toUpperCase() + '-XH'; }

  function runBondLevelModel(molecule, model, graphs) {
    const graph = graphs.forGraphType('explicit-h');
    const values = runOneBondLevel(model, graph);
    const bondValues = values.slice(0, graph.numRealBonds);
    const atomAggregate = graph.hBondCanonicalIndexByHeavyIndex.map(function (canonicalIdxs) {
      if (canonicalIdxs.length === 0) return NaN;
      return canonicalIdxs.reduce(function (min, idx) { return Math.min(min, values[idx]); }, Infinity);
    });
    return { bondValues: bondValues, bondIds: graph.bondIds, atomAggregate: atomAggregate, atomIds: graph.atomIds };
  }

  /**
   * Run one specific loaded model on a molecule.
   * Molecule-level: { molecularProperties: { [task]: number }, propertyMeta, atomIds, backend: 'chemprop', modelId }.
   * Atom-level:      { atomProperties: [{ [task]: number }, ...] (one per atom, matching atomIds), atomIds, backend: 'chemprop', modelId }.
   * Bond-level:      { bondProperties: [{ [task]: number }, ...] (one per bond, matching bondIds), bondIds,
   *                    atomProperties (the per-atom "weakest attached-H bond" aggregate, key `TASK + '-XH'`, e.g. "BDE-XH"), atomIds,
   *                    backend: 'chemprop', modelId }.
   */
  CC.GNN.predictChemprop = function (molecule, id) {
    const model = models.get(id);
    if (!model) throw new Error('No Chemprop model loaded under id "' + id + '"');
    const blocked = blockedReason(model, molecule);
    if (blocked) throw new Error('"' + id + '" refused to run: ' + blocked);

    const graphs = buildGraphsForMolecule(molecule);

    if (molecule.atoms.size === 0) {
      if (model.outputLevel === 'atom') return { atomProperties: [], atomIds: [], backend: 'chemprop', modelId: id };
      if (model.outputLevel === 'bond') return { bondProperties: [], bondIds: [], atomProperties: [], atomIds: [], backend: 'chemprop', modelId: id };
      return { molecularProperties: {}, propertyMeta: {}, atomIds: [], backend: 'chemprop', modelId: id };
    }

    if (model.outputLevel === 'bond') {
      const result = runBondLevelModel(molecule, model, graphs);
      const bondProperties = result.bondValues.map(function (v) {
        const obj = {};
        obj[model.task] = v;
        return obj;
      });
      const atomProperties = result.atomAggregate.map(function (v) {
        const obj = {};
        if (v === v && v !== Infinity) obj[weakestHKey(model)] = v; // v === v excludes NaN (no attached H's)
        return obj;
      });
      return { bondProperties: bondProperties, bondIds: result.bondIds, atomProperties: atomProperties, atomIds: result.atomIds, backend: 'chemprop', modelId: id };
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
    const out = runOneMolecule(model, graph);
    molecularProperties[model.task] = out.value;
    propertyMeta[model.task] = { taskType: model.taskType, modelId: model.id, confidence: confidenceMeta(model, out.pooled), uncertainty: out.uncertainty };
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
   * Returns { molecularProperties, propertyMeta, atomProperties, atomIds, bondProperties, bondIds, backend: 'chemprop' } --
   * molecularProperties/propertyMeta cover any molecule-level models loaded,
   * atomProperties (one object per atom, matching atomIds) covers any
   * atom-level models loaded, merged together if both kinds are loaded
   * at once (all atom-level models here share the SAME atomIds -- the
   * molecule's own heavy atoms -- regardless of which graph type each
   * individual model's own forward pass used internally). propertyMeta
   * lets callers (see app.js's renderGNNOutput) render a classification
   * model's output as Positive/Negative + score rather than a plain
   * regression number. bondProperties (one object per bond, matching
   * bondIds -- the molecule's own drawn bonds) covers any bond-level
   * models loaded (e.g. BDE); those also contribute a
   * `TASK + '-XH'` (e.g. "BDE-XH") entry into the SAME atomProperties as any
   * atom-level model would (the per-atom weakest-attached-H-bond
   * aggregate -- see runBondLevelModel()).
   */
  CC.GNN.predictAllChempropModels = function (molecule) {
    if (models.size === 0) throw new Error('No Chemprop models loaded');

    if (molecule.atoms.size === 0) {
      return { molecularProperties: {}, propertyMeta: {}, atomProperties: [], atomIds: [], bondProperties: [], bondIds: [], backend: 'chemprop' };
    }

    const graphs = buildGraphsForMolecule(molecule);
    const heavyAtomIds = Array.from(molecule.atoms.values()).map(function (a) { return a.id; });

    const molecularProperties = {};
    const propertyMeta = {};
    const warnings = [];
    let atomProperties = null; // built lazily -- most predictions won't have any atom-level (or bond-level) models loaded
    let bondProperties = null;
    let bondIds = [];

    models.forEach(function (model) {
      const blocked = blockedReason(model, molecule);
      if (blocked) {
        warnings.push('"' + model.id + '" skipped: ' + blocked);
        return;
      }
      if (model.outputLevel === 'bond') {
        const result = runBondLevelModel(molecule, model, graphs);
        if (!bondProperties) bondProperties = result.bondValues.map(function () { return {}; });
        result.bondValues.forEach(function (v, i) { bondProperties[i][model.task] = v; });
        bondIds = result.bondIds;
        if (!atomProperties) atomProperties = heavyAtomIds.map(function () { return {}; });
        result.atomAggregate.forEach(function (v, i) {
          if (v === v && v !== Infinity) atomProperties[i][weakestHKey(model)] = v;
        });
      } else if (model.outputLevel === 'atom') {
        const result = runAtomLevelModel(molecule, model, graphs);
        if (!atomProperties) atomProperties = heavyAtomIds.map(function () { return {}; });
        result.values.forEach(function (v, i) {
          if ((!result.mask || result.mask[i]) && v === v) atomProperties[i][model.task] = v;
        });
      } else {
        const graph = graphs.forGraphType('heavy');
        const out = runOneMolecule(model, graph);
        molecularProperties[model.task] = out.value;
        propertyMeta[model.task] = { taskType: model.taskType, modelId: model.id, confidence: confidenceMeta(model, out.pooled), uncertainty: out.uncertainty };
      }
    });

    return {
      molecularProperties: molecularProperties,
      propertyMeta: propertyMeta,
      atomProperties: atomProperties || [],
      atomIds: heavyAtomIds,
      bondProperties: bondProperties || [],
      bondIds: bondIds,
      backend: 'chemprop',
      warnings: warnings,
    };
  };

  // See model-adapters.js's header. `predict` here is
  // predictAllChempropModels itself (molecule) -> {...}, not
  // predict(molecule, id) -- it already aggregates across every
  // currently-loaded chemprop checkpoint in one call, a genuinely
  // different shape from the single-model-id adapters below, and
  // gnn-inference.js's predictMolecule treats it as its own first step
  // rather than folding it into the generic per-id loop for that reason.
  CC.ModelAdapters.register('chemprop', {
    kind: 'property',
    load: CC.GNN.loadChempropModel,
    unload: CC.GNN.clearChempropModel,
    hasModel: CC.GNN.hasChempropModel,
    getLoadedModelIds: CC.GNN.getLoadedChempropModelIds,
    validate: CC.GNN.checkChempropCompatibility,
    predict: CC.GNN.predictAllChempropModels,
  });
})();
