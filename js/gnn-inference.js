/**
 * gnn-inference.js
 *
 * Exposes the API the spec asked for:
 *   CC.GNN.predictMolecule(molecule) -> { molecularProperties, atomProperties }
 *   CC.GNN.predictAtoms(molecule)    -> atomProperties only
 *
 * Three backends, tried in this order:
 *
 *   1. "chemprop" — one or more real trained Chemprop checkpoints loaded
 *      via model-registry.js (the normal path) or directly via
 *      CC.GNN.loadChempropModel()/loadChempropModelFromBuffers()
 *      (chemprop-model.js). Every currently-loaded model runs and their
 *      outputs are merged. Runs entirely client-side with the bit-exact
 *      Chemprop featurizers (chemprop-features.js) and the fixed D-MPNN
 *      forward pass in dmpnn.js — no server, no ONNX Runtime. This is
 *      the path that gives real, numerically-correct predictions. Any
 *      loaded NAGL-MBIS charge model(s) (nagl-model.js) and pKa model(s)
 *      (pka-model.js) merge their atom-level results into the same
 *      atomProperties array here too -- a pKa model additionally needs
 *      its own required NAGL charge model loaded first (see
 *      pka-model.js's header), surfaced as a warning rather than an
 *      exception if it isn't.
 *
 *   2. "onnx" — an ONNX model loaded via CC.GNN.loadOnnxModel(), for
 *      anything exported that way instead. See buildOnnxTensors() below
 *      for the (unverified, PyG-convention) tensor layout this assumes.
 *
 *   3. "demo" — falls back to the hand-rolled D-MPNN in dmpnn.js with
 *      cached random weights when neither of the above is loaded.
 *      Architecturally real, numbers are meaningless (see dmpnn.js
 *      header) — proves the full pipeline (features -> graph -> message
 *      passing -> pooling -> heads -> UI) is wired up correctly even
 *      with zero trained models on hand.
 *
 * ON EXPORTING A TRAINED D-MPNN TO ONNX YOURSELF — a real caveat, not a
 * formality: D-MPNN message passing uses dynamic, variable-length
 * scatter/gather operations (each molecule has a different number of
 * atoms and bonds). That's exactly the kind of op that's historically
 * been painful to export cleanly to ONNX and run with fixed shapes.
 * There is no single "correct" input tensor layout this file can assume
 * — it uses the most common PyTorch Geometric convention (x, edge_index,
 * edge_attr, batch) as a default, but you will very likely need to
 * adjust buildOnnxTensors() below to match whatever your own export
 * actually produces. Inspect session.inputNames after loading to check.
 * This is exactly why the "chemprop" backend above exists as a
 * plain-JS alternative that sidesteps ONNX export entirely.
 */

window.CC = window.CC || {};
CC.GNN = window.CC.GNN || {};

(function () {
  const HIDDEN_SIZE = 32;
  const DEPTH = 3;
  const MOLECULAR_PROPERTIES = ['logP_demo', 'molWeight_demo', 'solubility_demo'];
  const ATOMIC_PROPERTIES = ['partialCharge_demo', 'reactivity_demo'];

  let demoWeightsCache = null; // keyed implicitly: rebuilt if dims change
  let demoMolHead = null;
  let demoAtomHead = null;
  let onnxSession = null;

  function getDemoWeights(atomFeatureDim, bondFeatureDim) {
    const key = atomFeatureDim + '_' + bondFeatureDim + '_' + HIDDEN_SIZE;
    if (!demoWeightsCache || demoWeightsCache.key !== key) {
      demoWeightsCache = {
        key: key,
        weights: CC.GNN.createDemoWeights(atomFeatureDim, bondFeatureDim, HIDDEN_SIZE),
      };
      demoMolHead = CC.GNN.createHeadWeights(HIDDEN_SIZE, MOLECULAR_PROPERTIES);
      demoAtomHead = CC.GNN.createHeadWeights(HIDDEN_SIZE, ATOMIC_PROPERTIES);
    }
    return demoWeightsCache.weights;
  }

  function buildGraphForMolecule(molecule) {
    const molblock = CC.moleculeToMolblock(molecule);
    const annotations = CC.GNN.getRDKitAnnotations(molblock);
    return CC.GNN.buildMolGraph(molecule, annotations);
  }

  function predictDemo(molecule) {
    const graph = buildGraphForMolecule(molecule);
    if (graph.numAtoms === 0) {
      return { molecularProperties: {}, atomProperties: [], backend: 'demo', atomIds: [] };
    }
    const weights = getDemoWeights(graph.atomFeatureDim, graph.bondFeatureDim);
    const out = CC.GNN.runDMPNN(graph, weights, { depth: DEPTH });
    const pooled = CC.GNN.poolMean(out.atomEmbeddings, HIDDEN_SIZE);

    return {
      molecularProperties: CC.GNN.runMolecularHead(pooled, demoMolHead),
      atomProperties: CC.GNN.runAtomicHead(out.atomEmbeddings, demoAtomHead),
      atomIds: graph.atomIds,
      backend: 'demo',
    };
  }

  // ---------- ONNX backend ----------

  CC.GNN.loadOnnxModel = function (arrayBuffer) {
    const ort = window.chemCanvasLibs && window.chemCanvasLibs.ort;
    if (!ort) return Promise.reject(new Error('ONNX Runtime Web is not loaded'));
    return ort.InferenceSession.create(arrayBuffer).then(function (session) {
      onnxSession = session;
      return session;
    });
  };

  CC.GNN.hasOnnxModel = function () {
    return !!onnxSession;
  };

  CC.GNN.getOnnxModelInfo = function () {
    if (!onnxSession) return null;
    return { inputNames: onnxSession.inputNames, outputNames: onnxSession.outputNames };
  };

  function buildOnnxTensors(graph) {
    const ort = window.chemCanvasLibs.ort;
    const flatX = [];
    graph.atomFeatures.forEach(function (row) { flatX.push.apply(flatX, row); });
    const x = new ort.Tensor('float32', Float32Array.from(flatX), [graph.numAtoms, graph.atomFeatureDim]);

    const edgeIndexFlat = graph.edgeSrc.concat(graph.edgeDst); // PyG layout: [2, numEdges] row-major = [src..., dst...]
    const edgeIndex = new ort.Tensor('int64', BigInt64Array.from(edgeIndexFlat.map(BigInt)), [2, graph.edgeSrc.length]);

    const flatEdgeAttr = [];
    graph.edgeFeatures.forEach(function (row) { flatEdgeAttr.push.apply(flatEdgeAttr, row); });
    const edgeAttr = new ort.Tensor('float32', Float32Array.from(flatEdgeAttr), [graph.edgeSrc.length, graph.bondFeatureDim || 1]);

    const batch = new ort.Tensor('int64', BigInt64Array.from(new Array(graph.numAtoms).fill(0n)), [graph.numAtoms]);

    return { x: x, edge_index: edgeIndex, edge_attr: edgeAttr, batch: batch };
  }

  function predictOnnx(molecule) {
    const graph = buildGraphForMolecule(molecule);
    const feeds = buildOnnxTensors(graph);
    return onnxSession.run(feeds).then(function (results) {
      // Output naming/shape is entirely dependent on your export — this
      // just hands back the raw named tensors rather than guessing which
      // one is "molecular" vs "atomic".
      const raw = {};
      Object.keys(results).forEach(function (name) {
        raw[name] = Array.from(results[name].data);
      });
      return { raw: raw, atomIds: graph.atomIds, backend: 'onnx' };
    });
  }

  // ---------- public API ----------

  CC.GNN.predictMolecule = function (molecule) {
    const chempropAdapter = CC.ModelAdapters.get('chemprop');
    const hasChemprop = chempropAdapter.hasModel();

    // Every OTHER registered kind:'property' engine (nagl, pka, and any
    // future one) shares the exact same single-model-id predict(molecule,
    // id) shape, so they're merged with one loop instead of one
    // hand-written block per engine -- see model-adapters.js's header for
    // why chemprop stays separate (it aggregates across every loaded
    // checkpoint internally, in one call, with a richer result shape).
    const otherPropertyEngines = CC.ModelAdapters.list().filter(function (name) {
      if (name === 'chemprop') return false;
      const adapter = CC.ModelAdapters.get(name);
      return adapter.kind === 'property' && adapter.hasModel();
    });

    if (hasChemprop || otherPropertyEngines.length > 0) {
      const merged = { molecularProperties: {}, propertyMeta: {}, atomProperties: [], atomIds: [], bondProperties: [], bondIds: [], backend: 'chemprop', warnings: [] };

      if (hasChemprop) {
        try {
          const cpResult = chempropAdapter.predict(molecule);
          Object.assign(merged.molecularProperties, cpResult.molecularProperties);
          Object.assign(merged.propertyMeta, cpResult.propertyMeta);
          if (cpResult.atomIds.length > 0) merged.atomIds = cpResult.atomIds;
          if (cpResult.atomProperties.length > 0) {
            merged.atomProperties = cpResult.atomProperties.map(function (p) { return Object.assign({}, p); });
          }
          if (cpResult.bondIds && cpResult.bondIds.length > 0) merged.bondIds = cpResult.bondIds;
          if (cpResult.bondProperties && cpResult.bondProperties.length > 0) {
            merged.bondProperties = cpResult.bondProperties.map(function (p) { return Object.assign({}, p); });
          }
        } catch (err) {
          merged.warnings.push('Chemprop models: ' + err.message);
        }
      }

      otherPropertyEngines.forEach(function (engineName) {
        const adapter = CC.ModelAdapters.get(engineName);
        adapter.getLoadedModelIds().forEach(function (id) {
          try {
            const result = adapter.predict(molecule, id);
            if (merged.atomIds.length === 0) merged.atomIds = result.atomIds;
            if (merged.atomProperties.length === 0) {
              merged.atomProperties = result.atomProperties.map(function (p) { return Object.assign({}, p); });
            } else {
              // Same atom ordering every engine produces (all iterate
              // molecule.atoms.values() directly, no reordering), so
              // this merges index-by-index rather than needing to
              // re-match by atom id.
              result.atomProperties.forEach(function (p, i) { Object.assign(merged.atomProperties[i], p); });
            }
          } catch (err) {
            // One incompatible/misconfigured model (e.g. an element
            // outside NAGL's vocabulary, or a pKa model missing its
            // required NAGL charge model) shouldn't sink predictions
            // from everything else that's loaded.
            merged.warnings.push(engineName + ' model "' + id + '": ' + err.message);
          }
        });
      });

      return Promise.resolve(merged);
    }

    if (onnxSession) return predictOnnx(molecule);
    return Promise.resolve(predictDemo(molecule));
  };

  CC.GNN.predictAtoms = function (molecule) {
    return CC.GNN.predictMolecule(molecule).then(function (result) {
      return result.atomProperties || (result.raw && result.raw) || [];
    });
  };
})();
