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
 *      the path that gives real, numerically-correct predictions.
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
    const hasChemprop = CC.GNN.hasChempropModel();
    const hasNagl = window.CC.NAGL && CC.NAGL.hasModel && CC.NAGL.hasModel();

    if (hasChemprop || hasNagl) {
      const merged = { molecularProperties: {}, propertyMeta: {}, atomProperties: [], atomIds: [], backend: 'chemprop', warnings: [] };

      if (hasChemprop) {
        try {
          const cpResult = CC.GNN.predictAllChempropModels(molecule);
          Object.assign(merged.molecularProperties, cpResult.molecularProperties);
          Object.assign(merged.propertyMeta, cpResult.propertyMeta);
          if (cpResult.atomIds.length > 0) merged.atomIds = cpResult.atomIds;
          if (cpResult.atomProperties.length > 0) {
            merged.atomProperties = cpResult.atomProperties.map(function (p) { return Object.assign({}, p); });
          }
        } catch (err) {
          merged.warnings.push('Chemprop models: ' + err.message);
        }
      }

      if (hasNagl) {
        CC.NAGL.getLoadedModelIds().forEach(function (id) {
          try {
            const naglResult = CC.NAGL.predict(molecule, id);
            if (merged.atomIds.length === 0) merged.atomIds = naglResult.atomIds;
            if (merged.atomProperties.length === 0) {
              merged.atomProperties = naglResult.atomProperties.map(function (p) { return Object.assign({}, p); });
            } else {
              // Same atom ordering either engine produces (both iterate
              // molecule.atoms.values() directly with no reordering), so
              // this merges index-by-index rather than needing to
              // re-match by atom id.
              naglResult.atomProperties.forEach(function (p, i) { Object.assign(merged.atomProperties[i], p); });
            }
          } catch (err) {
            // One incompatible NAGL model (e.g. an element outside its
            // vocabulary -- see CC.NAGL.checkCompatibility) shouldn't
            // sink predictions from everything else that's loaded.
            merged.warnings.push('NAGL model "' + id + '": ' + err.message);
          }
        });
      }

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
