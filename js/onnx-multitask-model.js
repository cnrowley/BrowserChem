/**
 * onnx-multitask-model.js
 *
 * Runs a genuine shared-encoder multi-task Chemprop D-MPNN (one CHEMELEON
 * fine-tune jointly predicting hERG-inhibition + 5 CYP450 inhibition
 * isoforms) via ONNX Runtime Web instead of a hand-rolled JS forward
 * pass. chemprop-model.js's own header explains why this project
 * normally avoids ONNX (D-MPNN's dynamic scatter/gather ops are fragile
 * to export, and this project's usual checkpoints are small enough --
 * hidden~300, depth~3 -- that plain JS is both simpler and fast enough).
 * CHEMELEON breaks that tradeoff: its encoder is d_h=2048/depth=6, ~9.3M
 * params, and a hand-rolled JS forward pass at that scale measured
 * 600-1300ms per molecule during this project's own benchmarking --
 * ONNX Runtime Web's WASM backend measured 15-40ms for the exact same
 * checkpoints (see scripts/export_onnx.py's docstring/commit history),
 * a real 20-30x win that justifies the ONNX path specifically here.
 *
 * A genuinely SHARED multi-task checkpoint (one encoder, N output
 * columns) can never go through scripts/convert_chemprop_checkpoint.py's
 * manifest.json/weights.bin format in the first place --
 * that converter explicitly refuses multi-task checkpoints
 * (`n_tasks != 1`), since chemprop-model.js's hand-rolled forward pass
 * and its whole per-model-id single-output convention were never built
 * to carry more than one output per loaded checkpoint. ONNX has no such
 * restriction, which is the other reason this model uses it.
 *
 * One physical .onnx file backs MULTIPLE registry ids (e.g.
 * herg-inhibition-v1, cyp1a2-inhibition-v1, ... all point at the same
 * weights file) -- each registry id gets its own tiny manifest.json
 * (see model/cyp-herg-chemeleon-multitask/*.json) naming which output
 * column ("onnxTargetTask") that id reads, plus the propertyKey-shaped
 * name ("task") the properties table already expects, matching
 * chemprop-model.js's own manifest.task convention exactly so nothing
 * downstream (CSV/XLSX/PDF export, radar chart) needs to know this
 * model is any different. Sessions are deduplicated by weights URL --
 * loading all 6 ids only fetches/parses the 37MB ONNX file once.
 *
 * Reuses the exact same featurizer as the JS D-MPNN path
 * (CC.GNN.buildMolGraphChemprop, from chemprop-features.js/
 * graph-builder.js) so predictions are computed from identical input
 * tensors either way -- verified bit-exact (diff ~5e-7, float32 noise)
 * against chemprop's own Python-side predictions using these same JS-
 * built tensors fed through onnxruntime-web, before this file existed.
 *
 * ort.InferenceSession.run() is inherently async (WASM), unlike every
 * other 'property'-kind adapter's predict() (nagl/pka/chemprop, all
 * synchronous JS math) -- see gnn-inference.js's predictMolecule, which
 * awaits every property adapter's result via a sequential promise chain
 * specifically so this adapter's async predict() merges in correctly
 * without breaking the deterministic base-array-then-merge order the
 * other synchronous adapters relied on.
 */

window.CC = window.CC || {};
CC.GNN = window.CC.GNN || {};

(function () {
  const sessions = new Map(); // weightsUrl -> Promise<ort.InferenceSession>
  const models = new Map();   // id -> { weightsUrl, taskIndex, task, taskType, dims }

  function loadSession(weightsUrl) {
    if (!sessions.has(weightsUrl)) {
      const ort = window.chemCanvasLibs && window.chemCanvasLibs.ort;
      if (!ort) {
        sessions.set(weightsUrl, Promise.reject(new Error('ONNX Runtime Web is not loaded')));
      } else {
        sessions.set(weightsUrl, fetch(weightsUrl).then(function (r) {
          if (!r.ok) throw new Error('failed to fetch ONNX weights: ' + r.status);
          return r.arrayBuffer();
        }).then(function (buf) {
          return ort.InferenceSession.create(new Uint8Array(buf), { executionProviders: ['wasm'] });
        }));
      }
    }
    return sessions.get(weightsUrl);
  }

  /**
   * Fetch this id's small manifest.json (names which ONNX output column
   * it reads) and the shared .onnx weights (reused across every id that
   * points at the same weightsUrl). Returns a Promise resolving to
   * { id, task, taskType } once ready.
   */
  CC.GNN.loadOnnxMultitaskModel = function (id, manifestUrl, weightsUrl) {
    return fetch(manifestUrl).then(function (r) {
      if (!r.ok) throw new Error('failed to fetch manifest: ' + r.status);
      return r.json();
    }).then(function (meta) {
      const taskIndex = (meta.onnxTaskNames || []).indexOf(meta.onnxTargetTask);
      if (taskIndex === -1) {
        throw new Error('manifest for "' + id + '" has onnxTargetTask ' + JSON.stringify(meta.onnxTargetTask) +
          ' not present in its own onnxTaskNames list');
      }
      return loadSession(weightsUrl).then(function () {
        models.set(id, {
          weightsUrl: weightsUrl,
          taskIndex: taskIndex,
          task: meta.task || meta.onnxTargetTask,
          taskType: meta.taskType || 'classification',
          dims: meta.dims,
        });
        return { id: id, task: meta.task, taskType: meta.taskType, dims: meta.dims };
      });
    });
  };

  CC.GNN.clearOnnxMultitaskModel = function (id) {
    const m = models.get(id);
    models.delete(id);
    if (m) {
      const stillReferenced = Array.from(models.values()).some(function (o) { return o.weightsUrl === m.weightsUrl; });
      if (!stillReferenced) sessions.delete(m.weightsUrl);
    }
  };

  CC.GNN.hasOnnxMultitaskModel = function () { return models.size > 0; };
  CC.GNN.getLoadedOnnxMultitaskModelIds = function () { return Array.from(models.keys()); };

  function buildTensors(graph) {
    const ort = window.chemCanvasLibs.ort;
    const flatV = [];
    graph.atomFeatures.forEach(function (row) { flatV.push.apply(flatV, row); });
    const flatE = [];
    graph.edgeFeatures.forEach(function (row) { flatE.push.apply(flatE, row); });
    const edgeIndexFlat = graph.edgeSrc.concat(graph.edgeDst);
    return {
      V: new ort.Tensor('float32', Float32Array.from(flatV), [graph.numAtoms, graph.atomFeatureDim]),
      E: new ort.Tensor('float32', Float32Array.from(flatE), [graph.edgeSrc.length, graph.bondFeatureDim]),
      edge_index: new ort.Tensor('int64', BigInt64Array.from(edgeIndexFlat.map(BigInt)), [2, graph.edgeSrc.length]),
      rev_edge_index: new ort.Tensor('int64', BigInt64Array.from(graph.revEdge.map(BigInt)), [graph.edgeSrc.length]),
      batch: new ort.Tensor('int64', BigInt64Array.from(new Array(graph.numAtoms).fill(0n)), [graph.numAtoms]),
    };
  }

  /**
   * Molecule-level: { molecularProperties: { [task]: number }, propertyMeta,
   * atomIds, backend: 'onnx-multitask', modelId }, matching
   * chemprop-model.js's runOneMolecule shape so gnn-inference.js's merge
   * logic doesn't need to special-case this engine. No confidence badge
   * (CC.AD.tierForEmbedding needs a JS-side pooled embedding this engine
   * doesn't compute; omitting applicabilityDomain from these registry
   * entries already makes that degrade to "no badge" rather than a wrong
   * one -- see model/registry.json's notes on these ids).
   */
  CC.GNN.predictOnnxMultitaskModel = function (molecule, id) {
    const model = models.get(id);
    if (!model) return Promise.reject(new Error('No onnx-multitask model loaded for id "' + id + '"'));

    const molblock = CC.moleculeToMolblock(molecule);
    const annotations = CC.GNN.getRDKitAnnotations(molblock);
    const graph = CC.GNN.buildMolGraphChemprop(molecule, annotations);
    if (graph.numAtoms === 0) {
      return Promise.resolve({ molecularProperties: {}, propertyMeta: {}, atomIds: [], backend: 'onnx-multitask', modelId: id });
    }

    return loadSession(model.weightsUrl).then(function (session) {
      return session.run(buildTensors(graph));
    }).then(function (results) {
      const value = results.output.data[model.taskIndex];
      const molecularProperties = {};
      molecularProperties[model.task] = value;
      const propertyMeta = {};
      propertyMeta[model.task] = { taskType: model.taskType, modelId: id, confidence: undefined, uncertainty: undefined };
      return {
        molecularProperties: molecularProperties, propertyMeta: propertyMeta,
        atomIds: graph.atomIds, backend: 'onnx-multitask', modelId: id,
      };
    });
  };

  CC.ModelAdapters.register('onnx-multitask', {
    kind: 'property',
    load: CC.GNN.loadOnnxMultitaskModel,
    unload: CC.GNN.clearOnnxMultitaskModel,
    hasModel: CC.GNN.hasOnnxMultitaskModel,
    getLoadedModelIds: CC.GNN.getLoadedOnnxMultitaskModelIds,
    validate: CC.GNN.checkChempropCompatibility, // same featurizer/vocabulary as the JS D-MPNN path
    predict: CC.GNN.predictOnnxMultitaskModel,
  });
})();
