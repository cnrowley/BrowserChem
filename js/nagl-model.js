/**
 * nagl-model.js
 *
 * Runs a NAGL-MBIS charge model (jthorton/nagl-mbis, e.g. the
 * "nagl-v1-mbis" checkpoint) fully client-side. A genuinely different
 * architecture from this project's Chemprop models (see
 * chemprop-model.js) -- a 5-layer GraphSAGE stack (DGL's
 * dgl.nn.pytorch.SAGEConv, 'mean' aggregator) over nagl-features.js's
 * 19-dim atom-only features (no bond features enter the convolution at
 * all), followed by a small per-atom FFN readout, followed by a
 * closed-form electronegativity-equalization step that converts the
 * FFN's raw (electronegativity, hardness) output pair into final
 * partial charges.
 *
 * Every piece of this was pulled from NAGL's and nagl-mbis's real
 * source (SimonBoothroyd/nagl, jthorton/nagl-mbis), not inferred from
 * general GNN knowledge -- see nagl-features.js's header for the same
 * note about the featurizer. In particular:
 *
 *  - SAGEConv 'mean' formula (per layer, per node i):
 *      h_self  = W_self  . x_i + b_self
 *      h_neigh = W_neigh . mean_{j in N(i)}(x_j)        (no bias -- DGL's
 *                                                         own default for
 *                                                         the neighbor
 *                                                         transform)
 *      h_i'    = ReLU(h_self + h_neigh)
 *    Confirmed against a real checkpoint's state_dict: fc_self has both
 *    weight and bias, fc_neigh has weight only -- exactly this shape.
 *
 *  - Readout: a plain per-atom MLP (128 -> 64 -> 64 -> 2, ReLU/ReLU/
 *    Identity for this checkpoint), no pooling at all (pooling: "atom"
 *    in the checkpoint's own config) -- same "apply the FFN directly to
 *    each atom's own embedding" shape as Chemprop's atom-level models,
 *    just a different-sized FFN and a 2-wide output instead of 1.
 *
 *  - Postprocess ("charges"): nagl's PartialChargeLayer
 *    (nagl/nn/postprocess.py) treats the readout's 2 outputs per atom as
 *    (electronegativity, hardness) and computes final charges via a
 *    closed-form electronegativity-equalization solve (Gilson, Gilson &
 *    Potter, J. Chem. Inf. Comput. Sci. 2003, 43, 6, 1982-1997) that
 *    analytically guarantees the charges sum to the molecule's total
 *    formal charge:
 *      invHardness_i = 1 / hardness_i
 *      charge_i = -invHardness_i * eneg_i
 *                 + invHardness_i * (sum_j invHardness_j*eneg_j + totalCharge)
 *                                   / sum_j invHardness_j
 *    This needs the molecule's total formal charge as an extra input,
 *    unlike a plain per-atom regression head.
 *
 * VALIDATION STATUS -- read before trusting this over the real model:
 * unlike every other model-execution file in this project, this one
 * could NOT be validated against real NAGL/DGL output directly. DGL is
 * pip-installable (unlike naglmbis's full conda-only environment), but
 * its precompiled C++ extension didn't have a build matching this
 * sandbox's PyTorch version, and downgrading risked destabilizing the
 * already-validated Chemprop pipeline sharing this environment. Every
 * formula above is transcribed directly from NAGL's and nagl-mbis's own
 * source with high confidence, and the state_dict shapes it was checked
 * against are consistent with it (right bias pattern, right dims) -- but
 * "consistent with" isn't "bit-exact confirmed" the way every Chemprop
 * model in this project was. Treat this as implemented-but-not-yet-
 * fully-validated until real reference values (see the accompanying
 * probe script) are diffed against it.
 */

window.CC = window.CC || {};
CC.NAGL = window.CC.NAGL || {};

(function () {
  const models = new Map(); // id -> parsed model

  function toRows(flat, shape) {
    const rows = shape[0];
    const cols = shape.length > 1 ? shape[1] : 1;
    const out = new Array(rows);
    for (let r = 0; r < rows; r++) out[r] = flat.subarray(r * cols, (r + 1) * cols);
    return out;
  }

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

  function addVec(a, b) { return a.map(function (v, i) { return v + b[i]; }); }
  function relu(v) { return v.map(function (x) { return Math.max(0, x); }); }

  CC.NAGL.loadModel = function (id, manifestUrl, binUrl) {
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
      return CC.NAGL.loadModelFromBuffers(id, results[0], results[1]);
    });
  };

  CC.NAGL.loadModelFromBuffers = function (id, manifest, binArrayBuffer) {
    if (!id) throw new Error('loadModelFromBuffers needs an id');
    const bytes = new Float32Array(binArrayBuffer);
    function tensor(name) {
      const t = manifest.tensors[name];
      if (!t) throw new Error('manifest is missing tensor "' + name + '"');
      return bytes.subarray(t.offset, t.offset + t.length);
    }
    function shape(name) { return manifest.tensors[name].shape; }

    const convLayers = manifest.convLayers.map(function (layer, i) {
      const prefix = 'conv' + i + '_';
      return {
        selfWeight: toRows(tensor(prefix + 'self_weight'), shape(prefix + 'self_weight')),
        selfBias: Array.from(tensor(prefix + 'self_bias')),
        neighWeight: toRows(tensor(prefix + 'neigh_weight'), shape(prefix + 'neigh_weight')),
      };
    });

    const readoutLayers = manifest.readoutLayers.map(function (layer, i) {
      const prefix = 'readout' + i + '_';
      return {
        weight: toRows(tensor(prefix + 'weight'), shape(prefix + 'weight')),
        bias: Array.from(tensor(prefix + 'bias')),
        activation: layer.activation, // 'relu' | 'identity'
      };
    });

    const model = {
      id: id,
      task: manifest.task || 'nagl-mbis-charges',
      atomFeatureDim: manifest.atomFeatureDim,
      convHiddenDim: manifest.convHiddenDim,
      convLayers: convLayers,
      readoutLayers: readoutLayers,
      postprocess: manifest.postprocess, // "charges" (electronegativity-equalization) -- only one implemented
    };
    if (model.postprocess !== 'charges') {
      throw new Error('Unsupported postprocess "' + model.postprocess + '" -- only "charges" (electronegativity-equalization) is implemented.');
    }

    models.set(id, model);
    return { id: id, task: model.task };
  };

  CC.NAGL.hasModel = function (id) { return id ? models.has(id) : models.size > 0; };
  CC.NAGL.getLoadedModelIds = function () { return Array.from(models.keys()); };
  CC.NAGL.clearModel = function (id) { if (id) models.delete(id); else models.clear(); };

  // One SAGEConv 'mean' layer over the whole graph. adjacency: array
  // (per atom) of neighbor atom indices (plain undirected connectivity
  // -- no bond features/order involved at all, matching nagl's graph
  // construction for this model).
  function sageConvLayer(layer, adjacency, features) {
    const numAtoms = features.length;
    const neighborMeans = new Array(numAtoms);
    for (let i = 0; i < numAtoms; i++) {
      const neighbors = adjacency[i];
      if (neighbors.length === 0) {
        neighborMeans[i] = new Array(features[i].length).fill(0);
        continue;
      }
      const dim = features[neighbors[0]].length;
      const sum = new Array(dim).fill(0);
      neighbors.forEach(function (j) {
        for (let d = 0; d < dim; d++) sum[d] += features[j][d];
      });
      neighborMeans[i] = sum.map(function (v) { return v / neighbors.length; });
    }

    const out = new Array(numAtoms);
    for (let i = 0; i < numAtoms; i++) {
      const hSelf = matVecBias(layer.selfWeight, layer.selfBias, features[i]);
      const hNeigh = matVecBias(layer.neighWeight, null, neighborMeans[i]);
      out[i] = relu(addVec(hSelf, hNeigh));
    }
    return out;
  }

  function applyReadout(model, embedding) {
    let x = embedding;
    model.readoutLayers.forEach(function (layer) {
      x = matVecBias(layer.weight, layer.bias, x);
      if (layer.activation === 'relu') x = relu(x);
      // 'identity' -- no-op, matches the checkpoint's final layer
    });
    return x; // [electronegativity, hardness] for the charges postprocess
  }

  // Closed-form electronegativity equalization -- see file header for
  // the formula and citation. `enegHardness` is an array of [eneg,
  // hardness] pairs, one per atom; `totalCharge` is the molecule's own
  // total formal charge (sum of every atom's formal charge).
  function equalizeCharges(enegHardness, totalCharge) {
    const invHardness = enegHardness.map(function (pair) { return 1 / pair[1]; });
    const eneg = enegHardness.map(function (pair) { return pair[0]; });

    let sumInvHardness = 0;
    let sumInvHardnessEneg = 0;
    for (let i = 0; i < invHardness.length; i++) {
      sumInvHardness += invHardness[i];
      sumInvHardnessEneg += invHardness[i] * eneg[i];
    }
    const correction = (sumInvHardnessEneg + totalCharge) / sumInvHardness;

    return invHardness.map(function (invH, i) {
      return -invH * eneg[i] + invH * correction;
    });
  }

  /**
   * Run a loaded NAGL-MBIS model on a molecule.
   * Returns { atomProperties: [{ [task]: charge }, ...], atomIds, backend: 'nagl', modelId }
   * -- same shape chemprop-model.js's atom-level output uses, so it
   * feeds the same existing atom-heatmap UI. atomProperties/atomIds
   * only cover the real, drawn (heavy) atoms -- synthetic explicit-H
   * nodes (see nagl-features.js's header) participate in the message
   * passing and the charge-conservation total, but there's nothing in
   * the drawn structure to attach their own result to, so they're not
   * surfaced individually.
   *
   * Throws if the molecule contains an atom outside this model's fixed
   * vocabulary (element or degree) -- see nagl-features.js's header for
   * why that's a real limitation to surface, not paper over.
   */
  /**
   * Checks whether the current molecule is within nagl-features.js's
   * fixed vocabulary (element set, degree range) -- usable even before
   * any NAGL model is actually loaded, since the vocabulary itself
   * doesn't depend on which specific checkpoint's weights are loaded
   * (see nagl-features.js's header for why this model family has a
   * hard, non-padding vocabulary unlike Chemprop's). Meant for a
   * proactive UI warning ("this structure has atoms the MBIS charge
   * model can't handle") rather than only finding out via a thrown
   * error after clicking Run.
   *
   * Returns { compatible: boolean, issues: string[] }.
   */
  CC.NAGL.checkCompatibility = function (molecule) {
    if (!molecule || molecule.atoms.size === 0) return { compatible: true, issues: [] };
    try {
      const molblock = CC.moleculeToMolblock(molecule);
      const annotations = CC.GNN.getRDKitAnnotations(molblock);
      const graph = CC.NAGL.buildExpandedGraph(molecule, annotations);
      return {
        compatible: graph.unsupportedAtoms.length === 0,
        issues: graph.unsupportedAtoms.map(function (u) { return u.reason; }),
      };
    } catch (err) {
      // A compatibility *check* failing shouldn't itself surface as a
      // scary error -- just don't warn if we couldn't determine an answer.
      return { compatible: true, issues: [] };
    }
  };

  CC.NAGL.predict = function (molecule, id) {
    const model = models.get(id);
    if (!model) throw new Error('No NAGL model loaded under id "' + id + '"');

    if (molecule.atoms.size === 0) return { atomProperties: [], atomIds: [], backend: 'nagl', modelId: id };

    const molblock = CC.moleculeToMolblock(molecule);
    const annotations = CC.GNN.getRDKitAnnotations(molblock);
    const graph = CC.NAGL.buildExpandedGraph(molecule, annotations);

    if (graph.unsupportedAtoms.length > 0) {
      const reasons = graph.unsupportedAtoms.map(function (u) { return u.reason; }).join('; ');
      throw new Error('This structure has atoms outside what this NAGL-MBIS model supports: ' + reasons);
    }

    let features = graph.rows;
    model.convLayers.forEach(function (layer) {
      features = sageConvLayer(layer, graph.adjacency, features);
    });

    // Total formal charge over the REAL atoms only -- the synthetic H
    // nodes are always neutral, so including or excluding them here
    // doesn't change the sum, but the real atoms are the ground truth.
    const heavyAtoms = Array.from(molecule.atoms.values());
    const totalCharge = heavyAtoms.reduce(function (sum, a) { return sum + a.charge; }, 0);

    const enegHardness = features.map(function (embedding) { return applyReadout(model, embedding); });
    const allCharges = equalizeCharges(enegHardness, totalCharge);

    // Only the leading numHeavyAtoms entries correspond to real, drawn
    // atoms (buildExpandedGraph appends synthetic H nodes after them).
    const atomProperties = allCharges.slice(0, graph.numHeavyAtoms).map(function (c) {
      const obj = {}; obj[model.task] = c; return obj;
    });

    return { atomProperties: atomProperties, atomIds: graph.atomIds, backend: 'nagl', modelId: id };
  };
})();
