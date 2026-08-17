/**
 * dmpnn.js
 *
 * A Directed Message Passing Neural Network forward pass, matching
 * Chemprop's core idea: hidden states live on directed bonds (not atoms),
 * and when updating the hidden state for edge i->j, the incoming message
 * excludes the reverse edge j->i (so a bond doesn't "hear its own echo").
 *
 * This is the same forward pass Chemprop's `BondMessagePassing` runs, and
 * it is numerically exact when fed real trained weights (see
 * chemprop-model.js, which loads a converted `best.pt` checkpoint and
 * calls this with the exact Chemprop featurizers). With
 * CC.GNN.createDemoWeights() instead, it's architecturally identical but
 * numerically meaningless (untrained) — a smoke test of the plumbing.
 *
 * Math per step (standard Chemprop BondMessagePassing, depth = T):
 *   h0raw[e]   = W_i . [atomFeat[src(e)], bondFeat[e]]                (no bias, no activation)
 *   h[e]_0     = ReLU(h0raw[e])
 *   for t = 1 .. T-1:
 *     msg[e]   = sum_{e' in incoming(src(e)) \ rev(e)} h[e']_(t-1)
 *     h[e]_t   = ReLU(h0raw[e] + W_h . msg[e])                        (no bias)
 *   atomEmb[v] = ReLU(W_o . [atomFeat[v], sum_{e: dst(e)=v} h[e]_(T-1)] + bias_o)
 *
 * A bond-level checkpoint (BDE) additionally needs a per-directed-edge
 * embedding Chemprop calls edge_finalize (confirmed directly from
 * chemprop/nn/message_passing/mol_atom_bond.py, MABBondMessagePassing's
 * base class): edgeEmb[e] = ReLU(W_eo . [bondFeat[e], h[e]_(T-1)] + bias_eo),
 * using the SAME final h[e] the atom-embedding step reads (no extra
 * message-passing round). Only computed when weights.Weo is supplied —
 * every existing molecule-/atom-level checkpoint's manifest has no W_eo
 * tensor at all, since only a bond-level head ever consumes it.
 *
 * Two things worth flagging since they're easy to get subtly wrong:
 *   1. The residual added back in at every step is the *unactivated*
 *      linear projection h0raw, not ReLU(h0raw) — only the running
 *      iterate gets activated.
 *   2. There are only depth-1 message/update rounds after the initial
 *      activation, not `depth` rounds (Chemprop's loop is
 *      `for _ in range(1, depth)`).
 */

window.CC = window.CC || {};
CC.GNN = window.CC.GNN || {};

(function () {
  function relu(v) { return v.map(function (x) { return Math.max(0, x); }); }

  function randMatrix(rows, cols, scale) {
    scale = scale || 0.1;
    const m = [];
    for (let r = 0; r < rows; r++) {
      const row = new Array(cols);
      for (let c = 0; c < cols; c++) row[c] = (Math.random() * 2 - 1) * scale;
      m.push(row);
    }
    return m;
  }

  // y = W . x  (W is rows x cols, x is length-cols vector, y is length-rows)
  function matVec(W, x) {
    const y = new Array(W.length).fill(0);
    for (let r = 0; r < W.length; r++) {
      let sum = 0;
      const row = W[r];
      for (let c = 0; c < x.length; c++) sum += row[c] * x[c];
      y[r] = sum;
    }
    return y;
  }

  // y = W . x + bias  (bias may be null/undefined for a bias-free layer)
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

  function addVec(a, b) {
    return a.map(function (v, i) { return v + b[i]; });
  }

  function sumVectors(vectors, dim) {
    const sum = new Array(dim).fill(0);
    vectors.forEach(function (v) {
      for (let i = 0; i < dim; i++) sum[i] += v[i];
    });
    return sum;
  }

  /**
   * Create a fresh set of randomly-initialized weights sized for a given
   * graph's feature dimensions. Call once per (atomFeatureDim,
   * bondFeatureDim, hiddenSize) combination and reuse — re-randomizing
   * per prediction would make repeated runs on the same molecule give
   * different (equally meaningless) answers, which is more confusing
   * than helpful even for a demo.
   *
   * Shape/bias convention matches Chemprop's BondMessagePassing exactly
   * (see chemprop-model.js for how a real checkpoint fills this same
   * shape): W_i and W_h have no bias, W_o does.
   */
  CC.GNN.createDemoWeights = function (atomFeatureDim, bondFeatureDim, hiddenSize) {
    return {
      hiddenSize: hiddenSize,
      Wi: randMatrix(hiddenSize, atomFeatureDim + bondFeatureDim), WiBias: null,
      Wh: randMatrix(hiddenSize, hiddenSize), WhBias: null,
      Wo: randMatrix(hiddenSize, atomFeatureDim + hiddenSize), WoBias: new Array(hiddenSize).fill(0),
    };
  };

  /**
   * Run the D-MPNN forward pass over a graph (from graph-builder.js).
   * Returns { atomEmbeddings: number[][], edgeEmbeddings: number[][] | null },
   * one embedding per atom (matching graph.atomFeatures order) and,
   * only when weights.Weo is supplied, one per directed edge (matching
   * graph.edgeFeatures order) — null otherwise, so existing
   * molecule-/atom-level callers see no change in shape.
   *
   * `weights` needs: hiddenSize, Wi/WiBias, Wh/WhBias — see
   * createDemoWeights() above for the shape convention (*Bias may be
   * null). Wo/WoBias and Weo/WeoBias are each independently optional: a
   * bond-level (BDE) checkpoint has no atom-output layer at all
   * (Chemprop never builds message_passing.W_vo when there's no
   * mol_predictor/atom_predictor attached), so Wo may be absent and
   * atomEmbeddings comes back null in that case; conversely Weo is only
   * present for a bond-level checkpoint.
   */
  CC.GNN.runDMPNN = function (graph, weights, opts) {
    opts = opts || {};
    const depth = opts.depth || 3;
    const hiddenSize = weights.hiddenSize;
    const numEdges = graph.edgeSrc.length;

    if (numEdges === 0) {
      // No bonds at all (single atom, e.g. methane drawn as a lone C):
      // atom embedding falls back to just the atom-feature projection.
      return {
        atomEmbeddings: weights.Wo ? graph.atomFeatures.map(function (feat) {
          return relu(matVecBias(weights.Wo, weights.WoBias, feat.concat(new Array(hiddenSize).fill(0))));
        }) : null,
        edgeEmbeddings: weights.Weo ? [] : null,
      };
    }

    // Unactivated linear projection per directed edge. This is kept
    // around (not just its ReLU) because Chemprop adds it back in
    // unactivated at every update step — see header comment.
    const h0raw = new Array(numEdges);
    for (let e = 0; e < numEdges; e++) {
      const input = graph.atomFeatures[graph.edgeSrc[e]].concat(graph.edgeFeatures[e]);
      h0raw[e] = matVecBias(weights.Wi, weights.WiBias, input);
    }

    let h = h0raw.map(relu); // H = tau(H_0): the first iterate used for messaging

    // depth-1 message/update rounds (Chemprop: `for _ in range(1, depth)`).
    for (let step = 1; step < depth; step++) {
      const hNext = new Array(numEdges);
      for (let e = 0; e < numEdges; e++) {
        const srcAtom = graph.edgeSrc[e];
        const rev = graph.revEdge[e];
        const incoming = graph.incomingEdgesByAtom[srcAtom].filter(function (e2) { return e2 !== rev; });
        const messageVectors = incoming.map(function (e2) { return h[e2]; });
        const message = sumVectors(messageVectors, hiddenSize);
        hNext[e] = relu(addVec(h0raw[e], matVecBias(weights.Wh, weights.WhBias, message)));
      }
      h = hNext;
    }

    let atomEmbeddings = null;
    if (weights.Wo) {
      atomEmbeddings = [];
      for (let v = 0; v < graph.numAtoms; v++) {
        const incoming = graph.incomingEdgesByAtom[v].map(function (e) { return h[e]; });
        const message = sumVectors(incoming, hiddenSize);
        const input = graph.atomFeatures[v].concat(message);
        atomEmbeddings.push(relu(matVecBias(weights.Wo, weights.WoBias, input)));
      }
    }

    let edgeEmbeddings = null;
    if (weights.Weo) {
      edgeEmbeddings = new Array(numEdges);
      for (let e = 0; e < numEdges; e++) {
        const input = graph.edgeFeatures[e].concat(h[e]);
        edgeEmbeddings[e] = relu(matVecBias(weights.Weo, weights.WeoBias, input));
      }
    }

    return { atomEmbeddings: atomEmbeddings, edgeEmbeddings: edgeEmbeddings };
  };
})();
