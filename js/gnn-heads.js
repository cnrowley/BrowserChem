/**
 * gnn-heads.js
 *
 * The two output heads: a molecular-property head reading the pooled
 * graph embedding, and an atomic-property head reading each atom's
 * embedding independently. Both are a single linear layer — real Chemprop
 * heads are usually a small MLP, but a linear layer is enough to prove
 * the shape of "dual heads reading the same backbone" without adding
 * more randomly-initialized layers that don't mean anything without
 * training anyway.
 */

window.CC = window.CC || {};
CC.GNN = window.CC.GNN || {};

(function () {
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

  function matVec(W, bias, x) {
    return W.map(function (row, r) {
      let sum = bias[r];
      for (let c = 0; c < x.length; c++) sum += row[c] * x[c];
      return sum;
    });
  }

  /**
   * propertyNames: e.g. ['logP', 'solubility'] — defines the output width
   * and gives the raw numbers labels in the UI.
   */
  CC.GNN.createHeadWeights = function (embeddingDim, propertyNames) {
    return {
      propertyNames: propertyNames,
      W: randMatrix(propertyNames.length, embeddingDim),
      b: new Array(propertyNames.length).fill(0),
    };
  };

  CC.GNN.runMolecularHead = function (pooledEmbedding, headWeights) {
    const values = matVec(headWeights.W, headWeights.b, pooledEmbedding);
    const result = {};
    headWeights.propertyNames.forEach(function (name, i) { result[name] = values[i]; });
    return result;
  };

  CC.GNN.runAtomicHead = function (atomEmbeddings, headWeights) {
    return atomEmbeddings.map(function (emb) {
      const values = matVec(headWeights.W, headWeights.b, emb);
      const result = {};
      headWeights.propertyNames.forEach(function (name, i) { result[name] = values[i]; });
      return result;
    });
  };
})();
