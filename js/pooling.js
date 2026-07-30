/**
 * pooling.js
 *
 * Reduces a set of per-atom embeddings down to a single molecule-level
 * embedding. Three strategies, matching what the spec asked for.
 */

window.CC = window.CC || {};
CC.GNN = window.CC.GNN || {};

(function () {
  function zeros(n) { return new Array(n).fill(0); }

  CC.GNN.poolMean = function (atomEmbeddings, dim) {
    if (atomEmbeddings.length === 0) return zeros(dim);
    const sum = zeros(dim);
    atomEmbeddings.forEach(function (v) {
      for (let i = 0; i < dim; i++) sum[i] += v[i];
    });
    return sum.map(function (x) { return x / atomEmbeddings.length; });
  };

  CC.GNN.poolSum = function (atomEmbeddings, dim) {
    const sum = zeros(dim);
    atomEmbeddings.forEach(function (v) {
      for (let i = 0; i < dim; i++) sum[i] += v[i];
    });
    return sum;
  };

  // Simple learned-attention pooling: score_v = w . atomEmb[v], softmax
  // over atoms, weighted sum. `attnWeight` is a length-dim vector — pass
  // one in (e.g. from CC.GNN.createDemoWeights-style random init) or
  // build your own; there's no canonical "right" one without training.
  CC.GNN.poolAttention = function (atomEmbeddings, dim, attnWeight) {
    if (atomEmbeddings.length === 0) return zeros(dim);
    const scores = atomEmbeddings.map(function (v) {
      let s = 0;
      for (let i = 0; i < dim; i++) s += v[i] * attnWeight[i];
      return s;
    });
    const maxScore = Math.max.apply(null, scores);
    const expScores = scores.map(function (s) { return Math.exp(s - maxScore); });
    const total = expScores.reduce(function (a, b) { return a + b; }, 0);
    const weights = expScores.map(function (e) { return e / total; });

    const pooled = zeros(dim);
    atomEmbeddings.forEach(function (v, idx) {
      for (let i = 0; i < dim; i++) pooled[i] += v[i] * weights[idx];
    });
    return { pooled: pooled, attentionWeights: weights };
  };
})();
