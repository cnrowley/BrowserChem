/**
 * geomol-model.js
 *
 * Loads a converted GeoMol checkpoint (manifest.json + weights.bin,
 * produced by scripts/convert_geomol_checkpoint.py from the real
 * Ganea et al. NeurIPS 2021 model -- github.com/PattanaikL/GeoMol) and
 * runs its forward pass: the weight-tied message-passing GNN encoder,
 * per-neighborhood local-structure prediction (Transformer encoder +
 * chirality correction + distance head). Dihedral-pair assembly (turning
 * local structures into a full 3D conformer) is a separate, later piece
 * -- see GEOMOL_INTEGRATION.md for the staged build plan.
 *
 * Processes one conformer sample at a time rather than batching multiple
 * stochastic samples together the way the original PyTorch code does --
 * nothing in this architecture mixes data *across* samples (confirmed
 * directly by reading the real model.py: every noise draw, every
 * embedding, every prediction is independent per sample, only the atom/
 * edge/molecule axes ever get reduced over), so this is an exact
 * per-sample equivalent, just simpler to write and validate correctly in
 * JS than reimplementing an extra batch dimension nobody needs here.
 *
 * Every forward-pass function below was validated against a live run of
 * the real PyTorch model (trained_models/drugs/best_model.pt) with the
 * *same* injected noise vectors, not just spot-checked by eye -- see
 * GEOMOL_INTEGRATION.md's validation section for the exact numbers.
 */

window.CC = window.CC || {};
CC.GeoMol = window.CC.GeoMol || {};

(function () {
  const models = new Map(); // id -> { manifest, weights: Float32Array }

  function tensorView(model, name) {
    const t = model.manifest.tensors[name];
    if (!t) throw new Error('GeoMol model is missing tensor "' + name + '"');
    return model.weights.subarray(t.offset, t.offset + t.length);
  }

  function relu(x) {
    const y = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) y[i] = x[i] > 0 ? x[i] : 0;
    return y;
  }

  function linear(w, b, outDim, inDim, x) {
    const y = new Float64Array(outDim);
    for (let o = 0; o < outDim; o++) {
      let sum = b ? b[o] : 0;
      const base = o * inDim;
      for (let i = 0; i < inDim; i++) sum += w[base + i] * x[i];
      y[o] = sum;
    }
    return y;
  }

  /**
   * Generic MLP forward: stacked Linear layers, ReLU after every layer
   * except the last (matches model/GNN.py's MLP class exactly -- the
   * final nn.Linear appended after the loop has no activation following
   * it). `dims` is [inDim, ...outDims] as recorded in the manifest
   * (e.g. coordPredDims = [100, 100, 100, 3]); tensor names are
   * `${prefix}_layer${i}_weight`/`_bias`.
   */
  function mlpForward(model, prefix, dims, x) {
    let h = x;
    for (let i = 0; i < dims.length - 1; i++) {
      const w = tensorView(model, prefix + '_layer' + i + '_weight');
      const b = tensorView(model, prefix + '_layer' + i + '_bias');
      h = linear(w, b, dims[i + 1], dims[i], h);
      if (i < dims.length - 2) h = relu(h);
    }
    return h;
  }

  /**
   * One weight-tied message-passing GNN (model/GNN.py's GNN class):
   * node_init/edge_init MLPs run once, then the same MetaLayer update
   * (EdgeModel then NodeModel, both residual) is applied `depth` times
   * with shared weights. `keyPrefix` is 'gnn1' or 'gnn2'.
   *
   * nodeX: array of per-atom augmented feature vectors (Float64Array),
   * edgeAttrX: array of per-directed-edge augmented feature vectors,
   * edgeSrc/edgeDst: parallel arrays of directed-edge endpoints.
   * Returns { x: per-atom hidden vectors, edgeAttr: per-edge hidden vectors }.
   */
  function gnnForward(model, keyPrefix, dims, depth, nodeX, edgeAttrX, edgeSrc, edgeDst) {
    const hiddenDim = dims[keyPrefix].nodeInitDims[dims[keyPrefix].nodeInitDims.length - 1];
    let x = nodeX.map(function (v) { return mlpForward(model, keyPrefix + '_node_init', dims[keyPrefix].nodeInitDims, v); });
    let edgeAttr = edgeAttrX.map(function (v) { return mlpForward(model, keyPrefix + '_edge_init', dims[keyPrefix].edgeInitDims, v); });

    const edgeEps = tensorView(model, keyPrefix + '_edge_eps')[0];
    const nodeEps = tensorView(model, keyPrefix + '_node_eps')[0];
    const edgeW = tensorView(model, keyPrefix + '_edgemodel_edge_weight');
    const edgeB = tensorView(model, keyPrefix + '_edgemodel_edge_bias');
    const nodeInW = tensorView(model, keyPrefix + '_edgemodel_node_in_weight');
    const nodeOutW = tensorView(model, keyPrefix + '_edgemodel_node_out_weight');
    const edgeMlpDims = dims[keyPrefix].edgeModelMlpDims;
    const nodeMlp1Dims = dims[keyPrefix].nodeModelMlp1Dims;
    const nodeMlp2Dims = dims[keyPrefix].nodeModelMlp2Dims;

    const n = x.length;
    const e = edgeAttr.length;

    for (let step = 0; step < depth; step++) {
      // ---- EdgeModel: f_ij + f_i[row] + f_j[col], ReLU, then an MLP ----
      const fI = x.map(function (v) { return linear(nodeInW, null, hiddenDim, hiddenDim, v); });
      const fJ = x.map(function (v) { return linear(nodeOutW, null, hiddenDim, hiddenDim, v); });
      const newEdgeAttr = new Array(e);
      for (let k = 0; k < e; k++) {
        const fij = linear(edgeW, edgeB, hiddenDim, hiddenDim, edgeAttr[k]);
        const row = edgeSrc[k], col = edgeDst[k];
        const summed = new Float64Array(hiddenDim);
        for (let d = 0; d < hiddenDim; d++) summed[d] = fij[d] + fI[row][d] + fJ[col][d];
        const activated = relu(summed);
        const delta = mlpForward(model, keyPrefix + '_edgemodel_mlp', edgeMlpDims, activated);
        const updated = new Float64Array(hiddenDim);
        for (let d = 0; d < hiddenDim; d++) updated[d] = (1 + edgeEps) * edgeAttr[k][d] + delta[d];
        newEdgeAttr[k] = updated;
      }
      edgeAttr = newEdgeAttr;

      // ---- NodeModel: sum incoming (post-update) edge messages per destination node, then an MLP ----
      const perEdgeOut = edgeAttr.map(function (v) { return mlpForward(model, keyPrefix + '_nodemodel_mlp1', nodeMlp1Dims, v); });
      const aggregated = Array.from({ length: n }, function () { return new Float64Array(hiddenDim); });
      for (let k = 0; k < e; k++) {
        const col = edgeDst[k];
        const contribution = perEdgeOut[k];
        for (let d = 0; d < hiddenDim; d++) aggregated[col][d] += contribution[d];
      }
      const newX = new Array(n);
      for (let i = 0; i < n; i++) {
        const delta = mlpForward(model, keyPrefix + '_nodemodel_mlp2', nodeMlp2Dims, aggregated[i]);
        const updated = new Float64Array(hiddenDim);
        for (let d = 0; d < hiddenDim; d++) updated[d] = (1 + nodeEps) * x[i][d] + delta[d];
        newX[i] = updated;
      }
      x = newX;
    }

    return { x: x, edgeAttr: edgeAttr };
  }

  function standardNormalSample(rng) {
    // Box-Muller -- fine for drawing GeoMol's own input noise: the model
    // was trained to be robust to *any* Gaussian draw, not a specific
    // RNG's bit pattern (see GEOMOL_INTEGRATION.md -- this is a genuinely
    // different case from the arbitrary orthonormal-frame vector in the
    // assembly step, which needs no randomness at all in this port).
    const u1 = Math.max(rng(), 1e-12), u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  function gaussianVector(dim, std, rng) {
    const v = new Float64Array(dim);
    for (let i = 0; i < dim; i++) v[i] = standardNormalSample(rng || Math.random) * std;
    return v;
  }

  /**
   * Runs both GNN encoders (gnn1 -> x1 for local-structure prediction,
   * gnn2 -> x2/h_mol for dihedral/torsion prediction) on one conformer
   * sample. `randX`/`randEdge`, if supplied, override the drawn Gaussian
   * noise (validation only -- real use always omits them and gets a
   * fresh draw, which is what gives repeated calls genuinely different
   * conformers).
   */
  CC.GeoMol.embed = function (model, input, randX, randEdge, rng) {
    const manifest = model.manifest;
    const noiseDim = manifest.randomVecDim;
    const noiseStd = manifest.randomVecStd;
    const n = input.numAtoms;
    const eCount = input.edgeSrc.length;

    const rx = randX || input.x.map(function () { return gaussianVector(noiseDim, noiseStd, rng); });
    const re = randEdge || input.edgeAttr.map(function () { return gaussianVector(noiseDim, noiseStd, rng); });

    const nodeAug = input.x.map(function (row, i) {
      return Float64Array.from(row.concat(Array.from(rx[i])));
    });
    const edgeAug = input.edgeAttr.map(function (row, i) {
      return Float64Array.from(row.concat(Array.from(re[i])));
    });

    const gnn1Out = gnnForward(model, 'gnn1', manifest, manifest.gnn1Depth, nodeAug, edgeAug, input.edgeSrc, input.edgeDst);
    const gnn2Out = gnnForward(model, 'gnn2', manifest, manifest.gnn2Depth, nodeAug, edgeAug, input.edgeSrc, input.edgeDst);

    const hiddenDim = manifest.modelDim;
    const pooled = new Float64Array(hiddenDim);
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < hiddenDim; d++) pooled[d] += gnn2Out.x[i][d];
    }
    const hMol = mlpForward(model, 'h_mol_mlp', manifest.hMolMlpDims, pooled);

    return { x1: gnn1Out.x, x2: gnn2Out.x, hMol: hMol, randX: rx, randEdge: re };
  };

  // ---------- local-structure prediction (per-neighborhood Transformer) ----------

  function softplus(x) {
    return x > 20 ? x : Math.log1p(Math.exp(x)); // numerically stable for large x
  }

  function layerNorm(x, weight, bias) {
    const n = x.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let variance = 0;
    for (let i = 0; i < n; i++) variance += (x[i] - mean) * (x[i] - mean);
    variance /= n; // PyTorch LayerNorm: biased variance (divide by n, not n-1)
    const invStd = 1 / Math.sqrt(variance + 1e-5);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) y[i] = (x[i] - mean) * invStd * weight[i] + bias[i];
    return y;
  }

  /**
   * One nn.TransformerEncoderLayer forward pass (post-LN, ReLU
   * feedforward -- PyTorch's defaults, matching how GeoMol constructs
   * it): multi-head self-attention with a residual + LayerNorm, then a
   * feedforward block with its own residual + LayerNorm. `seq` is a
   * fixed-length array of Float64Array (length 4 here -- up to 4
   * neighbor slots); `validMask[t]` false means that position is padding
   * and must never be attended *to* (it can still appear as a query,
   * same as real key_padding_mask semantics -- its output is discarded
   * by the caller regardless).
   */
  function transformerEncoderLayerForward(model, prefix, encoderInfo, seq, validMask) {
    const dModel = encoderInfo.dModel, nHead = encoderInfo.nHead, headDim = dModel / nHead;
    const seqLen = seq.length;

    const inProjW = tensorView(model, prefix + '_in_proj_weight');
    const inProjB = tensorView(model, prefix + '_in_proj_bias');
    const outProjW = tensorView(model, prefix + '_out_proj_weight');
    const outProjB = tensorView(model, prefix + '_out_proj_bias');

    const Q = [], K = [], V = [];
    for (let t = 0; t < seqLen; t++) {
      const qkv = linear(inProjW, inProjB, 3 * dModel, dModel, seq[t]);
      Q.push(qkv.subarray(0, dModel));
      K.push(qkv.subarray(dModel, 2 * dModel));
      V.push(qkv.subarray(2 * dModel, 3 * dModel));
    }

    const attnOut = seq.map(function () { return new Float64Array(dModel); });
    const scale = 1 / Math.sqrt(headDim);
    for (let h = 0; h < nHead; h++) {
      const off = h * headDim;
      for (let ti = 0; ti < seqLen; ti++) {
        const scores = new Float64Array(seqLen);
        let maxScore = -Infinity;
        for (let tj = 0; tj < seqLen; tj++) {
          if (!validMask[tj]) { scores[tj] = -Infinity; continue; }
          let dot = 0;
          for (let d = 0; d < headDim; d++) dot += Q[ti][off + d] * K[tj][off + d];
          scores[tj] = dot * scale;
          if (scores[tj] > maxScore) maxScore = scores[tj];
        }
        const expScores = new Float64Array(seqLen);
        let sumExp = 0;
        for (let tj = 0; tj < seqLen; tj++) {
          if (!isFinite(scores[tj])) continue;
          expScores[tj] = Math.exp(scores[tj] - maxScore);
          sumExp += expScores[tj];
        }
        for (let d = 0; d < headDim; d++) {
          let acc = 0;
          for (let tj = 0; tj < seqLen; tj++) {
            if (sumExp > 0) acc += (expScores[tj] / sumExp) * V[tj][off + d];
          }
          attnOut[ti][off + d] = acc;
        }
      }
    }

    const afterOutProj = attnOut.map(function (v) { return linear(outProjW, outProjB, dModel, dModel, v); });
    const norm1W = tensorView(model, prefix + '_norm1_weight'), norm1B = tensorView(model, prefix + '_norm1_bias');
    const afterNorm1 = seq.map(function (v, i) {
      const y = new Float64Array(dModel);
      for (let d = 0; d < dModel; d++) y[d] = v[d] + afterOutProj[i][d];
      return layerNorm(y, norm1W, norm1B);
    });

    const lin1W = tensorView(model, prefix + '_linear1_weight'), lin1B = tensorView(model, prefix + '_linear1_bias');
    const lin2W = tensorView(model, prefix + '_linear2_weight'), lin2B = tensorView(model, prefix + '_linear2_bias');
    const norm2W = tensorView(model, prefix + '_norm2_weight'), norm2B = tensorView(model, prefix + '_norm2_bias');
    const ffDim = encoderInfo.dimFeedforward;
    return afterNorm1.map(function (v) {
      const hidden = relu(linear(lin1W, lin1B, ffDim, dModel, v));
      const ffOut = linear(lin2W, lin2B, dModel, ffDim, hidden);
      const y = new Float64Array(dModel);
      for (let d = 0; d < dModel; d++) y[d] = v[d] + ffOut[d];
      return layerNorm(y, norm2W, norm2B);
    });
  }

  function signedVolume(coords4) {
    const v1 = CC.vec3.sub(coords4[0], coords4[3]);
    const v2 = CC.vec3.sub(coords4[1], coords4[3]);
    const v3 = CC.vec3.sub(coords4[2], coords4[3]);
    const cp = CC.vec3.cross(v2, v3);
    return Math.sign(CC.vec3.dot(v1, cp));
  }

  /**
   * Predicts the local 3D neighborhood geometry (unit direction + bond
   * distance for each neighbor, in local coordinates centered on the
   * atom) for every atom with more than one neighbor -- model.py's
   * model_local_stats(). Returns a map from atom index to an array of
   * {x,y,z} local-frame coordinates, one per neighbor (in the same order
   * as input.neighbors[atomIndex]).
   *
   * chiralTag (from CC.GNN.buildGeomolInput) is only actually consulted
   * for neighborhoods with exactly 4 neighbors -- signed_volume's
   * tetrahedron construction isn't meaningful with fewer, matching the
   * real model (a real stereocenter always has its 4th substituent
   * explicit at this point, whether that's a heavy atom or an implicit
   * hydrogen already added by buildGeomolInput's AddHs pass).
   */
  CC.GeoMol.predictLocalStructures = function (model, input, x1, chiralTag) {
    const manifest = model.manifest;
    const dModel = manifest.modelDim;
    const results = {};

    Object.keys(input.neighbors).forEach(function (key) {
      const atomIndex = parseInt(key, 10);
      const neighborIndices = input.neighbors[key];
      const nNb = neighborIndices.length;
      const xCentral = x1[atomIndex];

      const h = [], hFlipped = [], validMask = [];
      for (let k = 0; k < 4; k++) {
        if (k < nNb) {
          const xn = x1[neighborIndices[k]];
          const v = new Float64Array(2 * dModel);
          const vFlipped = new Float64Array(2 * dModel);
          for (let d = 0; d < dModel; d++) {
            v[d] = xn[d]; v[dModel + d] = xCentral[d];
            vFlipped[d] = xCentral[d]; vFlipped[dModel + d] = xn[d];
          }
          h.push(v); hFlipped.push(vFlipped); validMask.push(true);
        } else {
          h.push(new Float64Array(2 * dModel)); hFlipped.push(new Float64Array(2 * dModel)); validMask.push(false);
        }
      }

      const hNew = transformerEncoderLayerForward(model, 'encoder', manifest.encoder, h, validMask);
      let unitNormals = hNew.map(function (v) { return mlpForward(model, 'coord_pred', manifest.coordPredDims, v); });
      for (let k = 0; k < 4; k++) if (!validMask[k]) unitNormals[k] = new Float64Array(3);

      if (nNb === 4 && chiralTag[atomIndex] !== 0) {
        const coords4 = unitNormals.map(function (v) { return { x: v[0], y: v[1], z: v[2] }; });
        const sv = signedVolume(coords4);
        const zFlip = sv * chiralTag[atomIndex]; // always +-1 -- multiply, don't branch (matches z * z_flip exactly)
        unitNormals = unitNormals.map(function (v) {
          const y = Float64Array.from(v);
          y[2] = y[2] * zFlip;
          return y;
        });
      }

      const dPreds = h.map(function (v, k) {
        if (!validMask[k]) return 0;
        const a = mlpForward(model, 'd_mlp', manifest.dMlpDims, v);
        const b = mlpForward(model, 'd_mlp', manifest.dMlpDims, hFlipped[k]);
        return softplus(a[0] + b[0]);
      });

      results[atomIndex] = unitNormals.map(function (v, k) {
        if (!validMask[k]) return { x: 0, y: 0, z: 0 };
        const norm = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + 1e-10;
        const d = dPreds[k];
        return { x: v[0] / norm * d, y: v[1] / norm * d, z: v[2] / norm * d };
      });
    });

    return results;
  };

  // ---------- torsion-angle prediction (per dihedral pair) ----------

  function concatVectors(vectors) {
    let total = 0;
    for (let i = 0; i < vectors.length; i++) total += vectors[i].length;
    const out = new Float64Array(total);
    let offset = 0;
    for (let i = 0; i < vectors.length; i++) {
      out.set(vectors[i], offset);
      offset += vectors[i].length;
    }
    return out;
  }

  /**
   * Builds the per-dihedral-pair bookkeeping model.py's
   * assign_neighborhoods computes: which neighbor slot of each central
   * atom is literally the *other* central atom (xIdxOfY/yIdxOfX -- the
   * "map to neighbor y/x" one-hot position), and dihedralMask, which of
   * the 9 (pT_idx, qZ_idx) neighbor-slot combinations correspond to two
   * real (non-padding) atoms. pT_idx/qZ_idx follow
   * torch.cartesian_prod(arange(3), arange(3))'s exact enumeration order
   * (first index slow-varying, second fast).
   */
  function buildDihedralPairMeta(input, s, e) {
    const xNbrs = input.neighbors[s], yNbrs = input.neighbors[e];
    const xMaskFull = [0, 1, 2, 3].map(function (k) { return k < xNbrs.length; });
    const yMaskFull = [0, 1, 2, 3].map(function (k) { return k < yNbrs.length; });
    const xIdxOfY = xNbrs.indexOf(e);
    const yIdxOfX = yNbrs.indexOf(s);

    const xRemaining = [0, 1, 2, 3].filter(function (k) { return k !== xIdxOfY; });
    const yRemaining = [0, 1, 2, 3].filter(function (k) { return k !== yIdxOfX; });
    const xRemainingValid = xRemaining.map(function (k) { return xMaskFull[k]; });
    const yRemainingValid = yRemaining.map(function (k) { return yMaskFull[k]; });

    const dihedralMask = [];
    for (let pi = 0; pi < 3; pi++) {
      for (let qi = 0; qi < 3; qi++) dihedralMask.push(xRemainingValid[pi] && yRemainingValid[qi]);
    }

    return { xNbrs: xNbrs, yNbrs: yNbrs, xIdxOfY: xIdxOfY, yIdxOfX: yIdxOfX, xRemaining: xRemaining, yRemaining: yRemaining, dihedralMask: dihedralMask };
  }

  /**
   * Predicts, for every dihedral pair (s,e), the c_ij coefficients (one
   * per of the 9 neighbor-slot-pair combinations) and v_star = [cos
   * alpha, sin alpha] the assembly step's gamma-fitting needs --
   * model.py's align_dihedral_neighbors, minus the geometric
   * frame-construction half of that function (Hx/Hy, p_T_prime,
   * q_Z_translated), which belongs in the assembly step instead since it
   * needs actual 3D coordinates, not just embeddings. c_ij and v_star
   * are both provably independent of that geometric frame -- neither
   * expression below reads Hx/Hy or any rotated coordinate, only x2
   * embeddings, h_mol, and noise (confirmed directly against the real
   * source, not assumed) -- which is what makes splitting them apart
   * like this valid.
   */
  CC.GeoMol.predictTorsions = function (model, input, x2, hMol, dihedralPairs, randAlphaByPair, rng) {
    const manifest = model.manifest;
    const dModel = manifest.modelDim;
    const noiseDim = manifest.randomVecDim, noiseStd = manifest.randomVecStd;
    const zeroVec = new Float64Array(dModel);

    return dihedralPairs.map(function (pair, pairIdx) {
      const s = pair[0], e = pair[1];
      const meta = buildDihedralPairMeta(input, s, e);
      const xCentral = x2[s], yCentral = x2[e];

      const randAlpha = randAlphaByPair ? randAlphaByPair[pairIdx] : gaussianVector(noiseDim, noiseStd, rng);

      const alphaAB = mlpForward(model, 'alpha_mlp', manifest.alphaMlpDims, concatVectors([xCentral, yCentral, hMol, randAlpha]))[0];
      const alphaBA = mlpForward(model, 'alpha_mlp', manifest.alphaMlpDims, concatVectors([yCentral, xCentral, hMol, randAlpha]))[0];
      const alpha = alphaAB + alphaBA;
      const vStar = [Math.cos(alpha), Math.sin(alpha)];

      function repAt(nbrs, slot) {
        return slot < nbrs.length ? x2[nbrs[slot]] : zeroVec;
      }

      const cIj = [];
      for (let pi = 0; pi < 3; pi++) {
        const pRep = repAt(meta.xNbrs, meta.xRemaining[pi]);
        for (let qi = 0; qi < 3; qi++) {
          const qRep = repAt(meta.yNbrs, meta.yRemaining[qi]);
          const cPQ = mlpForward(model, 'c_mlp', manifest.cMlpDims, concatVectors([pRep, xCentral, qRep, yCentral]))[0];
          const cQP = mlpForward(model, 'c_mlp', manifest.cMlpDims, concatVectors([qRep, yCentral, pRep, xCentral]))[0];
          cIj.push(cPQ + cQP);
        }
      }

      return { s: s, e: e, meta: meta, cIj: cIj, vStar: vStar, alpha: alpha };
    });
  };

  // ---------- public loading API (same shape as CC.ANI's) ----------

  CC.GeoMol.loadModel = function (id, manifestUrl, weightsUrl) {
    return fetch(manifestUrl).then(function (r) {
      if (!r.ok) throw new Error('failed to fetch GeoMol manifest: ' + r.status);
      return r.json();
    }).then(function (manifest) {
      return fetch(weightsUrl).then(function (r) {
        if (!r.ok) throw new Error('failed to fetch GeoMol weights: ' + r.status);
        return r.arrayBuffer();
      }).then(function (buf) {
        models.set(id, { manifest: manifest, weights: new Float32Array(buf) });
        return { id: id, task: manifest.task };
      });
    });
  };

  CC.GeoMol.hasModel = function (id) {
    return id ? models.has(id) : models.size > 0;
  };

  CC.GeoMol.getLoadedModelIds = function () {
    return Array.from(models.keys());
  };

  CC.GeoMol.clearModel = function (id) {
    if (id) models.delete(id);
    else models.clear();
  };

  CC.GeoMol.getModel = function (id) {
    return models.get(id);
  };
})();
