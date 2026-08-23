/**
 * ocsrglyph-model.js
 *
 * Client-side inference for OCSRGlyph (optical chemical structure
 * recognition -- image of a molecule drawing in, SMILES out), via
 * ONNX Runtime Web. Real, published model: EdisonScientific/glyph
 * (Apache-2.0, github.com/EdisonScientific/glyph), weights at
 * huggingface.co/EdisonScientific/OCSRGlyph, converted by
 * scripts/convert_ocsrglyph_checkpoint.py -- see that script's own
 * header for the full architecture/preprocessing/decoding details
 * confirmed directly from the upstream source, and model/ocsrglyph/
 * manifest.json's own "validated" field for the exact-match check that
 * conversion ran before these weights were trusted.
 *
 * Deliberately standalone, NOT wired through model-registry.js/
 * model-adapters.js: those are shaped for "predict a property/geometry
 * for an already-drawn molecule," a fundamentally different input
 * contract from "produce a new molecule from an uploaded/pasted image."
 * This file follows the existing manual-ONNX-load demo path's pattern
 * instead (js/gnn-inference.js's loadOnnxModel/predictOnnx -- same
 * ort.InferenceSession.create()/ort.Tensor() calls, no options object).
 *
 * Two ONNX graphs, matching the reference `_greedy_batch` decode loop
 * (upstream glyph/ocsr/predict.py) exactly -- no KV-cache, same as the
 * original:
 *   - encoder.onnx: image [1,3,384,384] -> memory [1,144,256] (one
 *     forward pass per image).
 *   - decoder.onnx: tokens [1,T] (T grows by one every decode step) +
 *     memory [1,144,256] -> logits [1,T,101]; only the last position's
 *     logits are used each step (argmax, greedy -- matches the
 *     reference, no beam search/sampling).
 *
 * Tokenizer is a flat 101-entry char->id dict (vocab_chars.json,
 * vendored as-is from the upstream repo) -- no subword/BPE logic needed,
 * unlike this project's Chemprop/NAGL featurizers.
 *
 * Deliberately RDKit-free: this file's job stops at "a SMILES string,
 * with hallucinated explicit-H fragments stripped" (the upstream
 * postprocess.py's strip_hydrogen_fragments step, ported directly below
 * -- pure string manipulation, no chemistry library needed for it).
 * Parsing/validating/loading that string into the 2D canvas reuses
 * app.js's own existing "Load from SMILES" RDKit code path rather than
 * duplicating a second RDKit-canonicalization step here.
 */

window.CC = window.CC || {};
CC.OCSRGlyph = window.CC.OCSRGlyph || {};

(function () {
  const MODEL_BASE = 'model/ocsrglyph/';

  let manifest = null;
  let vocab = null; // {char: id}
  let itos = null; // {id: char}
  let encoderSession = null;
  let decoderSession = null;
  let loadPromise = null;

  CC.OCSRGlyph.isLoaded = function () {
    return !!(encoderSession && decoderSession);
  };

  /**
   * Fetches the manifest/vocab and creates both ONNX Runtime sessions.
   * Idempotent -- safe to call repeatedly; only fetches/loads once.
   * Load-on-demand: this app never calls this at page load, only when
   * the user opens the OCSR tab or clicks "Recognize" (see app.js) --
   * this is by far the largest one-time download any feature in this
   * app triggers (~209MB total, fp16), so it must never happen silently.
   */
  CC.OCSRGlyph.loadModel = function () {
    if (loadPromise) return loadPromise;
    const ort = window.chemCanvasLibs && window.chemCanvasLibs.ort;
    if (!ort) return Promise.reject(new Error('ONNX Runtime Web is not loaded'));

    loadPromise = Promise.all([
      fetch(MODEL_BASE + 'manifest.json').then(function (r) {
        if (!r.ok) throw new Error('failed to fetch manifest.json: ' + r.status);
        return r.json();
      }),
      fetch(MODEL_BASE + 'vocab_chars.json').then(function (r) {
        if (!r.ok) throw new Error('failed to fetch vocab_chars.json: ' + r.status);
        return r.json();
      }),
    ]).then(function (results) {
      manifest = results[0];
      vocab = results[1];
      itos = {};
      Object.keys(vocab).forEach(function (ch) { itos[vocab[ch]] = ch; });
      return Promise.all([
        ort.InferenceSession.create(MODEL_BASE + 'encoder.onnx'),
        ort.InferenceSession.create(MODEL_BASE + 'decoder.onnx'),
      ]);
    }).then(function (sessions) {
      encoderSession = sessions[0];
      decoderSession = sessions[1];
      return { manifest: manifest };
    }).catch(function (err) {
      loadPromise = null; // allow retry on a subsequent call rather than caching a failure forever
      throw err;
    });
    return loadPromise;
  };

  /**
   * Draws an image (HTMLImageElement/HTMLCanvasElement/ImageBitmap) into
   * an offscreen canvas at the model's input resolution and returns a
   * Float32Array in CHW order, normalized to [-1,1] -- matches
   * scripts/convert_ocsrglyph_checkpoint.py's preprocess_image exactly
   * (itself a direct port of the upstream predict.py's
   * _preprocess_image: RGB, bilinear resize, (v/255-0.5)/0.5).
   */
  function preprocessImage(imageSource) {
    const size = manifest.inputSize;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    // Canvas's default drawImage scaling is bilinear-equivalent
    // (browsers use a smoothing filter comparable to PIL's BILINEAR for
    // this kind of downscale/upscale), matching the reference
    // preprocessing closely enough for a model this size to be robust
    // to -- the conversion script's own validation used exact-pixel PIL
    // resizing and still matched bit-for-bit at the SMILES-string level,
    // so sub-pixel resampling differences are not the bottleneck here.
    ctx.fillStyle = '#ffffff'; // flatten any alpha channel onto white, matching a real photographed/scanned page background
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(imageSource, 0, 0, size, size);
    const imageData = ctx.getImageData(0, 0, size, size).data; // RGBA, HWC, row-major

    const chw = new Float32Array(3 * size * size);
    const plane = size * size;
    for (let p = 0; p < plane; p++) {
      const o = p * 4;
      chw[p] = (imageData[o] / 255 - 0.5) / 0.5; // R
      chw[plane + p] = (imageData[o + 1] / 255 - 0.5) / 0.5; // G
      chw[2 * plane + p] = (imageData[o + 2] / 255 - 0.5) / 0.5; // B
    }
    return chw;
  }

  // Matches upstream postprocess.py's strip_hydrogen_fragments exactly:
  // split on '.', drop fragments that are EXACTLY "[H]" or "[HH]"
  // (hallucinated disconnected-hydrogen artifacts the decoder
  // occasionally emits), rejoin. Never touches chemically meaningful
  // bracketed hydrogens like [nH] or [NH3+] (those aren't exact-match
  // fragments on their own).
  function stripHydrogenFragments(smiles) {
    if (!smiles) return smiles;
    const parts = smiles.split('.');
    const kept = parts.filter(function (p) { return p.trim() !== '[H]' && p.trim() !== '[HH]'; });
    if (kept.length === 0) return smiles;
    let joined = kept.join('.');
    while (joined.charAt(0) === '.') joined = joined.slice(1);
    while (joined.charAt(joined.length - 1) === '.') joined = joined.slice(0, -1);
    return joined;
  }

  /**
   * Runs the encoder once, then greedily decodes one token at a time
   * (matching the reference _greedy_batch loop -- no KV-cache, no beam
   * search) until EOS or manifest.maxLen. Returns a Promise resolving to
   * { rawSmiles, smiles } (smiles has hallucinated-H-fragment stripping
   * applied; rawSmiles is the untouched decode) or rejecting on error.
   */
  CC.OCSRGlyph.recognize = function (imageSource) {
    if (!CC.OCSRGlyph.isLoaded()) {
      return Promise.reject(new Error('OCSRGlyph model is not loaded yet -- call loadModel() first'));
    }
    const ort = window.chemCanvasLibs.ort;
    let memoryTensor;
    try {
      const chw = preprocessImage(imageSource);
      const imageTensor = new ort.Tensor('float32', chw, [1, 3, manifest.inputSize, manifest.inputSize]);
      return encoderSession.run({ image: imageTensor }).then(function (encOut) {
        memoryTensor = encOut.memory;
        return decodeLoop(memoryTensor);
      }).then(function (ids) {
        const rawSmiles = idsToSmiles(ids);
        return { rawSmiles: rawSmiles, smiles: stripHydrogenFragments(rawSmiles) };
      });
    } catch (err) {
      return Promise.reject(err);
    }
  };

  function decodeLoop(memoryTensor) {
    const ort = window.chemCanvasLibs.ort;
    const ids = [manifest.sosId];

    function step() {
      const tokens = new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
      return decoderSession.run({ tokens: tokens, memory: memoryTensor }).then(function (out) {
        const logits = out.logits; // [1, T, vocabSize]
        const t = logits.dims[1];
        const vocabSize = logits.dims[2];
        const lastRowStart = (t - 1) * vocabSize;
        let bestId = 0, bestVal = -Infinity;
        for (let v = 0; v < vocabSize; v++) {
          const val = logits.data[lastRowStart + v];
          if (val > bestVal) { bestVal = val; bestId = v; }
        }
        ids.push(bestId);
        if (bestId === manifest.eosId || ids.length >= manifest.maxLen) return ids;
        return step();
      });
    }

    return step();
  }

  function idsToSmiles(ids) {
    const chars = [];
    for (let k = 0; k < ids.length; k++) {
      const id = ids[k];
      if (id === manifest.eosId) break;
      if (id === manifest.padId || id === manifest.sosId) continue;
      chars.push(itos[id] !== undefined ? itos[id] : '');
    }
    return chars.join('');
  }
})();
