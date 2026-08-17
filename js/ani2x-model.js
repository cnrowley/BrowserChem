/**
 * ani2x-model.js
 *
 * Loads a converted ANI-2x ensemble (manifest.json + weights.bin, produced
 * by scripts/convert_ani2x_checkpoint.py) and runs it: per-atom AEVs (via
 * ani2x-features.js) through each ensemble member's per-element network,
 * averaged, plus a constant self-energy correction -- and, since geometry
 * optimization needs it, an analytic reverse-mode backprop of that whole
 * pipeline to get per-atom forces (not finite differences: an ANI
 * evaluation is expensive enough, and gets called often enough during
 * minimization, that 6N finite-difference energy evals per step would be
 * far slower than one analytic backward pass).
 *
 * Public API (same shape as CC.NAGL's, so model-registry.js can dispatch
 * to this engine uniformly): loadModel, hasModel, getLoadedModelIds,
 * clearModel, checkCompatibility, energyAndForces, optimizeGeometry.
 *
 * Energy is reported in Hartree (ANI-2x's native unit, same as the
 * self-energies baked into the checkpoint); forces in Hartree/Angstrom.
 */

window.CC = window.CC || {};
CC.ANI = window.CC.ANI || {};

(function () {
  const SUPPORTED_ELEMENTS = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl'];
  const CELU_ALPHA = 0.1; // torchani's TightCELU: CELU with alpha=0.1

  const models = new Map(); // id -> { manifest, weights: Float32Array }

  function tensorView(model, name) {
    const t = model.manifest.tensors[name];
    if (!t) throw new Error('ANI-2x model is missing tensor "' + name + '"');
    return model.weights.subarray(t.offset, t.offset + t.length);
  }

  function celu(x) {
    return x >= 0 ? x : CELU_ALPHA * (Math.exp(x / CELU_ALPHA) - 1);
  }

  function celuDeriv(x) {
    return x >= 0 ? 1 : Math.exp(x / CELU_ALPHA);
  }

  // Linear layer: y = W*x + b, W stored row-major (out, in) matching
  // PyTorch's nn.Linear.weight layout (and numpy's default .tobytes()).
  function linearForward(w, b, out, inDim, x) {
    const y = new Float64Array(out);
    for (let o = 0; o < out; o++) {
      let sum = b[o];
      const base = o * inDim;
      for (let i = 0; i < inDim; i++) sum += w[base + i] * x[i];
      y[o] = sum;
    }
    return y;
  }

  // dL/dx = W^T * dL/dy
  function linearBackward(w, out, inDim, dY) {
    const dX = new Float64Array(inDim);
    for (let o = 0; o < out; o++) {
      const dy = dY[o];
      if (dy === 0) continue;
      const base = o * inDim;
      for (let i = 0; i < inDim; i++) dX[i] += w[base + i] * dy;
    }
    return dX;
  }

  /**
   * Runs one (ensemble member, element) AtomicNetwork forward pass on one
   * atom's AEV, caching pre-activations for the backward pass.
   * layerDims: e.g. [1008, 256, 192, 160, 1] (input, ...hidden, output=1).
   */
  function atomicNetworkForward(model, memberIdx, element, layerDims, aev) {
    const numHidden = layerDims.length - 2; // hidden Linear+activation layers, excluding the final unactivated one
    let h = aev;
    const cache = { preAct: [], hidden: [], layerDims: layerDims };
    for (let i = 0; i < numHidden; i++) {
      const w = tensorView(model, 'm' + memberIdx + '_' + element + '_layer' + i + '_weight');
      const b = tensorView(model, 'm' + memberIdx + '_' + element + '_layer' + i + '_bias');
      const z = linearForward(w, b, layerDims[i + 1], layerDims[i], h);
      const a = new Float64Array(z.length);
      for (let j = 0; j < z.length; j++) a[j] = celu(z[j]);
      cache.preAct.push(z);
      cache.hidden.push(h);
      h = a;
    }
    const wf = tensorView(model, 'm' + memberIdx + '_' + element + '_final_weight');
    const bf = tensorView(model, 'm' + memberIdx + '_' + element + '_final_bias');
    cache.hidden.push(h); // input to final layer
    const out = linearForward(wf, bf, 1, layerDims[layerDims.length - 2], h)[0];
    return { output: out, cache: cache };
  }

  /** Backprop a scalar seed gradient (dL/d(output)) back to dL/d(aev). */
  function atomicNetworkBackward(model, memberIdx, element, cache, dOutput) {
    const layerDims = cache.layerDims;
    const numHidden = layerDims.length - 2;
    const wf = tensorView(model, 'm' + memberIdx + '_' + element + '_final_weight');
    let dH = linearBackward(wf, 1, layerDims[layerDims.length - 2], [dOutput]);

    for (let i = numHidden - 1; i >= 0; i--) {
      const z = cache.preAct[i];
      const dZ = new Float64Array(z.length);
      for (let j = 0; j < z.length; j++) dZ[j] = dH[j] * celuDeriv(z[j]);
      const w = tensorView(model, 'm' + memberIdx + '_' + element + '_layer' + i + '_weight');
      dH = linearBackward(w, layerDims[i + 1], layerDims[i], dZ);
    }
    return dH; // dL/d(aev)
  }

  // ---------- public loading API ----------

  CC.ANI.loadModel = function (id, manifestUrl, weightsUrl) {
    return fetch(manifestUrl).then(function (r) {
      if (!r.ok) throw new Error('failed to fetch ANI-2x manifest: ' + r.status);
      return r.json();
    }).then(function (manifest) {
      return fetch(weightsUrl).then(function (r) {
        if (!r.ok) throw new Error('failed to fetch ANI-2x weights: ' + r.status);
        return r.arrayBuffer();
      }).then(function (buf) {
        models.set(id, { manifest: manifest, weights: new Float32Array(buf) });
        return { id: id, task: manifest.task };
      });
    });
  };

  CC.ANI.hasModel = function (id) {
    return id ? models.has(id) : models.size > 0;
  };

  CC.ANI.getLoadedModelIds = function () {
    return Array.from(models.keys());
  };

  CC.ANI.clearModel = function (id) {
    if (id) models.delete(id);
    else models.clear();
  };

  /**
   * Returns { compatible: boolean, issues: string[] }. ANI-2x only
   * supports neutral molecules built from H/C/N/O/F/S/Cl -- checked
   * directly against the molecule's own formal charges/elements, not
   * against whichever specific model happens to be loaded (every ANI-2x
   * checkpoint has this same element/charge scope).
   */
  CC.ANI.checkCompatibility = function (molecule) {
    if (!molecule || molecule.atoms.size === 0) return { compatible: true, issues: [] };
    const issues = [];
    let totalCharge = 0;
    const badElements = new Set();
    molecule.atoms.forEach(function (atom) {
      totalCharge += atom.charge || 0;
      if (SUPPORTED_ELEMENTS.indexOf(atom.element) === -1) badElements.add(atom.element);
    });
    if (totalCharge !== 0) {
      issues.push('molecule has a net formal charge of ' + (totalCharge > 0 ? '+' : '') + totalCharge + ' -- ANI-2x only supports neutral molecules');
    }
    if (badElements.size > 0) {
      issues.push('unsupported element(s): ' + Array.from(badElements).join(', ') + ' -- ANI-2x only supports H, C, N, O, F, S, Cl');
    }
    return { compatible: issues.length === 0, issues: issues };
  };

  /**
   * atoms3d: array of {element, x, y, z} (Angstrom). Returns
   * { energy (Hartree), forces: [{x,y,z}, ...] (Hartree/Angstrom) }.
   */
  CC.ANI.energyAndForces = function (atoms3d, id) {
    const model = models.get(id);
    if (!model) throw new Error('No ANI-2x model loaded under id "' + id + '"');
    const manifest = model.manifest;
    const n = atoms3d.length;

    const speciesIdx = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const idx = manifest.species.indexOf(atoms3d[i].element);
      if (idx === -1) throw new Error('Element ' + atoms3d[i].element + ' is not supported by this ANI-2x model');
      speciesIdx[i] = idx;
    }

    const positions = atoms3d.map(function (a) { return { x: a.x, y: a.y, z: a.z }; });
    const geometry = CC.ANI.buildGeometry(positions, manifest.aev);
    const aevs = CC.ANI.computeAEV(n, speciesIdx, geometry, manifest.aev);

    const ensembleSize = manifest.ensembleSize;
    const dEdAEV = [];
    for (let i = 0; i < n; i++) dEdAEV.push(new Float64Array(manifest.aev.aevLen));

    let atomicEnergySum = 0;
    const seed = 1 / ensembleSize;
    for (let i = 0; i < n; i++) {
      const element = manifest.species[speciesIdx[i]];
      const layerDims = manifest.networks[element].layerDims;
      for (let m = 0; m < ensembleSize; m++) {
        const fwd = atomicNetworkForward(model, m, element, layerDims, aevs[i]);
        atomicEnergySum += fwd.output * seed;
        const dAev = atomicNetworkBackward(model, m, element, fwd.cache, seed);
        const dest = dEdAEV[i];
        for (let j = 0; j < dest.length; j++) dest[j] += dAev[j];
      }
    }

    let selfEnergySum = 0;
    for (let i = 0; i < n; i++) selfEnergySum += manifest.selfEnergies[speciesIdx[i]];

    const energy = atomicEnergySum + selfEnergySum;

    const posGrad = CC.ANI.backpropAEV(n, speciesIdx, geometry, manifest.aev, dEdAEV);
    const forces = posGrad.map(function (g) { return { x: -g.x, y: -g.y, z: -g.z }; });

    return { energy: energy, forces: forces };
  };

  function yieldToUI() {
    return new Promise(function (resolve) { setTimeout(resolve, 0); });
  }

  /**
   * Adaptive-step gradient-descent geometry optimization driven by
   * CC.ANI.energyAndForces' analytic gradient -- same step-size adaptation
   * style as embed3d.js's classical-force-field minimizer (grow on
   * improvement, shrink on rejection, honest exitReason), but without that
   * minimizer's staged LJ-ramp/torsion-conformer-search machinery: ANI's
   * potential energy surface doesn't have the initial-clash blowup problem
   * that staging exists to avoid, and this starts from whatever geometry
   * is already on screen (classically optimized or not).
   *
   * This optimizer is genuinely slow per iteration (each step needs two
   * full ANI forward+backward passes -- one for the gradient, one to
   * evaluate the trial step), confirmed directly: on a 15-heavy-atom
   * molecule it was still meaningfully improving (gradNorm ~0.01, nowhere
   * near the 1e-5 convergence target) after a full minute, at roughly
   * 3-5 iterations/second. A single call's time/iteration budget alone
   * can't responsibly cover real convergence without either a long
   * blocking wait or a fragile huge default, so this is designed to be
   * CALLED AGAIN to keep going instead: pass opts.iterationOffset (the
   * total iteration count so far) and opts.initialStep (the previous
   * call's returned finalStep) to continue the SAME trajectory -- both
   * the reported iteration numbers and the step-size tuning pick up
   * where the last call left off rather than resetting. See app.js's
   * "Optimize further" button for the actual resume wiring.
   *
   * opts.onProgress now also reports gradNorm (not just energy) every 5
   * iterations, for a live convergence chart -- see convergence-chart.js.
   *
   * Returns { atoms, bonds, energy, gradNorm, exitReason, converged,
   * finalStep } -- finalStep specifically for the "optimize further"
   * resume path above; everything else is the same shape
   * CC.render3D/app.js's renderResult already know how to consume.
   */
  CC.ANI.optimizeGeometry = function (atoms3d, bonds3d, id, opts) {
    opts = opts || {};
    const maxIterations = opts.maxIterations || 300;
    const deadline = opts.deadline || (performance.now() + (opts.timeBudgetMs || 30000));
    const iterationOffset = opts.iterationOffset || 0;

    let positions = atoms3d.map(function (a) { return { element: a.element, x: a.x, y: a.y, z: a.z }; });
    let energy = null;
    let step = opts.initialStep || 0.02;
    let exitReason = 'iteration-limit';
    let lastGradNorm = Infinity;

    let iterationsRun = 0;

    async function run() {
      for (let iter = 0; iter < maxIterations; iter++) {
        iterationsRun = iter + 1;
        if (deadline && performance.now() > deadline) { exitReason = 'deadline'; break; }

        const result = CC.ANI.energyAndForces(positions, id);
        if (energy === null) energy = result.energy;

        let gradNorm = 0;
        for (let i = 0; i < result.forces.length; i++) {
          const f = result.forces[i];
          gradNorm += f.x * f.x + f.y * f.y + f.z * f.z;
        }
        gradNorm = Math.sqrt(gradNorm);
        lastGradNorm = gradNorm;

        if (iter % 5 === 0) {
          await yieldToUI();
          if (opts.onProgress) {
            opts.onProgress({
              iteration: iterationOffset + iter,
              maxIterations: iterationOffset + maxIterations,
              energy: energy,
              gradNorm: gradNorm,
              step: step,
            });
          }
        }

        if (gradNorm < 1e-5) { exitReason = 'gradient-converged'; break; }

        const trial = positions.map(function (p, i) {
          const f = result.forces[i];
          return { element: p.element, x: p.x + step * f.x, y: p.y + step * f.y, z: p.z + step * f.z };
        });
        const trialEnergy = CC.ANI.energyAndForces(trial, id).energy;

        if (trialEnergy < result.energy) {
          const improvement = result.energy - trialEnergy;
          positions = trial;
          energy = trialEnergy;
          step *= 1.2;
          if (improvement < 1e-9) { exitReason = 'energy-plateau'; break; }
        } else {
          step *= 0.5;
          if (step < 1e-8) { exitReason = 'step-too-small'; break; }
        }
      }

      return {
        atoms: positions,
        bonds: bonds3d,
        energy: energy,
        gradNorm: lastGradNorm,
        exitReason: exitReason,
        converged: exitReason === 'gradient-converged' || exitReason === 'energy-plateau',
        finalStep: step,
        iterationsRun: iterationsRun, // this call's own iteration count -- add to iterationOffset for the next resumed call
      };
    }

    return run();
  };

  // See model-adapters.js's header. kind:'geometry' -- optimizeGeometry
  // takes (atoms, bonds, id, opts), not (molecule, id) the way the
  // property-predictor adapters' predict() does.
  CC.ModelAdapters.register('ani2x', {
    kind: 'geometry',
    load: CC.ANI.loadModel,
    unload: CC.ANI.clearModel,
    hasModel: CC.ANI.hasModel,
    getLoadedModelIds: CC.ANI.getLoadedModelIds,
    validate: CC.ANI.checkCompatibility,
    optimize: CC.ANI.optimizeGeometry,
  });
})();
