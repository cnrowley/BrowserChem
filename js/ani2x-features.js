/**
 * ani2x-features.js
 *
 * Builds ANI-2x's atomic environment vectors (AEVs) from Cartesian
 * coordinates + per-atom species indices, and (since geometry optimization
 * needs forces) analytically backpropagates a per-atom dE/dAEV gradient
 * back to a dE/dposition gradient through the same radial/angular
 * symmetry-function formulas.
 *
 * Formulas and constants here are transcribed directly from the real
 * `torchani` package's `AEVComputer`/`ANIRadial`/`ANIAngular` source
 * (torchani 2.8.4) -- not reimplemented from a paper or from memory -- so
 * they match convert_ani2x_checkpoint.py's manifest.json exactly:
 *
 *   - cutoff function: 0.5*cos(pi*r/Rc)+0.5 for r <= Rc, else the pair/
 *     triple is excluded entirely (torchani's neighborlist discards them,
 *     it does not rely on the cutoff function's shape past Rc).
 *   - radial term (per unordered pair a-b, distance r):
 *       0.25 * exp(-etaR*(r-shift_s)^2) * fc(r)   for each of 16 shifts
 *     added into BOTH atoms' AEVs, each indexed by the *other* atom's
 *     species block (torchani's `_collect_radial`).
 *   - angular term (per unordered pair of neighbors {i,k} of some center
 *     atom j, both within the smaller angular cutoff):
 *       mean_r = (r_ji + r_jk)/2
 *       theta  = acos(0.95 * cos(angle at j between j->i and j->k))
 *       term_{s,t} = exp(-etaA*(mean_r-ShfA_s)^2)
 *                    * 2*((1+cos(theta-ShfZ_t))/2)^zeta
 *                    * fc(r_ji) * fc(r_jk)
 *     added into ONLY the center atom j's AEV, indexed by the unordered
 *     species-pair block of {i,k} (torchani's `_collect_angular`). The
 *     0.95 factor before acos is torchani's own numerical-stability fudge
 *     (avoids acos(+-1) exactly) -- reproduced exactly, not an approximation.
 *   - total AEV = [radial_aev, angular_aev] concatenated, per atom.
 *
 * Species-pair -> angular-block-index uses the manifest's own
 * `aev.speciesPairIndex` table (torchani's `triu_index` buffer, read
 * directly by the converter) rather than re-deriving the enumeration
 * order of `torch.triu_indices` here.
 */

window.CC = window.CC || {};
CC.ANI = window.CC.ANI || {};

(function () {
  function cutoffFn(r, rc) {
    return 0.5 * Math.cos(r * (Math.PI / rc)) + 0.5;
  }

  function cutoffFnDeriv(r, rc) {
    return -0.5 * (Math.PI / rc) * Math.sin(r * (Math.PI / rc));
  }

  /**
   * Builds the radial-cutoff pair list and, from it, the angular-cutoff
   * triples list (every unordered pair of neighbors sharing a center atom,
   * both neighbors within the angular cutoff) -- the same geometry both
   * the forward AEV pass and the backward force pass walk.
   *
   * positions: array of {x,y,z}. Returns { pairs, triples }.
   */
  CC.ANI.buildGeometry = function (positions, aevParams) {
    const n = positions.length;
    const rCutR = aevParams.radial.cutoff;
    const rCutA = aevParams.angular.cutoff;

    const pairs = [];
    // neighborsWithinA[atomIdx] = list of {other, r, ux, uy, uz, fc, dfc} with
    // ux/uy/uz the unit vector pointing FROM atomIdx TO other.
    const neighborsWithinA = [];
    for (let i = 0; i < n; i++) neighborsWithinA.push([]);

    for (let a = 0; a < n; a++) {
      for (let b = a + 1; b < n; b++) {
        const dx = positions[a].x - positions[b].x;
        const dy = positions[a].y - positions[b].y;
        const dz = positions[a].z - positions[b].z;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (r > rCutR || r < 1e-6) continue;

        const ux = dx / r, uy = dy / r, uz = dz / r; // points from b to a
        pairs.push({ a: a, b: b, r: r, ux: ux, uy: uy, uz: uz });

        if (r <= rCutA) {
          // From a's perspective, neighbor b is at -[ux,uy,uz]*r away (a->b).
          neighborsWithinA[a].push({ other: b, r: r, ux: -ux, uy: -uy, uz: -uz });
          neighborsWithinA[b].push({ other: a, r: r, ux: ux, uy: uy, uz: uz });
        }
      }
    }

    const triples = [];
    for (let center = 0; center < n; center++) {
      const nbrs = neighborsWithinA[center];
      for (let p = 0; p < nbrs.length; p++) {
        for (let q = p + 1; q < nbrs.length; q++) {
          triples.push({ center: center, i: nbrs[p], k: nbrs[q] });
        }
      }
    }

    return { pairs: pairs, triples: triples };
  };

  /**
   * Forward AEV pass. speciesIdx: Int32Array/array of per-atom species
   * index (0-based, into aevParams' species order). Returns a plain array
   * of per-atom Float64Array(aevLen) AEVs.
   */
  CC.ANI.computeAEV = function (numAtoms, speciesIdx, geometry, aevParams) {
    const radial = aevParams.radial;
    const angular = aevParams.angular;
    const numSpecies = aevParams.numSpecies;
    const numSpeciesPairs = aevParams.numSpeciesPairs;
    const radialLen = aevParams.radialLen; // numSpecies * shifts.length
    const angularLen = aevParams.angularLen; // numSpeciesPairs * shifts.length * sections.length
    const aevLen = aevParams.aevLen;
    const speciesPairIndex = aevParams.speciesPairIndex;
    const rShifts = radial.shifts, rEta = radial.eta, rCut = radial.cutoff;
    const aShifts = angular.shifts, aSections = angular.sections;
    const aEta = angular.eta, aZeta = angular.zeta, aCut = angular.cutoff;
    const numRShifts = rShifts.length;
    const numAShifts = aShifts.length, numASections = aSections.length;
    const angularFeatsPerPair = numAShifts * numASections;

    const aevs = [];
    for (let i = 0; i < numAtoms; i++) aevs.push(new Float64Array(aevLen));

    // ---- radial ----
    for (let p = 0; p < geometry.pairs.length; p++) {
      const pr = geometry.pairs[p];
      const fc = cutoffFn(pr.r, rCut);
      const spA = speciesIdx[pr.a], spB = speciesIdx[pr.b];
      const aevA = aevs[pr.a], aevB = aevs[pr.b];
      const baseForA = spB * numRShifts; // atom a's block is indexed by b's species
      const baseForB = spA * numRShifts;
      for (let s = 0; s < numRShifts; s++) {
        const d = pr.r - rShifts[s];
        const term = 0.25 * Math.exp(-rEta * d * d) * fc;
        aevA[baseForA + s] += term;
        aevB[baseForB + s] += term;
      }
    }

    // ---- angular ----
    for (let t = 0; t < geometry.triples.length; t++) {
      const tr = geometry.triples[t];
      const i = tr.i, k = tr.k;
      const rJi = i.r, rJk = k.r;
      const cosTheta = i.ux * k.ux + i.uy * k.uy + i.uz * k.uz;
      const theta = Math.acos(Math.max(-1, Math.min(1, 0.95 * cosTheta)));
      const meanR = (rJi + rJk) / 2;
      const fcJi = cutoffFn(rJi, aCut), fcJk = cutoffFn(rJk, aCut);
      const cutoffProd = fcJi * fcJk;

      const spI = speciesIdx[i.other], spK = speciesIdx[k.other];
      const pairIdx = speciesPairIndex[spI][spK];
      const aevCenter = aevs[tr.center];
      const base = radialLen + pairIdx * angularFeatsPerPair;

      for (let s = 0; s < numAShifts; s++) {
        const dr = meanR - aShifts[s];
        const radialPart = Math.exp(-aEta * dr * dr);
        for (let sec = 0; sec < numASections; sec++) {
          const x = (1 + Math.cos(theta - aSections[sec])) / 2;
          const angularPart = 2 * Math.pow(x, aZeta);
          aevCenter[base + s * numASections + sec] += radialPart * angularPart * cutoffProd;
        }
      }
    }

    return aevs;
  };

  /**
   * Backward AEV pass: given dE/dAEV per atom (same shape as computeAEV's
   * output), walks the same geometry and returns dE/dposition per atom
   * (array of {x,y,z}, NOT yet negated into a force).
   */
  CC.ANI.backpropAEV = function (numAtoms, speciesIdx, geometry, aevParams, dEdAEV) {
    const radial = aevParams.radial;
    const angular = aevParams.angular;
    const radialLen = aevParams.radialLen;
    const speciesPairIndex = aevParams.speciesPairIndex;
    const rShifts = radial.shifts, rEta = radial.eta, rCut = radial.cutoff;
    const aShifts = angular.shifts, aSections = angular.sections;
    const aEta = angular.eta, aZeta = angular.zeta, aCut = angular.cutoff;
    const numRShifts = rShifts.length;
    const numAShifts = aShifts.length, numASections = aSections.length;
    const angularFeatsPerPair = numAShifts * numASections;

    const grad = [];
    for (let i = 0; i < numAtoms; i++) grad.push({ x: 0, y: 0, z: 0 });

    // ---- radial ----
    for (let p = 0; p < geometry.pairs.length; p++) {
      const pr = geometry.pairs[p];
      const r = pr.r;
      const fc = cutoffFn(r, rCut);
      const dfc = cutoffFnDeriv(r, rCut);
      const spA = speciesIdx[pr.a], spB = speciesIdx[pr.b];
      const dA = dEdAEV[pr.a], dB = dEdAEV[pr.b];
      const baseForA = spB * numRShifts;
      const baseForB = spA * numRShifts;

      let dEdr = 0;
      for (let s = 0; s < numRShifts; s++) {
        const d = r - rShifts[s];
        const gauss = Math.exp(-rEta * d * d);
        const term = 0.25 * gauss * fc;
        const dTermDr = 0.25 * (-2 * rEta * d * gauss * fc + gauss * dfc);
        const dEdTerm = dA[baseForA + s] + dB[baseForB + s];
        dEdr += dEdTerm * dTermDr;
        void term; // (kept only for readability/documentation of the forward value)
      }

      // dr/d(pos_a) = unit vector from b to a = (ux,uy,uz); dr/d(pos_b) = -that
      grad[pr.a].x += dEdr * pr.ux; grad[pr.a].y += dEdr * pr.uy; grad[pr.a].z += dEdr * pr.uz;
      grad[pr.b].x -= dEdr * pr.ux; grad[pr.b].y -= dEdr * pr.uy; grad[pr.b].z -= dEdr * pr.uz;
    }

    // ---- angular ----
    for (let t = 0; t < geometry.triples.length; t++) {
      const tr = geometry.triples[t];
      const i = tr.i, k = tr.k;
      const rJi = i.r, rJk = k.r;
      const cosTheta = i.ux * k.ux + i.uy * k.uy + i.uz * k.uz;
      const clamped095 = Math.max(-1, Math.min(1, 0.95 * cosTheta));
      const theta = Math.acos(clamped095);
      const meanR = (rJi + rJk) / 2;
      const fcJi = cutoffFn(rJi, aCut), fcJk = cutoffFn(rJk, aCut);
      const dfcJi = cutoffFnDeriv(rJi, aCut), dfcJk = cutoffFnDeriv(rJk, aCut);

      const spI = speciesIdx[i.other], spK = speciesIdx[k.other];
      const pairIdx = speciesPairIndex[spI][spK];
      const base = radialLen + pairIdx * angularFeatsPerPair;
      const dCenter = dEdAEV[tr.center];

      let dEdRadial = new Float64Array(numAShifts);
      let dEdAngular = new Float64Array(numASections);
      let dEdCutoffProd = 0;

      for (let s = 0; s < numAShifts; s++) {
        const dr = meanR - aShifts[s];
        const radialPart = Math.exp(-aEta * dr * dr);
        for (let sec = 0; sec < numASections; sec++) {
          const x = (1 + Math.cos(theta - aSections[sec])) / 2;
          const angularPart = 2 * Math.pow(x, aZeta);
          const dEdTerm = dCenter[base + s * numASections + sec];
          const cutoffProd = fcJi * fcJk;

          dEdRadial[s] += dEdTerm * angularPart * cutoffProd;
          dEdAngular[sec] += dEdTerm * radialPart * cutoffProd;
          dEdCutoffProd += dEdTerm * radialPart * angularPart;
        }
      }

      // dE/d(meanR) via the radial part's shift-sum
      let dEdMeanR = 0;
      for (let s = 0; s < numAShifts; s++) {
        const dr = meanR - aShifts[s];
        const radialPart = Math.exp(-aEta * dr * dr);
        dEdMeanR += dEdRadial[s] * radialPart * (-2 * aEta * dr);
      }

      // dE/d(theta) via the angular part's section-sum
      let dEdTheta = 0;
      for (let sec = 0; sec < numASections; sec++) {
        const x = (1 + Math.cos(theta - aSections[sec])) / 2;
        // d/dtheta [2*x^zeta] = 2*zeta*x^(zeta-1) * dx/dtheta, dx/dtheta = -sin(theta-ShfZ)/2
        const dxdtheta = -Math.sin(theta - aSections[sec]) / 2;
        const dAngularDTheta = x > 0 ? 2 * aZeta * Math.pow(x, aZeta - 1) * dxdtheta : 0;
        dEdTheta += dEdAngular[sec] * dAngularDTheta;
      }

      const dEdFcJi = dEdCutoffProd * fcJk;
      const dEdFcJk = dEdCutoffProd * fcJi;

      // theta = acos(0.95*cosTheta) -> dtheta/dcosTheta = -0.95/sqrt(1-(0.95*cosTheta)^2)
      const denom = Math.sqrt(Math.max(1e-12, 1 - clamped095 * clamped095));
      const dThetaDCos = -0.95 / denom;
      const dEdCos = dEdTheta * dThetaDCos;

      // e1 = i's unit vector (center->i), e2 = k's unit vector (center->k)
      const e1x = i.ux, e1y = i.uy, e1z = i.uz;
      const e2x = k.ux, e2y = k.uy, e2z = k.uz;

      // dcosTheta/d(pos_i) = (e2 - cosTheta*e1)/r_ji ; dcosTheta/d(pos_k) = (e1 - cosTheta*e2)/r_jk
      const dCosDI_x = (e2x - cosTheta * e1x) / rJi;
      const dCosDI_y = (e2y - cosTheta * e1y) / rJi;
      const dCosDI_z = (e2z - cosTheta * e1z) / rJi;
      const dCosDK_x = (e1x - cosTheta * e2x) / rJk;
      const dCosDK_y = (e1y - cosTheta * e2y) / rJk;
      const dCosDK_z = (e1z - cosTheta * e2z) / rJk;

      const gi = grad[i.other], gk = grad[k.other], gj = grad[tr.center];

      // meanR terms: dr_ji/d(pos_i)=e1, dr_ji/d(pos_center)=-e1; dr_jk/d(pos_k)=e2, dr_jk/d(pos_center)=-e2
      const halfDEdMeanR = 0.5 * dEdMeanR;
      gi.x += halfDEdMeanR * e1x; gi.y += halfDEdMeanR * e1y; gi.z += halfDEdMeanR * e1z;
      gk.x += halfDEdMeanR * e2x; gk.y += halfDEdMeanR * e2y; gk.z += halfDEdMeanR * e2z;
      gj.x -= halfDEdMeanR * (e1x + e2x); gj.y -= halfDEdMeanR * (e1y + e2y); gj.z -= halfDEdMeanR * (e1z + e2z);

      // cutoff terms
      const fcTermI = dEdFcJi * dfcJi;
      const fcTermK = dEdFcJk * dfcJk;
      gi.x += fcTermI * e1x; gi.y += fcTermI * e1y; gi.z += fcTermI * e1z;
      gk.x += fcTermK * e2x; gk.y += fcTermK * e2y; gk.z += fcTermK * e2z;
      gj.x -= fcTermI * e1x + fcTermK * e2x;
      gj.y -= fcTermI * e1y + fcTermK * e2y;
      gj.z -= fcTermI * e1z + fcTermK * e2z;

      // angle terms
      gi.x += dEdCos * dCosDI_x; gi.y += dEdCos * dCosDI_y; gi.z += dEdCos * dCosDI_z;
      gk.x += dEdCos * dCosDK_x; gk.y += dEdCos * dCosDK_y; gk.z += dEdCos * dCosDK_z;
      gj.x -= dEdCos * (dCosDI_x + dCosDK_x);
      gj.y -= dEdCos * (dCosDI_y + dCosDK_y);
      gj.z -= dEdCos * (dCosDI_z + dCosDK_z);
    }

    return grad;
  };
})();
