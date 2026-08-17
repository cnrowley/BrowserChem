/**
 * geomol-assembly.js
 *
 * Turns per-atom local-structure predictions (geomol-model.js's
 * CC.GeoMol.predictLocalStructures) into a full 3D conformer: which order
 * to walk the molecular graph in (getDihedralPairs, this file), the
 * torsion-angle prediction network for each dihedral pair, and the
 * geometric assembly that stitches local frames together bond by bond.
 * Ring closure (multi-start traversal + Kabsch averaging) is a separate,
 * later piece -- see GEOMOL_INTEGRATION.md.
 *
 * getDihedralPairs is a direct port of model/utils.py's
 * get_dihedral_pairs (+ cycle_utils.py's get_current_cycle_indices):
 * pick one representative directed edge per "interior" bond (both
 * endpoints have more than one neighbor), then expand any bond touching
 * a ring into a full traversal of that ring, consuming (removing) each
 * ring from the working set the first time any of its bonds is visited
 * so a fused ring system's shared atoms don't get double-processed.
 * Uses RDKit's SSSR (via CC.GNN.buildGeomolInput's `rings`) as the ring
 * basis rather than reimplementing networkx's cycle_basis algorithm --
 * both are valid minimal cycle bases for the same graph, but the
 * specific cycles chosen can differ for a fused ring system, which can
 * make this port's traversal order (and therefore its ring-closure
 * averaging path, once that's implemented) diverge from the reference
 * implementation's on such molecules. Validated bit-exact against a live
 * PyTorch run on both an acyclic and a fused-ring test molecule (see
 * GEOMOL_INTEGRATION.md) -- this note is about a difference that could in
 * principle arise on a different ring system, not one that was observed.
 */

window.CC = window.CC || {};
CC.GeoMol = window.CC.GeoMol || {};

(function () {
  function pairKey(a, b) { return a < b ? a + '_' + b : b + '_' + a; }

  CC.GeoMol.getDihedralPairs = function (input) {
    const n = input.numAtoms;
    const degree = new Array(n).fill(0);
    for (let k = 0; k < input.edgeDst.length; k++) degree[input.edgeDst[k]]++;

    const candidatePairs = [];
    for (let k = 0; k < input.edgeSrc.length; k++) {
      const s = input.edgeSrc[k], d = input.edgeDst[k];
      if (degree[s] <= 1 || degree[d] <= 1) continue;
      if (s < d) candidatePairs.push([s, d]);
    }

    const cyclesRemaining = (input.rings || []).map(function (r) { return r.slice(); });

    function getCurrentCycleIndices(idx) {
      let cycleAt = -1;
      for (let i = 0; i < cyclesRemaining.length; i++) {
        if (cyclesRemaining[i].indexOf(idx) !== -1) { cycleAt = i; break; }
      }
      if (cycleAt === -1) return [];
      const cycle = cyclesRemaining.splice(cycleAt, 1)[0];
      const startPos = cycle.indexOf(idx);
      const len = cycle.length;
      const edges = [];
      for (let i = 0; i < len; i++) {
        edges.push([cycle[(startPos + i) % len], cycle[(startPos + i + 1) % len]]);
      }
      return edges;
    }

    const keep = [];
    const sortedKeepSet = new Set();

    candidatePairs.forEach(function (pair) {
      const x = pair[0], y = pair[1];
      const key = pairKey(x, y);
      if (sortedKeepSet.has(key)) return;

      const xInCycle = cyclesRemaining.some(function (c) { return c.indexOf(x) !== -1; });
      const yInCycle = cyclesRemaining.some(function (c) { return c.indexOf(y) !== -1; });

      if (xInCycle && yInCycle) {
        getCurrentCycleIndices(x).forEach(function (e) {
          keep.push(e);
          sortedKeepSet.add(pairKey(e[0], e[1]));
        });
        return;
      }
      if (yInCycle) {
        keep.push([x, y]);
        sortedKeepSet.add(key);
        getCurrentCycleIndices(y).forEach(function (e) {
          keep.push(e);
          sortedKeepSet.add(pairKey(e[0], e[1]));
        });
        return;
      }
      keep.push([x, y]);
      sortedKeepSet.add(key);
    });

    return keep;
  };

  // ---------- geometric assembly (model/inference.py's construct_conformers_acyclic) ----------

  // Any vector perpendicular to u -- standard Gram-Schmidt against a seed
  // axis, falling back from z to x when u is itself nearly parallel to z.
  function arbitraryPerpendicular(u) {
    const seed = Math.abs(u.z) < 0.9 ? { x: 0, y: 0, z: 1 } : { x: 1, y: 0, z: 0 };
    return CC.vec3.normalize(CC.vec3.cross(u, seed));
  }

  // Real GeoMol (inference.py's rotation_matrix_inf_v2) draws this
  // perpendicular *randomly* (torch.rand_like) rather than via a fixed
  // Gram-Schmidt seed. Proven by hand (see the design discussion this
  // port followed) that the specific choice is geometrically irrelevant:
  // the later gamma-fitting step measures the "current" dihedral in
  // whatever frame this constructs and solves for the rotation that
  // makes it match the target v_star, so the final physical torsion
  // angle comes out identical regardless of which valid perpendicular
  // was used here. c_ij and v_star (the only real-model outputs this
  // step depends on) are both independently confirmed frame-independent
  // (see geomol-model.js's predictTorsions) -- this is why a deterministic
  // substitute is valid here without needing to match PyTorch's RNG.
  //
  // Builds the rows of a rotation matrix H such that H @ pY aligns with
  // the local +X axis -- pY is the local-frame coordinate of whichever
  // neighbor slot points toward the *other* central atom in this
  // dihedral pair.
  function rotationFrameFromNeighbor(pY) {
    const h1 = CC.vec3.normalize(pY);
    const eta = arbitraryPerpendicular(h1);
    const h3 = CC.vec3.normalize(CC.vec3.cross(pY, eta));
    const h2 = CC.vec3.scale(CC.vec3.cross(h1, h3), -1);
    return [h1, h2, h3];
  }

  function applyRotation(H, v) {
    return { x: CC.vec3.dot(H[0], v), y: CC.vec3.dot(H[1], v), z: CC.vec3.dot(H[2], v) };
  }

  // Rotation matrix around the local X axis by (gammaCos, gammaSin) --
  // model/inference.py's build_gamma_rotation_inf. This is the one
  // remaining degree of freedom after both fragments' shared bond axis
  // has been aligned to +X -- i.e. the actual torsion angle.
  function buildGammaRotation(gammaCos, gammaSin) {
    return [
      { x: 1, y: 0, z: 0 },
      { x: 0, y: gammaCos, z: -gammaSin },
      { x: 0, y: gammaSin, z: gammaCos },
    ];
  }

  // sin/cos of the dihedral angle p0-p1-p2-p3 -- model/utils.py's
  // batch_dihedrals (atan2-safe form: returns sin/cos directly rather
  // than the angle itself, since every caller here only ever needs
  // those, not the angle).
  function dihedralSinCos(p0, p1, p2, p3) {
    const s1 = CC.vec3.sub(p1, p0);
    const s2 = CC.vec3.sub(p2, p1);
    const s3 = CC.vec3.sub(p3, p2);
    const crossS1S2 = CC.vec3.cross(s1, s2);
    const crossS2S3 = CC.vec3.cross(s2, s3);
    const sinD = CC.vec3.length(s2) * CC.vec3.dot(s1, crossS2S3);
    const cosD = CC.vec3.dot(crossS1S2, crossS2S3);
    const den = CC.vec3.length(crossS1S2) * CC.vec3.length(crossS2S3) + 1e-10;
    return [sinD / den, cosD / den];
  }

  /**
   * Solves the small 2x2 linear system (model/inference.py's
   * calculate_gamma) that fits a single "gamma" rotation-around-the-bond-
   * axis correction, from up to 9 weighted neighbor-pair dihedral
   * observations (c_ij-weighted), to reproduce the predicted torsion
   * v_star = [cos(alpha), sin(alpha)]. `pTPrimeBySlot`/`qZTranslatedBySlot`
   * are each length-3 arrays (the dihedral pair's two "hinge" atoms' other
   * neighbors, already rotated/translated into the shared frame but
   * *before* this gamma correction is applied); `newPY` is the target
   * position of the pair's Y atom in that same frame.
   */
  function calculateGamma(dihedralMask, cIj, vStar, pTPrimeBySlot, qZTranslatedBySlot, newPY) {
    const origin = { x: 0, y: 0, z: 0 };
    let a00 = 0, a01 = 0, a10 = 0, a11 = 0;
    for (let pi = 0; pi < 3; pi++) {
      for (let qi = 0; qi < 3; qi++) {
        const combo = pi * 3 + qi;
        if (!dihedralMask[combo]) continue;
        const sc = dihedralSinCos(pTPrimeBySlot[pi], origin, newPY, qZTranslatedBySlot[qi]);
        const sinC = sc[0], cosC = sc[1];
        const c = cIj[combo];
        a00 += c * cosC; a01 += c * sinC; a10 += c * sinC; a11 += c * -cosC;
      }
    }
    const det = a00 * a11 - a01 * a10 + 1e-10;
    // 2x2 adjugate/det -- matches model/inference.py's explicit
    // [3,1,2,0]*[1,-1,-1,1] reindex exactly (that's [d,-b,-c,a]/det for
    // M=[[a,b],[c,d]]).
    const invA00 = a11 / det, invA01 = -a01 / det, invA10 = -a10 / det, invA11 = a00 / det;
    const vx = invA00 * vStar[0] + invA01 * vStar[1];
    const vy = invA10 * vStar[0] + invA11 * vStar[1];
    const norm = Math.sqrt(vx * vx + vy * vy) + 1e-10;
    return [vx / norm, vy / norm]; // [gammaCos, gammaSin]
  }

  /**
   * Sequential fragment-by-fragment assembly for a molecule with no rings
   * (model/inference.py's construct_conformers_acyclic) -- walks
   * dihedralPairs in order, and for each (x,y) pair: places x's local
   * structure (if x hasn't been placed yet, treats it as a fresh
   * fragment root at the origin; otherwise re-centers x's already-known
   * neighbor positions around its current absolute position), rotates
   * both x's and y's local frames so the shared x-y bond axis aligns to
   * local +X (rotationFrameFromNeighbor), mirrors+translates y's fragment
   * to attach at x's predicted bond vector, then applies the gamma
   * rotation (calculateGamma) that realizes the predicted torsion angle
   * around that shared bond axis.
   *
   * `localStructures` is CC.GeoMol.predictLocalStructures's output
   * (atom index -> array of up to 4 {x,y,z} local coordinates, zero for
   * padding slots); `torsions` is CC.GeoMol.predictTorsions's output for
   * the *same* dihedralPairs array (order must match, index for index).
   * Any dihedral pair whose y-side neighbors are all already placed gets
   * skipped -- that only happens for a ring bond, which this function
   * deliberately doesn't handle (see CC.GeoMol.constructConformers,
   * not yet implemented, for the ring-aware version).
   *
   * Returns an array of {x,y,z}, one per atom (only meaningful for atoms
   * actually reachable from dihedralPairs -- an isolated atom with no
   * dihedral pairs at all, e.g. a lone terminal substituent one bond
   * removed from everything else, is left at the origin).
   */
  CC.GeoMol.constructConformersAcyclic = function (input, dihedralPairs, localStructures, torsions) {
    const n = input.numAtoms;
    const newPos = new Array(n);
    for (let i = 0; i < n; i++) newPos[i] = { x: 0, y: 0, z: 0 };

    let Sx = [];
    let Sy = [];

    dihedralPairs.forEach(function (pair, pairIdx) {
      const xIndex = pair[0], yIndex = pair[1];
      const t = torsions[pairIdx];
      const meta = t.meta;
      const xNbrs = input.neighbors[xIndex], yNbrs = input.neighbors[yIndex];

      // Ring bond (all of y's neighbors already placed) -- not handled here.
      if (yNbrs.length > 0 && yNbrs.every(function (nb) { return Sx.indexOf(nb) !== -1; })) return;

      let pCoords;
      if (Sx.indexOf(xIndex) === -1) {
        Sx = [];
        pCoords = localStructures[xIndex];
        xNbrs.forEach(function (nb, k) { newPos[nb] = pCoords[k]; });
      } else {
        pCoords = [0, 1, 2, 3].map(function (k) {
          return k < xNbrs.length ? CC.vec3.sub(newPos[xNbrs[k]], newPos[xIndex]) : { x: 0, y: 0, z: 0 };
        });
      }

      Sx = Array.from(new Set(Sx.concat([xIndex]).concat(xNbrs)));
      Sy = Sy.concat([yIndex]).concat(yNbrs);

      const pX = newPos[xIndex];
      const newPosSx = Sx.map(function (idx) { return CC.vec3.sub(newPos[idx], pX); });

      const qCoords = localStructures[yIndex];
      yNbrs.forEach(function (nb, k) { newPos[nb] = qCoords[k]; });
      newPos[yIndex] = { x: 0, y: 0, z: 0 };
      const newPosSy = Sy.map(function (idx) { return newPos[idx]; });

      const hXY = rotationFrameFromNeighbor(pCoords[meta.xIdxOfY]);
      const hYX = rotationFrameFromNeighbor(qCoords[meta.yIdxOfX]);

      const newPosSx2 = newPosSx.map(function (v) { return applyRotation(hXY, v); });
      const newPosSy2 = newPosSy.map(function (v) { return applyRotation(hYX, v); });

      const newPY = newPosSx2[Sx.indexOf(yIndex)];

      const newPosSy3 = newPosSy2.map(function (v) {
        return CC.vec3.add({ x: -v.x, y: -v.y, z: v.z }, newPY);
      });

      const pTPrimeBySlot = meta.xRemaining.map(function (slot) {
        const nb = xNbrs[slot];
        if (nb === undefined) return { x: 0, y: 0, z: 0 };
        const idx = Sx.indexOf(nb);
        return idx === -1 ? { x: 0, y: 0, z: 0 } : newPosSx2[idx];
      });
      const qZTranslatedBySlot = meta.yRemaining.map(function (slot) {
        const nb = yNbrs[slot];
        if (nb === undefined) return { x: 0, y: 0, z: 0 };
        const idx = Sy.indexOf(nb);
        return idx === -1 ? { x: 0, y: 0, z: 0 } : newPosSy3[idx];
      });

      const gamma = calculateGamma(meta.dihedralMask, t.cIj, t.vStar, pTPrimeBySlot, qZTranslatedBySlot, newPY);
      const hGamma = buildGammaRotation(gamma[0], gamma[1]);
      const newPosSx3 = newPosSx2.map(function (v) { return applyRotation(hGamma, v); });

      Sy.forEach(function (idx, k) { newPos[idx] = newPosSy3[k]; });
      Sx.forEach(function (idx, k) { newPos[idx] = newPosSx3[k]; });

      Sx = Array.from(new Set(Sx.concat(Sy)));
      Sy = [];
    });

    return newPos;
  };

  // ---------- Kabsch alignment (needed for ring closure) ----------

  // Cyclic Jacobi eigenvalue algorithm for a symmetric 3x3 matrix --
  // a new numerical primitive this port hasn't needed anywhere else
  // (nothing else in GeoMol requires an SVD). Standard, textbook
  // formulation (Golub & Van Loan / Numerical Recipes' jacobi routine):
  // repeatedly zero the largest off-diagonal entries via plane rotations
  // until convergence. Reliable for 3x3 -- this only ever runs a handful
  // of sweeps for a matrix this small.
  function jacobiEigenSymmetric3x3(Ain) {
    const a = [Ain[0].slice(), Ain[1].slice(), Ain[2].slice()];
    const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const pairs = [[0, 1], [0, 2], [1, 2]];
    for (let sweep = 0; sweep < 60; sweep++) {
      const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
      if (off < 1e-14) break;
      for (let pi = 0; pi < pairs.length; pi++) {
        const p = pairs[pi][0], q = pairs[pi][1];
        const apq = a[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        const app = a[p][p], aqq = a[q][q];
        a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        a[p][q] = 0; a[q][p] = 0;
        for (let k = 0; k < 3; k++) {
          if (k === p || k === q) continue;
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq; a[p][k] = a[k][p];
          a[k][q] = s * akp + c * akq; a[q][k] = a[k][q];
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
    return { eigenvalues: [a[0][0], a[1][1], a[2][2]], eigenvectors: v };
  }

  // 3x3 SVD (H = U * diag(S) * V^T) via eigendecomposition of H^T H --
  // V's columns are the eigenvectors (right singular vectors), S their
  // square roots, U's columns recovered as H*v_i/sigma_i and
  // Gram-Schmidt-orthonormalized to stay numerically sound when a
  // singular value is at or near zero (a real possibility here: a
  // perfectly planar or otherwise degenerate small set of ring atoms).
  function svd3x3(H) {
    const HtH = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += H[k][i] * H[k][j];
      HtH[i][j] = s;
    }
    const eig = jacobiEigenSymmetric3x3(HtH);
    const order = [0, 1, 2].sort(function (a, b) { return eig.eigenvalues[b] - eig.eigenvalues[a]; });
    const V = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const S = [0, 0, 0];
    for (let newIdx = 0; newIdx < 3; newIdx++) {
      const oldIdx = order[newIdx];
      S[newIdx] = Math.sqrt(Math.max(0, eig.eigenvalues[oldIdx]));
      for (let r = 0; r < 3; r++) V[r][newIdx] = eig.eigenvectors[r][oldIdx];
    }

    function vnorm(v) { return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
    function vnormalize(v) { const n = vnorm(v); return n > 1e-10 ? [v[0] / n, v[1] / n, v[2] / n] : v; }
    function vdot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
    function vsub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
    function vscale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }

    const U = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let col = 0; col < 3; col++) {
      const v = [V[0][col], V[1][col], V[2][col]];
      const hv = [0, 0, 0];
      for (let r = 0; r < 3; r++) { let s = 0; for (let c = 0; c < 3; c++) s += H[r][c] * v[c]; hv[r] = s; }
      if (S[col] > 1e-10) for (let r = 0; r < 3; r++) U[r][col] = hv[r] / S[col];
    }

    let u0 = [U[0][0], U[1][0], U[2][0]];
    if (vnorm(u0) < 1e-10) u0 = [1, 0, 0];
    u0 = vnormalize(u0);
    let u1 = vsub([U[0][1], U[1][1], U[2][1]], vscale(u0, vdot(u0, [U[0][1], U[1][1], U[2][1]])));
    if (vnorm(u1) < 1e-10) {
      const seed = Math.abs(u0[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
      u1 = vsub(seed, vscale(u0, vdot(u0, seed)));
    }
    u1 = vnormalize(u1);
    const u2 = [u0[1] * u1[2] - u0[2] * u1[1], u0[2] * u1[0] - u0[0] * u1[2], u0[0] * u1[1] - u0[1] * u1[0]];
    for (let r = 0; r < 3; r++) { U[r][0] = u0[r]; U[r][1] = u1[r]; U[r][2] = u2[r]; }

    return { U: U, S: S, V: V };
  }

  function det3(M) {
    return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
      - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
      + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  }

  /**
   * Standard Kabsch algorithm (cycle_utils.py's align_coords_Kabsch):
   * finds the rigid rotation+translation that best-aligns `moving` onto
   * `fixed` (both arrays of {x,y,z}, same length) using only the points
   * where `mask[i]` is true to *fit* the transform, then applies it to
   * every point in `moving`. Includes the reflection guard
   * (d = sign(det(V U^T))) that forces a proper rotation rather than a
   * mirror image, even when the naive SVD solution would flip handedness
   * -- a real possibility for a small/near-planar fit set.
   */
  function kabschAlign(moving, fixed, mask) {
    const idxs = [];
    for (let i = 0; i < mask.length; i++) if (mask[i]) idxs.push(i);

    function centroidOf(pts) {
      const c = { x: 0, y: 0, z: 0 };
      idxs.forEach(function (i) { c.x += pts[i].x; c.y += pts[i].y; c.z += pts[i].z; });
      c.x /= idxs.length; c.y /= idxs.length; c.z /= idxs.length;
      return c;
    }
    const fixedCentroid = centroidOf(fixed);
    const movingCentroid = centroidOf(moving);

    const H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    idxs.forEach(function (i) {
      const p = [moving[i].x - movingCentroid.x, moving[i].y - movingCentroid.y, moving[i].z - movingCentroid.z];
      const q = [fixed[i].x - fixedCentroid.x, fixed[i].y - fixedCentroid.y, fixed[i].z - fixedCentroid.z];
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[r][c] += p[r] * q[c];
    });

    const svd = svd3x3(H);
    const U = svd.U, V = svd.V;
    const VUt = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += V[r][k] * U[c][k];
      VUt[r][c] = s;
    }
    const d = det3(VUt) >= 0 ? 1 : -1;

    const R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      R[r][c] = V[r][0] * U[c][0] + V[r][1] * U[c][1] + d * V[r][2] * U[c][2];
    }

    function applyR(v) {
      return {
        x: R[0][0] * v.x + R[0][1] * v.y + R[0][2] * v.z,
        y: R[1][0] * v.x + R[1][1] * v.y + R[1][2] * v.z,
        z: R[2][0] * v.x + R[2][1] * v.y + R[2][2] * v.z,
      };
    }
    const rotatedCentroid = applyR(movingCentroid);
    const b = {
      x: fixedCentroid.x - rotatedCentroid.x,
      y: fixedCentroid.y - rotatedCentroid.y,
      z: fixedCentroid.z - rotatedCentroid.z,
    };

    return moving.map(function (v) {
      const rv = applyR(v);
      return { x: rv.x + b.x, y: rv.y + b.y, z: rv.z + b.z };
    });
  }

  // ---------- ring closure (model/inference.py's smooth_cycle_coords) ----------

  /**
   * Builds cycleLen independent candidate assemblies of one ring -- one
   * starting at each possible rotational offset around the ring -- by
   * walking cycleLen-1 sequential dihedral-pair placement steps per
   * candidate (the same per-pair mechanics as the acyclic assembly:
   * build both rotation frames, mirror+translate, gamma-fit), all
   * starting-offsets processed in lockstep. A ring built by sequential
   * placement doesn't self-consistently close on its own -- different
   * starting points accumulate slightly different prediction error by
   * the time they reach the "closing" bond -- so this Kabsch-aligns
   * every candidate but the first onto the first (using only the
   * genuine ring atoms to fit the alignment, not their exocyclic
   * substituents) and averages them together. Each candidate excludes
   * its own final ("closing seam") atoms from its own contribution to
   * the average, since those were placed last in that candidate's own
   * sequence and are the least cross-validated there -- other
   * candidates, which place those same physical atoms mid-sequence
   * instead of last, cover them instead.
   *
   * `dihedralPairs`/`torsions` must be the full, parallel arrays this
   * ring's own cycleLen consecutive edges were pulled from (a property
   * getDihedralPairs's ring expansion guarantees); `cycleStartIdx` is
   * where they begin. Returns { avgCoords, avgIndices } -- avgIndices is
   * the ring's own canonical atom-index list (ring atoms plus their
   * immediate exocyclic substituents), avgCoords the corresponding
   * averaged local positions (still relative to whatever this ring's own
   * first-processed atom's frame ended up as -- the caller is
   * responsible for re-centering/aligning this into the rest of the
   * molecule, same as CC.GeoMol.constructConformers does).
   */
  function smoothCycleCoords(input, localStructures, dihedralPairs, torsions, cycleStartIdx, cycleLen) {
    const cyclePairs = dihedralPairs.slice(cycleStartIdx, cycleStartIdx + cycleLen);
    const cycleTorsions = torsions.slice(cycleStartIdx, cycleStartIdx + cycleLen);
    const n = input.numAtoms;

    const cyclePos = [];
    for (let t = 0; t < cycleLen; t++) {
      const arr = new Array(n);
      for (let i = 0; i < n; i++) arr[i] = { x: 0, y: 0, z: 0 };
      cyclePos.push(arr);
    }
    const SxCycle = [], SyCycle = [];
    for (let t = 0; t < cycleLen; t++) { SxCycle.push([]); SyCycle.push([]); }

    let cycleMask = null; // built on the final step -- cycleMask[t] = Set of atom indices traversal t excludes from its own average contribution

    for (let step = 0; step < cycleLen - 1; step++) {
      const stepPairs = [], stepTorsions = [], xNbrsByT = [], yNbrsByT = [];
      for (let t = 0; t < cycleLen; t++) {
        const srcIdx = (t + step) % cycleLen;
        stepPairs.push(cyclePairs[srcIdx]);
        stepTorsions.push(cycleTorsions[srcIdx]);
        xNbrsByT.push(input.neighbors[cyclePairs[srcIdx][0]]);
        yNbrsByT.push(input.neighbors[cyclePairs[srcIdx][1]]);
      }

      const pCoordsByT = [];
      for (let t = 0; t < cycleLen; t++) {
        const xIndex = stepPairs[t][0];
        if (step === 0) {
          const pCoords = localStructures[xIndex];
          pCoordsByT.push(pCoords);
          xNbrsByT[t].forEach(function (nb, k) { cyclePos[t][nb] = pCoords[k]; });
        } else {
          const xNbrs = xNbrsByT[t];
          pCoordsByT.push([0, 1, 2, 3].map(function (k) {
            return k < xNbrs.length ? CC.vec3.sub(cyclePos[t][xNbrs[k]], cyclePos[t][xIndex]) : { x: 0, y: 0, z: 0 };
          }));
        }
      }

      for (let t = 0; t < cycleLen; t++) {
        SxCycle[t] = Array.from(new Set(SxCycle[t].concat([stepPairs[t][0]]).concat(xNbrsByT[t])));
        SyCycle[t] = SyCycle[t].concat([stepPairs[t][1]]).concat(yNbrsByT[t]);
      }

      const newPosSxByT = SxCycle.map(function (sx, t) {
        const pX = cyclePos[t][stepPairs[t][0]];
        return sx.map(function (idx) { return CC.vec3.sub(cyclePos[t][idx], pX); });
      });

      const qCoordsByT = [];
      for (let t = 0; t < cycleLen; t++) {
        const yIndex = stepPairs[t][1];
        const qCoords = localStructures[yIndex];
        qCoordsByT.push(qCoords);
        yNbrsByT[t].forEach(function (nb, k) { cyclePos[t][nb] = qCoords[k]; });
        cyclePos[t][yIndex] = { x: 0, y: 0, z: 0 };
      }
      const newPosSyByT = SyCycle.map(function (sy, t) { return sy.map(function (idx) { return cyclePos[t][idx]; }); });

      const hXYByT = pCoordsByT.map(function (pCoords, t) { return rotationFrameFromNeighbor(pCoords[stepTorsions[t].meta.xIdxOfY]); });
      const hYXByT = qCoordsByT.map(function (qCoords, t) { return rotationFrameFromNeighbor(qCoords[stepTorsions[t].meta.yIdxOfX]); });

      const newPosSx2ByT = newPosSxByT.map(function (pts, t) { return pts.map(function (v) { return applyRotation(hXYByT[t], v); }); });
      const newPosSy2ByT = newPosSyByT.map(function (pts, t) { return pts.map(function (v) { return applyRotation(hYXByT[t], v); }); });

      for (let t = 0; t < cycleLen; t++) {
        const yIndex = stepPairs[t][1];
        const newPY = newPosSx2ByT[t][SxCycle[t].indexOf(yIndex)];
        const newPosSy3 = newPosSy2ByT[t].map(function (v) { return CC.vec3.add({ x: -v.x, y: -v.y, z: v.z }, newPY); });

        const meta = stepTorsions[t].meta;
        const xNbrs = xNbrsByT[t], yNbrs = yNbrsByT[t];
        const pTPrimeBySlot = meta.xRemaining.map(function (slot) {
          const nb = xNbrs[slot];
          if (nb === undefined) return { x: 0, y: 0, z: 0 };
          const idx = SxCycle[t].indexOf(nb);
          return idx === -1 ? { x: 0, y: 0, z: 0 } : newPosSx2ByT[t][idx];
        });
        const qZTranslatedBySlot = meta.yRemaining.map(function (slot) {
          const nb = yNbrs[slot];
          if (nb === undefined) return { x: 0, y: 0, z: 0 };
          const idx = SyCycle[t].indexOf(nb);
          return idx === -1 ? { x: 0, y: 0, z: 0 } : newPosSy3[idx];
        });

        const gamma = calculateGamma(meta.dihedralMask, stepTorsions[t].cIj, stepTorsions[t].vStar, pTPrimeBySlot, qZTranslatedBySlot, newPY);
        const hGamma = buildGammaRotation(gamma[0], gamma[1]);
        const newPosSx3 = newPosSx2ByT[t].map(function (v) { return applyRotation(hGamma, v); });

        SyCycle[t].forEach(function (idx, k) { cyclePos[t][idx] = newPosSy3[k]; });
        SxCycle[t].forEach(function (idx, k) { cyclePos[t][idx] = newPosSx3[k]; });
        SxCycle[t] = Array.from(new Set(SxCycle[t].concat(SyCycle[t])));
      }

      if (step < cycleLen - 2) {
        for (let t = 0; t < cycleLen; t++) SyCycle[t] = [];
      } else {
        cycleMask = [];
        for (let t = 0; t < cycleLen; t++) {
          const excluded = new Set();
          const yIndex = stepPairs[t][1];
          excluded.add(yIndex);
          const meta = stepTorsions[t].meta;
          yNbrsByT[t].forEach(function (nb, k) { if (k !== meta.yIdxOfX) excluded.add(nb); });
          cycleMask.push(excluded);
        }
      }
    }

    const refAtoms = SxCycle[0].slice();
    const refPos = refAtoms.map(function (idx) { return cyclePos[0][idx]; });

    const cycleAtomsSet = new Set();
    cyclePairs.forEach(function (p) { cycleAtomsSet.add(p[0]); cycleAtomsSet.add(p[1]); });
    const rmsdMask = refAtoms.map(function (idx) { return cycleAtomsSet.has(idx); });

    const sum = refAtoms.map(function () { return { x: 0, y: 0, z: 0 }; });
    const count = refAtoms.map(function () { return 0; });

    refAtoms.forEach(function (idx, k) {
      if (cycleMask[0].has(idx)) return;
      sum[k] = CC.vec3.add(sum[k], refPos[k]);
      count[k] += 1;
    });

    for (let t = 1; t < cycleLen; t++) {
      const movingPos = refAtoms.map(function (idx) { return cyclePos[t][idx]; });
      const aligned = kabschAlign(movingPos, refPos, rmsdMask);
      refAtoms.forEach(function (idx, k) {
        if (cycleMask[t].has(idx)) return;
        sum[k] = CC.vec3.add(sum[k], aligned[k]);
        count[k] += 1;
      });
    }

    const avgCoords = refAtoms.map(function (idx, k) {
      const c = count[k] > 0 ? count[k] : 1;
      return { x: sum[k].x / c, y: sum[k].y / c, z: sum[k].z / c };
    });

    return { avgCoords: avgCoords, avgIndices: refAtoms };
  }

  // ---------- full assembly, ring-aware (model/inference.py's construct_conformers) ----------

  /**
   * Full sequential assembly, handling both acyclic dihedral pairs
   * (same mechanics as CC.GeoMol.constructConformersAcyclic) and ring
   * bonds (via smoothCycleCoords above). Walks dihedralPairs in order,
   * checking each pair's endpoints against a *fresh* copy of the
   * molecule's rings (independent of whatever getDihedralPairs' own
   * traversal-ordering pass already consumed):
   *   - both endpoints still in an unconsumed ring: this is one of that
   *     ring's own edges -- run smoothCycleCoords for the whole ring in
   *     one shot, splice its averaged coordinates in (Kabsch-aligning
   *     onto whatever's already placed, for a fused ring system's second
   *     ring), then skip the ring's remaining already-expanded edges.
   *   - only the y-side endpoint is in an unconsumed ring: this is an
   *     acyclic bond *entering* a ring from outside -- place x normally,
   *     but instead of placing y from its own local structure, run
   *     smoothCycleCoords for y's whole ring and attach that.
   *   - neither endpoint in any (remaining) ring: identical to
   *     CC.GeoMol.constructConformersAcyclic's per-pair logic.
   */
  CC.GeoMol.constructConformers = function (input, dihedralPairs, localStructures, torsions) {
    const n = input.numAtoms;
    const newPos = new Array(n);
    for (let i = 0; i < n; i++) newPos[i] = { x: 0, y: 0, z: 0 };

    const cyclesRemaining = (input.rings || []).map(function (r) { return r.slice(); });
    function popCycleContaining(idx) {
      for (let i = 0; i < cyclesRemaining.length; i++) {
        if (cyclesRemaining[i].indexOf(idx) !== -1) return cyclesRemaining.splice(i, 1)[0];
      }
      return null;
    }
    function anyContains(idx) {
      return cyclesRemaining.some(function (c) { return c.indexOf(idx) !== -1; });
    }

    let Sx = [];
    let Sy = [];
    let inCycle = 0;

    for (let i = 0; i < dihedralPairs.length; i++) {
      const xIndex = dihedralPairs[i][0], yIndex = dihedralPairs[i][1];
      let cycleAdded = false;

      if (inCycle) inCycle -= 1;
      if (inCycle) continue;

      const xInCycle = anyContains(xIndex);
      const yInCycle = anyContains(yIndex);

      if (xInCycle && yInCycle) {
        const ring = popCycleContaining(xIndex);
        const cycleLen = ring.length;
        const result = smoothCycleCoords(input, localStructures, dihedralPairs, torsions, i, cycleLen);

        if (Sx.indexOf(xIndex) === -1) {
          result.avgIndices.forEach(function (idx, k) { newPos[idx] = result.avgCoords[k]; });
          Sx = [];
        } else {
          const fixedForRing = result.avgIndices.map(function (idx) { return Sx.indexOf(idx) !== -1 ? newPos[idx] : { x: 0, y: 0, z: 0 }; });
          const maskForRing = result.avgIndices.map(function (idx) { return Sx.indexOf(idx) !== -1; });
          const alignedRing = kabschAlign(result.avgCoords, fixedForRing, maskForRing);
          result.avgIndices.forEach(function (idx, k) { newPos[idx] = alignedRing[k]; });
        }

        Sx = Array.from(new Set(Sx.concat(result.avgIndices)));
        inCycle = cycleLen;
        continue;
      }

      let cycleIndices = null;
      if (yInCycle) {
        cycleIndices = popCycleContaining(yIndex);
        cycleAdded = true;
        inCycle = cycleIndices.length + 1;
      }

      const xNbrs = input.neighbors[xIndex], yNbrs = input.neighbors[yIndex];
      const t = torsions[i];
      const meta = t.meta;

      let pCoords;
      if (Sx.indexOf(xIndex) === -1) {
        Sx = [];
        pCoords = localStructures[xIndex];
        xNbrs.forEach(function (nb, k) { newPos[nb] = pCoords[k]; });
      } else {
        pCoords = [0, 1, 2, 3].map(function (k) {
          return k < xNbrs.length ? CC.vec3.sub(newPos[xNbrs[k]], newPos[xIndex]) : { x: 0, y: 0, z: 0 };
        });
      }

      Sx = Array.from(new Set(Sx.concat([xIndex]).concat(xNbrs)));
      Sy = Sy.concat([yIndex]).concat(yNbrs);

      const pX = newPos[xIndex];
      const newPosSx = Sx.map(function (idx) { return CC.vec3.sub(newPos[idx], pX); });

      let qCoords, newPosSy;
      if (cycleAdded) {
        const ringResult = smoothCycleCoords(input, localStructures, dihedralPairs, torsions, i + 1, cycleIndices.length);
        const yPosInRing = ringResult.avgCoords[ringResult.avgIndices.indexOf(yIndex)];
        const recentered = ringResult.avgCoords.map(function (v) { return CC.vec3.sub(v, yPosInRing); });
        qCoords = [0, 1, 2, 3].map(function (k) {
          if (k >= yNbrs.length) return { x: 0, y: 0, z: 0 };
          const idx = ringResult.avgIndices.indexOf(yNbrs[k]);
          return idx === -1 ? { x: 0, y: 0, z: 0 } : recentered[idx];
        });
        Sy = ringResult.avgIndices.slice();
        newPosSy = recentered;
      } else {
        qCoords = localStructures[yIndex];
        yNbrs.forEach(function (nb, k) { newPos[nb] = qCoords[k]; });
        newPos[yIndex] = { x: 0, y: 0, z: 0 };
        newPosSy = Sy.map(function (idx) { return newPos[idx]; });
      }

      const hXY = rotationFrameFromNeighbor(pCoords[meta.xIdxOfY]);
      const hYX = rotationFrameFromNeighbor(qCoords[meta.yIdxOfX]);

      const newPosSx2 = newPosSx.map(function (v) { return applyRotation(hXY, v); });
      const newPosSy2 = newPosSy.map(function (v) { return applyRotation(hYX, v); });

      const newPY = newPosSx2[Sx.indexOf(yIndex)];
      const newPosSy3 = newPosSy2.map(function (v) { return CC.vec3.add({ x: -v.x, y: -v.y, z: v.z }, newPY); });

      const pTPrimeBySlot = meta.xRemaining.map(function (slot) {
        const nb = xNbrs[slot];
        if (nb === undefined) return { x: 0, y: 0, z: 0 };
        const idx = Sx.indexOf(nb);
        return idx === -1 ? { x: 0, y: 0, z: 0 } : newPosSx2[idx];
      });
      const qZTranslatedBySlot = meta.yRemaining.map(function (slot) {
        const nb = yNbrs[slot];
        if (nb === undefined) return { x: 0, y: 0, z: 0 };
        const idx = Sy.indexOf(nb);
        return idx === -1 ? { x: 0, y: 0, z: 0 } : newPosSy3[idx];
      });

      const gamma = calculateGamma(meta.dihedralMask, t.cIj, t.vStar, pTPrimeBySlot, qZTranslatedBySlot, newPY);
      const hGamma = buildGammaRotation(gamma[0], gamma[1]);
      const newPosSx3 = newPosSx2.map(function (v) { return applyRotation(hGamma, v); });

      Sy.forEach(function (idx, k) { newPos[idx] = newPosSy3[k]; });
      Sx.forEach(function (idx, k) { newPos[idx] = newPosSx3[k]; });

      Sx = Array.from(new Set(Sx.concat(Sy)));
      Sy = [];
    }

    return newPos;
  };

  // ---------- public entry point ----------

  const BOND_TYPE_TO_ORDER = { single: 1, double: 2, triple: 3, aromatic: 4 }; // molfile V2000 bond order codes -- 4 is a real, valid V2000 code for "aromatic"

  /**
   * GeoMol needs at least a 4-atom path (matching the real featurizer's
   * own `HasSubstructMatch([*]~[*]~[*]~[*])` + `N >= 4` checks) -- there
   * has to be at least one dihedral to predict, or getDihedralPairs comes
   * back empty and CC.GeoMol.constructConformers would leave every atom
   * at its zero-initialized default position instead of a real
   * structure. Checked directly (build the real input, run the real
   * pair-finder) rather than guessed from atom/bond counts, since the
   * exact substructure-match semantics aren't worth reimplementing
   * separately just for a pre-flight check.
   */
  CC.GeoMol.checkCompatibility = function (molecule) {
    if (!molecule || molecule.atoms.size === 0) return { compatible: true, issues: [] };
    const issues = [];
    try {
      const input = CC.GNN.buildGeomolInput(molecule);
      if (input.numAtoms < 4) {
        issues.push('fewer than 4 atoms (including implicit hydrogens) -- GeoMol needs at least one 4-atom chain to predict a torsion for');
      } else if (CC.GeoMol.getDihedralPairs(input).length === 0) {
        issues.push('no bond with a real substituent on each side -- GeoMol has no torsion to predict for this structure');
      }
    } catch (err) {
      issues.push('could not featurize this structure for GeoMol: ' + err.message);
    }
    return { compatible: issues.length === 0, issues: issues };
  };

  /**
   * Runs the full pipeline -- featurize, GNN encode, predict local
   * structure and torsions, assemble (ring-aware) -- and returns
   * {atoms, bonds, energy, converged} in the same shape
   * CC.buildInitial3D/CC.optimize3D's results already have, so
   * app.js/CC.render3D can use it as a drop-in alternative 3D generation
   * method. `energy`/`converged` are always null/true -- GeoMol doesn't
   * produce an energy value the way the classical force field or ANI-2x
   * do (it's a single learned forward pass, not an iterative
   * minimization), so there's nothing meaningful to report there.
   *
   * Each call draws fresh Gaussian noise (see geomol-model.js's embed/
   * predictTorsions), so repeated calls on the same molecule give
   * genuinely different conformers -- the same "ensemble" behavior
   * that's the whole point of the real model.
   */
  CC.GeoMol.generateConformer = function (molecule, modelId, opts) {
    opts = opts || {};
    if (!molecule || molecule.isEmpty()) return { atoms: [], bonds: [], energy: null, converged: true };

    const model = CC.GeoMol.getModel(modelId);
    if (!model) throw new Error('No GeoMol model loaded under id "' + modelId + '"');

    const input = CC.GNN.buildGeomolInput(molecule);
    if (input.numAtoms === 0) return { atoms: [], bonds: [], energy: null, converged: true };

    const embedResult = CC.GeoMol.embed(model, input, null, null, opts.rng);
    const dihedralPairs = CC.GeoMol.getDihedralPairs(input);
    const torsions = CC.GeoMol.predictTorsions(model, input, embedResult.x2, embedResult.hMol, dihedralPairs, null, opts.rng);
    const localStructures = CC.GeoMol.predictLocalStructures(model, input, embedResult.x1, input.chiralTag);
    const positions = CC.GeoMol.constructConformers(input, dihedralPairs, localStructures, torsions);

    const atoms = positions.map(function (p, i) {
      return { element: input.elementByIndex[i], x: p.x, y: p.y, z: p.z };
    });

    const bonds = [];
    const seenBonds = new Set();
    for (let k = 0; k < input.edgeSrc.length; k++) {
      const s = input.edgeSrc[k], d = input.edgeDst[k];
      const key = pairKey(s, d);
      if (seenBonds.has(key)) continue;
      seenBonds.add(key);
      const typeIdx = input.edgeAttr[k].indexOf(1);
      const bondType = ['single', 'double', 'triple', 'aromatic'][typeIdx];
      bonds.push({ a1: s, a2: d, order: BOND_TYPE_TO_ORDER[bondType] });
    }

    return { atoms: atoms, bonds: bonds, energy: null, converged: true };
  };

  // Exposed only for GEOMOL_INTEGRATION.md's self-test (recovering a
  // known rotation) -- not part of the real assembly pipeline's public
  // surface, so not documented as a stable API.
  CC.GeoMol._internal = { svd3x3: svd3x3, kabschAlign: kabschAlign };

  // See model-adapters.js's header. kind:'geometry' -- generateConformer
  // takes (molecule, id, opts), a single learned prediction rather than
  // an iterative optimize() like ani2x's adapter.
  CC.ModelAdapters.register('geomol', {
    kind: 'geometry',
    load: CC.GeoMol.loadModel,
    unload: CC.GeoMol.clearModel,
    hasModel: CC.GeoMol.hasModel,
    getLoadedModelIds: CC.GeoMol.getLoadedModelIds,
    validate: CC.GeoMol.checkCompatibility,
    generate: CC.GeoMol.generateConformer,
  });
})();
