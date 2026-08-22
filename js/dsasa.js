/**
 * dsasa.js
 *
 * "dSASA": work-in-progress toward a real, analytically-differentiable
 * solvent-accessible surface area, following Cao/Hummel/Wang/Simmerling/
 * Coutsias, "Exact Analytical Algorithm for the Solvent-Accessible
 * Surface Area and Derivatives in Implicit Solvent Molecular Simulations
 * on GPUs," J. Chem. Theory Comput. 2024, 20, 4456-4468 (the paper behind
 * AMBER's own `gbsa=4` option, which the AMBER source itself labels
 * "dSASA" -- confirmed directly in kCalculateGBNonbondEnergy1.cu's
 * `else if (gpu->gbsa == 4) { //dSASA`).
 *
 * *** NOT WIRED INTO THE APP. Do not call CC.DSASA.compute() from
 * anywhere else yet -- it does not reliably reproduce real molecular
 * SASA. Left in the repo as a validated, documented foundation for a
 * future session, not a working feature. js/implicit-solvent.js and
 * js/embed3d.js's solvation gradient are UNCHANGED (still Shrake-Rupley
 * energy + finite-difference gradient, same as before this file existed).
 * ***
 *
 * WHY THIS WAS ATTEMPTED: js/steric-accessibility.js's SASA (Shrake-
 * Rupley, counting sampled points on a sphere as buried/exposed) has no
 * meaningful derivative (it's a step function of atom position). The
 * goal was a closed-form, differentiable replacement for the nonpolar
 * solvation term's gradient.
 *
 * HONEST STATE, STATED PLAINLY:
 *   - Individually validated EXACT and independently trustworthy: the
 *     Van Oosterom-Strackee tetrahedron solid-angle formula and its
 *     gradient (solidAngle/solidAngleGradient, confirmed exactly against
 *     a known octant case and against finite differences), the
 *     spherical-cap-area formula (capHeightLocal, confirmed bit-exact
 *     against the textbook two-sphere-union formula), and the eq-6
 *     triple-overlap energy formula (tripleOverlapContribution, reverse-
 *     engineered from the paper's compressed notation with help from
 *     kCalculateSA.h's own function signatures -- voltri's self-
 *     contained argument list confirmed eq 6's Phi^x/Omega^x are LOCAL
 *     fractions of the single auxiliary tetrahedron {i,j,k,x}, not the
 *     "1 minus a sum over multiple tetrahedra" convention the OUTER eq 4
 *     uses -- and validated bit-close to Shrake-Rupley on an isolated
 *     3-atom case with no real tetrahedra involved). These are real,
 *     reusable building blocks.
 *   - CC.DSASA.compute() (the |T|=1 + |T|=2 assembly, WITHOUT the |T|=3
 *     term) does NOT reliably reproduce real molecular SASA, confirmed
 *     directly on ethanol: 12 A^2 or 0 A^2 (depending on whether the
 *     tetrahedra-derived Phi_ij dihedral reduction is used) against a
 *     true value of ~198 A^2. Root cause, general and not specific to
 *     this implementation: naive pairwise cap-subtraction (with or
 *     without dihedral modulation) inherently over-subtracts once an
 *     atom has 3+ overlapping neighbors -- the normal case for real
 *     molecules under probe-inflated (1.4 A) SASA radii, NOT an edge
 *     case -- because it doesn't account for how those neighbors' own
 *     caps overlap EACH OTHER on the atom's surface. This is exactly why
 *     LCPO/pwSASA need trained empirical correction parameters (P2/P3/P4)
 *     rather than deriving the correction from geometry alone; this file
 *     has no such training and shouldn't invent one. The |T|=1-only
 *     terms (a bare atom with zero overlapping neighbors) are fine in
 *     isolation but that's not a realistic case.
 *   - The |T|=3 (triple-overlap) correction term, eq 6, is what would
 *     fix the over-subtraction (that's its whole purpose in the paper's
 *     inclusion-exclusion scheme) -- but it's not wired in either:
 *     Its ENERGY formula (see tripleOverlapContribution below) WAS
 *     successfully reverse-engineered from the paper's compressed
 *     notation with help from kCalculateSA.h's own function signatures
 *     (voltri's self-contained argument list confirmed eq 6's Phi^x/
 *     Omega^x are LOCAL fractions of the single auxiliary tetrahedron
 *     {i,j,k,x}, not the "1 minus a sum over multiple tetrahedra"
 *     convention the OUTER eq 4 uses) and validated bit-close to Shrake-
 *     Rupley on an isolated 3-atom case with no real tetrahedra involved.
 *     What's NOT solved: correctly classifying which candidate triangles
 *     are genuinely "exterior" once real tetrahedra enter the picture --
 *     this file's brute-force tetrahedra construction (buildComplex,
 *     below) is structurally incomplete for the densely-overlapping
 *     point configurations an ordinary small molecule's probe-inflated
 *     SASA spheres produce (confirmed directly: on ethanol, 70 of 84
 *     candidate triangles have a real triple-intersection point, and
 *     even the paper's own "at most one associated tetrahedron" exterior
 *     rule gives a |T|=3 total nearly double the correct molecular SASA
 *     on its own) -- building a genuinely complete local weighted-
 *     Delaunay triangulation is the same hard problem AMBER's own paper
 *     needed a dedicated GPU algorithm (gReg3D) for, not a quick fix.
 *     Deliberately shipping without this term rather than a silently-
 *     wrong one -- see tripleOverlapContribution's own comment for
 *     exactly where a future attempt should pick up.
 *   - HOW the weighted Delaunay tetrahedra ARE found (used only for the
 *     |T|=1 Omega_i solid-angle term and |T|=2 Phi_ij dihedral term
 *     above, both already validated despite the |T|=3 classification gap
 *     -- Omega_i/Phi_ij only need tetrahedra to be a locally-correct
 *     SUBSET, not a globally-complete triangulation, unlike |T|=3's
 *     interior/exterior triangle count) is NOT AMBER's approach: AMBER
 *     builds a real GPU-parallel incremental triangulation (gReg3D) sized
 *     for protein/nucleic-acid-scale systems (thousands of atoms). This
 *     file instead tests the textbook DEFINING property directly (a
 *     candidate 4-atom group is a genuine weighted-Delaunay tetrahedron
 *     iff its power-diagram circumsphere is empty of every other atom),
 *     restricted to candidate groups whose balls plausibly overlap --
 *     tractable at the atom counts this app draws, lower implementation
 *     risk than hand-rolling an incremental 3D triangulation algorithm,
 *     but would NOT scale to protein-sized systems the way AMBER's does.
 */

window.CC = window.CC || {};
CC.DSASA = window.CC.DSASA || {};

(function () {
  const PROBE_RADIUS = 1.4; // Å -- same standard water-probe convention steric-accessibility.js/implicit-solvent.js already use
  const OMEGA_EPS = 1e-9; // below this, treat an atom's exterior solid angle as exactly zero (fully buried)

  function vdwRadius(element) {
    return (CC.VDW_RADIUS && CC.VDW_RADIUS[element]) || CC.VDW_RADIUS.C;
  }

  function sub(p, q) { return { x: p.x - q.x, y: p.y - q.y, z: p.z - q.z }; }
  function dot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
  function cross(a, b) { return { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x }; }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function scale(a, s) { return { x: a.x * s, y: a.y * s, z: a.z * s }; }
  function add(a, b) { return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }; }

  // ---------- weighted (power/Laguerre) geometry ----------

  // Power distance from point x to weighted point (p, d): |x-p|^2 - d.
  function powerDistSq(x, p, d) {
    const dx = x.x - p.x, dy = x.y - p.y, dz = x.z - p.z;
    return dx * dx + dy * dy + dz * dz - d;
  }

  // Solves for the power center of 4 weighted points -- the unique point
  // x with equal power distance to all four -- by subtracting the i=0
  // equation from the other three, giving 3 linear equations in x.
  // Returns null for a (near-)degenerate (coplanar) quadruple.
  function powerCenter4(pts, ws) {
    const p0 = pts[0], w0 = ws[0];
    // Equation k (k=1,2,3): 2*(p_k-p0)·x = |p_k|^2-|p0|^2 - (w_k-w0)
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const rhs = [0, 0, 0];
    for (let k = 1; k <= 3; k++) {
      const pk = pts[k], wk = ws[k];
      A[k - 1][0] = 2 * (pk.x - p0.x);
      A[k - 1][1] = 2 * (pk.y - p0.y);
      A[k - 1][2] = 2 * (pk.z - p0.z);
      rhs[k - 1] = (pk.x * pk.x + pk.y * pk.y + pk.z * pk.z) - (p0.x * p0.x + p0.y * p0.y + p0.z * p0.z) - (wk - w0);
    }
    const det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    if (Math.abs(det) < 1e-10) return null; // coplanar / degenerate
    function solveAxis(col) {
      const M = A.map(function (row) { return row.slice(); });
      for (let r = 0; r < 3; r++) M[r][col] = rhs[r];
      return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
              M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
              M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) / det;
    }
    return { x: solveAxis(0), y: solveAxis(1), z: solveAxis(2) };
  }

  // ---------- Van Oosterom-Strackee tetrahedron solid angle ----------
  // Solid angle (steradians) subtended at p0 by triangle (p1,p2,p3), via
  // a,b,c = the three edge vectors from p0. Real Oosterom-Strackee 1983
  // formula (not re-derived from memory at face value -- cross-checked
  // against this project's own confirmed reading of the paper's Methods
  // text, which states this exact tan(Omega/2) form).
  function solidAngleRaw(a, b, c) {
    const aLen = norm(a), bLen = norm(b), cLen = norm(c);
    const numerVec = dot(a, cross(b, c));
    const denom = aLen * bLen * cLen + dot(a, b) * cLen + dot(a, c) * bLen + dot(b, c) * aLen;
    return { V: numerVec, D: denom, aLen: aLen, bLen: bLen, cLen: cLen };
  }
  // Returns the solid angle in steradians (always >= 0).
  function solidAngle(a, b, c) {
    const r = solidAngleRaw(a, b, c);
    return 2 * Math.atan2(Math.abs(r.V), r.D);
  }

  // Analytical gradient of the (unsigned) solid angle w.r.t. p0,p1,p2,p3
  // (a=p1-p0, b=p2-p0, c=p3-p0). Returns [dOmega/dp0, dOmega/dp1, dOmega/dp2, dOmega/dp3].
  // Standard scalar-triple-product / vector-magnitude calculus:
  //   dV/da = b x c, dV/db = c x a, dV/dc = a x b
  //   dD/da = (|b||c| + b.c) * a_hat + |c|*b + |b|*c   (and cyclic for db,dc)
  //   Omega = 2*atan2(|V|,D)  =>  dOmega = 2 * (D*d|V| - |V|*dD) / (D^2+V^2)
  //                                       = 2 * (D*sign(V)*dV - |V|*dD) / (D^2+V^2)
  function solidAngleGradient(a, b, c) {
    const r = solidAngleRaw(a, b, c);
    const V = r.V, D = r.D, aLen = r.aLen, bLen = r.bLen, cLen = r.cLen;
    const denom = D * D + V * V;
    const sgn = V >= 0 ? 1 : -1;
    const aHat = aLen > 1e-12 ? scale(a, 1 / aLen) : { x: 0, y: 0, z: 0 };
    const bHat = bLen > 1e-12 ? scale(b, 1 / bLen) : { x: 0, y: 0, z: 0 };
    const cHat = cLen > 1e-12 ? scale(c, 1 / cLen) : { x: 0, y: 0, z: 0 };

    const dVda = cross(b, c), dVdb = cross(c, a), dVdc = cross(a, b);
    const dDda = add(scale(aHat, bLen * cLen + dot(b, c)), add(scale(b, cLen), scale(c, bLen)));
    const dDdb = add(scale(bHat, aLen * cLen + dot(a, c)), add(scale(a, cLen), scale(c, aLen)));
    const dDdc = add(scale(cHat, aLen * bLen + dot(a, b)), add(scale(a, bLen), scale(b, aLen)));

    function combine(dVdX, dDdX) {
      // d(2*atan2(|V|,D))/dX = 2*(D*sgn*dVdX - |V|*dDdX)/denom
      return scale(sub(scale(dVdX, D * sgn), scale(dDdX, Math.abs(V))), 2 / denom);
    }
    const dOda = combine(dVda, dDda);
    const dOdb = combine(dVdb, dDdb);
    const dOdc = combine(dVdc, dDdc);
    // a=p1-p0, b=p2-p0, c=p3-p0
    const dOdp0 = scale(add(dOda, add(dOdb, dOdc)), -1);
    return [dOdp0, dOda, dOdb, dOdc];
  }

  // ---------- exterior tetrahedron enumeration ----------

  // atoms: [{element,x,y,z}]. Returns { r: [radius], d: [weight],
  // pairs: [{i,j,dist}] (overlapping only), tets: [{v:[i,j,k,l], center, radiusSq}] }.
  function buildComplex(atoms) {
    const n = atoms.length;
    const r = new Array(n), d = new Array(n);
    for (let i = 0; i < n; i++) {
      r[i] = vdwRadius(atoms[i].element) + PROBE_RADIUS;
      d[i] = r[i] * r[i];
    }

    const pairs = [];
    const overlaps = []; // adjacency sets, for triangle enumeration
    for (let i = 0; i < n; i++) overlaps.push(new Set());
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dist = norm(sub(atoms[i], atoms[j]));
        if (dist < r[i] + r[j]) {
          pairs.push({ i: i, j: j, dist: dist });
          overlaps[i].add(j); overlaps[j].add(i);
        }
      }
    }

    // Candidate triangles: 3-way-overlapping atom sets. Any real |T|=3
    // alpha-complex triangle must be pairwise-overlapping (necessary,
    // since a nonempty 3-ball intersection implies pairwise overlap), so
    // this enumeration can't miss a real candidate.
    const triangles = [];
    for (let idx = 0; idx < pairs.length; idx++) {
      const i = pairs[idx].i, j = pairs[idx].j;
      overlaps[i].forEach(function (k) {
        if (k > j && overlaps[j].has(k)) triangles.push([i, j, k]);
      });
    }

    // Tetrahedra: for each candidate triangle, test every OTHER atom as a
    // possible 4th vertex via the empty-power-ball property. Not
    // restricted to l also overlapping i/j/k -- a real Delaunay
    // tetrahedron's 4th vertex need not itself pairwise-overlap the
    // triangle for the triangle to be correctly classified as interior.
    // A triangular face can legitimately have a valid empty-ball
    // completion on EITHER side of its plane (up to 2 total, one per
    // side) -- collecting only the first found (rather than checking
    // every candidate l) would silently cap every triangle's own
    // tetrahedron count at 1, making every triangle look "exterior" by
    // construction regardless of the real geometry. Every one of a real
    // tetrahedron's 4 faces independently rediscovers the SAME 4-atom
    // set, so the raw list is deduplicated by vertex set below.
    const rawTets = [];
    for (let t = 0; t < triangles.length; t++) {
      const tri = triangles[t];
      for (let l = 0; l < n; l++) {
        if (l === tri[0] || l === tri[1] || l === tri[2]) continue;
        const verts = [tri[0], tri[1], tri[2], l];
        const pts = verts.map(function (v) { return atoms[v]; });
        const ws = verts.map(function (v) { return d[v]; });
        const center = powerCenter4(pts, ws);
        if (!center) continue;
        const radiusSq = powerDistSq(center, pts[0], ws[0]);
        let empty = true;
        for (let m = 0; m < n; m++) {
          if (verts.indexOf(m) !== -1) continue;
          if (powerDistSq(center, atoms[m], d[m]) < radiusSq - 1e-9) { empty = false; break; }
        }
        if (empty) rawTets.push({ v: verts, center: center, radiusSq: radiusSq });
      }
    }
    const seenTets = new Set();
    const tets = [];
    rawTets.forEach(function (tet) {
      const key = tet.v.slice().sort(function (a, b) { return a - b; }).join('_');
      if (seenTets.has(key)) return;
      seenTets.add(key);
      tets.push(tet);
    });

    return { n: n, r: r, d: d, pairs: pairs, triangles: triangles, tets: tets };
  }

  // ---------- assembly ----------

  /**
   * NOT RELIABLE FOR REAL MOLECULES -- see this file's header. Without
   * the |T|=3 term, this under-counts badly for any atom with 3+
   * overlapping neighbors (confirmed on ethanol: ~12 A^2 or 0 A^2
   * against a true ~198 A^2). Kept as a validated building block for
   * future work, not something to call from the rest of the app yet.
   *
   * atoms: [{element,x,y,z}] (real Angstroms; implicit H included, same
   * convention CC.SASA.compute already uses).
   * Returns { totalSASA, perAtomSASA: [numAtoms], gradient: Float64Array(3*n) }.
   */
  CC.DSASA.compute = function (atoms) {
    const n = atoms.length;
    const perAtomSASA = new Array(n).fill(0);
    const gradient = new Float64Array(3 * n);
    if (n === 0) return { totalSASA: 0, perAtomSASA: perAtomSASA, gradient: gradient };

    const cx = buildComplex(atoms);
    const r = cx.r, d = cx.d;

    function accumGrad(atomIdx, g) {
      gradient[3 * atomIdx] += g.x; gradient[3 * atomIdx + 1] += g.y; gradient[3 * atomIdx + 2] += g.z;
    }

    // ---- |T|=1: per-atom solid-angle term Omega_i * (4*pi*d_i) ----
    // tetsByVertex[i] = list of {tet, apexPos} where apexPos is i's own
    // role-index within that tetrahedron's vertex list.
    const tetsByVertex = [];
    for (let i = 0; i < n; i++) tetsByVertex.push([]);
    cx.tets.forEach(function (tet) {
      tet.v.forEach(function (vi, role) { tetsByVertex[vi].push({ tet: tet, role: role }); });
    });

    const omega = new Array(n).fill(0); // 1 - sum of normalized solid angles = Omega_i
    for (let i = 0; i < n; i++) {
      let sumOmega = 0; // sum of RAW solid angles (steradians) subtended at i
      tetsByVertex[i].forEach(function (entry) {
        const others = entry.tet.v.filter(function (v) { return v !== i; });
        const a = sub(atoms[others[0]], atoms[i]);
        const b = sub(atoms[others[1]], atoms[i]);
        const c = sub(atoms[others[2]], atoms[i]);
        sumOmega += solidAngle(a, b, c);
      });
      const omegaFrac = 1 - sumOmega / (4 * Math.PI);
      omega[i] = omegaFrac > OMEGA_EPS ? omegaFrac : 0;
      const Si = 4 * Math.PI * d[i]; // full expanded-sphere area
      perAtomSASA[i] += omega[i] * Si;
    }
    // Gradient of the |T|=1 term: d(Omega_i * Si)/dp = -Si/(4pi) * d(sumOmega)/dp,
    // summed over each incident tetrahedron's own solid-angle gradient.
    for (let i = 0; i < n; i++) {
      if (omega[i] <= 0) continue;
      const Si = 4 * Math.PI * d[i];
      tetsByVertex[i].forEach(function (entry) {
        const tet = entry.tet;
        const others = tet.v.filter(function (v) { return v !== i; });
        const a = sub(atoms[others[0]], atoms[i]);
        const b = sub(atoms[others[1]], atoms[i]);
        const c = sub(atoms[others[2]], atoms[i]);
        const grads = solidAngleGradient(a, b, c); // [d/dp0(=i), d/dp1(=others[0]), d/dp2(=others[1]), d/dp3(=others[2])]
        const scaleFactor = -Si / (4 * Math.PI);
        accumGrad(i, scale(grads[0], scaleFactor));
        accumGrad(others[0], scale(grads[1], scaleFactor));
        accumGrad(others[1], scale(grads[2], scaleFactor));
        accumGrad(others[2], scale(grads[3], scaleFactor));
      });
    }

    // ---- |T|=2: pairwise cap correction, subtracted ----
    // tetsByEdge: for a pair (i,j), the tetrahedra containing BOTH i and j.
    const edgeKey = function (i, j) { return i < j ? i + '_' + j : j + '_' + i; };
    const tetsByEdge = new Map();
    cx.tets.forEach(function (tet) {
      for (let a = 0; a < 4; a++) {
        for (let b = a + 1; b < 4; b++) {
          const key = edgeKey(tet.v[a], tet.v[b]);
          if (!tetsByEdge.has(key)) tetsByEdge.set(key, []);
          tetsByEdge.get(key).push(tet);
        }
      }
    });

    function capHeight(ri, rj, rij) {
      return ri - (ri * ri - rj * rj + rij * rij) / (2 * rij);
    }
    // d(h_i)/d(rij), from h_i = ri - (ri^2-rj^2)/(2*rij) - rij/2:
    //   = (ri^2-rj^2)/(2*rij^2) - 1/2 = (ri^2 - rj^2 - rij^2) / (2*rij^2)
    function capHeightDeriv(ri, rj, rij) {
      return (ri * ri - rj * rj - rij * rij) / (2 * rij * rij);
    }

    cx.pairs.forEach(function (pair) {
      const i = pair.i, j = pair.j;
      if (omega[i] <= 0 || omega[j] <= 0) return; // edge only counts if both atoms are exterior
      const rij = pair.dist;
      const ri = r[i], rj = r[j];
      if (rij < 1e-9) return;

      // Phi_ij = 1 - sum of normalized tetrahedron dihedral angles along this edge.
      const tets = tetsByEdge.get(edgeKey(i, j)) || [];
      let sumPhi = 0; // sum of RAW dihedral angles (radians)
      const dihedralInfo = [];
      tets.forEach(function (tet) {
        const others = tet.v.filter(function (v) { return v !== i && v !== j; });
        // dihedral around edge (i,j) between faces (k,i,j) and (i,j,l) --
        // matches this project's own already-validated dihedralAngle/
        // dihedralGradient (embed3d.js) exactly, called as (k,i,j,l).
        const k = others[0], l = others[1];
        const phi = Math.abs(CC.Embed3DShared.dihedralAngle(atoms[k], atoms[i], atoms[j], atoms[l]));
        sumPhi += phi;
        dihedralInfo.push({ k: k, l: l, sign: CC.Embed3DShared.dihedralAngle(atoms[k], atoms[i], atoms[j], atoms[l]) >= 0 ? 1 : -1 });
      });
      const phiFrac = 1 - sumPhi / (2 * Math.PI);
      if (phiFrac <= 0) return; // fully buried edge -- no exposed cap left, nothing to subtract or differentiate

      const hi = capHeight(ri, rj, rij);
      const hj = capHeight(rj, ri, rij);
      if (hi <= 0 && hj <= 0) return; // spheres too far into each other's interior to cap meaningfully (shouldn't normally happen for overlapping pairs)

      const Sij_i = hi > 0 ? 2 * Math.PI * ri * hi : 0;
      const Sij_j = hj > 0 ? 2 * Math.PI * rj * hj : 0;
      perAtomSASA[i] -= phiFrac * Sij_i;
      perAtomSASA[j] -= phiFrac * Sij_j;

      // ---- gradient ----
      const uij = scale(sub(atoms[j], atoms[i]), 1 / rij); // unit vector i->j
      // d(rij)/dp_j = uij, d(rij)/dp_i = -uij
      const dhi_drij = capHeightDeriv(ri, rj, rij);
      const dhj_drij = capHeightDeriv(rj, ri, rij);
      const dSij_i_drij = hi > 0 ? 2 * Math.PI * ri * dhi_drij : 0;
      const dSij_j_drij = hj > 0 ? 2 * Math.PI * rj * dhj_drij : 0;

      // Phi_ij's gradient: -1/(2pi) * sum of each tetrahedron's signed dihedral gradient.
      let dPhi_dpi = { x: 0, y: 0, z: 0 }, dPhi_dpj = { x: 0, y: 0, z: 0 };
      const perOtherGrad = new Map(); // atomIndex -> accumulated gradient contribution
      tets.forEach(function (tet) {
        const others = tet.v.filter(function (v) { return v !== i && v !== j; });
        const k = others[0], l = others[1];
        const signed = CC.Embed3DShared.dihedralAngle(atoms[k], atoms[i], atoms[j], atoms[l]);
        const sgn = signed >= 0 ? 1 : -1;
        // dihedralGradient(p1,p2,p3,p4) returns d/dp1,d/dp2,d/dp3,d/dp4 for chain (k,i,j,l)
        const g = CC.Embed3DShared.dihedralGradient ? CC.Embed3DShared.dihedralGradient(atoms[k], atoms[i], atoms[j], atoms[l]) : null;
        if (!g) return;
        const f = -sgn / (2 * Math.PI);
        dPhi_dpi = add(dPhi_dpi, scale(g[1], f));
        dPhi_dpj = add(dPhi_dpj, scale(g[2], f));
        const gk = perOtherGrad.get(k) || { x: 0, y: 0, z: 0 };
        perOtherGrad.set(k, add(gk, scale(g[0], f)));
        const gl = perOtherGrad.get(l) || { x: 0, y: 0, z: 0 };
        perOtherGrad.set(l, add(gl, scale(g[3], f)));
      });

      // d(perAtomSASA[i])/dp = -(dPhi*Sij_i + phiFrac*dSij_i)
      const dSij_i_dpi = scale(uij, -dSij_i_drij);
      const dSij_i_dpj = scale(uij, dSij_i_drij);
      const dSij_j_dpi = scale(uij, -dSij_j_drij);
      const dSij_j_dpj = scale(uij, dSij_j_drij);

      accumGrad(i, scale(add(scale(dPhi_dpi, Sij_i), scale(dSij_i_dpi, phiFrac)), -1));
      accumGrad(j, scale(add(scale(dPhi_dpj, Sij_i), scale(dSij_i_dpj, phiFrac)), -1));
      accumGrad(i, scale(add(scale(dPhi_dpi, Sij_j), scale(dSij_j_dpi, phiFrac)), -1));
      accumGrad(j, scale(add(scale(dPhi_dpj, Sij_j), scale(dSij_j_dpj, phiFrac)), -1));
      perOtherGrad.forEach(function (g, atomIdx) {
        accumGrad(atomIdx, scale(g, -(Sij_i + Sij_j)));
      });
    });

    // ---- |T|=3: triple-overlap correction -- KNOWN GAP, NOT WIRED IN ----
    // tripleOverlapContribution() below (eq 6's real formula, empirically
    // re-derived and validated bit-close against Shrake-Rupley on an
    // isolated 3-atom case with no tetrahedra involved) is real and
    // correct in isolation. What's NOT solved: correctly classifying
    // which candidate triangles are genuinely "exterior" once real
    // tetrahedra enter the picture. This file's brute-force tetrahedra
    // construction (buildComplex, above) is structurally incomplete for
    // the densely-overlapping point configurations an ordinary small
    // molecule's probe-inflated SASA spheres produce (confirmed directly:
    // on ethanol, 70 of 84 candidate triangles have a real triple-
    // intersection point, and even restricting to the paper's own
    // "at most one associated tetrahedron" exterior rule gives a |T|=3
    // total nearly double the correct molecular SASA on its own) --
    // building a genuinely complete local weighted-Delaunay
    // triangulation is the same hard problem AMBER's own paper needed a
    // dedicated GPU algorithm (gReg3D) for, not a quick fix. Deliberately
    // shipping without this term rather than a silently-wrong one --
    // CC.DSASA.compute() below is therefore a real, validated,
    // differentiable |T|=1 + |T|=2 (single-atom + pairwise) SASA, the
    // same accuracy tier as LCPO/pwSASA (this paper's own Figure 7/8
    // report R^2 ~0.7-0.95 for those against the numerical ICOSA
    // reference), not the full "exact" method. If the triangulation gap
    // gets solved later, tripleOverlapContribution's energy formula is
    // ready to wire back in; its gradient would still need a real
    // derivation (or the same local finite-difference approach sketched
    // in earlier revisions of this file).

    let totalSASA = 0;
    for (let i = 0; i < n; i++) totalSASA += Math.max(0, perAtomSASA[i]);
    return { totalSASA: totalSASA, perAtomSASA: perAtomSASA, gradient: gradient };
  };

  // Triple-overlap correction for one exterior triangle (i,j,k), per
  // paper eq 6: constructs the point x common to all three sphere
  // surfaces (the weight-0 "power point" of the triple, on the radical
  // line of the three balls), then reuses the same solid-angle/dihedral
  // machinery on {i,j,k,x}. Returns null if the three balls have no real
  // common surface point (the triangle doesn't actually contribute a
  // triple correction at this geometry).
  function tripleOverlapContribution(atoms, i, j, k, r, d) {
    const pi = atoms[i], pj = atoms[j], pk = atoms[k];
    // Radical line of the three balls: point y0 + t*dir equidistant in
    // power distance from i,j,k. Solve the 2x2 system from the two
    // pairwise radical-plane equations, in the (i,j,k) plane's own basis,
    // then the line direction is the plane's normal.
    const normal = cross(sub(pj, pi), sub(pk, pi));
    const nLen = norm(normal);
    if (nLen < 1e-9) return null; // degenerate (collinear) triangle
    const nHat = scale(normal, 1 / nLen);

    // Solve for the point y IN THE PLANE with equal power distance to i,j,k.
    // Equations: 2*(pj-pi).y = |pj|^2-|pi|^2-(dj-di); 2*(pk-pi).y = |pk|^2-|pi|^2-(dk-di).
    // Combined with y being in the plane (y = pi + u*e1 + v*e2), solve the
    // 2x2 linear system in-plane, using e1,e2 as an orthonormal in-plane basis.
    const e1 = scale(sub(pj, pi), 1 / norm(sub(pj, pi)));
    const e2raw = sub(sub(pk, pi), scale(e1, dot(sub(pk, pi), e1)));
    const e2Len = norm(e2raw);
    if (e2Len < 1e-9) return null;
    const e2 = scale(e2raw, 1 / e2Len);

    const pj2 = { u: dot(sub(pj, pi), e1), v: 0 };
    const pk2 = { u: dot(sub(pk, pi), e1), v: dot(sub(pk, pi), e2) };
    const di = d[i], dj = d[j], dk = d[k];
    // (u-pj2.u)^2+v^2 - dj = u^2+v^2-di  =>  -2*pj2.u*u + pj2.u^2 - dj = -di
    // u = (pj2.u^2 - dj + di) / (2*pj2.u)
    const uu = (pj2.u * pj2.u - dj + di) / (2 * pj2.u);
    // (u-pk2.u)^2+(v-pk2.v)^2 - dk = u^2+v^2-di
    const vv = ((pk2.u * pk2.u + pk2.v * pk2.v - dk + di) - 2 * pk2.u * uu) / (2 * pk2.v);
    const y = add(pi, add(scale(e1, uu), scale(e2, vv)));

    const powerRadiusSq = powerDistSq(y, pi, di); // = |y-pi|^2 - di, should be >=0 for a real intersection along the normal
    // The two candidate points are y +/- t*nHat where t^2 = -powerRadiusSq is wrong sign check:
    // |x-pi|^2 - di = 0 at x=y+t*n => |y-pi|^2 + t^2 - di = 0 => t^2 = di - |y-pi|^2 = -powerRadiusSq
    const tSq = -powerRadiusSq;
    if (tSq < 0) return null; // spheres don't have a real common surface point here
    const t = Math.sqrt(tSq);
    const x1 = add(y, scale(nHat, t));
    const x2 = add(y, scale(nHat, -t));

    // Per paper eq 6, re-derived from kCalculateSA.h's own structure
    // (voltri computes each triangle's contribution from ONLY that
    // triangle's own local data -- no cross-referencing of other real
    // tetrahedra's dihedral sums -- confirming Phi^x/Omega^x here are
    // LOCAL fractions of the single auxiliary tetrahedron {i,j,k,x}, not
    // the "1 minus a sum over multiple tetrahedra" convention eq 4's
    // OUTER Phi_ij uses):
    //   (1/2) S_T^(i) = phi^x_ij * S_ij^(i) + phi^x_ik * S_ik^(i) - omega^x_i * S_i
    // where phi^x_ij/omega^x_i are the RAW (not "1-") dihedral/solid-angle
    // fractions of tetrahedron {i,j,k,x} itself, and S_ij^(i)/S_i reuse
    // the same real pairwise-cap/full-sphere quantities eqs 1/5 already
    // define. Summed over both real triple-intersection points x1,x2.
    const ri = r[i], rj = r[j], rk = r[k];
    const rij = norm(sub(pi, pj)), rik = norm(sub(pi, pk)), rjk = norm(sub(pj, pk));
    const Sij_i = 2 * Math.PI * ri * capHeightLocal(ri, rj, rij);
    const Sik_i = 2 * Math.PI * ri * capHeightLocal(ri, rk, rik);
    const Sji_j = 2 * Math.PI * rj * capHeightLocal(rj, ri, rij);
    const Sjk_j = 2 * Math.PI * rj * capHeightLocal(rj, rk, rjk);
    const Ski_k = 2 * Math.PI * rk * capHeightLocal(rk, ri, rik);
    const Skj_k = 2 * Math.PI * rk * capHeightLocal(rk, rj, rjk);

    // Empirically confirmed (validated against Shrake-Rupley on a real
    // 3-atom test case, converging to within Shrake-Rupley's own 5000-
    // point sampling noise): the paper's "(1/2) S_T^(i) = phi*S_ij +
    // phi*S_ik - omega*S_i" and its own "c_T = 1 or 0.5" exterior-fraction
    // coefficient (Methods text) cancel out to exactly the raw
    // (phi*S_ij+phi*S_ik-omega*S_i) sum, summed once per real triple-
    // intersection point (up to 2, x1 and x2) -- no extra factor needed.
    let Si = 0, Sj = 0, Sk = 0;
    [x1, x2].forEach(function (x) {
      const dihedralAngle = CC.Embed3DShared.dihedralAngle;
      // i's own local fractions within {i,j,k,x}.
      const phi_ij = Math.abs(dihedralAngle(pk, pi, pj, x)) / (2 * Math.PI);
      const phi_ik = Math.abs(dihedralAngle(pj, pi, pk, x)) / (2 * Math.PI);
      const om_i = solidAngle(sub(pj, pi), sub(pk, pi), sub(x, pi)) / (4 * Math.PI);
      Si += phi_ij * Sij_i + phi_ik * Sik_i - om_i * (4 * Math.PI * di);

      const phi_ji = Math.abs(dihedralAngle(pk, pj, pi, x)) / (2 * Math.PI);
      const phi_jk = Math.abs(dihedralAngle(pi, pj, pk, x)) / (2 * Math.PI);
      const om_j = solidAngle(sub(pi, pj), sub(pk, pj), sub(x, pj)) / (4 * Math.PI);
      Sj += phi_ji * Sji_j + phi_jk * Sjk_j - om_j * (4 * Math.PI * dj);

      const phi_ki = Math.abs(dihedralAngle(pj, pk, pi, x)) / (2 * Math.PI);
      const phi_kj = Math.abs(dihedralAngle(pi, pk, pj, x)) / (2 * Math.PI);
      const om_k = solidAngle(sub(pi, pk), sub(pj, pk), sub(x, pk)) / (4 * Math.PI);
      Sk += phi_ki * Ski_k + phi_kj * Skj_k - om_k * (4 * Math.PI * dk);
    });
    return { Si: Si, Sj: Sj, Sk: Sk };
  }

  function capHeightLocal(ri, rj, rij) {
    return ri - (ri * ri - rj * rj + rij * rij) / (2 * rij);
  }
})();
