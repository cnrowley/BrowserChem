/**
 * internal-coords.js
 *
 * A redundant-internal-coordinate (RIC) geometry optimizer: bonds, angles,
 * and dihedrals built from the real bond graph, a quasi-Newton RFO
 * (rational function optimization) step taken in that internal-coordinate
 * space, converted back to a Cartesian displacement via an iterative
 * pseudo-inverse back-transformation. This exists because this app's other
 * optimizers (embed3d.js's classical path, openff-forcefield.js's SMIRNOFF
 * path) both take L-BFGS steps directly in raw Cartesian x/y/z -- a
 * reasonable, standard choice for classical force-field minimization (it's
 * what OpenMM/GROMACS/LAMMPS do too), but a poorly-conditioned one for
 * molecules with a wide spread of bond-stretch vs. angle-bend vs. torsion
 * stiffness, where L-BFGS's limited-memory Hessian struggles to capture the
 * whole spread from scratch. Real ab initio geometry optimizers (Gaussian,
 * ORCA, Psi4's optking) use exactly this redundant-internal-coordinate
 * approach for that reason.
 *
 * ALGORITHM SOURCE: ported (not copied verbatim -- this is JS, the
 * reference is Python/NumPy) from PyBerny (github.com/jhrmnn/pyberny,
 * MPL-2.0), specifically its coordinate value/gradient formulas, its
 * rho-scaled empirical guess Hessian (bond/angle/dihedral force constants
 * derived from a covalentness measure -- Lindh, Bernhardsson, Karlstrom,
 * Malmqvist, Chem. Phys. Lett. 1995, 241, 423), its BFGS Hessian update,
 * and its augmented-matrix RFO step with trust-radius on-sphere handling
 * (Banerjee et al., J. Phys. Chem. 1985, 89, 52; Baker, J. Comput. Chem.
 * 1986, 7, 385). PyBerny's guess-Hessian/convergence constants are in
 * atomic units (Hartree, bohr) since it's built for QM gradients; this file
 * converts them to kcal/mol and Angstrom (1 Hartree = 627.509474 kcal/mol,
 * 1 bohr = 0.529177210903 Angstrom) to match this app's force-field energy
 * units throughout.
 *
 * WHAT'S DELIBERATELY SIMPLIFIED vs. the real PyBerny algorithm (honest
 * gaps, not hidden ones -- see CLAUDE.md's own documentation norm):
 *   - No dummy-atom handling for near-linear angles (PyBerny inserts
 *     perpendicular dummy atoms so an sp-like center still gets a
 *     well-defined angle coordinate). This file just OMITS an angle/
 *     dihedral through any atom triple/quadruple where the flanking angle
 *     exceeds ~170 degrees. Fine for typical drug-like molecules (a
 *     genuinely linear heavy-atom chain is rare outside alkynes/nitriles);
 *     a real linear or near-linear backbone segment will be under-
 *     constrained by this coordinate set.
 *   - No dynamic coordinate-set rebuild mid-optimization (PyBerny
 *     re-detects near-linear topology changes and carries over the
 *     Hessian). This file builds the coordinate set once, from the
 *     starting geometry, and keeps it fixed for the whole run.
 *   - No dihedral-swap/periodicity edge cases beyond simple (-pi,pi]
 *     wrapping of each dihedral's OWN step-to-step change (see wrapAngle).
 *     PyBerny's eval_geom() also un-swaps a flanking angle when a
 *     dihedral flips by pi -- not implemented here.
 *   - No cubic/quartic-interpolated line search between the current and
 *     best-so-far points (PyBerny's linear_search()) before taking the
 *     next quadratic step. This file always steps from the latest
 *     accepted point.
 *   - No dead-code coordinate weighting either, unlike an earlier version
 *     of this file: an earlier header here claimed PyBerny's
 *     coords.weight()/weights() -- a per-coordinate "how well-defined is
 *     this bond/angle/dihedral" measure -- biases its RFO step away from
 *     weak auxiliary coordinates. That's what the weight() methods and
 *     their docstrings suggest, but checked directly against source
 *     (github.com/pyberny/pyberny, src/berny/berny.py, 2026-08): weights
 *     is computed, threaded through Berny's whole optimizer state, and
 *     passed as `w` into quadratic_step() -- then never referenced
 *     anywhere in that function's body. It's vestigial in the current
 *     source; there was never a real mechanism here to port.
 *
 * WHAT IS now ported, beyond the core algorithm above (both verified
 * against the same current PyBerny source, not assumed):
 *   - Hessian projection (projectedHessian): PyBerny computes
 *     `H_proj = proj.H.proj + PENALTY*(I - proj)` (proj = B.B+) before
 *     every RFO step, forcing a redundant coordinate set's null-space
 *     directions (which don't correspond to any real Cartesian motion)
 *     to look artificially, deliberately very stiff instead of leaving
 *     them ill-defined by the raw Hessian.
 *   - Fletcher-ratio trust update (fletcherTrustUpdate): compares the
 *     quadratic model's PREDICTED energy change to what actually
 *     happened to size the next trust radius, replacing an earlier flat
 *     retry-count-based grow/shrink heuristic this file used before.
 * None of these change the core method (redundant internal coordinates +
 * RFO), only its robustness on pathological/edge-case geometries -- see
 * this file's own validation notes in the header of internalOptimize()
 * below for what WAS checked before this shipped.
 *
 * Needs window.mlMatrix (ml-matrix, MIT-licensed, loaded via CDN in
 * index.html) for the general N x N linear algebra a redundant coordinate
 * set needs (Wilson B-matrix pseudo-inverse, Hessian eigendecomposition for
 * the RFO step) -- this app's own hand-rolled linear algebra elsewhere
 * (molecular-shape.js, geomol-assembly.js) only ever handles fixed 3x3
 * matrices, not a general size that scales with the number of internal
 * coordinates (order 100-300 for a real drug-sized molecule).
 */

window.CC = window.CC || {};
CC.InternalOpt = window.CC.InternalOpt || {};

(function () {
  const vec3 = CC.vec3;
  const HARTREE_TO_KCAL = 627.509474;

  // Guess-Hessian force constants (PyBerny coords.py's Bond/Angle/Dihedral
  // .hessian()), converted from Hartree/bohr^2 (bonds) or Hartree/rad^2
  // (angles, dihedrals -- radians are dimensionless, so only the energy
  // unit converts) to kcal/mol. 1 Hartree/bohr^2 = 627.509474 /
  // 0.529177210903^2 kcal/mol/Angstrom^2 = 2240.87 kcal/mol/Angstrom^2.
  const BOND_HESS_K = 0.45 * 2240.87; // ~1008.4 kcal/mol/Angstrom^2
  const ANGLE_HESS_K = 0.15 * HARTREE_TO_KCAL; // ~94.13 kcal/mol/rad^2
  const DIHEDRAL_HESS_K = 0.005 * HARTREE_TO_KCAL; // ~3.14 kcal/mol/rad^2

  // Angles/dihedrals through a triple whose flanking angle exceeds this are
  // skipped entirely at coordinate-construction time (see file header --
  // no dummy-atom replacement for near-linear centers in this port).
  const LIN_THRESHOLD = (170 * Math.PI) / 180;

  // Hard cap on the REAL Cartesian consequence of a single trial step,
  // checked directly against the back-transformed geometry rather than
  // trusted as implied by the trust radius -- see the retry loop's own
  // comment for why this exists: this file's trust region is a single
  // Euclidean norm over dq, mixing bond-Angstrom, angle-radian, and
  // dihedral-radian components as if they were the same kind of unit.
  // For a bond or a local angle bend that's a reasonable approximation,
  // but a DIHEDRAL rotation swings every atom on the far side of that
  // bond by (angle change x distance from the rotation axis) -- for a
  // heavy substituent or a fused-ring branch several bonds away, that
  // lever arm can be many Angstroms, so the SAME nominal ||dq|| can mean
  // a tiny bond adjustment or a wildly larger real displacement,
  // depending entirely on which coordinate type the step happens to be
  // dominated by. Reproduced directly: a real fused-ring molecule's
  // first optimization step needed 11 retries at trust=4.9e-5, yet still
  // produced a 0.0030 Angstrom Cartesian displacement -- ~60x what that
  // trust value would suggest -- entirely from an angle/dihedral-
  // dominated step (95% angle fraction) swinging a ring-fused branch.
  // Matches embed3d.js's/openff-forcefield.js's own MAX_STEP_ANGSTROM,
  // the same real, load-bearing cap their Cartesian L-BFGS line search
  // already uses for exactly this reason (see that constant's own
  // comment) -- reusing its value keeps the two optimizers' notion of
  // "how far is too far in one step" consistent.
  const MAX_CART_STEP = 0.2;

  // ---------- topology ----------

  function neighborsOf(bonds3d, atomIdx) {
    const result = [];
    bonds3d.forEach(function (b) {
      if (b.a1 === atomIdx) result.push(b.a2);
      else if (b.a2 === atomIdx) result.push(b.a1);
    });
    return result;
  }

  function angleValue(atoms, i, j, k) {
    const v1 = vec3.sub(atoms[i], atoms[j]);
    const v2 = vec3.sub(atoms[k], atoms[j]);
    const n1 = vec3.length(v1), n2 = vec3.length(v2);
    if (n1 < 1e-9 || n2 < 1e-9) return 0;
    let c = vec3.dot(v1, v2) / (n1 * n2);
    c = Math.max(-1, Math.min(1, c));
    return Math.acos(c);
  }

  // bonds/angles/dihedrals from the REAL bond graph (this app always has
  // one -- unlike PyBerny, which has to perceive bonds from geometry itself
  // via covalent radii, since it's meant to work from bare QM gradient
  // calculations). Returns { bonds, angles, dihedrals, all }.
  function buildCoordSet(atoms3d, bonds3d) {
    const n = atoms3d.length;
    const nbrs = [];
    for (let i = 0; i < n; i++) nbrs.push(neighborsOf(bonds3d, i));

    const bonds = bonds3d.map(function (b) {
      return { type: 'bond', i: b.a1, j: b.a2 };
    });

    const angles = [];
    for (let j = 0; j < n; j++) {
      const nb = nbrs[j];
      for (let a = 0; a < nb.length; a++) {
        for (let b = a + 1; b < nb.length; b++) {
          const i = nb[a], k = nb[b];
          if (angleValue(atoms3d, i, j, k) > LIN_THRESHOLD) continue; // near-linear -- see file header
          angles.push({ type: 'angle', i: i, j: j, k: k });
        }
      }
    }

    // ONE representative dihedral per bond (first valid i, first valid l),
    // not the full i x l product PyBerny's own get_dihedrals also mostly
    // avoids (via its own "weak > 1" cap) -- deliberately simplified here
    // for a stronger reason than fidelity: the extra i/l choices around the
    // SAME central bond are almost entirely redundant once the angle
    // coordinates are already in the set (they fix each substituent's own
    // position relative to its bonded neighbors; a second dihedral through
    // the same bond mostly just repeats that constraint at an angular
    // offset). Generating all of them made the coordinate count blow up
    // combinatorially at any branching atom (a tert-butyl-adjacent bond
    // alone contributes 3x as many), which is real, measured, and
    // significant: pseudo-inverse cost scales with matrix size, and this
    // exact combinatorial blow-up (351 internal coordinates on a 60-atom
    // molecule, dihedral-dominated) made a single pseudoInverse call take
    // ~330ms before this fix -- see project memory for the actual numbers.
    const dihedrals = [];
    bonds3d.forEach(function (b) {
      const j = b.a1, k = b.a2;
      const isAtoms = nbrs[j].filter(function (x) { return x !== k; });
      const ksAtoms = nbrs[k].filter(function (x) { return x !== j; });
      if (isAtoms.length === 0 || ksAtoms.length === 0) return;
      const i = isAtoms.find(function (x) { return angleValue(atoms3d, x, j, k) <= LIN_THRESHOLD; });
      if (i === undefined) return;
      const l = ksAtoms.find(function (x) { return x !== i && angleValue(atoms3d, j, k, x) <= LIN_THRESHOLD; });
      if (l === undefined) return;
      dihedrals.push({ type: 'dihedral', i: i, j: j, k: k, l: l });
    });

    return { bonds: bonds, angles: angles, dihedrals: dihedrals, all: bonds.concat(angles, dihedrals) };
  }

  // ---------- coordinate value + Cartesian gradient (B-matrix row) ----------
  // Bond/angle formulas ported directly from PyBerny's Bond.eval/Angle.eval
  // (coords.py); dihedral reuses this app's OWN already-validated
  // dihedralAngle/dihedralGradient (CC.Embed3DShared, embed3d.js) rather
  // than re-deriving PyBerny's separate dihedral formula from scratch --
  // same physical quantity (a signed torsion angle and its Cartesian
  // gradient), no reason to risk a second, independent derivation.

  function evalBond(atoms, c) {
    const v = vec3.sub(atoms[c.i], atoms[c.j]);
    const r = vec3.length(v);
    const u = r > 1e-9 ? vec3.scale(v, 1 / r) : { x: 1, y: 0, z: 0 };
    return { value: r, grads: [{ atom: c.i, vec: u }, { atom: c.j, vec: vec3.scale(u, -1) }] };
  }

  function evalAngle(atoms, c) {
    const pi = atoms[c.i], pj = atoms[c.j], pk = atoms[c.k];
    const v1 = vec3.sub(pi, pj), v2 = vec3.sub(pk, pj);
    const n1 = vec3.length(v1), n2 = vec3.length(v2);
    let cosPhi = (n1 > 1e-9 && n2 > 1e-9) ? vec3.dot(v1, v2) / (n1 * n2) : 0;
    cosPhi = Math.max(-1, Math.min(1, cosPhi));
    const phi = Math.acos(cosPhi);
    const sinPhi = Math.sin(phi);
    let gi, gj, gk;
    if (sinPhi < 1e-6 || n1 < 1e-9 || n2 < 1e-9) {
      // Degenerate (should be rare -- near-linear triples are already
      // excluded at construction time); zero gradient rather than a NaN.
      gi = { x: 0, y: 0, z: 0 }; gk = { x: 0, y: 0, z: 0 }; gj = { x: 0, y: 0, z: 0 };
    } else {
      const cotPhi = cosPhi / sinPhi;
      gi = vec3.sub(vec3.scale(v1, cotPhi / (n1 * n1)), vec3.scale(v2, 1 / (n1 * n2 * sinPhi)));
      gk = vec3.sub(vec3.scale(v2, cotPhi / (n2 * n2)), vec3.scale(v1, 1 / (n1 * n2 * sinPhi)));
      gj = vec3.scale(vec3.add(gi, gk), -1); // translation invariance: the 3 gradients sum to zero
    }
    return { value: phi, grads: [{ atom: c.i, vec: gi }, { atom: c.j, vec: gj }, { atom: c.k, vec: gk }] };
  }

  function evalDihedral(atoms, c) {
    const shared = CC.Embed3DShared;
    const phi = shared.dihedralAngle(atoms[c.i], atoms[c.j], atoms[c.k], atoms[c.l]);
    const g = shared.dihedralGradient(atoms[c.i], atoms[c.j], atoms[c.k], atoms[c.l]);
    return {
      value: phi,
      grads: [{ atom: c.i, vec: g[0] }, { atom: c.j, vec: g[1] }, { atom: c.k, vec: g[2] }, { atom: c.l, vec: g[3] }],
    };
  }

  function evalCoord(atoms, c) {
    if (c.type === 'bond') return evalBond(atoms, c);
    if (c.type === 'angle') return evalAngle(atoms, c);
    return evalDihedral(atoms, c);
  }

  function wrapAngle(a) {
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a < -Math.PI) a += 2 * Math.PI;
    return a;
  }

  // Values only (no B-matrix) -- used inside the back-transformation's
  // inner loop, where only the achieved coordinate change matters.
  function evalCoordValues(atoms, coordList) {
    const q = new Float64Array(coordList.length);
    for (let r = 0; r < coordList.length; r++) q[r] = evalCoord(atoms, coordList[r]).value;
    return q;
  }

  // Real, reproduced failure mode this guards: the coordinate SET is built
  // once from the starting geometry and never rebuilt (see file header),
  // so nothing stops an angle that started safely under LIN_THRESHOLD from
  // drifting past it as the geometry evolves. As sin(phi) -> 0, evalAngle's
  // 1/sin(phi) terms blow up, corrupting the B-matrix (and therefore the
  // pseudo-inverse and the Hessian built from projected gradients) for
  // EVERY coordinate, not just the one near-linear angle -- reproduced
  // directly as a real optimization run's energy jumping from ~150 to
  // 538537 kcal/mol in a single accepted step. A small safety MARGIN
  // (5 degrees short of the construction-time LIN_THRESHOLD) means a trial
  // step that's heading toward that singularity gets caught and rejected
  // here, before it corrupts anything, same as an energy-increasing step.
  const LIN_SAFETY_MARGIN = (5 * Math.PI) / 180;

  // The SAME 1/sin(phi) singularity that blows up as phi -> 180 degrees
  // (LIN_THRESHOLD, above) ALSO blows up as phi -> 0 -- sin(phi) is zero
  // at BOTH ends, not just the obtuse one, and evalAngle's cotPhi/sinPhi
  // terms don't care which direction phi is approaching zero from. This
  // guard used to only check the upper bound, which is the common case
  // (a near-linear heavy-atom chain), but a real regression this session
  // reproduced the SAME corruption pattern from the OTHER end: an aspirin
  // run's angle-dominated steps needed 8+ retries per iteration with the
  // reported Cartesian-displacement-to-trust ratio climbing into the
  // tens of thousands (vs. the already-large ~60x this file's
  // MAX_CART_STEP cap was built to catch) as trust collapsed toward
  // TRUST_FLOOR without ever finding an accepted step -- exactly the
  // signature of a coordinate approaching ITS OWN singularity, just from
  // below instead of above. No real, stable molecular angle needs to go
  // this acute (even cyclopropane's ~60 degree ring angle has enormous
  // headroom above this), so a low threshold with the same safety margin
  // catches the genuine numerical case without ever false-triggering on
  // real chemistry.
  const LOW_LIN_THRESHOLD = (10 * Math.PI) / 180;

  function hasNearLinearAngle(atoms, coordList) {
    for (let r = 0; r < coordList.length; r++) {
      const c = coordList[r];
      if (c.type !== 'angle') continue;
      const theta = angleValue(atoms, c.i, c.j, c.k);
      if (theta > LIN_THRESHOLD - LIN_SAFETY_MARGIN) return true;
      if (theta < LOW_LIN_THRESHOLD + LIN_SAFETY_MARGIN) return true;
    }
    return false;
  }

  // B: (numCoords x 3*numAtoms) Wilson B-matrix. q: coordinate values.
  function buildBMatrix(atoms, coordList) {
    const n = atoms.length;
    const m = coordList.length;
    const B = new mlMatrix.Matrix(m, 3 * n);
    const q = new Float64Array(m);
    for (let r = 0; r < m; r++) {
      const ev = evalCoord(atoms, coordList[r]);
      q[r] = ev.value;
      ev.grads.forEach(function (g) {
        B.set(r, 3 * g.atom, B.get(r, 3 * g.atom) + g.vec.x);
        B.set(r, 3 * g.atom + 1, B.get(r, 3 * g.atom + 1) + g.vec.y);
        B.set(r, 3 * g.atom + 2, B.get(r, 3 * g.atom + 2) + g.vec.z);
      });
    }
    return { B: B, q: q };
  }

  // ---------- guess Hessian (Lindh-style rho covalentness measure) ----------

  function rho(atoms, i, j) {
    const ri = CC.elementData(atoms[i].element).radius;
    const rj = CC.elementData(atoms[j].element).radius;
    const rij = vec3.distance(atoms[i], atoms[j]);
    return Math.exp(-rij / (ri + rj) + 1);
  }

  function guessHessianDiag(atoms, coordList) {
    const m = coordList.length;
    const diag = new Float64Array(m);
    for (let r = 0; r < m; r++) {
      const c = coordList[r];
      if (c.type === 'bond') diag[r] = BOND_HESS_K * rho(atoms, c.i, c.j);
      else if (c.type === 'angle') diag[r] = ANGLE_HESS_K * rho(atoms, c.i, c.j) * rho(atoms, c.j, c.k);
      else diag[r] = DIHEDRAL_HESS_K * rho(atoms, c.i, c.j) * rho(atoms, c.j, c.k) * rho(atoms, c.k, c.l);
    }
    return diag;
  }

  // ---------- Cartesian preconditioner (Packwood-style, an alternative to full RIC+RFO) ----------
  //
  // Everything above this point runs the WHOLE optimization in redundant
  // internal coordinates (RFO steps, iterative back-transformation, near-
  // linear-angle guards, coordinate-set fragility -- see this file's own
  // header for the honest list of what that costs in robustness). This
  // section is a structurally much simpler alternative that gets a
  // similar conditioning benefit a different way: use guessHessianDiag's
  // bond/angle/dihedral force constants only ONCE, to build a Cartesian-
  // space approximate Hessian M = B^T diag(k) B, then run PLAIN L-BFGS
  // directly in Cartesian coordinates with M as the initial inverse-
  // Hessian guess (via CC.Embed3DShared.lbfgsDirectionWithH0) instead of
  // the usual isotropic gamma*I. No back-transformation, no internal-
  // coordinate retry loop, no near-linear-angle special-casing -- L-BFGS's
  // own rank-2 curvature updates do the real refinement, same as they
  // always do; this only fixes L-BFGS's worst blind spot (starting from
  // an isotropic guess when the true stiffness spans ~300x between a bond
  // stretch and a torsion) without changing the SEARCH SPACE at all.
  //
  // This is the well-known "preconditioned L-BFGS" scheme: Packwood,
  // Kermode, Mones, Bernstein, Woolley, Gould, Ortner, Csanyi, J. Chem.
  // Phys. 144, 164109 (2016), "A universal preconditioner for simulating
  // condensed phase materials" -- also shipped in ASE's own `precon`
  // package. Their finding (reused here rather than re-derived): a
  // STATIC, approximate preconditioner built once from the starting
  // geometry captures most of the benefit; it doesn't need rebuilding
  // every iteration the way this file's own RFO Hessian does.
  function buildCartesianPreconditioner(atoms3d, bonds3d) {
    const coordSet = buildCoordSet(atoms3d, bonds3d).all;
    if (coordSet.length === 0) return null;
    const k = guessHessianDiag(atoms3d, coordSet);
    const built = buildBMatrix(atoms3d, coordSet);
    const B = built.B;
    const n3 = 3 * atoms3d.length;

    const kDiagB = new mlMatrix.Matrix(B.rows, B.columns);
    for (let r = 0; r < B.rows; r++) {
      const kr = k[r];
      for (let c = 0; c < B.columns; c++) kDiagB.set(r, c, B.get(r, c) * kr);
    }
    const M = B.transpose().mmul(kDiagB); // = B^T diag(k) B

    // M is EXACTLY singular in exact arithmetic, not just ill-conditioned:
    // no bond/angle/dihedral value changes under a whole-molecule
    // translation or rotation, so those 6 Cartesian directions are
    // genuinely in its null space. REGULARIZE_EPS lifts every eigenvalue
    // by a small constant (small next to a bond's ~1000 or even a
    // dihedral's ~3, but enough to clear exact zero) so the Cholesky
    // solve below succeeds; L-BFGS's own curvature-pair updates correct
    // any resulting mismatch in those directions quickly in practice --
    // translation/rotation are the CHEAPEST, flattest directions in the
    // whole landscape, not the ones that need a precise preconditioner.
    // Also re-symmetrizes: B^T diag(k) B is symmetric in exact math, but
    // ml-matrix's CholeskyDecomposition checks isSymmetric() with exact
    // (!==) equality, which floating-point matrix-multiply round-off can
    // violate by ~1e-13 even for a mathematically symmetric product.
    const REGULARIZE_EPS = 1.0; // kcal/mol/Angstrom^2
    for (let i = 0; i < n3; i++) {
      for (let j = i + 1; j < n3; j++) {
        const avg = (M.get(i, j) + M.get(j, i)) / 2;
        M.set(i, j, avg);
        M.set(j, i, avg);
      }
      M.set(i, i, M.get(i, i) + REGULARIZE_EPS);
    }

    const chol = new mlMatrix.CholeskyDecomposition(M);
    if (!chol.isPositiveDefinite()) return null;
    return {
      applyH0: function (q) {
        const qM = mlMatrix.Matrix.columnVector(Array.from(q));
        return Float64Array.from(chol.solve(qM).to1DArray());
      },
    };
  }

  // ---------- Cartesian gradient <-> internal-coordinate gradient ----------
  // g_q = (B+)^T g_x, where B+ is the Moore-Penrose pseudo-inverse of B
  // (ml-matrix's pseudoInverse handles the rank-deficiency a REDUNDANT
  // coordinate set always has via truncated SVD, same as PyBerny's own
  // Math.pinv). The same B+ also converts an internal-coordinate STEP back
  // to a Cartesian one (dx = B+ . dq), used by backTransform below.

  function projectGradient(Bpinv, gradX) {
    const rows = Bpinv.rows, cols = Bpinv.columns;
    const gq = new Float64Array(cols);
    for (let a = 0; a < cols; a++) {
      let s = 0;
      for (let k = 0; k < rows; k++) s += Bpinv.get(k, a) * gradX[k];
      gq[a] = s;
    }
    return gq;
  }

  // ---------- BFGS Hessian update (PyBerny berny.py's update_hessian) ----------

  function bfgsUpdate(H, dq, dg) {
    const m = dq.length;
    let dqDotDg = 0;
    for (let a = 0; a < m; a++) dqDotDg += dq[a] * dg[a];
    // Standard BFGS curvature condition: the rank-2 update only preserves
    // positive-definiteness when dq.dg > 0. PyBerny's own trust-
    // radius/line-search machinery (not fully ported here -- see file
    // header) mostly keeps this satisfied in practice; this file's
    // simpler accept/reject scheme doesn't guarantee it as reliably, so
    // the check is explicit rather than assumed. Skipping a violating
    // update (not just an exactly-zero one) is what stops H from
    // silently drifting non-positive-definite over many iterations --
    // reproduced directly: without this guard, a real optimization run
    // stayed well-behaved for ~100 iterations, then the Hessian
    // degenerated and the very next RFO step sent the energy from ~150
    // kcal/mol to 538537 kcal/mol in one step.
    if (!(dqDotDg > 1e-10)) return H;
    const dqM = mlMatrix.Matrix.columnVector(dq);
    const dgM = mlMatrix.Matrix.columnVector(dg);
    const term1 = dgM.mmul(dgM.transpose()).mul(1 / dqDotDg);
    const Hdq = H.mmul(dqM);
    const dqHdq = dqM.transpose().mmul(H).mmul(dqM).get(0, 0);
    if (!(dqHdq > 1e-10)) return H;
    const term2 = Hdq.mmul(Hdq.transpose()).mul(1 / dqHdq);
    return H.add(term1).sub(term2);
  }

  // ---------- Hessian projection (real PyBerny mechanism, verified against source) ----------
  //
  // A REDUNDANT coordinate set's B-matrix has a nontrivial null space
  // (rank m - (3n-6) for a real molecule, not m) -- the raw BFGS Hessian
  // above is defined over the FULL m-dimensional q-space, but any
  // component along that null space doesn't correspond to any real
  // Cartesian motion at all, so it's fundamentally undefined/arbitrary,
  // not just "possibly wrong." PyBerny addresses this directly (checked
  // against src/berny/berny.py, not assumed -- see file header for a
  // correction to an earlier, wrong assumption about a DIFFERENT PyBerny
  // mechanism, coordinate weighting, which turned out to be vestigial):
  // before every RFO step it computes `H_proj = proj.H.proj +
  // PENALTY*(I - proj)` where `proj = B.B+`, projecting H onto B's actual
  // row space and making the complementary null-space directions
  // artificially, deliberately very stiff. This stops RFO from ever
  // "spending" trust-region budget along a direction with no real
  // geometric meaning, which a raw, unprojected Hessian has no way to
  // rule out. PROJ_PENALTY: PyBerny's own constant is 1000 in its native
  // Hartree/bohr^2 Hessian units; converted via the SAME
  // HARTREE_TO_KCAL/bohr-to-Angstrom^2 factor BOND_HESS_K itself already
  // uses (627.509474 / 0.529177210903^2 = 2240.87) to land in this
  // file's kcal/mol/Angstrom^2-equivalent units -- deliberately far
  // stiffer than even a real bond's guess stiffness (~1008), which is
  // the point: a null-space direction should never look worth moving
  // along, by a wide margin.
  // NOTE: PyBerny's own 1000-Hartree/bohr^2 value converts to ~2.24e6 in
  // this file's units via the same factor BOND_HESS_K uses -- tested
  // directly and found to occasionally destabilize this file's
  // eigendecomposition (mlMatrix.EigenvalueDecomposition, not the same
  // LAPACK routine PyBerny's NumPy calls): a real aspirin run regressed
  // to gradNorm 14.4 (worse than any failure seen before this projection
  // was added) with that value. A ~45x smaller penalty -- still ~50x
  // stiffer than even a real bond (BOND_HESS_K ~1008), which is all the
  // underlying purpose needs (null-space directions just need to look
  // clearly unattractive next to every real coordinate, not
  // astronomically so) -- keeps the matrix's dynamic range far enough
  // from destabilizing this implementation's eigensolver while still
  // fully serving PyBerny's own reasoning for why this exists.
  const PROJ_PENALTY = 50000;

  // Fletcher's ratio (ported directly from PyBerny's update_trust(),
  // checked against source): compares the quadratic model's PREDICTED
  // energy change (from g, H, and the step just taken) to what actually
  // happened, to size the NEXT iteration's trust radius. More principled
  // than a flat retry-count-based heuristic because it directly measures
  // how much to trust the quadratic model itself, not just whether THIS
  // particular step happened to be downhill: r = dE_actual / dE_predicted
  // near 1 means the model was a good local predictor (safe to grow,
  // if the step used the full trust budget); r well below 1 means the
  // model badly overestimated the improvement (shrink to a quarter of
  // whatever step size was ACTUALLY achievable, not a fixed multiplier
  // of the old trust). ENERGY_NOISE guards the r = dE/dE_predicted
  // division when both are near numerical noise (PyBerny's own default,
  // 2e-8 Hartree, converted via HARTREE_TO_KCAL).
  const ENERGY_NOISE = 2e-8 * HARTREE_TO_KCAL;

  function fletcherTrustUpdate(trust, dEActual, dEPredicted, dqNorm) {
    if (Math.abs(dEPredicted) < 10 * ENERGY_NOISE) {
      return Math.abs(dqNorm - trust) < 1e-10 ? 2 * trust : trust;
    }
    const r = dEActual !== 0 ? dEActual / dEPredicted : 1.0;
    if (r < 0.25) return dqNorm / 4;
    if (r > 0.75 && Math.abs(dqNorm - trust) < 1e-10) return 2 * trust;
    return trust;
  }

  function projectedHessian(H, B, Bpinv, m) {
    const proj = B.mmul(Bpinv); // (m x 3n)(3n x m) = m x m
    const HProj = proj.mmul(H).mmul(proj);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        const complement = (i === j ? 1 : 0) - proj.get(i, j);
        if (complement !== 0) HProj.set(i, j, HProj.get(i, j) + PROJ_PENALTY * complement);
      }
    }
    return HProj;
  }

  // ---------- RFO step with trust radius ----------
  // In-trust case: augmented-matrix RFO (Banerjee 1985 / Baker 1986) --
  // eigenvector of [[H,g],[g^T,0]] belonging to its SMALLEST eigenvalue,
  // normalized by its own last component. On-sphere case: find the
  // Lagrange multiplier lambda (below H's smallest eigenvalue) such that
  // ||(lambda*I - H)^-1 g|| == trust, via bisection on H's own
  // eigendecomposition (avoids re-inverting a shifted matrix at every
  // trial lambda).

  function smallestEigenIndex(evals) {
    let idx = 0;
    for (let i = 1; i < evals.length; i++) if (evals[i] < evals[idx]) idx = i;
    return idx;
  }

  // H and g are the SAME across every retry within one outer iteration --
  // only trust changes. Eigendecomposing H (O(m^3)) is by far the most
  // expensive step here, and redoing it on every one of up to 10 retries
  // per iteration was reproduced as a real, severe cost: on a 60-atom
  // molecule (~200 internal coordinates), 300 iterations of the caller's
  // retry loop took OVER 8 MINUTES wall-clock (vs. well under a minute for
  // the same iteration budget on the existing Cartesian L-BFGS optimizer)
  // -- this cache is what makes that tractable. computeHEigen does the
  // decomposition once per outer iteration; onSphereStepFromEigen below
  // reuses it for however many trust values that iteration's retry loop
  // tries (the root-finding bisection itself is cheap, O(m) per trial
  // lambda, so only the decomposition needed hoisting).
  function computeHEigen(H, g, m) {
    const evd = new mlMatrix.EigenvalueDecomposition(H, { assumeSymmetric: true });
    const evals = evd.realEigenvalues;
    const V = evd.eigenvectorMatrix;
    const gEig = new Float64Array(m);
    for (let a = 0; a < m; a++) {
      let s = 0;
      for (let b = 0; b < m; b++) s += V.get(b, a) * g[b];
      gEig[a] = s;
    }
    let minEval = evals[0];
    for (let a = 1; a < m; a++) if (evals[a] < minEval) minEval = evals[a];
    return { evals: evals, V: V, gEig: gEig, minEval: minEval };
  }

  function onSphereStepFromEigen(hEigen, g, trust, m) {
    const evals = hEigen.evals, V = hEigen.V, gEig = hEigen.gEig, minEval = hEigen.minEval;

    function stepNormMinusTrust(lambda) {
      let sumsq = 0;
      for (let a = 0; a < m; a++) {
        const denom = lambda - evals[a];
        const t = gEig[a] / denom;
        sumsq += t * t;
      }
      return Math.sqrt(sumsq) - trust;
    }

    // Bracket-search for f(lo)<=0 is capped, not open-ended: f should
    // always go negative as lambda -> -infinity (every term's denominator
    // grows without bound), but a corrupted/non-finite H (e.g. a
    // near-degenerate BFGS update -- see the caller's own guard against
    // that) could hand back NaN/Infinity eigenvalues that never satisfy
    // the comparison either way, which would otherwise spin forever. Falls
    // back to a plain steepest-descent step scaled to the trust radius --
    // safe, if not optimal -- rather than hang.
    let lo = minEval - Math.max(1, Math.abs(minEval)) - 1;
    let bracketOk = false;
    for (let guard = 0; guard < 200; guard++) {
      const f = stepNormMinusTrust(lo);
      if (!isFinite(f)) break;
      if (f <= 0) { bracketOk = true; break; }
      lo -= Math.max(1, Math.abs(lo));
    }
    if (!bracketOk) {
      let gNorm = 0;
      for (let a = 0; a < m; a++) gNorm += g[a] * g[a];
      gNorm = Math.sqrt(gNorm) || 1;
      const dq = new Float64Array(m);
      for (let a = 0; a < m; a++) dq[a] = -g[a] * (trust / gNorm);
      return dq;
    }
    let hi = minEval - 1e-8;
    for (let iter = 0; iter < 100; iter++) {
      const mid = (lo + hi) / 2;
      if (stepNormMinusTrust(mid) > 0) hi = mid; else lo = mid;
    }
    const lambda = (lo + hi) / 2;
    const dqEig = new Float64Array(m);
    for (let a = 0; a < m; a++) dqEig[a] = gEig[a] / (lambda - evals[a]);
    const dq = new Float64Array(m);
    for (let b = 0; b < m; b++) {
      let s = 0;
      for (let a = 0; a < m; a++) s += V.get(b, a) * dqEig[a];
      dq[b] = s;
    }
    return dq;
  }

  // The augmented-matrix eigendecomposition (the in-trust RFO candidate)
  // is ALSO trust-independent -- same reasoning as computeHEigen above:
  // g and H don't change across retries, so this candidate step and its
  // norm are the same for every retry too. Only whether norm <= trust
  // (i.e. whether the in-trust candidate is used as-is, or the on-sphere
  // fallback is needed) depends on the per-retry trust value.
  function computeRfoCandidate(g, H, m) {
    const aug = new mlMatrix.Matrix(m + 1, m + 1);
    for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) aug.set(a, b, H.get(a, b));
    for (let a = 0; a < m; a++) { aug.set(a, m, g[a]); aug.set(m, a, g[a]); }
    aug.set(m, m, 0);
    const evd = new mlMatrix.EigenvalueDecomposition(aug, { assumeSymmetric: true });
    const idx = smallestEigenIndex(evd.realEigenvalues);
    const V = evd.eigenvectorMatrix;
    const last = V.get(m, idx);
    const dq = new Float64Array(m);
    if (Math.abs(last) > 1e-10) {
      for (let a = 0; a < m; a++) dq[a] = V.get(a, idx) / last;
    }
    let normDq = 0;
    for (let a = 0; a < m; a++) normDq += dq[a] * dq[a];
    normDq = Math.sqrt(normDq);
    return { dq: dq, norm: normDq };
  }

  // rfoCandidate/hEigenCache: computed once per outer iteration by the
  // caller (see minimize()'s retry loop) and threaded through every retry
  // within that iteration -- hEigenCache is a { value: null } box the
  // caller passes fresh each outer iteration, filled in lazily here only
  // if/when the on-sphere fallback is actually needed (many retries never
  // need it: if the FIRST trial trust already accepts the in-trust
  // candidate, computeHEigen is never called at all).
  function rfoStepCached(rfoCandidate, hEigenCache, g, H, trust, m) {
    if (rfoCandidate.norm > 1e-10 && rfoCandidate.norm <= trust) return rfoCandidate.dq;
    if (!hEigenCache.value) hEigenCache.value = computeHEigen(H, g, m);
    return onSphereStepFromEigen(hEigenCache.value, g, trust, m);
  }

  // ---------- iterative internal -> Cartesian back-transformation ----------
  // Ported from PyBerny's InternalCoords.update_geom (coords.py): B+ is
  // held fixed at the STARTING geometry for the whole inner loop (cheaper
  // than re-differentiating B every sub-iteration; fine since dq is
  // usually modest within a trust region) -- apply the full remaining dq,
  // measure how much of it was actually achieved (internal coordinates are
  // curvilinear, so a linear B+ . dq step never lands exactly on target),
  // subtract that off, repeat until the Cartesian displacement itself goes
  // to ~zero or 20 iterations are exhausted (falling back to the first
  // iteration's result, same as PyBerny's own keep_first).

  function backTransform(atoms, coordList, Bpinv, dqTarget, qCurrent) {
    let working = atoms.map(function (a) { return { element: a.element, x: a.x, y: a.y, z: a.z }; });
    const n = atoms.length;
    let q = qCurrent.slice();
    let dq = dqTarget.slice();
    let firstResult = null;
    for (let iter = 0; iter < 20; iter++) {
      const dqM = mlMatrix.Matrix.columnVector(dq);
      const dxM = Bpinv.mmul(dqM);
      let dcartSumSq = 0;
      for (let a = 0; a < n; a++) {
        const dx = dxM.get(3 * a, 0), dy = dxM.get(3 * a + 1, 0), dz = dxM.get(3 * a + 2, 0);
        working[a].x += dx; working[a].y += dy; working[a].z += dz;
        dcartSumSq += dx * dx + dy * dy + dz * dz;
      }
      const dcartRms = Math.sqrt(dcartSumSq / (3 * n));
      const qNew = evalCoordValues(working, coordList);
      const nextDq = new Float64Array(coordList.length);
      for (let r = 0; r < coordList.length; r++) {
        let achieved = qNew[r] - q[r];
        if (coordList[r].type === 'dihedral') achieved = wrapAngle(achieved);
        nextDq[r] = dq[r] - achieved;
      }
      q = qNew;
      dq = nextDq;
      if (iter === 0) {
        firstResult = { atoms: working.map(function (a) { return { element: a.element, x: a.x, y: a.y, z: a.z }; }), q: q.slice() };
      }
      if (dcartRms < 1e-7) return { atoms: working, q: q, converged: true };
    }
    return firstResult ? { atoms: firstResult.atoms, q: firstResult.q, converged: false } : { atoms: working, q: q, converged: false };
  }

  // ---------- main loop ----------

  /**
   * energyGradFn(atoms3d) -> { energy, grad: Float64Array(3*n) } -- the same
   * shape as this app's other optimizers' energy/gradient calls, so a
   * caller can plug in EITHER force field (SMIRNOFF's computeEnergySMIRNOFF/
   * gradientSMIRNOFF, or embed3d.js's classical ones) without this file
   * knowing anything about SMIRNOFF/classical specifics.
   *
   * VALIDATED before shipping: buildBMatrix's per-coordinate-type gradient
   * (bond/angle analytic per PyBerny's own formulas, dihedral reusing this
   * app's already-validated dihedralGradient) checked against central
   * finite differences of evalCoord's own value, and projectGradient's
   * B+ -based Cartesian<->internal round-trip checked for self-consistency,
   * on real molecules before wiring into openff-forcefield.js -- see
   * project memory / commit message for the specific numbers.
   *
   * Returns { atoms, energy, gradNorm, converged, exitReason, iterationsRun,
   * gradNormHistory }. gradNorm/converged use the CARTESIAN gradient norm
   * with the SAME 1e-5 threshold this app's Cartesian L-BFGS optimizers
   * already use, so results are directly comparable between optimizers.
   */
  async function _minimizeCore(atoms3d, bonds3d, energyGradFn, opts) {
    opts = opts || {};
    const maxIterations = opts.maxIterations || 100;
    const gradTol = opts.gradTol || 1e-5;
    let trust = opts.trustInit || 0.1;
    const onProgress = opts.onProgress;
    const stopToken = opts.stopToken;

    let atoms = atoms3d.map(function (a) { return { element: a.element, x: a.x, y: a.y, z: a.z }; });
    const coordSet = buildCoordSet(atoms, bonds3d).all;
    if (coordSet.length === 0) {
      const seed = energyGradFn(atoms);
      return { atoms: atoms, energy: seed.energy, gradNorm: 0, converged: true, exitReason: 'no-internal-coordinates', iterationsRun: 0, gradNormHistory: [] };
    }
    const m = coordSet.length;

    let H = mlMatrix.Matrix.diag(Array.from(guessHessianDiag(atoms, coordSet)));
    let built = buildBMatrix(atoms, coordSet);
    let q = built.q;
    let B = built.B;
    let Bpinv = mlMatrix.pseudoInverse(built.B);
    let current = energyGradFn(atoms);
    let energy = current.energy;
    let grad = current.grad;
    let gq = projectGradient(Bpinv, grad);

    const gradNormHistory = [];
    let exitReason = 'iteration-limit';
    let iterationsRun = 0;

    function cartGradNorm(g) {
      let s = 0;
      for (let i = 0; i < g.length; i++) s += g[i] * g[i];
      return Math.sqrt(s);
    }

    for (let iter = 0; iter < maxIterations; iter++) {
      iterationsRun = iter + 1;
      if (stopToken && stopToken.stopped) { exitReason = 'user-stopped'; break; }
      const gradNorm = cartGradNorm(grad);
      const shouldReport = iter > 0 && iter % 5 === 0;
      if (shouldReport) {
        await CC.Embed3DShared.yieldToUI();
        if (onProgress) onProgress({ iteration: iter, gradNorm: gradNorm, energy: energy });
      }
      gradNormHistory.push({ iteration: iter, gradNorm: gradNorm, energy: energy });
      if (gradNorm < gradTol) { exitReason = 'gradient-converged'; break; }

      let accepted = false;
      let attemptTrust = trust;
      let acceptedTrust = trust;
      let newAtoms = null, newEnergy = null, newGrad = null, newQ = null, newB = null, newBpinv = null, newGq = null;
      // Project H onto B's actual row space before every RFO step (see
      // projectedHessian's own comment) -- once per outer iteration, same
      // caching reasoning as rfoCandidate/hEigenCache below: B/Bpinv are
      // fixed for the whole retry loop, only attemptTrust changes.
      const Hproj = projectedHessian(H, B, Bpinv, m);
      // gq/Hproj are fixed for the whole retry loop below -- only
      // attemptTrust changes per retry -- so both eigendecompositions
      // rfoStepCached needs are computed AT MOST once per outer iteration,
      // not once per retry (see rfoStepCached's own comment for why that
      // mattered).
      const rfoCandidate = computeRfoCandidate(gq, Hproj, m);
      const hEigenCache = { value: null };
      // Depth matters, not just presence of a retry loop: embed3d.js's/
      // openff-forcefield.js's own Cartesian line search backtracks its
      // step length from 1.0 down to a hard 1e-10 floor before giving up
      // (~33 halvings) -- this loop used to cap at a flat 10 retries
      // (~1024x total shrink from whatever `trust` started at), nowhere
      // near as fine-grained, and measurably the reason this optimizer
      // gave up via 'step-too-small' well before the Cartesian optimizer's
      // line search would have on the SAME landscape: reproduced directly
      // on several real molecules where this optimizer stalled at a
      // meaningfully larger residual gradient than the Cartesian one
      // reached on an equal iteration budget, immediately after the Berny
      // convergence-criteria fix made that gap visible for the first time
      // (previously masked by the Cartesian optimizer's OWN premature-
      // convergence bug). TRUST_FLOOR matches that same 1e-10 order of
      // magnitude; the retry-count cap alongside it is a pure safety
      // bound (never expected to bind -- halving 0.3 down past 1e-10
      // takes ~32 retries), not a second, tighter limit competing with it.
      const TRUST_FLOOR = 1e-10;
      for (let retry = 0; retry < 60 && !accepted && attemptTrust > TRUST_FLOOR; retry++) {
        const dq = rfoStepCached(rfoCandidate, hEigenCache, gq, Hproj, attemptTrust, m);
        const bt = backTransform(atoms, coordSet, Bpinv, dq, q);
        if (hasNearLinearAngle(bt.atoms, coordSet)) { attemptTrust *= 0.5; continue; }
        // Check the REAL Cartesian consequence directly (see MAX_CART_STEP's
        // own comment) rather than trusting attemptTrust's internal-
        // coordinate norm to predict it -- cheap, since bt.atoms is already
        // computed above; catches an angle/dihedral-dominated step's
        // lever-arm amplification before wasting an energy/gradient
        // evaluation on a step already known to be too large.
        let maxCartStep = 0;
        for (let a = 0; a < atoms.length; a++) {
          const dx = bt.atoms[a].x - atoms[a].x, dy = bt.atoms[a].y - atoms[a].y, dz = bt.atoms[a].z - atoms[a].z;
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (d > maxCartStep) maxCartStep = d;
        }
        if (maxCartStep > MAX_CART_STEP) { attemptTrust *= 0.5; continue; }
        const evalResult = energyGradFn(bt.atoms);
        const dE = evalResult.energy - energy;
        if (isFinite(dE) && dE <= 1e-8) {
          accepted = true;
          acceptedTrust = attemptTrust;
          newAtoms = bt.atoms;
          newEnergy = evalResult.energy;
          newGrad = evalResult.grad;
          newQ = bt.q;
          const rebuilt = buildBMatrix(newAtoms, coordSet);
          newB = rebuilt.B;
          newBpinv = mlMatrix.pseudoInverse(rebuilt.B);
          newGq = projectGradient(newBpinv, newGrad);
          // Trust for the NEXT iteration is now set after this loop via
          // fletcherTrustUpdate (ported from PyBerny's own update_trust,
          // checked against source) -- comparing actual vs. quadratic-
          // model-predicted energy change, not a flat retry-count-based
          // grow/shrink heuristic. See that function's own comment for why
          // this is more principled; see file header for the debugging
          // history that motivated the EARLIER flat-heuristic version this
          // replaces.
        } else {
          attemptTrust *= 0.5;
        }
      }
      if (!accepted) {
        // Never force-accept a worse step (a real, reproduced failure
        // mode: near a coordinate that's drifted close to its own
        // near-linear singularity -- see LIN_THRESHOLD and the file
        // header's note on not dynamically rebuilding the coordinate set
        // -- the pseudo-inverse becomes so ill-conditioned that even the
        // smallest retried trust radius still maps to a wildly distorted
        // Cartesian step, and blindly accepting it sent a real
        // optimization run's energy from ~150 kcal/mol to 538537 in one
        // step before this fix). Same "stop rather than corrupt the
        // state" precedent embed3d.js's/openff-forcefield.js's own
        // Cartesian L-BFGS already uses for an exhausted line search.
        exitReason = 'step-too-small';
        break;
      }

      const dq = new Float64Array(m);
      const dgq = new Float64Array(m);
      for (let a = 0; a < m; a++) {
        let d = newQ[a] - q[a];
        if (coordSet[a].type === 'dihedral') d = wrapAngle(d);
        dq[a] = d;
        dgq[a] = newGq[a] - gq[a];
      }
      H = bfgsUpdate(H, dq, dgq);

      // dE predicted by the SAME quadratic model (gq, Hproj) the accepted
      // step was actually chosen from -- dEPredicted = g.dq + 0.5 dq.H.dq,
      // exactly PyBerny's own quadratic_step() formula.
      let dqDotHproj = 0;
      const HprojDq = Hproj.mmul(mlMatrix.Matrix.columnVector(dq));
      for (let a = 0; a < m; a++) dqDotHproj += dq[a] * HprojDq.get(a, 0);
      let gDotDq = 0;
      for (let a = 0; a < m; a++) gDotDq += gq[a] * dq[a];
      const dEPredicted = gDotDq + 0.5 * dqDotHproj;
      const dqNorm = cartGradNorm(dq);
      // A sane ceiling PyBerny's own update_trust doesn't need (its trust
      // radius is the ONLY thing standing between a bad quadratic model
      // and an oversized step) but this file still benefits from as a
      // second layer: MAX_CART_STEP now independently catches any
      // oversized step's real Cartesian consequence regardless of what
      // internal-coordinate trust produced it, so an unbounded trust
      // isn't directly dangerous anymore -- this cap is just defensive
      // headroom, not the primary safety mechanism it used to be.
      trust = Math.min(fletcherTrustUpdate(acceptedTrust, newEnergy - energy, dEPredicted, dqNorm), 2.0);

      // Standard 4-criterion Berny/Gaussian check (see its definition in
      // embed3d.js for why this replaces trusting a raw gradient-norm
      // threshold alone). Deliberately uses the CARTESIAN displacement
      // this internal-coordinate step actually produced (newAtoms -
      // atoms), not dq (the internal-coordinate step) -- PyBerny's own
      // convergence check works the same way: internal coordinates are
      // the SEARCH space, but "has the geometry stopped moving" is a
      // Cartesian question, and mixing units (bond-Angstrom, angle-
      // radian, dihedral-radian all in the same dq vector) into one
      // max/RMS pair wouldn't mean anything physical anyway.
      const cartStep = new Float64Array(atoms.length * 3);
      for (let a = 0; a < atoms.length; a++) {
        cartStep[3 * a] = newAtoms[a].x - atoms[a].x;
        cartStep[3 * a + 1] = newAtoms[a].y - atoms[a].y;
        cartStep[3 * a + 2] = newAtoms[a].z - atoms[a].z;
      }
      const bernyDone = CC.Embed3DShared.bernyConvergence(newGrad, cartStep).converged;

      atoms = newAtoms; energy = newEnergy; grad = newGrad; gq = newGq; q = newQ; B = newB; Bpinv = newBpinv;

      if (bernyDone) { exitReason = 'gradient-converged'; break; }

      if (dqNorm < 1e-8 && gradNorm > gradTol) {
        exitReason = 'step-too-small';
        break;
      }
    }

    return {
      atoms: atoms,
      energy: energy,
      gradNorm: cartGradNorm(grad),
      converged: exitReason === 'gradient-converged',
      exitReason: exitReason,
      iterationsRun: iterationsRun,
      gradNormHistory: gradNormHistory,
    };
  }

  // Verification restart wrapping _minimizeCore -- a real, reproduced
  // false-convergence failure motivated this, not a hypothetical one: on
  // a hindered biaryl (2,2'-dimethylbiphenyl), _minimizeCore sometimes
  // exits 'gradient-converged' at a geometry ~16 kcal/mol ABOVE the real
  // nearby minimum, and simply calling _minimizeCore AGAIN from that
  // exact "converged" geometry (fresh Hessian, fresh coordinate set)
  // immediately drops straight back down to the real minimum -- proven
  // directly: perturbing that "converged" geometry's central dihedral by
  // anywhere from -20 to +20 degrees and re-relaxing landed at the SAME
  // real minimum every time, none stayed near the false one. That rules
  // out a genuine second atropisomer basin; the Berny 4-criterion check
  // (gradient AND displacement both small) was satisfied at a point that
  // wasn't actually a stationary point, most likely because a
  // near-degenerate/ridge-like region of the PES let trust collapse to
  // something tiny before the gradient had actually gone to zero, which
  // reads as "converged" under a criterion that never checks curvature
  // sign (no check that this is truly a minimum, not a saddle-like
  // shoulder). A single re-run from a FRESH Hessian is a standard,
  // pragmatic verification technique for exactly this failure mode
  // (analogous to why real QM codes report imaginary frequencies at a
  // nominal "converged" structure) -- cheaper and more robust than trying
  // to diagnose and fix whatever specific numerical path let trust
  // collapse prematurely in the first place.
  CC.InternalOpt.minimize = async function (atoms3d, bonds3d, energyGradFn, opts) {
    opts = opts || {};
    const maxIterations = opts.maxIterations || 100;
    let remaining = maxIterations;
    let result = await _minimizeCore(atoms3d, bonds3d, energyGradFn, Object.assign({}, opts, { maxIterations: remaining }));
    remaining -= result.iterationsRun;

    const MAX_VERIFICATIONS = 3;
    const IMPROVEMENT_TOLERANCE = 1e-4; // kcal/mol -- below this, treat as genuinely converged
    for (let v = 0; v < MAX_VERIFICATIONS && result.exitReason === 'gradient-converged' && remaining > 5; v++) {
      const verifyIters = Math.min(remaining, Math.max(50, Math.round(maxIterations * 0.2)));
      const retry = await _minimizeCore(atoms3d, bonds3d, energyGradFn, Object.assign({}, opts, { maxIterations: verifyIters }));
      remaining -= retry.iterationsRun;
      const improved = result.energy - retry.energy > IMPROVEMENT_TOLERANCE;
      if (improved) {
        result = { atoms: retry.atoms, energy: retry.energy, gradNorm: retry.gradNorm, converged: retry.converged, exitReason: retry.exitReason, iterationsRun: result.iterationsRun + retry.iterationsRun, gradNormHistory: result.gradNormHistory.concat(retry.gradNormHistory) };
      } else {
        break; // verification found no real further improvement -- genuinely converged
      }
    }
    return result;
  };

  CC.InternalOpt.buildCartesianPreconditioner = buildCartesianPreconditioner;

  // Exposed for testing/validation only.
  CC.InternalOpt._internal = {
    buildCoordSet: buildCoordSet,
    evalCoord: evalCoord,
    buildBMatrix: buildBMatrix,
    guessHessianDiag: guessHessianDiag,
    projectGradient: projectGradient,
    backTransform: backTransform,
  };
})();
