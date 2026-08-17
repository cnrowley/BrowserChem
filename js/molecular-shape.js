/**
 * molecular-shape.js
 *
 * Quantitative 3D shape descriptors computed directly from a real 3D
 * conformer's atom positions -- pure linear algebra (eigenanalysis of a
 * 3x3 symmetric matrix), not a trained model or an RDKit call.
 *
 * - PBF (Plane of Best Fit): Firth, Brown & Blagg 2012, "Plane of Best
 *   Fit: A Novel Method to Characterize the Three-Dimensionality of
 *   Molecules" -- the field's standard quantitative "how non-planar is
 *   this molecule" metric (the same one behind Lovering et al.'s
 *   "Escape from Flatland" framing). Fits the least-squares plane
 *   through every atom (the plane whose normal is the eigenvector of the
 *   position covariance matrix with the SMALLEST eigenvalue -- the
 *   direction of least spread), then reports the mean absolute distance
 *   of every atom from that plane, in Å. 0 = perfectly flat; RDKit's own
 *   CalcPBF computes the identical quantity the identical way (confirmed
 *   against its documented definition), but isn't implemented here via
 *   RDKit.js since it's cheap enough, and this app's atom positions are
 *   already sitting in a plain JS array from embed3d.js.
 *
 * - NPR1/NPR2 (Normalized Principal Moment Ratios): Sauer & Schwarz
 *   2003, "Molecular Shape Diversity of Combinatorial Libraries: A
 *   Prospective Analysis" -- from the mass-weighted inertia tensor's
 *   three principal moments I1<=I2<=I3, NPR1=I1/I3 and NPR2=I2/I3 place
 *   a molecule on the classic rod (0,1) / disc (0.5,0.5) / sphere (1,1)
 *   shape triangle. Complementary to PBF: PBF asks "how far off a single
 *   plane", NPR asks "what's the overall 3D shape" (a rod and a sphere
 *   can both have zero-ish PBF... no, actually a rod is 1D so it IS
 *   planar in infinitely many planes containing its axis, while a sphere
 *   is not planar at all -- NPR tells those apart, PBF alone can't).
 *
 * Both need the SAME eigenanalysis machinery (closed-form eigenvalues of
 * a symmetric 3x3 matrix, the standard trigonometric/Smith-1961 method),
 * shared here rather than duplicated.
 *
 * Correctness of the eigenvector extraction (not just the eigenvalues)
 * was NOT obvious to get right on the first pass: a fully isotropic
 * point cloud (e.g. atoms at alternating corners of a cube -- a regular
 * tetrahedron) makes (covariance - lambda*I) the exact zero matrix for
 * every eigenvalue, so the usual "cross two rows of (M - lambda*I)"
 * eigenvector trick degenerates to the zero vector -- which silently
 * zeroed out PBF for that case (reporting a genuinely 3D tetrahedron as
 * perfectly flat) until caught by testing against a regular tetrahedron
 * specifically (expected PBF clearly > 0, NPR1=NPR2=1.0 exactly by
 * symmetry) and fixed by falling back to an arbitrary fixed unit vector
 * in that case -- mathematically correct, since a fully isotropic
 * distribution's mean-absolute-plane-distance is the same in every
 * direction by symmetry. Validated against five hand-checkable cases
 * (planar point set, regular tetrahedron, a linear "rod", a flat "disc")
 * before shipping, not just eyeballed.
 */

window.CC = window.CC || {};
CC.Shape = window.CC.Shape || {};

(function () {
  function cross(u, v) {
    return { x: u.y * v.z - u.z * v.y, y: u.z * v.x - u.x * v.z, z: u.x * v.y - u.y * v.x };
  }

  function normalize(v) {
    const len = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return len < 1e-12 ? { x: 0, y: 0, z: 0 } : { x: v.x / len, y: v.y / len, z: v.z / len };
  }

  // Closed-form eigenvalues of the symmetric 3x3 matrix
  // [[a,d,e],[d,b,f],[e,f,c]] -- the standard trigonometric method
  // (equivalent to Smith 1961's algorithm). Returns [big, mid, small]
  // (descending).
  function symmetricEigenvalues3x3(a, b, c, d, e, f) {
    const offDiagSq = d * d + e * e + f * f;
    if (offDiagSq < 1e-12) {
      // Already diagonal -- eigenvalues are just the diagonal entries.
      return [a, b, c].sort(function (x, y) { return y - x; });
    }
    const q = (a + b + c) / 3;
    const p2 = (a - q) * (a - q) + (b - q) * (b - q) + (c - q) * (c - q) + 2 * offDiagSq;
    const p = Math.sqrt(p2 / 6);
    // B = (M - q*I) / p
    const ba = (a - q) / p, bb = (b - q) / p, bc = (c - q) / p;
    const bd = d / p, be = e / p, bf = f / p;
    const detB = ba * (bb * bc - bf * bf) - bd * (bd * bc - bf * be) + be * (bd * bf - bb * be);
    const r = Math.max(-1, Math.min(1, detB / 2));
    const phi = Math.acos(r) / 3;
    const eig1 = q + 2 * p * Math.cos(phi);
    const eig3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
    const eig2 = 3 * q - eig1 - eig3; // trace is preserved
    return [eig1, eig2, eig3];
  }

  // Eigenvector of the same matrix for a given (already-computed)
  // eigenvalue -- the null space of (M - lambda*I), found as the cross
  // product of two of its rows (whichever pair is least parallel, for
  // numerical stability). See file header for the degenerate-matrix
  // fallback this needed.
  function symmetricEigenvector3x3(a, b, c, d, e, f, lambda) {
    const r0 = { x: a - lambda, y: d, z: e };
    const r1 = { x: d, y: b - lambda, z: f };
    const r2 = { x: e, y: f, z: c - lambda };
    const candidates = [cross(r0, r1), cross(r0, r2), cross(r1, r2)];
    let best = null, bestLenSq = 1e-9;
    candidates.forEach(function (v) {
      const lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
      if (lenSq > bestLenSq) { bestLenSq = lenSq; best = v; }
    });
    return best ? normalize(best) : { x: 0, y: 0, z: 1 };
  }

  /**
   * Plane of Best Fit, in Å. `atoms`: array of {x,y,z} (element not
   * needed -- unlike NPR, PBF doesn't mass-weight). Uses every atom
   * passed in, including implicit H if present in the array (matching
   * RDKit's own CalcPBF convention) -- callers decide whether to pass a
   * heavy-only or heavy+H atom list.
   */
  CC.Shape.planeOfBestFit = function (atoms) {
    const n = atoms.length;
    if (n < 3) return 0;

    let cx = 0, cy = 0, cz = 0;
    atoms.forEach(function (a) { cx += a.x; cy += a.y; cz += a.z; });
    cx /= n; cy /= n; cz /= n;

    let Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;
    atoms.forEach(function (a) {
      const x = a.x - cx, y = a.y - cy, z = a.z - cz;
      Sxx += x * x; Syy += y * y; Szz += z * z;
      Sxy += x * y; Sxz += x * z; Syz += y * z;
    });

    const eigs = symmetricEigenvalues3x3(Sxx, Syy, Szz, Sxy, Sxz, Syz); // descending
    const lambdaMin = eigs[2];
    const normal = symmetricEigenvector3x3(Sxx, Syy, Szz, Sxy, Sxz, Syz, lambdaMin);

    let sumAbsDist = 0;
    atoms.forEach(function (a) {
      const x = a.x - cx, y = a.y - cy, z = a.z - cz;
      sumAbsDist += Math.abs(x * normal.x + y * normal.y + z * normal.z);
    });
    return sumAbsDist / n;
  };

  // Real atomic masses (amu) -- same values as this project's other
  // mass-weighted calculations (chemprop-features.js's mass feature).
  const ATOMIC_MASS = {
    H: 1.008, C: 12.011, N: 14.007, O: 15.999, F: 18.998,
    P: 30.974, S: 32.06, Cl: 35.45, Br: 79.904, I: 126.904,
  };

  /**
   * { npr1, npr2 } -- Normalized Principal Moment Ratios from the
   * mass-weighted inertia tensor. `atoms`: array of {x,y,z,element}.
   */
  CC.Shape.principalMomentRatios = function (atoms) {
    const n = atoms.length;
    if (n < 2) return { npr1: 0, npr2: 0 };

    let totalMass = 0, cx = 0, cy = 0, cz = 0;
    atoms.forEach(function (a) {
      const m = ATOMIC_MASS[a.element] || ATOMIC_MASS.C;
      totalMass += m; cx += m * a.x; cy += m * a.y; cz += m * a.z;
    });
    cx /= totalMass; cy /= totalMass; cz /= totalMass;

    let Ixx = 0, Iyy = 0, Izz = 0, Ixy = 0, Ixz = 0, Iyz = 0;
    atoms.forEach(function (a) {
      const m = ATOMIC_MASS[a.element] || ATOMIC_MASS.C;
      const x = a.x - cx, y = a.y - cy, z = a.z - cz;
      Ixx += m * (y * y + z * z); Iyy += m * (x * x + z * z); Izz += m * (x * x + y * y);
      Ixy -= m * x * y; Ixz -= m * x * z; Iyz -= m * y * z;
    });

    const eigs = symmetricEigenvalues3x3(Ixx, Iyy, Izz, Ixy, Ixz, Iyz);
    const sorted = eigs.slice().sort(function (a, b) { return a - b; }); // ascending: I1 <= I2 <= I3
    const I1 = sorted[0], I2 = sorted[1], I3 = sorted[2];
    if (I3 < 1e-9) return { npr1: 0, npr2: 0 };
    return { npr1: I1 / I3, npr2: I2 / I3 };
  };

  /**
   * Best-fit RMSD (Angstrom) between two conformers of the SAME molecule
   * -- same atom count, same index correspondence (both must come from
   * the same withImplicitHydrogens()-style atom ordering; no graph
   * matching is attempted). Used by js/conformer-search.js to deduplicate
   * an ensemble the way real conformer search tools do: superpose the two
   * structures with a least-squares (Kabsch) rotation+translation fit
   * first, THEN measure the residual atom-position spread -- comparing
   * raw, unaligned coordinates would report two identical conformers that
   * simply ended up translated/rotated in space as wildly different.
   *
   * Reuses geomol-assembly.js's own Kabsch/SVD implementation
   * (CC.GeoMol._internal, already validated there for stitching local
   * conformer fragments together) rather than a second from-scratch
   * SVD -- that's the only nontrivial numerical piece this needs.
   *
   * opts.heavyOnly (default false) restricts both the fit and the
   * reported RMSD to non-hydrogen atoms -- CREST's own CREGEN conformer
   * pruning defaults to all-atom RMSD (heavy-atom is an opt-in flag
   * there too, confirmed directly against its source), which is this
   * function's default for the same reason: freely-rotating terminal
   * methyl/hydroxyl hydrogens genuinely do occupy different positions
   * between two otherwise-identical heavy-atom conformers, and CREST
   * doesn't pretend that's not real geometry by default either.
   */
  CC.Shape.rmsd = function (atomsA, atomsB, opts) {
    opts = opts || {};
    if (atomsA.length !== atomsB.length || atomsA.length === 0) return Infinity;
    const mask = atomsA.map(function (a) { return !opts.heavyOnly || a.element !== 'H'; });
    if (!mask.some(Boolean)) return 0; // nothing to compare (e.g. heavyOnly on a bare H2 fragment)

    const aligned = CC.GeoMol._internal.kabschAlign(atomsB, atomsA, mask);

    let sumSq = 0, count = 0;
    for (let i = 0; i < atomsA.length; i++) {
      if (!mask[i]) continue;
      const dx = atomsA[i].x - aligned[i].x, dy = atomsA[i].y - aligned[i].y, dz = atomsA[i].z - aligned[i].z;
      sumSq += dx * dx + dy * dy + dz * dz;
      count++;
    }
    return Math.sqrt(sumSq / count);
  };
})();
