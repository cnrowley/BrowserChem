/**
 * embed3d.js
 *
 * Turns a 2D Molecule into a 3D structure:
 *   1. Add implicit hydrogens based on default valence.
 *   2. Seed 3D coordinates from the 2D layout.
 *   3. For each conformer attempt, randomize the dihedral angle of every
 *      real rotatable bond (single, acyclic, both ends substituted) --
 *      genuine conformer *search*, not just numerical noise -- then
 *      relax with a UFF/MMFF-style force field: harmonic bond stretching,
 *      harmonic angle bending, periodic torsion terms, and a proper
 *      12-6 Lennard-Jones nonbonded potential (with 1-4 scaling) built
 *      on real van der Waals radii.
 *   4. Keep the lowest-energy conformer found across all attempts.
 *
 * This is NOT literally UFF or MMFF94 -- it doesn't use either force
 * field's actual per-atom-type parameter tables (which are large,
 * published, and not something to approximate from memory without
 * risking silently-wrong numbers). It uses the same *functional form*
 * every real force field of this style uses (harmonic bonds/angles,
 * periodic torsions, 12-6 LJ with 1-4 scaling) with hand-tuned constants
 * aimed at "produces a chemically sane, non-overlapping 3D structure,"
 * not quantitative accuracy. If you need real UFF/MMFF94 energies (not
 * just a reasonable-looking structure), that needs an actual
 * cheminformatics toolkit's compiled force field code -- RDKit.js's
 * browser build doesn't include one (checked directly: no 3D embedding
 * or force-field method exists anywhere in its API, confirmed against
 * the actual @rdkit/rdkit package, not assumed).
 *
 * The whole thing uses numeric (finite-difference) gradients rather than
 * hand-derived analytic ones -- more compute per step, much easier to
 * keep correct across four different energy terms, and still fast enough
 * for editor-sized (including drug-sized) molecules given the time-budget
 * approach below.
 */

window.CC = window.CC || {};

(function () {
  const DEG2RAD = Math.PI / 180;
  const K_BOND = 300;
  const K_ANGLE = 40;
  const GRAD_H = 1e-4;

  // Periodic torsion force constants -- see file header: hand-tuned for
  // "chemically sane structure," not sourced from a published parameter
  // table. Relative *ordering* is the chemically important part: a
  // conjugated pi system should resist twisting much more strongly than
  // a freely-rotating single bond, which these reflect.
  const V_TORSION_CONJUGATED = 8; // double bond / aromatic -- strong planarity preference
  const V_TORSION_SP3 = 1.2; // sp3-sp3 single bond -- staggered-preferring, real barrier
  const V_TORSION_LOW = 0.3; // sp2-sp3 (e.g. exocyclic ring bond) -- nearly free rotation

  const LJ_EPSILON = 0.15; // uniform well depth -- see file header, not per-element real data
  const LJ_SCALE_14 = 0.5; // standard force-field convention: soften (don't exclude) 1-4 nonbonded pairs

  // Below this fraction of (rm = sum of vdW radii), the raw 12-6 shape is
  // replaced by a quadratic continuation -- see ljShape() below for why.
  const LJ_FLOOR_FRAC = 0.85;
  // (rm/rFloor) is the same fixed ratio (1/LJ_FLOOR_FRAC) for every atom
  // pair, since rFloor is always LJ_FLOOR_FRAC*rm -- so the shape
  // function's value/slope/curvature *coefficients* at the floor are
  // universal constants, only the per-pair rFloor scale differs.
  const LJ_FLOOR_SR6 = Math.pow(1 / LJ_FLOOR_FRAC, 6);
  const LJ_FLOOR_F0 = LJ_FLOOR_SR6 * LJ_FLOOR_SR6 - 2 * LJ_FLOOR_SR6;
  const LJ_FLOOR_SLOPE_COEF = -12 * LJ_FLOOR_SR6 * (LJ_FLOOR_SR6 - 1); // slope at floor = this / rFloor
  const LJ_FLOOR_CURV_COEF = 156 * LJ_FLOOR_SR6 * LJ_FLOOR_SR6 - 84 * LJ_FLOOR_SR6; // curvature at floor = this / rFloor^2

  const K_OOP = 60; // out-of-plane (improper) force constant -- real sp2 centers are quite
                     // rigidly planar, comparable in stiffness to angle bending, not a soft preference

  // Same lone-pair-conjugation check already validated in
  // chemprop-features.js (ported here rather than shared, since
  // embed3d.js works on the post-H-expansion bonds3d/index
  // representation, not the app's own Molecule/atom-id model) --
  // deliberately restricted to N/O, not S: RDKit itself doesn't extend
  // this promotion to a singly-bonded exocyclic S (a thiophenol-type
  // sulfur stays sp3 even attached straight to an aromatic ring), a real
  // distinction found and confirmed against real RDKit output earlier in
  // this project, not a simplification made here.
  function hasConjugatedNeighbor3d(atoms3d, bonds3d, atomIndex, aromaticSet) {
    const nbrs = neighborsOf(bonds3d, atomIndex);
    return nbrs.some(function (n) {
      if (aromaticSet.has(n)) return true;
      const nNbrs = neighborsOf(bonds3d, n);
      // A hypervalent neighbor's formal double bond (phosphate P=O,
      // sulfonyl S=O -- anything with 4+ heavy-atom neighbors) isn't
      // true pi-conjugation the way a carbonyl's is.
      if (nNbrs.length >= 4) return false;
      return bonds3d.some(function (b) {
        if (b.order < 2) return false;
        const isThisBond = (b.a1 === n && b.a2 === atomIndex) || (b.a2 === n && b.a1 === atomIndex);
        if (isThisBond) return false;
        return b.a1 === n || b.a2 === n;
      });
    });
  }

  // atomIndex here is always a heavy-atom index (0..numHeavyAtoms-1,
  // withImplicitHydrogens always places heavy atoms first) -- aromaticSet
  // and the lone-pair check are only meaningful/needed for those; a
  // synthetic H node is never aromatic and never gets promoted by
  // conjugation, so this is never called for one in practice (H's bond
  // order is always 1, always sp3 via the plain bond-order path below).
  function hybridizationOf(atoms3d, bonds3d, atomIndex, aromaticSet) {
    const orders = bonds3d
      .filter(function (b) { return b.a1 === atomIndex || b.a2 === atomIndex; })
      .map(function (b) { return b.order; });
    if (orders.indexOf(3) !== -1) return 'sp';
    if (orders.indexOf(2) !== -1) return 'sp2';
    if (aromaticSet && aromaticSet.has(atomIndex)) return 'sp2';

    const element = atoms3d[atomIndex].element;
    if ((element === 'N' || element === 'O') && aromaticSet &&
        hasConjugatedNeighbor3d(atoms3d, bonds3d, atomIndex, aromaticSet)) {
      return 'sp2';
    }
    return 'sp3';
  }

  function idealAngleForAtom(atoms3d, bonds3d, atomIndex, aromaticSet) {
    const hyb = hybridizationOf(atoms3d, bonds3d, atomIndex, aromaticSet);
    if (hyb === 'sp') return 180 * DEG2RAD;
    if (hyb === 'sp2') return 120 * DEG2RAD;
    return 109.5 * DEG2RAD;
  }

  function neighborsOf(bonds3d, atomIndex) {
    const result = [];
    bonds3d.forEach(function (b) {
      if (b.a1 === atomIndex) result.push(b.a2);
      else if (b.a2 === atomIndex) result.push(b.a1);
    });
    return result;
  }

  // ---------- step 1: implicit hydrogens ----------

  // An arbitrary unit vector perpendicular to u, plus a second one
  // completing an orthonormal basis with it -- standard Gram-Schmidt
  // against a seed axis, falling back from z to x when u is itself
  // nearly parallel to z (the only case the z seed fails on).
  function computePerpendicularBasis(u) {
    const seed = Math.abs(u.z) < 0.9 ? { x: 0, y: 0, z: 1 } : { x: 1, y: 0, z: 0 };
    const p1 = CC.vec3.normalize(CC.vec3.cross(u, seed));
    const p2 = CC.vec3.cross(u, p1); // u, p1 already orthonormal -> p2 unit length free
    return [p1, p2];
  }

  // Real vector-geometry placement for implicit hydrogens, keyed off the
  // central atom's hybridization and how many *heavy* neighbor directions
  // are already known -- not random jitter. Each case is the standard
  // tetrahedral/trigonal-planar completion construction (the same one
  // real "add explicit hydrogens with 3D coordinates" toolkit code uses):
  // given the known bond direction(s), the missing one(s) are fully
  // determined up to an arbitrary azimuthal rotation when more than one
  // is missing and only one direction is known (e.g. a methyl group's
  // three H's can spin freely around the C-X axis before any force-field
  // relaxation -- any consistent choice is an equally valid starting
  // point). Verified by hand: every formula below reproduces the exact
  // ideal angle (dot product -1/3, i.e. 109.4712 degrees) between any two
  // placed H directions, and between each placed H and a known neighbor
  // direction where geometry pins that down (methyl, methylene).
  //
  // Returns null for any (hybridization, known-count, missing-count)
  // combination not explicitly covered below (e.g. a charged/hypervalent
  // center) -- the caller falls back to the previous
  // average-neighbor-direction-plus-jitter behavior for those, so
  // something reasonable always gets placed rather than throwing.
  function computeImplicitHDirections(neighborDirs, hybridization, count) {
    const known = neighborDirs.length;
    const TETRA_ANGLE = 109.4712206 * DEG2RAD;
    const TETRA_HALF = TETRA_ANGLE / 2;

    // sp, one known neighbor (triple bond or cumulated double bonds),
    // one H missing -- linear, straight out the back (terminal alkyne CH).
    if (hybridization === 'sp' && known === 1 && count === 1) {
      return [CC.vec3.scale(neighborDirs[0], -1)];
    }

    // sp2, two known neighbors, one H missing -- trigonal planar,
    // opposite the bisector (aromatic ring CH, vinyl CH, aldehyde CH).
    if (hybridization === 'sp2' && known === 2 && count === 1) {
      const bisector = CC.vec3.add(neighborDirs[0], neighborDirs[1]);
      if (CC.vec3.length(bisector) < 1e-6) return null; // antiparallel neighbors -- degenerate
      return [CC.vec3.scale(CC.vec3.normalize(bisector), -1)];
    }

    // sp2, one known neighbor, two H missing -- trigonal planar tripod
    // in an arbitrary plane containing the known bond (terminal =CH2).
    if (hybridization === 'sp2' && known === 1 && count === 2) {
      const u = neighborDirs[0];
      const p1 = computePerpendicularBasis(u)[0];
      const cos120 = Math.cos(120 * DEG2RAD), sin120 = Math.sin(120 * DEG2RAD);
      return [
        CC.vec3.normalize(CC.vec3.add(CC.vec3.scale(u, cos120), CC.vec3.scale(p1, sin120))),
        CC.vec3.normalize(CC.vec3.add(CC.vec3.scale(u, cos120), CC.vec3.scale(p1, -sin120))),
      ];
    }

    // sp3, three known neighbors, one H missing -- the fourth tetrahedron
    // corner is exactly opposite the other three's vector sum (methine).
    if (hybridization === 'sp3' && known === 3 && count === 1) {
      let sum = { x: 0, y: 0, z: 0 };
      neighborDirs.forEach(function (d) { sum = CC.vec3.add(sum, d); });
      if (CC.vec3.length(sum) < 1e-6) return null; // perfectly symmetric -- degenerate, no unique direction
      return [CC.vec3.scale(CC.vec3.normalize(sum), -1)];
    }

    // sp3, two known neighbors, two H missing -- the two new bonds sit
    // symmetric about the plane perpendicular to the two knowns' bisector,
    // tilted by half the tetrahedral angle either side of straight-back
    // (methylene).
    if (hybridization === 'sp3' && known === 2 && count === 2) {
      const u1 = neighborDirs[0], u2 = neighborDirs[1];
      const bisectorSum = CC.vec3.add(u1, u2);
      if (CC.vec3.length(bisectorSum) < 1e-6) return null; // antiparallel -- degenerate
      let n = CC.vec3.cross(u1, u2);
      if (CC.vec3.length(n) < 1e-6) return null; // parallel -- no well-defined perpendicular plane
      const negB = CC.vec3.scale(CC.vec3.normalize(bisectorSum), -1);
      n = CC.vec3.normalize(n);
      const cosH = Math.cos(TETRA_HALF), sinH = Math.sin(TETRA_HALF);
      return [
        CC.vec3.normalize(CC.vec3.add(CC.vec3.scale(negB, cosH), CC.vec3.scale(n, sinH))),
        CC.vec3.normalize(CC.vec3.add(CC.vec3.scale(negB, cosH), CC.vec3.scale(n, -sinH))),
      ];
    }

    // sp3, one known neighbor, three H missing -- a tetrahedral tripod
    // around the known bond axis, each H at the ideal 109.4712 degree
    // angle from it and 120 degrees apart from each other azimuthally
    // (methyl).
    if (hybridization === 'sp3' && known === 1 && count === 3) {
      const u = neighborDirs[0];
      const basis = computePerpendicularBasis(u);
      const p1 = basis[0], p2 = basis[1];
      const cosT = Math.cos(TETRA_ANGLE), sinT = Math.sin(TETRA_ANGLE);
      const dirs = [];
      for (let k = 0; k < 3; k++) {
        const az = k * 120 * DEG2RAD;
        const perp = CC.vec3.add(CC.vec3.scale(p1, Math.cos(az)), CC.vec3.scale(p2, Math.sin(az)));
        dirs.push(CC.vec3.normalize(CC.vec3.add(CC.vec3.scale(u, cosT), CC.vec3.scale(perp, sinT))));
      }
      return dirs;
    }

    // No known neighbors at all, four H missing (isolated methane) --
    // no reference direction to build from; a canonical tetrahedron
    // orientation (alternating cube corners) is as good as any other.
    if (known === 0 && count === 4) {
      return [
        CC.vec3.normalize({ x: 1, y: 1, z: 1 }),
        CC.vec3.normalize({ x: 1, y: -1, z: -1 }),
        CC.vec3.normalize({ x: -1, y: 1, z: -1 }),
        CC.vec3.normalize({ x: -1, y: -1, z: 1 }),
      ];
    }

    return null;
  }

  function withImplicitHydrogens(molecule, aromaticSet) {
    const heavyAtoms = Array.from(molecule.atoms.values());
    const idToIndex = new Map();
    const atoms3d = [];
    const bonds3d = [];

    heavyAtoms.forEach(function (a, i) {
      idToIndex.set(a.id, i);
      atoms3d.push({
        element: a.element,
        x: a.x / CC.BOND_LENGTH,
        y: -a.y / CC.BOND_LENGTH,
        z: (Math.random() - 0.5) * 0.3,
      });
    });

    molecule.bonds.forEach(function (b) {
      bonds3d.push({ a1: idToIndex.get(b.a1), a2: idToIndex.get(b.a2), order: b.order });
    });

    heavyAtoms.forEach(function (a) {
      const index = idToIndex.get(a.id);
      const data = CC.elementData(a.element);
      const bondedOrders = molecule.getBondsForAtom(a.id).map(function (b) { return b.order; });
      const usedValence = bondedOrders.reduce(function (sum, o) { return sum + o; }, 0);
      const implicitH = Math.max(0, data.valence - usedValence);
      if (implicitH === 0) return;

      const neighborIds = molecule.getBondsForAtom(a.id).map(function (b) {
        return b.a1 === a.id ? b.a2 : b.a1;
      });
      // Directions off the already-placed (jittered-z) 3D positions, not
      // a fresh z=0 read of the original 2D coordinates -- keeps this
      // consistent with where atoms3d[index] itself actually ended up.
      // Convention every computeImplicitHDirections() formula assumes:
      // neighborDirs[k] points FROM the central atom OUT TO neighbor k
      // (neighbor minus center), not the reverse.
      const neighborDirs = neighborIds.map(function (id) {
        const nIndex = idToIndex.get(id);
        return CC.vec3.normalize(CC.vec3.sub(atoms3d[nIndex], atoms3d[index]));
      });

      const hybridization = hybridizationOf(atoms3d, bonds3d, index, aromaticSet);
      const rHX = CC.idealBondLength(a.element, 'H', 1);
      const dirs = computeImplicitHDirections(neighborDirs, hybridization, implicitH);

      for (let h = 0; h < implicitH; h++) {
        let dir;
        if (dirs) {
          dir = dirs[h];
        } else {
          // Fallback for anything computeImplicitHDirections doesn't
          // explicitly cover (charged/hypervalent centers, unusual
          // valence/hybridization combinations) -- the same
          // average-neighbor-direction-plus-jitter approach this file
          // always used, not a hard failure.
          let baseDir = { x: 0, y: 0, z: 1 };
          if (neighborDirs.length > 0) {
            let sum = { x: 0, y: 0, z: 0 };
            neighborDirs.forEach(function (d) { sum = CC.vec3.add(sum, d); });
            if (CC.vec3.length(sum) > 1e-6) baseDir = CC.vec3.normalize(sum);
          }
          const jitter = {
            x: (Math.random() - 0.5) * 0.8,
            y: (Math.random() - 0.5) * 0.8,
            z: (Math.random() - 0.5) * 0.8 + 0.3,
          };
          dir = CC.vec3.normalize(CC.vec3.add(baseDir, jitter));
        }
        const hPos = CC.vec3.add(atoms3d[index], CC.vec3.scale(dir, rHX));
        const hIndex = atoms3d.length;
        atoms3d.push({ element: 'H', x: hPos.x, y: hPos.y, z: hPos.z });
        bonds3d.push({ a1: index, a2: hIndex, order: 1 });
      }
    });

    return { atoms3d: atoms3d, bonds3d: bonds3d };
  }

  // ---------- step 2: angle / torsion / nonbonded topology ----------

  function buildAngles(atoms3d, bonds3d, atomCount, aromaticSet) {
    const angles = [];
    for (let j = 0; j < atomCount; j++) {
      const nbrs = neighborsOf(bonds3d, j);
      const theta0 = idealAngleForAtom(atoms3d, bonds3d, j, aromaticSet);
      for (let a = 0; a < nbrs.length; a++) {
        for (let b = a + 1; b < nbrs.length; b++) {
          angles.push({ i: nbrs[a], j: j, k: nbrs[b], theta0: theta0 });
        }
      }
    }
    return angles;
  }

  // Proper torsions i-j-k-l for every bond (j,k) that has substituents on
  // both sides. Parameterized off the (j,k) bond order and hybridization
  // -- see the V_TORSION_* constants' comment for what this is and isn't
  // based on. Skipped entirely when either central atom is sp (linear --
  // "dihedral around a linear center" isn't a meaningful constraint, the
  // angle term already pins it straight).
  function buildTorsions(atoms3d, bonds3d, atomCount, aromaticSet) {
    const torsions = [];
    bonds3d.forEach(function (centralBond) {
      const j = centralBond.a1, k = centralBond.a2;
      const hybJ = hybridizationOf(atoms3d, bonds3d, j, aromaticSet);
      const hybK = hybridizationOf(atoms3d, bonds3d, k, aromaticSet);
      if (hybJ === 'sp' || hybK === 'sp') return;

      const isAtoms = neighborsOf(bonds3d, j).filter(function (n) { return n !== k; });
      const lsAtoms = neighborsOf(bonds3d, k).filter(function (n) { return n !== j; });
      if (isAtoms.length === 0 || lsAtoms.length === 0) return;

      let n, phi0, V;
      if (centralBond.order >= 2) {
        n = 2; phi0 = Math.PI; V = V_TORSION_CONJUGATED; // planar minima at 0, 180
      } else if (hybJ === 'sp3' && hybK === 'sp3') {
        n = 3; phi0 = 0; V = V_TORSION_SP3; // staggered minima at 60, 180, 300
      } else {
        n = 6; phi0 = 0; V = V_TORSION_LOW; // near-free rotation
      }

      isAtoms.forEach(function (i) {
        lsAtoms.forEach(function (l) {
          if (i === l) return; // degenerate in a 3-membered ring
          torsions.push({ i: i, j: j, k: k, l: l, n: n, phi0: phi0, V: V });
        });
      });
    });
    return torsions;
  }

  // Out-of-plane (improper) terms: one per sp2 atom with exactly 3
  // neighbors -- a real aromatic ring or carbonyl carbon, not "any atom
  // that happens to have 3 heavy neighbors" (a genuine sp3 atom can also
  // have 3 heavy neighbors plus an implicit H; hybridizationOf already
  // correctly excludes those, since it's driven by bond order/aromaticity/
  // lone-pair conjugation, not neighbor count). Without this term the
  // angle-bending energy alone can't prevent a planar sp2 center from
  // pyramidalizing (three ~120 degree bond angles at an atom are
  // satisfiable by many non-planar arrangements too, not just the flat
  // one) -- this was a real reported bug (a phenyl ring, and separately a
  // conjugated/amide-type nitrogen, coming out non-planar), not a
  // hypothetical one. The nitrogen case specifically needed the
  // lone-pair-conjugation check in hybridizationOf, not just this term --
  // a plain bond-order check never marks an amide/aniline nitrogen sp2 at
  // all (no double bond sits on the nitrogen itself), so it never even
  // reached this loop before that fix.
  function buildImproperTerms(atoms3d, bonds3d, atomCount, aromaticSet) {
    const impropers = [];
    for (let i = 0; i < atomCount; i++) {
      if (hybridizationOf(atoms3d, bonds3d, i, aromaticSet) !== 'sp2') continue;
      const nbrs = neighborsOf(bonds3d, i);
      if (nbrs.length !== 3) continue;
      impropers.push({ i: i, a: nbrs[0], b: nbrs[1], c: nbrs[2] });
    }
    return impropers;
  }

  function buildNonBondedPairs(bonds3d, angles, torsions, atomCount) {
    const excluded = new Set();
    function key(a, b) { return a < b ? a + '_' + b : b + '_' + a; }
    bonds3d.forEach(function (b) { excluded.add(key(b.a1, b.a2)); });
    angles.forEach(function (a) { excluded.add(key(a.i, a.k)); });

    const pairs14 = new Set();
    torsions.forEach(function (t) { pairs14.add(key(t.i, t.l)); });

    const pairs = [];
    for (let a = 0; a < atomCount; a++) {
      for (let b = a + 1; b < atomCount; b++) {
        const k = a + '_' + b;
        if (excluded.has(k)) continue;
        pairs.push({ a: a, b: b, scale: pairs14.has(k) ? LJ_SCALE_14 : 1.0 });
      }
    }
    return pairs;
  }

  // ---------- step 3: energy ----------

  function angleBetween(p1, p2, p3) {
    const u = CC.vec3.sub(p1, p2);
    const v = CC.vec3.sub(p3, p2);
    const lu = CC.vec3.length(u);
    const lv = CC.vec3.length(v);
    if (lu < 1e-9 || lv < 1e-9) return 0;
    let cosT = CC.vec3.dot(u, v) / (lu * lv);
    cosT = Math.max(-1, Math.min(1, cosT));
    return Math.acos(cosT);
  }

  // Standard atan2-based dihedral formula (robust near 0/180 degrees,
  // unlike an acos-of-dot-product approach): angle of the i-j-k-l torsion
  // around the j-k axis.
  function dihedralAngle(pi, pj, pk, pl) {
    const b1 = CC.vec3.sub(pj, pi);
    const b2 = CC.vec3.sub(pk, pj);
    const b3 = CC.vec3.sub(pl, pk);
    const n1 = CC.vec3.cross(b1, b2);
    const n2 = CC.vec3.cross(b2, b3);
    const m1 = CC.vec3.cross(n1, CC.vec3.normalize(b2));
    const x = CC.vec3.dot(n1, n2);
    const y = CC.vec3.dot(m1, n2);
    if (Math.abs(x) < 1e-12 && Math.abs(y) < 1e-12) return 0; // degenerate (collinear atoms)
    return Math.atan2(y, x);
  }

  // Signed perpendicular distance of point p from the plane through
  // (a, b, c). Used for the out-of-plane term: for a true sp2 center,
  // the central atom should lie exactly in the plane of its three
  // substituents, i.e. this should be ~0. Chosen over an "improper
  // dihedral angle" formulation deliberately -- a plane-deviation
  // distance is smooth and well-behaved right at the planar minimum
  // (where a dihedral-angle-based measure would sit at a
  // coordinate-singularity, the same atan2(0,0)-adjacent instability
  // dihedralAngle() above already has to guard against for torsions).
  function planeDeviation(p, a, b, c) {
    const normal = CC.vec3.cross(CC.vec3.sub(b, a), CC.vec3.sub(c, a));
    const normalLen = CC.vec3.length(normal);
    if (normalLen < 1e-9) return 0; // a, b, c nearly collinear -- no well-defined plane
    return CC.vec3.dot(CC.vec3.sub(p, a), normal) / normalLen;
  }

  // 12-6 shape (minimum -1 at r=rm), continued below r=LJ_FLOOR_FRAC*rm by
  // a quadratic matching the true curve's value/slope/curvature exactly at
  // the floor. A hard clamp on r itself (the previous approach here) makes
  // the *numeric* gradient exactly zero for any pair closer than the floor
  // -- computeEnergy's numericGradient perturbs positions by +-GRAD_H
  // (1e-4 A), far smaller than the gap between a badly-clashed distance and
  // the floor, so both perturbed evaluations clamp to the identical value
  // and the finite difference is 0/(2*GRAD_H) = 0. That silences the
  // repulsive force precisely when two atoms need it most, letting a bad
  // torsion-randomized starting conformer (or a chance downhill move
  // elsewhere in the energy) leave them sitting on top of each other with
  // nothing pushing them apart -- a real observed failure, not a
  // hypothetical one (e.g. two ring systems of
  // Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C ending up
  // superimposed and staying that way through the whole optimization). The
  // quadratic keeps a real, ever-steepening, never-zero gradient all the
  // way to r=0 while still bounding the curvature the true r^-12 term would
  // otherwise have right at r=0 (which is what the floor was added to
  // avoid in the first place).
  function ljShape(r, rm) {
    const rFloor = LJ_FLOOR_FRAC * rm;
    if (r >= rFloor) {
      const sr6 = Math.pow(rm / r, 6);
      return sr6 * sr6 - 2 * sr6;
    }
    const dr = r - rFloor; // <= 0
    const slope = LJ_FLOOR_SLOPE_COEF / rFloor;
    const curv = LJ_FLOOR_CURV_COEF / (rFloor * rFloor);
    return LJ_FLOOR_F0 + slope * dr + 0.5 * curv * dr * dr;
  }

  // GB/SA implicit solvation (see implicit-solvent.js) added on top of
  // whichever force field's own vacuum energy computeEnergy/
  // computeEnergySMIRNOFF already computed -- a real, independent physics
  // term (Born radii computed fresh from the current positions every call),
  // not a fixed offset. Needs per-atom partial charges (this project's
  // NAGL-MBIS model); if none are loaded, solvent.charges is simply absent
  // and this contributes nothing rather than faking a charge-free
  // approximation. Ramped in via `strength` (the same stage.lj/stage.
  // nonbonded ramp fraction already used for the LJ/electrostatics
  // nonbonded terms) so it doesn't fight the staged minimization's own
  // "introduce stiff terms gradually" principle -- Born radii depend on
  // every pairwise distance, exactly the kind of term that's badly behaved
  // against a still-clashy freshly-randomized structure.
  function solvationEnergy(positions, atoms3d, solvent, strength) {
    // strength===0 during the bonds/angles and torsion-only stages
    // (before the nonbonded ramp starts) -- skip the O(n^2) Born-radii
    // computation entirely rather than computing it just to multiply by
    // zero. Not just an optimization: computeEnergy runs inside
    // numericGradient's finite differences (2*3*numAtoms calls per
    // optimizer iteration), so a skippable O(n^2) term left unskipped
    // would materially slow down every stage regardless of whether
    // solvation is actually contributing anything at that stage yet.
    if (!solvent || !solvent.enabled || !solvent.charges || strength <= 0) return 0;
    const atoms = atoms3d.map(function (a, i) {
      return { element: a.element, x: positions[i].x, y: positions[i].y, z: positions[i].z };
    });
    return strength * CC.Solvent.predict(atoms, solvent.charges, solvent.epsSolvent).total;
  }

  function computeEnergy(positions, atoms3d, bonds3d, angles, torsions, impropers, pairs, stage, solvent) {
    let energy = 0;

    for (let i = 0; i < bonds3d.length; i++) {
      const b = bonds3d[i];
      const r = CC.vec3.distance(positions[b.a1], positions[b.a2]);
      const r0 = CC.idealBondLength(atoms3d[b.a1].element, atoms3d[b.a2].element, b.order);
      const dr = r - r0;
      energy += 0.5 * K_BOND * dr * dr;
    }

    for (let i = 0; i < angles.length; i++) {
      const a = angles[i];
      const theta = angleBetween(positions[a.i], positions[a.j], positions[a.k]);
      const dTheta = theta - a.theta0;
      energy += 0.5 * K_ANGLE * dTheta * dTheta;
    }

    if (stage.torsion) {
      for (let i = 0; i < torsions.length; i++) {
        const t = torsions[i];
        const phi = dihedralAngle(positions[t.i], positions[t.j], positions[t.k], positions[t.l]);
        energy += 0.5 * t.V * (1 + Math.cos(t.n * phi - t.phi0));
      }
      for (let i = 0; i < impropers.length; i++) {
        const imp = impropers[i];
        const dev = planeDeviation(positions[imp.i], positions[imp.a], positions[imp.b], positions[imp.c]);
        energy += 0.5 * K_OOP * dev * dev;
      }
    }

    if (stage.lj > 0) {
      for (let i = 0; i < pairs.length; i++) {
        const p = pairs[i];
        const rm = CC.VDW_RADIUS[atoms3d[p.a].element] + CC.VDW_RADIUS[atoms3d[p.b].element];
        const r = CC.vec3.distance(positions[p.a], positions[p.b]);
        // stage.lj ramps 0->1 over the first several iterations of the
        // nonbonded stage (see minimizeStaged) -- introducing the
        // stiffest term gradually rather than at full strength against
        // whatever (possibly clashy) structure the previous stage left,
        // same "soft start" principle real minimization protocols use.
        energy += stage.lj * p.scale * LJ_EPSILON * ljShape(r, rm); // AMBER-style 12-6: minimum (-eps) exactly at r=rm
      }
    }

    energy += solvationEnergy(positions, atoms3d, solvent, stage.lj);

    return energy;
  }

  function yieldToUI() {
    return new Promise(function (resolve) { setTimeout(resolve, 0); });
  }

  function flatten(positions) {
    const flat = new Float64Array(positions.length * 3);
    for (let i = 0; i < positions.length; i++) {
      flat[i * 3] = positions[i].x;
      flat[i * 3 + 1] = positions[i].y;
      flat[i * 3 + 2] = positions[i].z;
    }
    return flat;
  }

  function unflatten(flat) {
    const positions = [];
    for (let i = 0; i < flat.length; i += 3) {
      positions.push({ x: flat[i], y: flat[i + 1], z: flat[i + 2] });
    }
    return positions;
  }

  function numericGradient(flat, atoms3d, bonds3d, angles, torsions, impropers, pairs, stage, solvent) {
    const grad = new Float64Array(flat.length);
    for (let i = 0; i < flat.length; i++) {
      const original = flat[i];

      flat[i] = original + GRAD_H;
      const ePlus = computeEnergy(unflatten(flat), atoms3d, bonds3d, angles, torsions, impropers, pairs, stage, solvent);

      flat[i] = original - GRAD_H;
      const eMinus = computeEnergy(unflatten(flat), atoms3d, bonds3d, angles, torsions, impropers, pairs, stage, solvent);

      flat[i] = original;
      grad[i] = (ePlus - eMinus) / (2 * GRAD_H);
    }
    return grad;
  }

  async function minimize(atoms3d, bonds3d, angles, torsions, impropers, pairs, iterations, deadline, stage, startFlat, onProgress, solvent) {
    let flat = startFlat || flatten(atoms3d);
    let energy = computeEnergy(unflatten(flat), atoms3d, bonds3d, angles, torsions, impropers, pairs, stage, solvent);
    let step = 0.02;
    let lastGradNorm = Infinity;
    // What actually stopped the loop -- the honest signal for whether
    // this settled on its own or got cut off while still improving.
    // 'gradient-converged' / 'energy-plateau' mean the optimizer decided
    // it was done; 'iteration-limit' / 'deadline' / 'step-too-small'
    // mean it was interrupted. A raw final gradient norm doesn't
    // generalize well across molecules of different size/complexity
    // (measured directly: ethanol's gradient norm at a clearly
    // well-converged structure was ~0.03, nowhere near a naive 1e-3
    // "converged" cutoff) -- the exit reason is what's actually honest.
    let exitReason = 'iteration-limit';

    for (let iter = 0; iter < iterations; iter++) {
      if (deadline && performance.now() > deadline) { exitReason = 'deadline'; break; }

      // Yield every ~15 iterations so the browser can repaint (a progress
      // bar, a "still working" indicator) instead of freezing the tab for
      // the whole multi-second optimization -- this used to run as one
      // long synchronous call with zero feedback, which was a real
      // regression once the force field got expensive enough to take
      // several seconds to tens of seconds on bigger molecules.
      if (iter > 0 && iter % 15 === 0) {
        await yieldToUI();
        if (onProgress) onProgress();
      }

      const grad = numericGradient(flat, atoms3d, bonds3d, angles, torsions, impropers, pairs, stage, solvent);
      const gradNorm = Math.sqrt(grad.reduce(function (s, g) { return s + g * g; }, 0));
      lastGradNorm = gradNorm;
      if (gradNorm < 1e-5) { exitReason = 'gradient-converged'; break; }

      const trial = new Float64Array(flat.length);
      for (let i = 0; i < flat.length; i++) trial[i] = flat[i] - step * grad[i];

      const trialEnergy = computeEnergy(unflatten(trial), atoms3d, bonds3d, angles, torsions, impropers, pairs, stage, solvent);

      if (trialEnergy < energy) {
        flat = trial;
        const improvement = energy - trialEnergy;
        energy = trialEnergy;
        step *= 1.2;
        if (improvement < 1e-7) { exitReason = 'energy-plateau'; break; }
      } else {
        step *= 0.5;
        if (step < 1e-7) { exitReason = 'step-too-small'; break; }
      }
    }

    // If the loop above never actually ran (deadline already passed on
    // entry, or iterations=0), report a real gradient norm rather than
    // the Infinity placeholder -- one extra evaluation is cheap.
    if (!isFinite(lastGradNorm)) {
      const grad = numericGradient(flat, atoms3d, bonds3d, angles, torsions, impropers, pairs, stage, solvent);
      lastGradNorm = Math.sqrt(grad.reduce(function (s, g) { return s + g * g; }, 0));
    }

    return {
      positions: unflatten(flat), energy: energy, flat: flat, gradNorm: lastGradNorm,
      exitReason: exitReason,
      settled: exitReason === 'gradient-converged' || exitReason === 'energy-plateau',
    };
  }

  /**
   * Staged minimization: bonds+angles only -> add torsions -> ramp in
   * full nonbonded LJ. A real, standard technique (not something
   * specific to this project) -- throwing every energy term at a simple
   * gradient-descent optimizer simultaneously from a cold (especially a
   * freshly torsion-randomized) start makes for a very stiff, poorly
   * conditioned landscape; relaxing the cheap/well-behaved terms first
   * gives the expensive/stiff ones a much saner structure to start from.
   */
  async function minimizeStaged(atoms3d, bonds3d, angles, torsions, impropers, pairs, totalIterations, deadline, onProgress, solvent) {
    const stage1Iters = Math.round(totalIterations * 0.25);
    const stage2Iters = Math.round(totalIterations * 0.15);
    const rampSteps = 6;
    const rampItersEach = Math.max(3, Math.round(totalIterations * 0.05));
    let remainingIters = totalIterations - stage1Iters - stage2Iters - rampSteps * rampItersEach;
    if (remainingIters < 0) remainingIters = Math.round(totalIterations * 0.1);

    function report(label) { if (onProgress) onProgress(label); }

    // Stage 1: bonds + angles only.
    report('bonds & angles');
    let result = await minimize(atoms3d, bonds3d, angles, torsions, impropers, pairs, stage1Iters, deadline, { torsion: false, lj: 0 }, undefined, function () { report('bonds & angles'); });

    // Stage 2: bring in torsions (bounded gradients, well-behaved).
    report('torsions & ring planarity');
    result = await minimize(atoms3d, bonds3d, angles, torsions, impropers, pairs, stage2Iters, deadline, { torsion: true, lj: 0 }, result.flat, function () { report('torsions & ring planarity'); });

    // Stage 3: ramp LJ (and, if enabled, GB/SA solvation -- see
    // solvationEnergy) in gradually rather than switching either on at
    // full strength in one step.
    for (let r = 1; r <= rampSteps; r++) {
      if (deadline && performance.now() > deadline) break;
      report('steric relaxation');
      const ljStrength = r / rampSteps;
      result = await minimize(atoms3d, bonds3d, angles, torsions, impropers, pairs, rampItersEach, deadline, { torsion: true, lj: ljStrength }, result.flat, function () { report('steric relaxation'); }, solvent);
    }

    // Stage 4: full-strength final polish with whatever time remains.
    report('final polish');
    result = await minimize(atoms3d, bonds3d, angles, torsions, impropers, pairs, remainingIters, deadline, { torsion: true, lj: 1 }, result.flat, function () { report('final polish'); }, solvent);

    // Only the *final* stage's outcome speaks to real convergence --
    // earlier stages are expected to hand off to the next one, not fully
    // settle on their own (stage 1's "converged" bond+angle-only
    // structure isn't the real answer once torsions/LJ are added).
    result.converged = result.settled;
    return result;
  }

  // ---------- quick preoptimization pass (deliberately simpler model) ----------
  //
  // Runs automatically as part of CC.buildInitial3D, right after implicit
  // hydrogens are placed by real vector geometry (see above) -- a cheap,
  // fixed-iteration sanity pass so the *very first* structure shown (before
  // any explicit "Optimize geometry..." click) is already roughly
  // declashed, not a raw 2D-projected structure with only its hydrogens
  // freshly placed. Deliberately a separate, simpler model from
  // minimizeStaged above, not a cut-down call into it: softened harmonic
  // bond/angle constants (this only needs to roughly correct gross
  // distortions, not settle them precisely) and a genuinely soft-core
  // repulsive-only nonbonded term -- no attractive well, no torsions, no
  // 1-4 scaling (buildNonBondedPairs is reused with an empty torsions list
  // specifically to get its 1-2/1-3 exclusion logic without any 1-4
  // distinction, so every remaining pair is treated uniformly) -- rather
  // than the full 12-6 LJ. Fixed at 5 iterations, no early exit, no time
  // budget: this is meant to be cheap enough to always run synchronously,
  // not a real minimization to convergence.
  const PREOPT_K_BOND = 100;
  const PREOPT_K_ANGLE = 15;
  const PREOPT_ITERATIONS = 5;
  const PREOPT_SOFTCORE_EPSILON = 0.3;
  const PREOPT_SOFTCORE_DELTA = 0.5; // Angstrom -- softening length scale

  // Purely repulsive, soft-core by construction: unlike a bare inverse-
  // power term (or even the main model's floor-patched LJ above), this
  // needs no clamp/floor hack to stay smooth at extreme overlap -- the
  // +delta^2 term keeps the denominator away from zero everywhere, so the
  // gradient stays real and repulsive all the way to r=0.
  function softCoreRepulsion(r, rm) {
    const ratio = (rm * rm) / (r * r + PREOPT_SOFTCORE_DELTA * PREOPT_SOFTCORE_DELTA);
    return PREOPT_SOFTCORE_EPSILON * Math.pow(ratio, 6);
  }

  function computePreoptEnergy(positions, atoms3d, bonds3d, angles, pairs) {
    let energy = 0;
    for (let i = 0; i < bonds3d.length; i++) {
      const b = bonds3d[i];
      const r = CC.vec3.distance(positions[b.a1], positions[b.a2]);
      const r0 = CC.idealBondLength(atoms3d[b.a1].element, atoms3d[b.a2].element, b.order);
      const dr = r - r0;
      energy += 0.5 * PREOPT_K_BOND * dr * dr;
    }
    for (let i = 0; i < angles.length; i++) {
      const a = angles[i];
      const theta = angleBetween(positions[a.i], positions[a.j], positions[a.k]);
      const dTheta = theta - a.theta0;
      energy += 0.5 * PREOPT_K_ANGLE * dTheta * dTheta;
    }
    for (let i = 0; i < pairs.length; i++) {
      const p = pairs[i];
      const rm = CC.VDW_RADIUS[atoms3d[p.a].element] + CC.VDW_RADIUS[atoms3d[p.b].element];
      const r = CC.vec3.distance(positions[p.a], positions[p.b]);
      energy += softCoreRepulsion(r, rm);
    }
    return energy;
  }

  function preoptNumericGradient(flat, atoms3d, bonds3d, angles, pairs) {
    const grad = new Float64Array(flat.length);
    for (let i = 0; i < flat.length; i++) {
      const original = flat[i];
      flat[i] = original + GRAD_H;
      const ePlus = computePreoptEnergy(unflatten(flat), atoms3d, bonds3d, angles, pairs);
      flat[i] = original - GRAD_H;
      const eMinus = computePreoptEnergy(unflatten(flat), atoms3d, bonds3d, angles, pairs);
      flat[i] = original;
      grad[i] = (ePlus - eMinus) / (2 * GRAD_H);
    }
    return grad;
  }

  // Synchronous on purpose (no yielding, no time budget, fixed 5
  // iterations) -- cheap enough to always run as part of the fast phase
  // CC.buildInitial3D already promises. Returns new positions; doesn't
  // mutate atoms3d/bonds3d.
  function quickPreoptimize(atoms3d, bonds3d, aromaticSet) {
    const angles = buildAngles(atoms3d, bonds3d, atoms3d.length, aromaticSet);
    const pairs = buildNonBondedPairs(bonds3d, angles, [], atoms3d.length);
    let flat = flatten(atoms3d);
    let energy = computePreoptEnergy(unflatten(flat), atoms3d, bonds3d, angles, pairs);
    let step = 0.02;

    for (let iter = 0; iter < PREOPT_ITERATIONS; iter++) {
      const grad = preoptNumericGradient(flat, atoms3d, bonds3d, angles, pairs);
      const trial = new Float64Array(flat.length);
      for (let i = 0; i < flat.length; i++) trial[i] = flat[i] - step * grad[i];
      const trialEnergy = computePreoptEnergy(unflatten(trial), atoms3d, bonds3d, angles, pairs);
      if (trialEnergy < energy) {
        flat = trial;
        energy = trialEnergy;
        step *= 1.2;
      } else {
        step *= 0.5;
      }
    }
    return unflatten(flat);
  }

  // ---------- conformer seeding: real torsion-driven search ----------

  // A bond is "rotatable" for conformer-search purposes using the
  // standard cheminformatics definition: a single bond, not in any ring,
  // with a real substituent on each side to actually rotate. Ring
  // membership comes from RDKit's own ring perception (same pipeline
  // chemprop-features.js and sascorer.js already use) rather than
  // reimplementing ring detection here -- reusing an already-validated
  // source beats a second, possibly-inconsistent one.
  function findRotatableBonds(molecule, ringBondPairs, idToIndex) {
    const rotatable = [];
    molecule.bonds.forEach(function (b) {
      if (b.order !== 1) return;
      const i = idToIndex.get(b.a1), j = idToIndex.get(b.a2);
      const key = i < j ? i + '_' + j : j + '_' + i;
      if (ringBondPairs.has(key)) return;
      if (molecule.getDegree(b.a1) < 2 || molecule.getDegree(b.a2) < 2) return; // terminal bond, nothing to rotate
      rotatable.push({ j: i, k: j });
    });
    return rotatable;
  }

  // All atoms on the "k side" of a rotatable bond (j,k) -- BFS from k
  // without crossing back over the (j,k) edge. Safe (splits the graph
  // into exactly two pieces) specifically because rotatable bonds are
  // pre-filtered to non-ring bonds -- a ring bond wouldn't have this
  // property and must never be passed in here.
  function sideAtoms(bonds3d, atomCount, j, k) {
    const adjacency = Array.from({ length: atomCount }, function () { return []; });
    bonds3d.forEach(function (b) {
      if ((b.a1 === j && b.a2 === k) || (b.a1 === k && b.a2 === j)) return; // exclude the pivot bond itself
      adjacency[b.a1].push(b.a2);
      adjacency[b.a2].push(b.a1);
    });
    const visited = new Set([k]);
    const queue = [k];
    while (queue.length > 0) {
      const cur = queue.shift();
      adjacency[cur].forEach(function (n) {
        if (!visited.has(n)) { visited.add(n); queue.push(n); }
      });
    }
    return visited;
  }

  // Rotate every atom on the "k side" of each rotatable bond by an angle
  // sampled from the real staggered minima (60/180/300 degrees) most
  // sp3-sp3 single bonds actually prefer -- not a fully continuous
  // uniform draw. This is the actual conformer *search* part: different
  // attempts start from genuinely different rotamer combinations, not
  // just numerical jitter, and seeding at physically sensible minima
  // (rather than arbitrary angles the optimizer then has to discover on
  // its own) matters a lot for molecules with many rotatable bonds at
  // once -- a long alkyl chain's low-energy conformers are overwhelmingly
  // at or near all-staggered states, not scattered uniformly across
  // rotamer space, so seeding there directly is a much better-informed
  // starting guess than pure randomness, not just a smaller search space.
  const STAGGERED_ANGLES = [60 * Math.PI / 180, 180 * Math.PI / 180, 300 * Math.PI / 180];

  function randomizeRotatableBonds(atoms3d, bonds3d, rotatableBonds) {
    rotatableBonds.forEach(function (rb) {
      const moving = sideAtoms(bonds3d, atoms3d.length, rb.j, rb.k);
      const angle = STAGGERED_ANGLES[Math.floor(Math.random() * STAGGERED_ANGLES.length)];
      const origin = atoms3d[rb.j];
      const axis = CC.vec3.sub(atoms3d[rb.k], atoms3d[rb.j]);
      moving.forEach(function (idx) {
        const rotated = CC.vec3.rotateAroundAxis(atoms3d[idx], origin, axis, angle);
        atoms3d[idx].x = rotated.x; atoms3d[idx].y = rotated.y; atoms3d[idx].z = rotated.z;
      });
    });
  }

  // ---------- entry points ----------

  /**
   * Fast, synchronous phase: adds implicit hydrogens, seeds 3D
   * coordinates from the 2D layout, and works out which bonds are
   * rotatable (needs a quick RDKit ring-perception call, but that's the
   * only non-trivial cost here -- no force field optimization at all).
   * Meant to give an instant "here's a rough 3D structure" result before
   * the separate, potentially many-second CC.optimize3D() pass.
   */
  CC.buildInitial3D = function (molecule) {
    if (molecule.isEmpty()) {
      return { atoms: [], bonds: [], molecule: molecule, rotatableBonds: [] };
    }

    const heavyAtoms = Array.from(molecule.atoms.values());
    const idToIndex = new Map();
    heavyAtoms.forEach(function (a, i) { idToIndex.set(a.id, i); });

    let ringBondPairs = new Set();
    let aromaticSet = new Set();
    try {
      const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
      if (RDKit) {
        const molblock = CC.moleculeToMolblock(molecule);
        const annotations = CC.GNN.getRDKitAnnotations(molblock);
        ringBondPairs = annotations.ringBondPairs;
        aromaticSet = annotations.aromaticAtomIndices;
      }
    } catch (err) {
      // Ring/aromaticity info is a real-structure-quality input (correct
      // planarity, better conformer seeding), not a correctness
      // requirement for 3D generation to run at all -- if RDKit isn't
      // ready yet or the structure can't be parsed, fall back to "no
      // bonds treated as rotatable, no atoms treated as aromatic" rather
      // than failing 3D generation entirely.
    }
    const rotatableBonds = findRotatableBonds(molecule, ringBondPairs, idToIndex);

    const built = withImplicitHydrogens(molecule, aromaticSet);
    // Quick 5-iteration decrappification pass (see the section above) --
    // still cheap enough to keep this whole function synchronous, so the
    // structure shown immediately after "Generate 3D structure" is
    // already roughly declashed rather than a raw jittered seed.
    const preoptimized = quickPreoptimize(built.atoms3d, built.bonds3d, aromaticSet);

    return {
      atoms: built.atoms3d.map(function (a, i) {
        return { element: a.element, x: preoptimized[i].x, y: preoptimized[i].y, z: preoptimized[i].z };
      }),
      bonds: built.bonds3d,
      molecule: molecule,
      rotatableBonds: rotatableBonds,
      aromaticSet: aromaticSet,
      energy: null,
      converged: null,
    };
  };

  /**
   * The slow part: multi-attempt conformer search + staged force-field
   * minimization (see CC.buildInitial3D for the fast/cheap phase this
   * continues from). Async and yields periodically (see minimize()) so
   * the UI can show real progress instead of freezing for however many
   * seconds this takes on a bigger or more flexible molecule.
   *
   * opts.onProgress, if given, is called as
   *   onProgress({ attempt, totalAttempts, stage, bestEnergySoFar })
   * roughly every 15 optimizer iterations.
   *
   * Result includes `converged` (true if the best attempt's final
   * gradient norm was actually small, not just "ran out of time/
   * iterations") -- real information, not just "here's a structure,
   * trust it."
   */
  // The single-seed body CC.optimize3D's attempt loop below runs once per
  // attempt -- factored out so js/conformer-search.js can run the exact
  // same, already-validated classical energy/minimization path against
  // its own seed geometries (systematic/randomized torsion states, not
  // just the per-attempt randomization this file's own loop does)
  // without duplicating buildAngles/buildTorsions/buildImproperTerms/
  // buildNonBondedPairs/minimizeStaged wiring a second time.
  async function optimizeGivenSeed(atoms3d, bonds3d, aromaticSet, iterations, deadline, onProgress, solvent) {
    const angles = buildAngles(atoms3d, bonds3d, atoms3d.length, aromaticSet);
    const torsions = buildTorsions(atoms3d, bonds3d, atoms3d.length, aromaticSet);
    const impropers = buildImproperTerms(atoms3d, bonds3d, atoms3d.length, aromaticSet);
    const pairs = buildNonBondedPairs(bonds3d, angles, torsions, atoms3d.length);
    const result = await minimizeStaged(atoms3d, bonds3d, angles, torsions, impropers, pairs, iterations, deadline, onProgress, solvent);
    return {
      energy: result.energy,
      converged: result.converged,
      gradNorm: result.gradNorm,
      exitReason: result.exitReason,
      atoms: atoms3d.map(function (a, i) {
        return { element: a.element, x: result.positions[i].x, y: result.positions[i].y, z: result.positions[i].z };
      }),
      bonds: bonds3d,
    };
  }

  CC.optimize3D = async function (initial, opts) {
    opts = opts || {};
    const molecule = initial.molecule;
    const rotatableBonds = initial.rotatableBonds || [];
    const aromaticSet = initial.aromaticSet || new Set();

    if (!molecule || molecule.isEmpty()) {
      return { atoms: [], bonds: [], energy: 0, converged: true };
    }

    const heavyAtomCount = molecule.atoms.size;
    const totalBudgetMs = opts.timeBudgetMs || Math.min(25000, Math.max(5000, heavyAtomCount * 900));
    const attempts = opts.attempts || Math.max(3, Math.min(8, 3 + rotatableBonds.length));
    const perAttemptBudgetMs = Math.max(3500, totalBudgetMs / attempts);
    const iterations = opts.iterations || 400;

    let best = null;
    const overallDeadline = performance.now() + totalBudgetMs;

    for (let attempt = 0; attempt < attempts; attempt++) {
      if (performance.now() > overallDeadline) break;

      const built = withImplicitHydrogens(molecule, aromaticSet);
      const atoms3d = built.atoms3d;
      const bonds3d = built.bonds3d;

      // First attempt keeps the 2D-seeded layout as-is (a good starting
      // point on its own for many molecules); later attempts randomize
      // rotamer state for real conformer diversity.
      if (attempt > 0) randomizeRotatableBonds(atoms3d, bonds3d, rotatableBonds);

      const attemptDeadline = Math.min(overallDeadline, performance.now() + perAttemptBudgetMs);
      const result = await optimizeGivenSeed(atoms3d, bonds3d, aromaticSet, iterations, attemptDeadline, function (stage) {
        if (opts.onProgress) {
          opts.onProgress({
            attempt: attempt + 1,
            totalAttempts: attempts,
            stage: stage,
            bestEnergySoFar: best ? best.energy : null,
          });
        }
      }, opts.solvent);

      if (!best || result.energy < best.energy) best = result;
    }

    return best;
  };

  /**
   * Convenience wrapper for anything that just wants a finished, fully
   * optimized structure in one call (runs both phases back to back).
   * The app's own 3D panel UI calls buildInitial3D/optimize3D
   * separately instead, specifically so "see a rough structure" and
   * "spend several/tens of seconds refining it" are two distinct,
   * separately-triggered actions rather than one long blocking call.
   */
  CC.embed3D = async function (molecule, opts) {
    const initial = CC.buildInitial3D(molecule);
    if (initial.atoms.length === 0) return { atoms: [], bonds: [], energy: 0, converged: true };
    return CC.optimize3D(initial, opts);
  };

  // Exposed for js/openff-forcefield.js to reuse this file's already-
  // validated implicit-H placement, rotatable-bond detection/seeding, and
  // generic numeric-optimization plumbing -- rather than re-deriving a
  // second copy of BFS side-splitting, the atan2 dihedral formula, etc.
  // for a different energy model. The hand-tuned bonds/angles/torsions/LJ
  // energy functions above stay private to this file; only the
  // model-independent pieces are shared here.
  CC.Embed3DShared = {
    withImplicitHydrogens: withImplicitHydrogens,
    findRotatableBonds: findRotatableBonds,
    randomizeRotatableBonds: randomizeRotatableBonds,
    flatten: flatten,
    unflatten: unflatten,
    yieldToUI: yieldToUI,
    angleBetween: angleBetween,
    dihedralAngle: dihedralAngle,
    planeDeviation: planeDeviation,
    ljShape: ljShape,
    optimizeSeedClassical: optimizeGivenSeed,
    sideAtoms: sideAtoms,
  };
})();
