/**
 * steric-accessibility.js
 *
 * Per-atom steric accessibility via real Solvent-Accessible Surface Area
 * (SASA), computed with the classic Shrake-Rupley rolling-sphere
 * algorithm on a real 3D conformer -- a deterministic geometric
 * calculation, not a trained model. This is this app's first atom-level
 * property that genuinely needs a 3D structure rather than just the 2D
 * graph/topology every other atom-level property (NAGL charges, pKa,
 * NMR shifts, BDE-XH) works from.
 *
 * Pipeline: a Shrake-Rupley pass directly over whatever 3D atom array the
 * caller hands in -- CC.SASA.predictFromAtoms() is the only entry point,
 * and it does NOT build or optimize a conformer itself (see that
 * function's own doc comment for why: an earlier version did, and that
 * "quick" build+optimize routinely still didn't converge in its own time
 * budget, silently producing unreliable numbers). app.js's "Compute
 * (SASA)" button gets its atoms from the 3D tab's own already-optimized
 * structure (CC.buildInitial3D() + CC.optimize3D(), the same UFF/MMFF-
 * style force field "Optimize geometry…" uses), and requires one to
 * exist first rather than generating one on demand. embed3d.js's
 * internal coordinate system is already real Angstroms (confirmed: its
 * own nonbonded term uses CC.VDW_RADIUS and CC.idealBondLength()'s real
 * covalent radii directly, unscaled, in the same coordinate space as
 * atom positions -- see elements.js's own header comment), so no unit
 * conversion is needed before computing SASA in Å².
 *
 * Two properties are surfaced per heavy atom, both from the SAME
 * underlying calculation:
 *   - `sasa`: the raw exposed surface area, in Å² (probe-radius-expanded
 *     sphere), meaningful on its own but not directly comparable across
 *     elements (a fully exposed Br has a bigger raw SASA than a fully
 *     exposed H just from being a bigger sphere).
 *   - `steric_accessibility`: that same area as a FRACTION (0 = fully
 *     buried, 1 = fully exposed) of this atom's own probe-expanded
 *     sphere -- normalized per atom, so it's the one that's actually
 *     comparable across different elements and is what "how accessible
 *     is this atom" usually means.
 *
 * Implicit hydrogens are included as real occluding spheres in the SASA
 * calculation (they physically block solvent/reagent access just like
 * any other atom) but, like every other atom-level property in this
 * project, are not individually surfaced -- only the numHeavyAtoms
 * leading entries (heavy atoms, in molecule.atoms order) are returned.
 */

window.CC = window.CC || {};
CC.SASA = window.CC.SASA || {};

(function () {
  // Standard SASA convention: a probe radius approximating a water
  // molecule (Lee & Richards 1971; still the field's default even when
  // "solvent" isn't literally the point -- it's just the standard scale
  // steric-accessibility numbers are reported on).
  const DEFAULT_PROBE_RADIUS = 1.4; // Å
  const DEFAULT_NUM_POINTS = 300; // per-atom sphere sample density

  // Deterministic, evenly-distributed points on a unit sphere (golden-
  // angle spiral) -- the standard fast alternative to a real geodesic
  // (icosahedral) sphere for Shrake-Rupley, and memoized per point count
  // since callers running many atoms all use the same count.
  const sphereCache = new Map();
  function unitSpherePoints(n) {
    if (sphereCache.has(n)) return sphereCache.get(n);
    const points = new Array(n);
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < n; i++) {
      const y = 1 - (i / (n - 1)) * 2; // 1 .. -1
      const radiusAtY = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = goldenAngle * i;
      points[i] = { x: Math.cos(theta) * radiusAtY, y: y, z: Math.sin(theta) * radiusAtY };
    }
    sphereCache.set(n, points);
    return points;
  }

  function vdwRadius(element) {
    return CC.VDW_RADIUS[element] || CC.VDW_RADIUS.C;
  }

  /**
   * Pure Shrake-Rupley SASA over a plain array of {element, x, y, z}
   * positions (real Angstroms). Returns one { sasa, fraction } object per
   * atom, same order as `atoms`. O(numAtoms^2 * numPoints) -- all-pairs,
   * no spatial acceleration structure -- which is fast enough (well
   * under 100ms) at the atom counts this app ever draws (a few hundred
   * atoms including implicit H at most), so not worth the complexity.
   */
  CC.SASA.compute = function (atoms, opts) {
    opts = opts || {};
    const probeRadius = opts.probeRadius === undefined ? DEFAULT_PROBE_RADIUS : opts.probeRadius;
    const numPoints = opts.numPoints || DEFAULT_NUM_POINTS;
    const spherePoints = unitSpherePoints(numPoints);

    const radii = atoms.map(function (a) { return vdwRadius(a.element) + probeRadius; });

    return atoms.map(function (atom, i) {
      const ri = radii[i];
      let exposedCount = 0;
      for (let p = 0; p < spherePoints.length; p++) {
        const sp = spherePoints[p];
        const tx = atom.x + sp.x * ri, ty = atom.y + sp.y * ri, tz = atom.z + sp.z * ri;
        let buried = false;
        for (let j = 0; j < atoms.length; j++) {
          if (j === i) continue;
          const rj = radii[j];
          const dx = tx - atoms[j].x, dy = ty - atoms[j].y, dz = tz - atoms[j].z;
          if (dx * dx + dy * dy + dz * dz < rj * rj) { buried = true; break; }
        }
        if (!buried) exposedCount++;
      }
      const fraction = exposedCount / spherePoints.length;
      const sphereArea = 4 * Math.PI * ri * ri;
      return { sasa: fraction * sphereArea, fraction: fraction };
    });
  };

  /**
   * Computes SASA directly from an already-built 3D atom array (real
   * Angstroms, heavy atoms in molecule.atoms order followed by implicit
   * H's -- the same shape CC.buildInitial3D()/CC.optimize3D()/
   * CC.ANI.optimizeGeometry() all produce and preserve). This is the
   * ONLY way to get SASA in this app -- it deliberately does not build
   * or optimize a conformer of its own (an earlier version did; that
   * path was slow -- a full conformer search, several seconds to tens
   * of seconds -- AND its own time budget routinely still wasn't enough
   * to reach real convergence, silently producing unreliable numbers).
   * A real, already-relaxed conformer is a precondition, not something
   * this file will go generate on your behalf: app.js's "Compute
   * (SASA)" button requires one to already exist (built via the 3D
   * tab's "Generate 3D structure" + "Optimize geometry…"/"Optimize with
   * ANI-2x…") and prompts the user to go make one instead of silently
   * building its own.
   *
   * opts.probeRadius/opts.numPoints are forwarded to CC.SASA.compute().
   * Has no opinion on whether atoms3D is itself well-relaxed -- that's
   * on the caller (app.js surfaces the source geometry's own convergence
   * flag alongside this result for exactly that reason).
   */
  CC.SASA.predictFromAtoms = function (molecule, atoms3D, opts) {
    opts = opts || {};
    if (molecule.isEmpty() || !atoms3D || atoms3D.length === 0) {
      return { atomProperties: [], atomIds: [], backend: 'sasa' };
    }

    const perAtom = CC.SASA.compute(atoms3D, { probeRadius: opts.probeRadius, numPoints: opts.numPoints });

    const heavyAtoms = Array.from(molecule.atoms.values());
    const atomIds = heavyAtoms.map(function (a) { return a.id; });
    const atomProperties = heavyAtoms.map(function (a, i) {
      return { sasa: perAtom[i].sasa, steric_accessibility: perAtom[i].fraction };
    });

    return {
      atomProperties: atomProperties,
      atomIds: atomIds,
      backend: 'sasa',
      numAtomsIncludingH: atoms3D.length,
    };
  };
})();
