/**
 * xyz.js
 *
 * Import/export for the plain-text XYZ coordinate format (element + x/y/z
 * per atom, Angstrom -- no bonding information at all, unlike a molfile).
 *
 * Import has to do real work an XYZ file doesn't supply: infer which
 * atoms are bonded and at what order from geometry alone (distances vs.
 * covalent radii, valence balancing), then hand that guess to RDKit.js
 * to validate/canonicalize (aromaticity, Kekulization) and to generate a
 * real 2D layout for the drawn structure -- confirmed live against the
 * bundled RDKit build, not assumed: `RDKit.get_mol(molblock)` returns
 * `null` (not a throw) on a genuine valence violation, and
 * `RDKit.get_mol(smiles)` + `mol.set_new_coords()` + `mol.get_molblock()`
 * produces real, non-degenerate 2D coordinates from bonds alone.
 * RDKit is NOT doing any geometry-based bond perception itself here --
 * this project's bundled minimal build has no DetermineBonds-equivalent
 * (checked directly against the actual API surface, not assumed) -- it
 * only validates/canonicalizes whatever bond orders this file guesses.
 */

window.CC = window.CC || {};
CC.XYZ = window.CC.XYZ || {};

(function () {
  // Covalent-radius-sum-plus-tolerance: a standard, commonly-used style
  // of geometric bond perception (not claimed to be a byte-for-byte
  // match of any specific external tool's constants -- this project's
  // own honesty convention about what's verified vs. approximated, see
  // CHEMPROP_INTEGRATION.md). 1.25x the sum of CC.ELEMENT_DATA's
  // covalent radii (elements.js) is generous enough to catch real
  // elongated/strained bonds without also catching ordinary non-bonded
  // contacts.
  const BOND_TOLERANCE = 1.25;

  // Below this, two atoms are almost certainly overlapping/bad data, not
  // a real (if unusually short) bond -- skip rather than propose a
  // physically absurd bond.
  const MIN_BOND_DISTANCE = 0.4;

  // Hypervalent-capable elements: a phosphate/sulfone/sulfonamide center
  // routinely exceeds its "standard" neutral valence (CC.ELEMENT_DATA's
  // P:3, S:2) while still being a completely ordinary, real structure --
  // the same exception js/structure-validation.js's own STANDARD_VALENCE
  // table already documents and checks for (hypervalent P/S flagged as
  // "info", not an error). Real pentavalent P / hexavalent S, not
  // invented here.
  const EXPANDED_VALENCE = { P: 5, S: 6 };

  const MAX_FORMAL_CHARGE_MAGNITUDE = 2;

  // ---------- parsing ----------

  /**
   * text: raw file contents.
   * Returns { atoms: [{element, x, y, z}], title }.
   * Tolerates the standard 2-line header (atom count, then a title/
   * comment line) OR bare element/x/y/z lines with no header at all --
   * detected by whether the first non-blank line is a lone integer.
   * Throws a clear, specific Error on anything unparseable (empty file,
   * an atom line missing a coordinate, a declared header count that
   * doesn't match what actually follows).
   */
  CC.XYZ.parse = function (text) {
    const lines = String(text).split(/\r?\n/)
      .map(function (l) { return l.trim(); })
      .filter(function (l) { return l.length > 0; });
    if (lines.length === 0) throw new Error('Empty file -- no atom lines found.');

    const firstTokens = lines[0].split(/\s+/);
    const hasHeader = firstTokens.length === 1 && /^\d+$/.test(firstTokens[0]);

    let title = '';
    let atomLines;
    if (hasHeader) {
      const declaredCount = parseInt(firstTokens[0], 10);
      title = lines.length > 1 ? lines[1] : '';
      atomLines = lines.slice(2);
      if (declaredCount > 0 && atomLines.length < declaredCount) {
        throw new Error('Header declares ' + declaredCount + ' atom(s) but only ' + atomLines.length + ' atom line(s) follow.');
      }
      if (declaredCount > 0) atomLines = atomLines.slice(0, declaredCount);
    } else {
      atomLines = lines;
    }

    const atoms = atomLines.map(function (line, idx) {
      const tokens = line.split(/\s+/);
      if (tokens.length < 4) throw new Error('Could not parse atom line ' + (idx + 1) + ': "' + line + '" (expected element x y z).');
      const rawEl = tokens[0];
      const element = rawEl.charAt(0).toUpperCase() + rawEl.slice(1).toLowerCase();
      const x = parseFloat(tokens[1]), y = parseFloat(tokens[2]), z = parseFloat(tokens[3]);
      if (!isFinite(x) || !isFinite(y) || !isFinite(z)) throw new Error('Could not parse coordinates on atom line ' + (idx + 1) + ': "' + line + '".');
      return { element: element, x: x, y: y, z: z };
    });

    if (atoms.length === 0) throw new Error('No atom lines found.');
    return { atoms: atoms, title: title };
  };

  // ---------- bond/order inference ----------

  /**
   * atoms: [{element, x, y, z}].
   * Returns { bonds3d: [{a1, a2, order}], warnings: [string],
   * charges: {atomIndex: integerCharge} }.
   *
   * 1. Connectivity from covalent-radius-sum distance cutoff.
   * 2. Every detected bond starts at order 1.
   * 3. Greedy multi-bond upgrade, shortest-relative-to-an-ideal-single-
   *    bond first (real double/triple bonds are measurably shorter --
   *    same CC.BOND_ORDER_FACTOR table already used for 3D embedding),
   *    upgrading a bond only while BOTH endpoints still have spare
   *    valence. Because of that "both endpoints" rule, a bond can only
   *    be promoted once its neighbor's spare valence is still available
   *    -- around a simple ring this naturally alternates single/double
   *    (a valid Kekulé structure for e.g. benzene) without any separate
   *    aromatic-ring-detection step: once one ring bond consumes an
   *    atom's spare valence, its OTHER ring bond can no longer be
   *    upgraded.
   * 4. Any atom left with spare valence after step 3 gets a formal
   *    charge for the difference (the standard "one Kekulé resonance
   *    contributor" convention -- e.g. one O of a carboxylate ends up
   *    single-bonded with charge -1, exactly how this app's own molfile
   *    writer already represents a charged resonance form) -- capped at
   *    +/-2; beyond that, left uncharged with a warning rather than
   *    guessing wildly. This is NOT a general resonance/charge solver
   *    (no attempt to average/delocalize across equivalent resonance
   *    forms) -- a known, documented scope boundary, same honesty
   *    convention CHEMPROP_INTEGRATION.md's "known approximations"
   *    section already uses elsewhere in this project.
   */
  CC.XYZ.inferBonds = function (atoms) {
    const n = atoms.length;
    const warnings = [];
    const bonds = [];

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = atoms[i].x - atoms[j].x, dy = atoms[i].y - atoms[j].y, dz = atoms[i].z - atoms[j].z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < MIN_BOND_DISTANCE) {
          warnings.push('Atoms ' + (i + 1) + ' and ' + (j + 1) + ' (' + atoms[i].element + ', ' + atoms[j].element + ') are only ' + dist.toFixed(2) + ' Å apart -- skipped as likely overlapping/bad data rather than a real bond.');
          continue;
        }
        const cutoff = (CC.elementData(atoms[i].element).radius + CC.elementData(atoms[j].element).radius) * BOND_TOLERANCE;
        if (dist <= cutoff) bonds.push({ a1: i, a2: j, order: 1, dist: dist });
      }
    }

    function orderSum(atomIdx) {
      let sum = 0;
      for (let k = 0; k < bonds.length; k++) {
        const b = bonds[k];
        if (b.a1 === atomIdx || b.a2 === atomIdx) sum += b.order;
      }
      return sum;
    }

    // Effective valence cap per atom, fixed once up front from the
    // connectivity step alone (before any multi-bond upgrades): the
    // element's normal valence, expanded to EXPANDED_VALENCE for P/S
    // whose single-bond connectivity ALONE already exceeds their normal
    // valence (a real hypervalent center -- a phosphate P routinely has
    // 4 single-bonded oxygens before any bond even gets upgraded), and
    // never LOWER than the raw connectivity count for any element (so a
    // merely-generous distance cutoff on an ordinary atom doesn't get
    // misread as a formal charge later -- see step 4).
    const caps = atoms.map(function (a, idx) {
      const standard = CC.elementData(a.element).valence;
      const neighborCount = orderSum(idx); // all bonds still order 1 here
      const expanded = EXPANDED_VALENCE[a.element];
      if (expanded && neighborCount > standard) return expanded;
      return Math.max(standard, neighborCount);
    });

    const candidates = bonds
      .filter(function (b) { return atoms[b.a1].element !== 'H' && atoms[b.a2].element !== 'H'; })
      .map(function (b) {
        const idealSingle = CC.idealBondLength(atoms[b.a1].element, atoms[b.a2].element, 1);
        return { bond: b, ratio: idealSingle > 0 ? b.dist / idealSingle : Infinity };
      })
      .sort(function (a, b) { return a.ratio - b.ratio; });

    candidates.forEach(function (entry) {
      const b = entry.bond;
      while (b.order < 3) {
        const roomA = caps[b.a1] - orderSum(b.a1);
        const roomB = caps[b.a2] - orderSum(b.a2);
        if (roomA > 0 && roomB > 0) b.order += 1;
        else break;
      }
    });

    // Repair pass: the plain greedy walk above can leave a RING in a
    // maximal-but-not-fully-alternating state (a real, confirmed failure
    // mode of greedy edge selection on an even cycle, not a theoretical
    // concern -- e.g. picking benzene's (0,1) and (3,4) ring bonds first,
    // both individually valid greedy choices, structurally prevents any
    // bond touching atoms 2 or 5 from ever being reachable, leaving those
    // two atoms under-bonded even though a fully alternating 1-2-1-2-1-2
    // Kekulé pattern exists). Standard graph-matching augmenting-path
    // search fixes this: from a still-deficient atom, walk an alternating
    // path of not-yet-upgraded/already-upgraded bonds; if it reaches
    // ANOTHER deficient atom, flipping every bond along that path
    // (+1 order on the not-yet-upgraded bonds walked, -1 on the
    // already-upgraded ones) resolves both atoms' deficiency at once
    // without changing anything else's total order. Bounded to the
    // molecule's own bond count (small for anything this app draws), and
    // only ever needed for the specific case above -- unambiguous
    // multi-bond candidates (a real carbonyl, nitrile, etc.) are already
    // resolved correctly by the plain greedy pass and never trigger this.
    // Not a general (blossom-handling) matching solver -- a real,
    // documented scope boundary consistent with this function's other
    // known limitations (see the file header and this function's own
    // doc comment) -- but does correctly resolve the common even-ring
    // case this project's own aromatic molecules actually hit.
    function tryAugmentingUpgrade() {
      for (let start = 0; start < n; start++) {
        if (caps[start] - orderSum(start) <= 0) continue;
        const visited = new Set([start]);
        const queue = [{ atom: start, path: [] }];
        while (queue.length) {
          const cur = queue.shift();
          const needUnmatched = cur.path.length % 2 === 0; // next edge must be not-yet-upgraded (order 1) on even steps, already-upgraded (order>1) on odd steps
          for (let k = 0; k < candidates.length; k++) {
            const b = candidates[k].bond;
            if (b.a1 !== cur.atom && b.a2 !== cur.atom) continue;
            if (cur.path.indexOf(b) !== -1) continue;
            if ((b.order > 1) === needUnmatched) continue;
            const other = b.a1 === cur.atom ? b.a2 : b.a1;
            const newPath = cur.path.concat([b]);
            if (needUnmatched && other !== start && caps[other] - orderSum(other) > 0) {
              newPath.forEach(function (pb, idx) { pb.order += (idx % 2 === 0) ? 1 : -1; });
              return true;
            }
            if (!visited.has(other)) { visited.add(other); queue.push({ atom: other, path: newPath }); }
          }
        }
      }
      return false;
    }
    let augmentGuard = 0;
    while (tryAugmentingUpgrade() && augmentGuard < bonds.length) augmentGuard++;

    const charges = {};
    for (let idx = 0; idx < n; idx++) {
      const charge = orderSum(idx) - caps[idx];
      if (charge === 0) continue;
      if (Math.abs(charge) > MAX_FORMAL_CHARGE_MAGNITUDE) {
        warnings.push('Atom ' + (idx + 1) + ' (' + atoms[idx].element + ') has an unusual apparent valence (bonded order sum ' + orderSum(idx) + ') -- left uncharged rather than guessing a formal charge of ' + charge + '.');
        continue;
      }
      charges[idx] = charge;
    }

    const bonds3d = bonds.map(function (b) { return { a1: b.a1, a2: b.a2, order: b.order }; });
    return { bonds3d: bonds3d, warnings: warnings, charges: charges };
  };

  // ---------- RDKit validation + 2D layout ----------

  /**
   * RDKit: window.chemCanvasLibs.RDKit (may be undefined/not-ready yet --
   * handled the same as a validation failure, 3D-only fallback below).
   * atoms: [{element, x, y, z}] (as from CC.XYZ.parse).
   * Returns { atoms3d, bonds3d, molecule: CC.Molecule|null, warnings }.
   * `molecule` is null when bonding couldn't be valence-validated (or
   * RDKit isn't available) -- callers should still show atoms3d/bonds3d
   * in the 3D viewer either way (same graceful-degradation posture as
   * gnn-inference.js's per-model failure handling), just skip populating
   * the 2D canvas.
   */
  CC.XYZ.buildFromAtoms = function (RDKit, atoms) {
    const inferred = CC.XYZ.inferBonds(atoms);
    const bonds3d = inferred.bonds3d;
    const warnings = inferred.warnings.slice();

    let molecule = null;
    if (RDKit) {
      const molblock = CC.atoms3DToMolblock(atoms, bonds3d, 'xyz-import', inferred.charges);
      let mol = null;
      try {
        mol = RDKit.get_mol(molblock);
      } catch (err) {
        mol = null;
      }

      if (mol && mol.is_valid()) {
        let mol2 = null;
        try {
          const smiles = mol.get_smiles();
          mol2 = RDKit.get_mol(smiles);
          mol2.set_new_coords();
          molecule = CC.molblockToMolecule(mol2.get_molblock());
        } catch (err) {
          warnings.push('Bonding was inferred and valence-validated, but building a 2D layout failed (' + err.message + ') -- showing the 3D structure only.');
        } finally {
          if (mol2) mol2.delete();
        }
      } else {
        warnings.push('Bonding was inferred from geometry, but could not be fully valence-validated -- showing the 3D structure only; you may need to fix bonds manually.');
      }
      if (mol) mol.delete();
    } else {
      warnings.push('RDKit is not ready yet -- showing the 3D structure only; bonding could not be valence-validated.');
    }

    return { atoms3d: atoms, bonds3d: bonds3d, molecule: molecule, warnings: warnings };
  };

  // ---------- export ----------

  /** atoms: [{element, x, y, z}]. Returns standard XYZ text, always with the header. */
  CC.XYZ.exportXYZ = function (atoms, title) {
    const lines = [String(atoms.length), title || 'ChemCanvas export'];
    atoms.forEach(function (a) {
      lines.push(a.element + '\t' + a.x.toFixed(6) + '\t' + a.y.toFixed(6) + '\t' + a.z.toFixed(6));
    });
    return lines.join('\n') + '\n';
  };
})();
