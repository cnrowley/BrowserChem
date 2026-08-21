/**
 * molfile.js
 *
 * Converts between our Molecule graph and MDL Molfile V2000 text.
 * This is a plain format encoder/decoder — no chemistry validation here,
 * that's chemistry.js's job. Kept dependency-free so Save/Open work even
 * before RDKit finishes initializing.
 *
 * Coordinate scale: our SVG uses a fixed bond length of 42 px (see
 * geometry.js). Molfiles conventionally use ~1.5 units per bond, so we
 * divide by SCALE on write and multiply back on read — purely cosmetic,
 * it cancels out on round-trip through this app, but keeps exported
 * files closer to what other chemistry tools expect to see.
 */

window.CC = window.CC || {};

(function () {
  const SCALE = 30; // px per molfile unit, chosen so BOND_LENGTH(42)/SCALE ≈ 1.4

  function padLeft(value, width) {
    const s = String(value);
    return s.length >= width ? s : ' '.repeat(width - s.length) + s;
  }

  function padRight(value, width) {
    const s = String(value);
    return s.length >= width ? s : s + ' '.repeat(width - s.length);
  }

  function fixed(num, width, decimals) {
    return padLeft(num.toFixed(decimals), width);
  }

  // The 12 fixed-width fields after an atom's element symbol, per the
  // V2000 spec: mass-diff(2), charge(3), stereo-parity(3), h-count(3),
  // stereo-care(3), valence(3), H0-designator(3), unused(3), unused(3),
  // atom-map(3), inversion-flag(3), exact-change(3) -- 2 + 3*11 = 35
  // characters total. A real, previously-shipped bug here (one field
  // short -- "0  0  0  ..." repeated 12 times, i.e. treating the first
  // field as 3 chars wide like the rest instead of 2) silently shifted
  // every subsequent column left by one when RDKit re-parsed a molblock
  // this app had written: verified directly by diffing against a real
  // RDKit-written reference molblock for the same molecule, byte for
  // byte, after RDKit's get_smiles() on the re-parsed result started
  // showing spurious ":0" atom-map annotations on every atom (harmless
  // to valence/charge perception for ordinary atoms, since misreading
  // one all-zero field as another is invisible -- but it broke this
  // app's own downstream fragment-SMILES comparisons, e.g.
  // structure-validation.js's counterion recognition, and would
  // misalign a real non-zero field like the special zero-valence "15"
  // sentinel RDKit writes for bare ions).
  const ATOM_BLOCK_TAIL_DEFAULT = ' 0' + '  0'.repeat(11);

  CC.moleculeToMolblock = function (molecule, name) {
    const atoms = Array.from(molecule.atoms.values());
    const bonds = Array.from(molecule.bonds.values());
    const idToIndex = new Map();
    atoms.forEach(function (a, i) { idToIndex.set(a.id, i + 1); });

    const lines = [];
    lines.push(name || '');
    lines.push('  ChemCanvas');
    lines.push('');
    lines.push(
      padLeft(atoms.length, 3) + padLeft(bonds.length, 3) +
      '  0  0  0  0  0  0  0  0999 V2000'
    );

    // SVG y grows downward; molfile y grows upward, so flip on write.
    atoms.forEach(function (a) {
      lines.push(
        fixed(a.x / SCALE, 10, 4) +
        fixed(-a.y / SCALE, 10, 4) +
        fixed(0, 10, 4) +
        ' ' + padRight(a.element, 3) +
        ATOM_BLOCK_TAIL_DEFAULT
      );
    });

    bonds.forEach(function (b) {
      const i1 = idToIndex.get(b.a1);
      const i2 = idToIndex.get(b.a2);
      // MDL bond stereo flag meaning depends on bond order:
      //   single bond: 0 = none, 1 = wedge (up, narrow end at atom1),
      //                4 = either (defined stereocenter, unspecified which),
      //                6 = hash (down, narrow end at atom1).
      //   double bond: 0 = none, 3 = "either" (cis/trans genuinely
      //                unspecified -- e.g. loaded from a SMILES with no
      //                explicit / \ marks on a stereogenic double bond).
      // Losing that double-bond "either" flag on a round-trip is a real
      // bug, not a cosmetic one: re-parsing a molblock that dropped it
      // makes RDKit fall back to inferring cis/trans from the 2D
      // coordinates alone, silently inventing a stereo assignment where
      // none was intended -- exactly the trap RDKit's own molblock
      // writer sets this flag specifically to avoid. Atom1 is the
      // stereocenter by convention for wedge/hash, matching how we draw
      // and store them.
      let stereoFlag = 0;
      if (b.order === 1) {
        stereoFlag = b.stereo === 'wedge' ? 1 : b.stereo === 'hash' ? 6 : b.stereo === 'either' ? 4 : 0;
      } else if (b.order === 2) {
        stereoFlag = b.stereo === 'either' ? 3 : 0;
      }
      lines.push(padLeft(i1, 3) + padLeft(i2, 3) + padLeft(b.order, 3) + padLeft(stereoFlag, 3));
    });

    const charged = atoms.filter(function (a) { return a.charge; });
    for (let i = 0; i < charged.length; i += 8) {
      const chunk = charged.slice(i, i + 8);
      let line = 'M  CHG' + padLeft(chunk.length, 3);
      chunk.forEach(function (a) {
        line += padLeft(idToIndex.get(a.id), 4) + padLeft(a.charge, 4);
      });
      lines.push(line);
    }

    // M RAD (radical) lines -- one atom per M RAD entry (not chunked
    // 8-per-line like M CHG above) matches real MDL writers' own
    // convention closely enough for round-tripping; radicals are rare
    // enough in practice that this app never needs to write many.
    atoms.forEach(function (a) {
      if (a.radical) lines.push('M  RAD  1' + padLeft(idToIndex.get(a.id), 4) + padLeft(a.radical, 4));
      if (a.isotope) lines.push('M  ISO  1' + padLeft(idToIndex.get(a.id), 4) + padLeft(a.isotope, 4));
    });

    lines.push('M  END');
    return lines.join('\n') + '\n';
  };

  /**
   * Writes a real 3D V2000 molblock from CC.embed3D()'s output shape
   * (flat { element, x, y, z } atoms array + flat { a1, a2, order } bonds
   * array, both index-based -- not the app's own Molecule/atom-id model,
   * which moleculeToMolblock above expects). Exists specifically so
   * viewer3d.js can hand 3Dmol.js the *real* bond topology instead of
   * letting it re-guess bonds from interatomic distances in the
   * optimized structure -- a real bug, not a style choice: distance-based
   * bond perception can silently draw bonds between atoms that are just
   * spatially close (a folded conformation, or simply two non-bonded
   * atoms that ended up near each other) rather than the bonds that
   * actually exist. We already know the exact topology -- it's the same
   * bond list embed3D() built the whole 3D structure around -- so there's
   * no reason to throw it away and re-infer it.
   */
  // charges (optional): {atomIndex: integerCharge}, 0-indexed, same
  // indexing as `atoms`/`bonds`. Only used by js/xyz.js's geometry-based
  // bond/charge inference (a resonance-structure O- on a carboxylate,
  // etc.) -- every other existing caller (viewer3d.js, openff-forcefield.js)
  // omits it and gets today's all-neutral behavior unchanged. Written as
  // a standard "M  CHG" property line (confirmed live against the bundled
  // RDKit.js build: a molblock with "M  CHG  1   3  -1" parses to the
  // expected [O-] SMILES), not the V2000 atom-block charge-code column
  // (RDKit writes/reads M CHG in preference to that column when both
  // could apply, and M CHG supports the full integer range directly
  // rather than the atom block's compressed 7-value charge code).
  CC.atoms3DToMolblock = function (atoms, bonds, name, charges) {
    const lines = [];
    lines.push(name || '');
    lines.push('  ChemCanvas3D');
    lines.push('');
    lines.push(
      padLeft(atoms.length, 3) + padLeft(bonds.length, 3) +
      '  0  0  0  0  0  0  0  0999 V2000'
    );

    atoms.forEach(function (a) {
      lines.push(
        fixed(a.x, 10, 4) +
        fixed(a.y, 10, 4) +
        fixed(a.z, 10, 4) +
        ' ' + padRight(a.element, 3) +
        ATOM_BLOCK_TAIL_DEFAULT
      );
    });

    bonds.forEach(function (b) {
      // embed3D()'s bond list is 0-indexed; molfile atom numbering is 1-indexed.
      lines.push(padLeft(b.a1 + 1, 3) + padLeft(b.a2 + 1, 3) + padLeft(b.order, 3) + '  0');
    });

    if (charges) {
      const entries = Object.keys(charges)
        .map(function (k) { return { index: parseInt(k, 10), charge: charges[k] }; })
        .filter(function (e) { return e.charge; });
      for (let i = 0; i < entries.length; i += 8) {
        const chunk = entries.slice(i, i + 8);
        let line = 'M  CHG' + padLeft(chunk.length, 3);
        chunk.forEach(function (e) { line += ' ' + padLeft(e.index + 1, 3) + ' ' + padLeft(e.charge, 3); });
        lines.push(line);
      }
    }

    lines.push('M  END');
    return lines.join('\n') + '\n';
  };

  CC.molblockToMolecule = function (text) {
    const lines = text.split(/\r?\n/);
    const countsLine = lines[3] || '';
    const numAtoms = parseInt(countsLine.slice(0, 3), 10) || 0;
    const numBonds = parseInt(countsLine.slice(3, 6), 10) || 0;

    const molecule = new CC.Molecule();
    const indexToId = [];

    for (let i = 0; i < numAtoms; i++) {
      const line = lines[4 + i] || '';
      const x = parseFloat(line.slice(0, 10)) || 0;
      const y = parseFloat(line.slice(10, 20)) || 0;
      const element = (line.slice(31, 34) || 'C').trim() || 'C';
      const atom = molecule.addAtom(element, x * SCALE, -y * SCALE);
      indexToId.push(atom.id);
    }

    for (let i = 0; i < numBonds; i++) {
      const line = lines[4 + numAtoms + i] || '';
      const a1 = parseInt(line.slice(0, 3), 10);
      const a2 = parseInt(line.slice(3, 6), 10);
      const order = parseInt(line.slice(6, 9), 10) || 1;
      const stereoFlag = parseInt(line.slice(9, 12), 10) || 0;
      // See the writer's comment in moleculeToMolblock for why flag
      // meaning depends on bond order -- 3 only means "either" on a
      // double bond, 4 only means "either" on a single bond.
      let stereo = 'none';
      if (order === 1) {
        if (stereoFlag === 1) stereo = 'wedge';
        else if (stereoFlag === 6) stereo = 'hash';
        else if (stereoFlag === 4) stereo = 'either';
      } else if (order === 2 && stereoFlag === 3) {
        stereo = 'either';
      }
      if (a1 && a2 && indexToId[a1 - 1] && indexToId[a2 - 1]) {
        molecule.addBond(indexToId[a1 - 1], indexToId[a2 - 1], order, { stereo: stereo });
      }
    }

    for (let i = 4 + numAtoms + numBonds; i < lines.length; i++) {
      const line = lines[i];
      if (!line || line.indexOf('M  END') === 0) break;
      if (line.indexOf('M  CHG') === 0) {
        const count = parseInt(line.slice(6, 9), 10) || 0;
        for (let k = 0; k < count; k++) {
          const start = 9 + k * 8;
          const idx = parseInt(line.slice(start, start + 4), 10);
          const chg = parseInt(line.slice(start + 4, start + 8), 10);
          const atomId = indexToId[idx - 1];
          const atom = atomId && molecule.atoms.get(atomId);
          if (atom) atom.charge = chg;
        }
      } else if (line.indexOf('M  RAD') === 0) {
        const count = parseInt(line.slice(6, 9), 10) || 0;
        for (let k = 0; k < count; k++) {
          const start = 9 + k * 8;
          const idx = parseInt(line.slice(start, start + 4), 10);
          const rad = parseInt(line.slice(start + 4, start + 8), 10);
          const atomId = indexToId[idx - 1];
          const atom = atomId && molecule.atoms.get(atomId);
          if (atom) atom.radical = rad;
        }
      } else if (line.indexOf('M  ISO') === 0) {
        const count = parseInt(line.slice(6, 9), 10) || 0;
        for (let k = 0; k < count; k++) {
          const start = 9 + k * 8;
          const idx = parseInt(line.slice(start, start + 4), 10);
          const iso = parseInt(line.slice(start + 4, start + 8), 10);
          const atomId = indexToId[idx - 1];
          const atom = atomId && molecule.atoms.get(atomId);
          if (atom) atom.isotope = iso;
        }
      }
    }

    return molecule;
  };
})();
