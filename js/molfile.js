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
        '0  0  0  0  0  0  0  0  0  0  0  0'
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
  CC.atoms3DToMolblock = function (atoms, bonds, name) {
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
        '0  0  0  0  0  0  0  0  0  0  0  0'
      );
    });

    bonds.forEach(function (b) {
      // embed3D()'s bond list is 0-indexed; molfile atom numbering is 1-indexed.
      lines.push(padLeft(b.a1 + 1, 3) + padLeft(b.a2 + 1, 3) + padLeft(b.order, 3) + '  0');
    });

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
      }
    }

    return molecule;
  };
})();
