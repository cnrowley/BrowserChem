/**
 * stereo2d.js
 *
 * Two things, both computed directly from OUR OWN 2D coordinates and bond
 * data — no RDKit dependency, and no claim to full CIP R/S or E/Z labels:
 *
 *   1. Wedge/hash lookup — which atoms have an adjacent stereo bond, and
 *      which kind. This is real, unambiguous data straight from what the
 *      user drew.
 *
 *   2. Double-bond cis/trans classification — for a C=C (or similar) with
 *      exactly one other substituent on each end, whether those two
 *      substituents sit on the same side of the double bond axis (cis) or
 *      opposite sides (trans). This is plain 2D geometry (a cross-product
 *      sign test), not a CIP priority calculation — "cis/trans" here means
 *      the geometric relationship of the two substituents actually drawn,
 *      which is exactly what a 2D drawing communicates. True CIP E/Z
 *      additionally requires ranking substituents by priority when either
 *      end has more than one non-H substituent; that ranking algorithm
 *      isn't implemented here, so double bonds with two substituents per
 *      end are reported as 'unknown' rather than guessing.
 */

window.CC = window.CC || {};

(function () {
  /**
   * Returns a Map atomId -> 'wedge' | 'hash' for every atom that is the
   * narrow (stereocenter) end of a wedge or hash bond. An atom can only
   * sensibly have one such bond in a typical 2D drawing; if more than one
   * is present, the first one found wins.
   */
  CC.getWedgeHashByAtom = function (molecule) {
    const result = new Map();
    molecule.bonds.forEach(function (b) {
      if (b.stereo === 'wedge' || b.stereo === 'hash') {
        if (!result.has(b.a1)) result.set(b.a1, b.stereo);
      }
    });
    return result;
  };

  function crossSign(ax, ay, bx, by) {
    const cross = ax * by - ay * bx;
    if (Math.abs(cross) < 1e-6) return 0;
    return cross > 0 ? 1 : -1;
  }

  /**
   * Returns a Map bondId -> 'cis' | 'trans' | 'unknown' for every
   * double-order bond in the molecule. 'unknown' covers: terminal double
   * bonds (nothing to compare), either end having 0 or 2+ other
   * substituents, or a degenerate (near-zero-length) geometry.
   */
  CC.getDoubleBondStereo = function (molecule) {
    const result = new Map();

    molecule.bonds.forEach(function (bond) {
      if (bond.order !== 2) return;

      const a1 = molecule.atoms.get(bond.a1);
      const a2 = molecule.atoms.get(bond.a2);
      if (!a1 || !a2) return;

      const a1Others = molecule.getBondsForAtom(a1.id)
        .filter(function (b) { return b.id !== bond.id; })
        .map(function (b) { return (b.a1 === a1.id ? b.a2 : b.a1); });
      const a2Others = molecule.getBondsForAtom(a2.id)
        .filter(function (b) { return b.id !== bond.id; })
        .map(function (b) { return (b.a1 === a2.id ? b.a2 : b.a1); });

      if (a1Others.length !== 1 || a2Others.length !== 1) {
        result.set(bond.id, 'unknown');
        return;
      }

      const sub1 = molecule.atoms.get(a1Others[0]);
      const sub2 = molecule.atoms.get(a2Others[0]);
      if (!sub1 || !sub2) {
        result.set(bond.id, 'unknown');
        return;
      }

      const axisX = a2.x - a1.x;
      const axisY = a2.y - a1.y;
      const side1 = crossSign(axisX, axisY, sub1.x - a1.x, sub1.y - a1.y);
      const side2 = crossSign(axisX, axisY, sub2.x - a2.x, sub2.y - a2.y);

      if (side1 === 0 || side2 === 0) {
        result.set(bond.id, 'unknown'); // substituent collinear with the double bond — degenerate drawing
        return;
      }

      result.set(bond.id, side1 === side2 ? 'cis' : 'trans');
    });

    return result;
  };
})();
