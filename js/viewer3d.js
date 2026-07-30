/**
 * viewer3d.js
 *
 * Takes the output of CC.embed3D() (a flat atom list with x/y/z, plus the
 * real bond list) and renders it in an existing 3Dmol.js viewer instance.
 *
 * Uses a real V2000 molblock (see CC.atoms3DToMolblock in molfile.js),
 * not plain XYZ -- XYZ carries no bond information at all, which forces
 * the viewer to *re-guess* bonds from interatomic distances in the
 * optimized 3D structure. That's a real correctness problem, not just a
 * style choice: distance-based bond perception can draw a bond between
 * two atoms that just happen to be spatially close (a folded
 * conformation, or two genuinely non-bonded atoms sitting near each
 * other) rather than the bonds that actually exist. We already know the
 * exact topology -- it's the same bond list embed3D() built the entire
 * 3D structure around -- so there's no reason to throw it away.
 */

window.CC = window.CC || {};

// Explicit element colors, rather than trusting 3Dmol.js's own default
// palette -- a real, reported problem, not a hypothetical one: F and S
// were visually too close to tell apart at a glance in the default
// scheme. Numeric 0xRRGGBB (3Dmol's elementColors map format, confirmed
// against its own docs -- a plain '#rrggbb' string is NOT what
// colorscheme:{prop:'elem', map:...} expects). Kept in the same hue
// family as the 2D editor's own ELEMENT_COLORS (render.js) for
// consistency between the two views, but deliberately NOT identical
// hex-for-hex to render.js's table: F and Cl share render.js's exact
// color there too (also a real bug, fixed separately in render.js), and
// this table gives every element -- including C/H, which the 2D editor
// leaves uncolored since it usually doesn't label them -- its own
// distinguishable color, which the 3D view needs since every atom is
// always rendered here regardless of label visibility.
const ELEMENT_COLORS_3D = {
  H: 0xdcdcdc,
  C: 0x4a4a4a,
  N: 0x2f6fed,
  O: 0xd9432f,
  F: 0x1fa8a0,  // teal-cyan, clearly blue-leaning
  Cl: 0x3dae2e, // true green, clearly yellow-leaning -- kept well apart from F's teal
  S: 0xc99a1e,
  P: 0xd9832f,
  Br: 0xa0522d,
  I: 0x7b3fa0,
};

CC.render3D = function (viewer, embedResult) {
  viewer.clear();
  if (!embedResult || embedResult.atoms.length === 0) {
    viewer.render();
    return;
  }
  const molblock = CC.atoms3DToMolblock(embedResult.atoms, embedResult.bonds);
  viewer.addModel(molblock, 'mol');
  const colorscheme = { prop: 'elem', map: ELEMENT_COLORS_3D };
  viewer.setStyle({}, {
    stick: { radius: 0.14, colorscheme: colorscheme },
    sphere: { scale: 0.28, colorscheme: colorscheme },
  });
  viewer.zoomTo();
  viewer.render();
};
