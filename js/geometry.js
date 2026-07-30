/**
 * geometry.js
 *
 * Small, dependency-free geometry helpers shared by the renderer and the
 * interaction tools: converting pointer events into SVG coordinates, and
 * snapping freehand bond drags to standard chemistry drawing angles.
 */

window.CC = window.CC || {};

CC.BOND_LENGTH = 42;

// Convert a client-space pointer position into the SVG's own coordinate
// system, accounting for the viewBox and any CSS scaling.
CC.svgPoint = function (svg, clientX, clientY) {
  const pt = svg.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  const ctm = svg.getScreenCTM();
  if (!ctm) return { x: 0, y: 0 };
  const loc = pt.matrixTransform(ctm.inverse());
  return { x: loc.x, y: loc.y };
};

CC.distance = function (p1, p2) {
  return Math.hypot(p2.x - p1.x, p2.y - p1.y);
};

CC.DEFAULT_VIEWBOX = { x: 0, y: 0, width: 600, height: 440 };

/**
 * Compute a viewBox that fits every atom in `molecule` with padding,
 * preserving the SVG's own aspect ratio (read from its current viewBox).
 * Returns CC.DEFAULT_VIEWBOX for an empty molecule -- there's nothing to
 * fit to, and resetting to the default is more useful than leaving
 * whatever pan/zoom state was active before.
 *
 * This exists because a hand-drawn molecule always stays within the
 * default viewBox by construction (you draw incrementally on what's
 * visible), but a molecule loaded from SMILES uses RDKit's own
 * auto-generated 2D coordinates, which have no relationship to this
 * app's canvas origin at all -- without this, a loaded structure can
 * land partly or entirely outside the visible area with no way to see
 * the rest of it.
 */
CC.computeFitViewBox = function (svg, molecule, padding) {
  padding = padding === undefined ? 60 : padding;
  const atoms = Array.from(molecule.atoms.values());
  if (atoms.length === 0) return Object.assign({}, CC.DEFAULT_VIEWBOX);

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  atoms.forEach(function (a) {
    if (a.x < minX) minX = a.x;
    if (a.x > maxX) maxX = a.x;
    if (a.y < minY) minY = a.y;
    if (a.y > maxY) maxY = a.y;
  });

  minX -= padding; minY -= padding; maxX += padding; maxY += padding;
  const contentWidth = Math.max(1, maxX - minX);
  const contentHeight = Math.max(1, maxY - minY);

  // Preserve the SVG's own aspect ratio (its rendered box, not whatever
  // the current viewBox happens to be) so content doesn't look stretched.
  const rect = svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width
    ? svg.viewBox.baseVal
    : CC.DEFAULT_VIEWBOX;
  const aspect = rect.width / rect.height;

  let width = contentWidth, height = contentHeight;
  if (width / height > aspect) {
    height = width / aspect;
  } else {
    width = height * aspect;
  }
  // Never zoom in tighter than the default view, even for a single atom --
  // a fit that's too close feels jarring for trivially small molecules.
  if (width < CC.DEFAULT_VIEWBOX.width) {
    const scale = CC.DEFAULT_VIEWBOX.width / width;
    width *= scale; height *= scale;
  }

  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  return { x: cx - width / 2, y: cy - height / 2, width: width, height: height };
};

CC.setViewBox = function (svg, vb) {
  svg.setAttribute('viewBox', vb.x + ' ' + vb.y + ' ' + vb.width + ' ' + vb.height);
};

CC.fitViewToContent = function (svg, molecule, padding) {
  CC.setViewBox(svg, CC.computeFitViewBox(svg, molecule, padding));
};

/**
 * Zoom the SVG's viewBox by `factor` (>1 zooms out, <1 zooms in), keeping
 * the point under (clientX, clientY) stationary on screen -- the usual
 * scroll-to-zoom feel. Clamped to a sane range so the view can't be
 * zoomed into uselessness or out to nothing.
 */
CC.zoomViewBoxAt = function (svg, clientX, clientY, factor) {
  const vb = svg.viewBox.baseVal;
  const focal = CC.svgPoint(svg, clientX, clientY);

  let newWidth = vb.width * factor;
  let newHeight = vb.height * factor;
  const minWidth = 80, maxWidth = 4000;
  if (newWidth < minWidth) { const s = minWidth / newWidth; newWidth *= s; newHeight *= s; }
  if (newWidth > maxWidth) { const s = maxWidth / newWidth; newWidth *= s; newHeight *= s; }

  const newX = focal.x - (focal.x - vb.x) * (newWidth / vb.width);
  const newY = focal.y - (focal.y - vb.y) * (newHeight / vb.height);
  CC.setViewBox(svg, { x: newX, y: newY, width: newWidth, height: newHeight });
};

/**
 * Given a fixed origin and a freehand target point, return a point at the
 * standard bond length, with its angle snapped to the nearest `stepDeg`
 * increment. This is what makes bond drawing feel like a chemistry tool
 * instead of a freehand line tool.
 */
CC.snapToAngle = function (origin, target, stepDeg, length) {
  stepDeg = stepDeg || 15;
  length = length || CC.BOND_LENGTH;
  const dx = target.x - origin.x;
  const dy = target.y - origin.y;
  let angle = Math.atan2(dy, dx);
  const step = (stepDeg * Math.PI) / 180;
  angle = Math.round(angle / step) * step;
  return {
    x: origin.x + Math.cos(angle) * length,
    y: origin.y + Math.sin(angle) * length,
  };
};
