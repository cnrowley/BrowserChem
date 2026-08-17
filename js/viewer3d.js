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
 *
 * Also owns the 3D distance-measurement tool (CC.Viewer3D.setMeasureMode/
 * clearMeasurements) -- kept in this file rather than app.js since this
 * is the one place that's allowed to call the raw 3Dmol.js API directly
 * (see CC.render3D's own precedent). Click two atoms to draw a dashed
 * line + distance label between them; repeatable without re-toggling.
 */

window.CC = window.CC || {};

(function () {
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

  // ---------- distance measurement state ----------
  //
  // Module-level, not per-call: atom clicks arrive asynchronously from
  // 3Dmol's own event handling, not from a single synchronous call this
  // file controls the lifetime of, so there's no natural closure to hang
  // "which atom was clicked first" off of other than the module itself.
  let measureModeEnabled = false;
  let measureCallback = null; // (result) => void, result = {element1, element2, distanceAngstrom}
  let pendingAtom = null;     // the first-clicked atom (3Dmol atom object), or null
  let pendingMarker = null;   // temporary highlight sphere shown for pendingAtom

  function clearPendingMarker(viewer) {
    if (pendingMarker && viewer) viewer.removeShape(pendingMarker);
    pendingMarker = null;
    pendingAtom = null;
  }

  function distance3D(a, b) {
    const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  function midpoint3D(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
  }

  function handleAtomClick(viewer, atom) {
    if (!pendingAtom) {
      pendingAtom = atom;
      pendingMarker = viewer.addSphere({
        center: { x: atom.x, y: atom.y, z: atom.z },
        radius: 0.45,
        color: 'yellow',
        opacity: 0.55,
      });
      viewer.render();
      return;
    }

    // Clicking the same atom again cancels the pending pick instead of
    // reporting a meaningless zero-length measurement.
    if (atom.serial === pendingAtom.serial) {
      clearPendingMarker(viewer);
      viewer.render();
      return;
    }

    const d = distance3D(pendingAtom, atom);
    viewer.addCylinder({
      start: { x: pendingAtom.x, y: pendingAtom.y, z: pendingAtom.z },
      end: { x: atom.x, y: atom.y, z: atom.z },
      radius: 0.05,
      color: 'yellow',
      dashed: true,
    });
    viewer.addLabel(d.toFixed(2) + ' Å', {
      position: midpoint3D(pendingAtom, atom),
      backgroundColor: 'black',
      backgroundOpacity: 0.7,
      fontColor: 'white',
      fontSize: 12,
      showBackground: true,
    });

    const result = { element1: pendingAtom.elem, element2: atom.elem, distanceAngstrom: d };
    clearPendingMarker(viewer);
    viewer.render();
    if (measureCallback) measureCallback(result);
  }

  // Re-applied after every CC.render3D call (below) as well as whenever
  // measure mode is toggled -- 3Dmol's clickable/callback state lives on
  // the model's own atom objects, which addModel() replaces wholesale on
  // every render, so "was clickable before" doesn't survive a re-render
  // on its own.
  function applyClickable(viewer) {
    if (!viewer) return;
    viewer.setClickable({}, measureModeEnabled, function (atom) { handleAtomClick(viewer, atom); });
  }

  CC.render3D = function (viewer, embedResult) {
    viewer.clear(); // also clears any shapes/labels from a previous measurement -- those referred to the previous conformer's atom positions and would be silently wrong overlaid on a new one
    pendingAtom = null;
    pendingMarker = null; // clear() already dropped it; keep this module's own bookkeeping in sync rather than holding a dangling reference
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
    applyClickable(viewer);
    viewer.zoomTo();
    viewer.render();
  };

  CC.Viewer3D = CC.Viewer3D || {};

  /**
   * enabled: whether clicking atoms starts/continues measuring distances.
   * onMeasured (optional): called with {element1, element2, distanceAngstrom}
   * each time a pair is completed, so a caller (app.js) can show it as
   * text alongside the 3D label. Cancels any in-progress (first-atom-
   * picked) measurement when called, whichever direction it's toggling.
   */
  CC.Viewer3D.setMeasureMode = function (viewer, enabled, onMeasured) {
    measureModeEnabled = !!enabled;
    measureCallback = onMeasured || null;
    clearPendingMarker(viewer);
    applyClickable(viewer);
    if (viewer) viewer.render();
  };

  CC.Viewer3D.isMeasureModeEnabled = function () {
    return measureModeEnabled;
  };

  /** Removes every drawn distance line/label (and any in-progress pick) without touching measure-mode itself. */
  CC.Viewer3D.clearMeasurements = function (viewer) {
    if (!viewer) return;
    clearPendingMarker(viewer);
    viewer.removeAllShapes();
    viewer.removeAllLabels();
    viewer.render();
  };
})();
