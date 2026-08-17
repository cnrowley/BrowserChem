/**
 * render.js
 *
 * Pure(ish) rendering: takes a Molecule and draws it into an <svg> element.
 * No event handling lives here — tools.js owns interaction, this file only
 * turns molecule state into SVG markup.
 */

window.CC = window.CC || {};

(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const LABEL_GAP = 11;

  const ELEMENT_COLORS = {
    N: '#2f6fed',
    O: '#d9432f',
    S: '#c99a1e',
    P: '#d9832f',
    F: '#1fa8a0',
    Cl: '#3dae2e',
    Br: '#a0522d',
    I: '#7b3fa0',
  };

  function el(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const key in attrs) {
        node.setAttribute(key, attrs[key]);
      }
    }
    return node;
  }

  function createLine(p1, p2, attrs) {
    const line = el('line', Object.assign({
      x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y,
    }, attrs || {}));
    return line;
  }

  function shouldLabelAtom(molecule, atom) {
    return atom.element !== 'C' || molecule.getDegree(atom.id) === 0;
  }

  function insetPoint(from, to, dist) {
    if (dist <= 0) return { x: from.x, y: from.y };
    const d = CC.distance(from, to);
    if (d === 0) return { x: from.x, y: from.y };
    const t = dist / d;
    return {
      x: from.x + (to.x - from.x) * t,
      y: from.y + (to.y - from.y) * t,
    };
  }

  function perpUnit(a1, a2) {
    const dx = a2.x - a1.x;
    const dy = a2.y - a1.y;
    const len = Math.hypot(dx, dy) || 1;
    return { dx: -dy / len, dy: dx / len };
  }

  function formatCharge(charge) {
    if (charge === 0) return '';
    const magnitude = Math.abs(charge);
    const sign = charge > 0 ? '+' : '\u2212';
    return magnitude === 1 ? sign : magnitude + sign;
  }

  function drawWedgeBond(layer, p1, p2, perp) {
    // Filled triangle: narrow point at p1 (the stereocenter), wide base at p2.
    const halfWidth = 3.6;
    const b1 = { x: p2.x + perp.dx * halfWidth, y: p2.y + perp.dy * halfWidth };
    const b2 = { x: p2.x - perp.dx * halfWidth, y: p2.y - perp.dy * halfWidth };
    const points = p1.x + ',' + p1.y + ' ' + b1.x + ',' + b1.y + ' ' + b2.x + ',' + b2.y;
    layer.appendChild(el('polygon', { points: points, class: 'bond-wedge' }));
  }

  function drawHashBond(layer, p1, p2, perp) {
    // A series of parallel bars, growing from a point at p1 to full width
    // at p2 — the standard "dashed wedge" convention for a bond receding
    // away from the viewer.
    const segments = 6;
    for (let i = 1; i <= segments; i++) {
      const t = i / segments;
      const center = { x: p1.x + (p2.x - p1.x) * t, y: p1.y + (p2.y - p1.y) * t };
      const halfWidth = 3.6 * t;
      const o1 = { x: center.x + perp.dx * halfWidth, y: center.y + perp.dy * halfWidth };
      const o2 = { x: center.x - perp.dx * halfWidth, y: center.y - perp.dy * halfWidth };
      layer.appendChild(createLine(o1, o2, { class: 'bond-hash' }));
    }
  }

  function drawBond(layer, molecule, bond, a1, a2) {
    const label1 = shouldLabelAtom(molecule, a1);
    const label2 = shouldLabelAtom(molecule, a2);
    const p1 = insetPoint(a1, a2, label1 ? LABEL_GAP : 0);
    const p2 = insetPoint(a2, a1, label2 ? LABEL_GAP : 0);

    if (bond.stereo === 'wedge') {
      drawWedgeBond(layer, p1, p2, perpUnit(a1, a2));
      return;
    }
    if (bond.stereo === 'hash') {
      drawHashBond(layer, p1, p2, perpUnit(a1, a2));
      return;
    }

    if (bond.order <= 1) {
      layer.appendChild(createLine(p1, p2, { class: 'bond-line' }));
      return;
    }

    const offsets = bond.order === 2 ? [-2.6, 2.6] : [-4.4, 0, 4.4];
    const perp = perpUnit(a1, a2);
    for (const off of offsets) {
      const o1 = { x: p1.x + perp.dx * off, y: p1.y + perp.dy * off };
      const o2 = { x: p2.x + perp.dx * off, y: p2.y + perp.dy * off };
      layer.appendChild(createLine(o1, o2, { class: 'bond-line' }));
    }
  }

  // Implicit H count for label purposes: valence minus the sum of this
  // atom's own bond orders. A fast, local, valence-table estimate (same
  // convention chemprop-features.js's fallback uses) -- fine for display,
  // where the goal is "a reasonable label as you draw," not a
  // RDKit-verified count.
  function implicitHCountForLabel(molecule, atom) {
    const data = CC.elementData(atom.element);
    const usedValence = molecule.getBondsForAtom(atom.id).reduce(function (sum, b) { return sum + b.order; }, 0);
    return Math.max(0, data.valence - usedValence);
  }

  function drawAtomLabel(layer, molecule, atom) {
    const text = el('text', {
      x: atom.x,
      y: atom.y,
      class: 'atom-label',
      'text-anchor': 'middle',
      'dominant-baseline': 'central',
    });
    if (ELEMENT_COLORS[atom.element]) {
      text.style.fill = ELEMENT_COLORS[atom.element];
    }
    text.textContent = atom.element;

    // Implicit-H suffix (NH2, OH, CH3, ...) -- element symbol first, then
    // H, then a subscripted count if more than one. Always element-then-H
    // regardless of which side the bond enters from; a fully
    // direction-aware layout (H2N- when the label reads right-to-left)
    // is a real refinement but not implemented here.
    const hCount = implicitHCountForLabel(molecule, atom);
    if (hCount > 0) {
      const hSpan = el('tspan', { class: 'atom-label-h' });
      hSpan.textContent = 'H';
      text.appendChild(hSpan);
      if (hCount > 1) {
        const countSpan = el('tspan', { class: 'atom-label-h-count', dy: '3' });
        countSpan.textContent = String(hCount);
        text.appendChild(countSpan);
        // reset baseline for anything appended after (the charge tspan
        // below) so it doesn't inherit this subscript's lowered dy
        const resetSpan = el('tspan', { dy: '-3' });
        resetSpan.textContent = '';
        text.appendChild(resetSpan);
      }
    }

    if (atom.charge) {
      const tspan = el('tspan', { class: 'atom-charge', dy: '-6' });
      tspan.textContent = formatCharge(atom.charge);
      text.appendChild(tspan);
    }
    layer.appendChild(text);
  }

  CC.render = function (svg, molecule, state) {
    state = state || {};
    const selectedAtomIds = state.selectedAtomIds || new Set();

    while (svg.firstChild) svg.removeChild(svg.firstChild);

    if (molecule.isEmpty()) {
      const placeholder = el('text', {
        x: 300, y: 220, 'text-anchor': 'middle', class: 'canvas-placeholder-text',
      });
      placeholder.textContent = 'Select the atom tool, then click here to place your first atom';
      svg.appendChild(placeholder);
      svg.__overlayLayer = null;
      return;
    }

    const bondLayer = el('g', { class: 'layer-bonds' });
    const atomLayer = el('g', { class: 'layer-atoms' });
    const hitLayer = el('g', { class: 'layer-hits' });
    const overlayLayer = el('g', { class: 'layer-overlay' });

    svg.appendChild(bondLayer);
    svg.appendChild(atomLayer);
    svg.appendChild(hitLayer);
    svg.appendChild(overlayLayer);

    for (const bond of molecule.bonds.values()) {
      const a1 = molecule.atoms.get(bond.a1);
      const a2 = molecule.atoms.get(bond.a2);
      if (!a1 || !a2) continue;
      drawBond(bondLayer, molecule, bond, a1, a2);
      hitLayer.appendChild(createLine(a1, a2, {
        class: 'bond-hit', 'data-bond-id': bond.id,
      }));
    }

    for (const atom of molecule.atoms.values()) {
      if (selectedAtomIds.has(atom.id)) {
        atomLayer.appendChild(el('circle', {
          cx: atom.x, cy: atom.y, r: 12, class: 'atom-selection-halo',
        }));
      }
      if (shouldLabelAtom(molecule, atom)) {
        drawAtomLabel(atomLayer, molecule, atom);
      }
      hitLayer.appendChild(el('circle', {
        cx: atom.x, cy: atom.y, r: 12, class: 'atom-hit', 'data-atom-id': atom.id,
      }));
    }

    svg.__overlayLayer = overlayLayer;
  };

  CC.renderPreview = function (svg, molecule, drag) {
    const layer = svg.__overlayLayer;
    if (!layer) return;

    if (drag.type === 'bond' && drag.previewEnd) {
      const start = molecule.atoms.get(drag.startAtomId);
      if (!start) return;
      layer.appendChild(createLine(start, drag.previewEnd, { class: 'bond-preview' }));
      if (!drag.hoverAtomId) {
        layer.appendChild(el('circle', {
          cx: drag.previewEnd.x, cy: drag.previewEnd.y, r: 3, class: 'atom-ghost',
        }));
      }
      return;
    }

    if (drag.type === 'chain' && drag.previewPoints) {
      const pts = drag.previewPoints;
      for (let i = 0; i < pts.length - 1; i++) {
        layer.appendChild(createLine(pts[i], pts[i + 1], { class: 'bond-preview' }));
      }
      const last = pts[pts.length - 1];
      if (!drag.snapAtomId) {
        layer.appendChild(el('circle', { cx: last.x, cy: last.y, r: 3, class: 'atom-ghost' }));
      }
    }
  };

  // A single persistent, non-interactive circle showing whatever atom or
  // bond midpoint the pointer is currently near -- independent of
  // CC.render/CC.renderPreview so it can be repositioned on every
  // pointermove without rebuilding the whole molecule's SVG each time.
  // Because CC.render() tears down and rebuilds the SVG's own children on
  // every molecule edit, this element is looked up lazily rather than
  // cached, and re-appended as needed -- it heals itself on the next
  // pointermove after any full re-render.
  CC.updateHoverHighlight = function (svg, target) {
    let circle = svg.querySelector('#hover-highlight');
    if (!target) {
      if (circle) circle.style.display = 'none';
      return;
    }
    if (!circle) {
      circle = el('circle', { id: 'hover-highlight', class: 'hover-highlight' });
      svg.appendChild(circle);
    } else if (svg.lastChild !== circle) {
      svg.appendChild(circle); // keep it topmost
    }
    circle.setAttribute('cx', target.x);
    circle.setAttribute('cy', target.y);
    circle.setAttribute('r', target.r);
    circle.style.display = '';
  };
})();
