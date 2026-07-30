/**
 * atom-heatmap.js
 *
 * Colors atoms on the 2D canvas by a per-atom scalar value (e.g. one
 * property from a GNN atomic-property head), with a diverging colormap
 * and a hover tooltip showing the exact value.
 *
 * Two things this handles specifically for signed physical quantities
 * like partial charges, not just "any number":
 *
 *   1. Color direction matches the standard electrostatic-potential-map
 *      convention (red = negative/electron-rich, blue = positive/
 *      electron-poor) -- the same convention every ESP surface in
 *      Spartan, Gaussian, PyMOL, etc. uses. Getting this backwards for
 *      charges specifically would be actively misleading to anyone who's
 *      seen an ESP map before, not just a cosmetic choice.
 *   2. Zero-centered normalization, not molecule min/max-centered. A
 *      plain linear min->max map puts "white" (the visually neutral
 *      midpoint) at whatever this molecule's min/max happen to average
 *      to -- e.g. aspirin's charges run roughly -0.7 to +1.0, so a naive
 *      min/max scale would show white around +0.15, not at an actual
 *      charge of zero. For a signed quantity where zero has real
 *      physical meaning, that's a real correctness issue, not styling.
 *
 * Both are auto-detected from the property name (anything containing
 * "charge") rather than needing the caller to know to ask for them --
 * other atom-level properties (which may have no natural zero, or where
 * "high/low" rather than "positive/negative" is the meaningful framing)
 * keep the original low->blue/high->red, min/max-normalized behavior.
 *
 * This draws on top of whatever render.js already produced for the
 * current frame — call it after CC.render(), not instead of it.
 */

window.CC = window.CC || {};

(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';

  function lerp(a, b, t) { return a + (b - a) * t; }

  function isChargeLikeProperty(name) {
    return /charge/i.test(name || '');
  }

  // Diverging colormap, t in [0,1], t=0.5 is the neutral midpoint.
  // `reversed`: false = low->blue/high->red (generic numeric property);
  // true = low->red/high->blue (ESP-map convention, for charges).
  function heatColor(t, reversed) {
    t = Math.max(0, Math.min(1, t));
    const lowColor = reversed ? [217, 60, 60] : [60, 90, 220];
    const highColor = reversed ? [60, 90, 220] : [217, 60, 60];
    let r, g, b;
    if (t < 0.5) {
      const k = t / 0.5;
      r = lerp(lowColor[0], 245, k); g = lerp(lowColor[1], 245, k); b = lerp(lowColor[2], 245, k);
    } else {
      const k = (t - 0.5) / 0.5;
      r = lerp(245, highColor[0], k); g = lerp(245, highColor[1], k); b = lerp(245, highColor[2], k);
    }
    return 'rgb(' + Math.round(r) + ',' + Math.round(g) + ',' + Math.round(b) + ')';
  }

  /**
   * atomValues: Map or plain object of atomId -> number.
   * propertyName: used to auto-detect charge-like display conventions
   * (color direction, zero-centering) -- pass whatever key this value
   * came from (e.g. "mbis-charges").
   * Removes any previous heatmap layer before drawing a new one, and does
   * nothing (leaves canvas as-is) if atomValues is empty/null — call
   * CC.clearAtomHeatmap() explicitly to remove an existing overlay.
   *
   * Returns the {min, max, zeroCentered, reversed} scale actually used,
   * so a caller can draw a matching legend (see CC.renderHeatmapLegend).
   */
  CC.renderAtomHeatmap = function (svg, molecule, atomValues, propertyName) {
    CC.clearAtomHeatmap(svg);
    if (!atomValues) return null;

    const values = molecule ? Array.from(molecule.atoms.keys())
      .map(function (id) { return atomValues[id]; })
      .filter(function (v) { return typeof v === 'number' && !isNaN(v); }) : [];
    if (values.length === 0) return null;

    const chargeLike = isChargeLikeProperty(propertyName);
    const rawMin = Math.min.apply(null, values);
    const rawMax = Math.max.apply(null, values);

    // Zero-centered: scale symmetrically around 0 by the larger-magnitude
    // extreme, so a charge of 0 always lands exactly at the white
    // midpoint regardless of how skewed this particular molecule's
    // charges happen to be.
    let min = rawMin, max = rawMax;
    if (chargeLike) {
      const absExtent = Math.max(Math.abs(rawMin), Math.abs(rawMax)) || 1;
      min = -absExtent; max = absExtent;
    }
    const range = (max - min) || 1;

    const layer = document.createElementNS(SVG_NS, 'g');
    layer.setAttribute('class', 'layer-heatmap');
    layer.setAttribute('data-heatmap', 'true');

    molecule.atoms.forEach(function (atom) {
      const value = atomValues[atom.id];
      if (typeof value !== 'number' || isNaN(value)) return;
      const t = (value - min) / range;

      const circle = document.createElementNS(SVG_NS, 'circle');
      circle.setAttribute('cx', atom.x);
      circle.setAttribute('cy', atom.y);
      circle.setAttribute('r', 13);
      circle.setAttribute('fill', heatColor(t, chargeLike));
      circle.setAttribute('opacity', '0.55');
      circle.setAttribute('class', 'heatmap-atom');

      const title = document.createElementNS(SVG_NS, 'title');
      title.textContent = atom.element + ': ' + value.toFixed(2);
      circle.appendChild(title);

      layer.appendChild(circle);
    });

    // Insert behind the atom/bond layers so labels stay legible on top,
    // but after the background so it's actually visible.
    svg.insertBefore(layer, svg.firstChild);

    return { min: min, max: max, zeroCentered: chargeLike, reversed: chargeLike };
  };

  CC.clearAtomHeatmap = function (svg) {
    const existing = svg.querySelector('[data-heatmap="true"]');
    if (existing) existing.remove();
  };

  /**
   * Secondary display for the same per-atom values the heatmap colors --
   * small numeric text labels positioned near (not on top of) each atom,
   * for reading exact numbers at a glance instead of hovering each atom
   * individually or eyeballing a color gradient. Meant to be layered
   * together with the color heatmap, not a replacement for it -- call
   * both, or either alone.
   *
   * Positioned below-right of each atom by default, offset further out
   * along that atom's own outward direction (away from the molecule's
   * centroid) so labels on the "outside" of a ring or chain don't
   * overlap bonds pointing inward -- a fixed offset would put labels on
   * top of bonds for atoms on the left/top of a structure.
   */
  CC.renderAtomValueLabels = function (svg, molecule, atomValues, decimals) {
    CC.clearAtomValueLabels(svg);
    if (!atomValues || !molecule || molecule.atoms.size === 0) return;
    decimals = decimals === undefined ? 2 : decimals;

    const atoms = Array.from(molecule.atoms.values());
    const centroid = atoms.reduce(function (acc, a) {
      return { x: acc.x + a.x / atoms.length, y: acc.y + a.y / atoms.length };
    }, { x: 0, y: 0 });

    const layer = document.createElementNS(SVG_NS, 'g');
    layer.setAttribute('class', 'layer-heatmap-text');
    layer.setAttribute('data-heatmap-text', 'true');

    atoms.forEach(function (atom) {
      const value = atomValues[atom.id];
      if (typeof value !== 'number' || isNaN(value)) return;

      let dx = atom.x - centroid.x, dy = atom.y - centroid.y;
      const len = Math.hypot(dx, dy) || 1;
      dx = (dx / len) * 16 + 10; // push outward, plus a fixed offset for atoms near the centroid
      dy = (dy / len) * 16 + 14;

      const text = document.createElementNS(SVG_NS, 'text');
      text.setAttribute('x', atom.x + dx);
      text.setAttribute('y', atom.y + dy);
      text.setAttribute('class', 'heatmap-value-label');
      text.setAttribute('text-anchor', 'middle');
      text.textContent = value.toFixed(decimals);
      layer.appendChild(text);
    });

    svg.appendChild(layer); // on top, unlike the color heatmap -- text needs to stay legible
  };

  CC.clearAtomValueLabels = function (svg) {
    const existing = svg.querySelector('[data-heatmap-text="true"]');
    if (existing) existing.remove();
  };

  /**
   * Draws a small horizontal gradient legend into `container` (any DOM
   * element, not the SVG canvas -- meant for the side panel, next to the
   * heatmap property dropdown) matching a scale returned by
   * CC.renderAtomHeatmap. Replaces any previously-rendered legend in the
   * same container.
   */
  CC.renderHeatmapLegend = function (container, scale) {
    container.innerHTML = '';
    if (!scale) return;

    const wrap = document.createElement('div');
    wrap.className = 'heatmap-legend';

    const gradientDiv = document.createElement('div');
    gradientDiv.className = 'heatmap-legend-gradient';
    const stops = [];
    for (let i = 0; i <= 10; i++) {
      stops.push(heatColor(i / 10, scale.reversed) + ' ' + (i * 10) + '%');
    }
    gradientDiv.style.background = 'linear-gradient(to right, ' + stops.join(', ') + ')';
    wrap.appendChild(gradientDiv);

    const labelsDiv = document.createElement('div');
    labelsDiv.className = 'heatmap-legend-labels';
    const lowLabel = document.createElement('span');
    lowLabel.textContent = scale.min.toFixed(2);
    const midLabel = document.createElement('span');
    midLabel.textContent = scale.zeroCentered ? '0' : ((scale.min + scale.max) / 2).toFixed(2);
    const highLabel = document.createElement('span');
    highLabel.textContent = scale.max.toFixed(2);
    labelsDiv.appendChild(lowLabel);
    labelsDiv.appendChild(midLabel);
    labelsDiv.appendChild(highLabel);
    wrap.appendChild(labelsDiv);

    container.appendChild(wrap);
  };
})();
