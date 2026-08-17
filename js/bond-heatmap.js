/**
 * bond-heatmap.js
 *
 * Colors bonds on the 2D canvas by a per-bond scalar value (e.g. ALFABET-
 * style bond dissociation enthalpy from a bond-level GNN head), with
 * numeric labels at each bond's midpoint -- this app's first bond-level
 * (as opposed to atom- or molecule-level) visualization, mirroring
 * atom-heatmap.js's structure closely and reusing its CC.heatColor /
 * CC.renderHeatmapLegend so both heatmaps share one colormap.
 *
 * Unlike atom-heatmap.js's charge-detection, bond dissociation enthalpy
 * has an unambiguous "interesting" direction regardless of which
 * specific bond-level property is loaded: a LOW value is the weak,
 * reactive, chemically interesting bond (ALFABET's whole point is
 * finding these), so this always uses the reversed colormap (low->red,
 * high->blue) -- the same "hot spot" convention, not the charge-specific
 * zero-centering atom-heatmap.js applies.
 *
 * This draws on top of whatever render.js already produced for the
 * current frame — call it after CC.render(), not instead of it. Reads
 * bond endpoints directly from the molecule (same a1/a2 atom lookup
 * render.js's own drawBond() uses) rather than the hit-test layer.
 */

window.CC = window.CC || {};

(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';

  /**
   * bondValues: Map or plain object of bondId -> number.
   * Removes any previous bond heatmap layer before drawing a new one, and
   * does nothing (leaves canvas as-is) if bondValues is empty/null — call
   * CC.clearBondHeatmap() explicitly to remove an existing overlay.
   *
   * Returns the {min, max, zeroCentered, reversed} scale actually used,
   * so a caller can draw a matching legend (see CC.renderHeatmapLegend).
   */
  CC.renderBondHeatmap = function (svg, molecule, bondValues, propertyName) {
    CC.clearBondHeatmap(svg);
    if (!bondValues) return null;

    const values = molecule ? Array.from(molecule.bonds.keys())
      .map(function (id) { return bondValues[id]; })
      .filter(function (v) { return typeof v === 'number' && !isNaN(v); }) : [];
    if (values.length === 0) return null;

    const min = Math.min.apply(null, values);
    const max = Math.max.apply(null, values);
    const range = (max - min) || 1;
    const decimals = CC.propertyDecimals(propertyName, 1);

    const layer = document.createElementNS(SVG_NS, 'g');
    layer.setAttribute('class', 'layer-bond-heatmap');
    layer.setAttribute('data-bond-heatmap', 'true');

    molecule.bonds.forEach(function (bond) {
      const value = bondValues[bond.id];
      if (typeof value !== 'number' || isNaN(value)) return;
      const a1 = molecule.atoms.get(bond.a1);
      const a2 = molecule.atoms.get(bond.a2);
      if (!a1 || !a2) return;
      const t = (value - min) / range;

      const line = document.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', a1.x);
      line.setAttribute('y1', a1.y);
      line.setAttribute('x2', a2.x);
      line.setAttribute('y2', a2.y);
      line.setAttribute('stroke', CC.heatColor(t, true)); // reversed: low (weak bond) -> red
      line.setAttribute('stroke-width', 7);
      line.setAttribute('stroke-linecap', 'round');
      line.setAttribute('opacity', '0.55');
      line.setAttribute('class', 'heatmap-bond');

      const title = document.createElementNS(SVG_NS, 'title');
      title.textContent = a1.element + '-' + a2.element + ': ' + value.toFixed(decimals);
      line.appendChild(title);

      layer.appendChild(line);
    });

    // Behind the atom/bond layers (same reasoning as atom-heatmap.js):
    // labels and the real bond lines stay legible on top of the overlay.
    svg.insertBefore(layer, svg.firstChild);

    return { min: min, max: max, zeroCentered: false, reversed: true, decimals: decimals };
  };

  CC.clearBondHeatmap = function (svg) {
    const existing = svg.querySelector('[data-bond-heatmap="true"]');
    if (existing) existing.remove();
  };

  /**
   * Secondary display for the same per-bond values the heatmap colors --
   * numeric text at each bond's midpoint. Meant to be layered together
   * with the color heatmap, not a replacement for it.
   */
  CC.renderBondValueLabels = function (svg, molecule, bondValues, propertyName, decimals) {
    CC.clearBondValueLabels(svg);
    if (!bondValues || !molecule || molecule.bonds.size === 0) return;
    decimals = decimals === undefined ? CC.propertyDecimals(propertyName, 1) : decimals;

    const layer = document.createElementNS(SVG_NS, 'g');
    layer.setAttribute('class', 'layer-bond-heatmap-text');
    layer.setAttribute('data-bond-heatmap-text', 'true');

    molecule.bonds.forEach(function (bond) {
      const value = bondValues[bond.id];
      if (typeof value !== 'number' || isNaN(value)) return;
      const a1 = molecule.atoms.get(bond.a1);
      const a2 = molecule.atoms.get(bond.a2);
      if (!a1 || !a2) return;

      const text = document.createElementNS(SVG_NS, 'text');
      text.setAttribute('x', (a1.x + a2.x) / 2);
      text.setAttribute('y', (a1.y + a2.y) / 2);
      text.setAttribute('class', 'heatmap-value-label heatmap-bond-value-label');
      text.setAttribute('text-anchor', 'middle');
      text.textContent = value.toFixed(decimals);
      layer.appendChild(text);
    });

    svg.appendChild(layer); // on top, unlike the color heatmap -- text needs to stay legible
  };

  CC.clearBondValueLabels = function (svg) {
    const existing = svg.querySelector('[data-bond-heatmap-text="true"]');
    if (existing) existing.remove();
  };
})();
