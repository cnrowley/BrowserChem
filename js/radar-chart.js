/**
 * radar-chart.js
 *
 * A 15-axis property radar/spider chart. Pure data-in, SVG-out module --
 * gathering the actual property values (RDKit descriptors, QED, GNN
 * model predictions) is app.js's job, same separation this project uses
 * elsewhere (atom-heatmap.js is the same kind of pure rendering module).
 *
 * Each axis has its own reference range used only for *visual*
 * normalization (mapping a raw value to a 0-1 radius) -- these are rough,
 * roughly drug-like-space ranges, not a scientific claim about where
 * "good" and "bad" sit. A value outside its range is clamped, not
 * distorting the rest of the chart.
 *
 * Three axes (MP, logP, logS) depend on GNN models that may not be
 * loaded/run -- those are drawn as a distinct "no data" marker (a small
 * ring at the axis, not silently plotted at 0) rather than pretending
 * "unknown" and "worst possible value" are the same thing.
 */

window.CC = window.CC || {};

(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';

  // { key, label, unit, min, max, invert } -- invert=true means a LOWER
  // raw value maps to a LARGER radius (used for SA score, where 1=easy
  // and 10=hard, so "more accessible" should read as "bigger on the
  // chart" like every other axis here).
  CC.RADAR_AXES = [
    { key: 'mp', label: 'Melt. pt.', unit: 'K', min: 250, max: 600, source: 'gnn' },
    { key: 'logP', label: 'logP', unit: '', min: -3, max: 7, source: 'gnn-or-rdkit' },
    { key: 'logS', label: 'logS', unit: '', min: -8, max: 2, source: 'gnn' },
    { key: 'mw', label: 'MW', unit: 'g/mol', min: 100, max: 600, source: 'rdkit' },
    { key: 'tpsa', label: 'TPSA', unit: '\u00c5\u00b2', min: 0, max: 180, source: 'rdkit' },
    { key: 'hbd', label: 'HBD', unit: '', min: 0, max: 8, source: 'rdkit' },
    { key: 'hba', label: 'HBA', unit: '', min: 0, max: 12, source: 'rdkit' },
    { key: 'fsp3', label: 'Fsp3', unit: '', min: 0, max: 1, source: 'rdkit' },
    { key: 'rotb', label: 'Rot. bonds', unit: '', min: 0, max: 15, source: 'rdkit' },
    { key: 'qed', label: 'QED', unit: '', min: 0, max: 1, source: 'rdkit' },
    { key: 'rings', label: 'Rings', unit: '', min: 0, max: 6, source: 'rdkit' },
    { key: 'aromaticFraction', label: 'Aromatic frac.', unit: '', min: 0, max: 1, source: 'rdkit' },
    { key: 'heteroFraction', label: 'Heteroatom frac.', unit: '', min: 0, max: 0.5, source: 'rdkit' },
    { key: 'complexity', label: 'Complexity*', unit: '', min: 0, max: 1.5, source: 'rdkit' },
    { key: 'saScore', label: 'Synth. access.', unit: '', min: 1, max: 10, invert: true, source: 'rdkit' },
  ];

  /**
   * Gathers and normalizes radar data from already-computed descriptors
   * (CC.DESCRIPTOR_FIELDS-shaped, i.e. what analyzeMolblock/chemistry.js
   * produces) plus optional GNN molecular properties (propName -> value,
   * matched by registry propertyKey -- see app.js's refreshRadarChart for
   * how mp/logP/logS get located among whatever's currently loaded).
   *
   * Returns an array parallel to CC.RADAR_AXES: { axis, raw, t, available }
   * -- t is the normalized [0,1] radius (only meaningful if available).
   */
  CC.buildRadarData = function (descriptors, gnnValues) {
    gnnValues = gnnValues || {};
    if (!descriptors) return null;

    const heavyAtoms = descriptors.NumHeavyAtoms || 0;
    const raw = {
      mp: typeof gnnValues.mp === 'number' ? gnnValues.mp : null,
      logP: typeof gnnValues.logP === 'number' ? gnnValues.logP
        : (typeof descriptors.CrippenClogP === 'number' ? descriptors.CrippenClogP : null),
      logS: typeof gnnValues.logS === 'number' ? gnnValues.logS : null,
      mw: descriptors.amw,
      tpsa: descriptors.tpsa,
      hbd: descriptors.NumHBD,
      hba: descriptors.NumHBA,
      fsp3: descriptors.FractionCSP3,
      rotb: descriptors.NumRotatableBonds,
      qed: descriptors.qed,
      rings: descriptors.NumRings,
      aromaticFraction: heavyAtoms > 0 && typeof descriptors.aromaticAtomCount === 'number'
        ? descriptors.aromaticAtomCount / heavyAtoms : null,
      heteroFraction: heavyAtoms > 0 && typeof descriptors.NumHeteroatoms === 'number'
        ? descriptors.NumHeteroatoms / heavyAtoms : null,
      // Structural-complexity proxy, NOT RDKit's BertzCT -- this minimal
      // WASM build doesn't expose BertzCT (checked directly, not
      // assumed). Rings + stereocenters + heteroatoms per heavy atom is
      // a simple, transparent stand-in, marked with an asterisk in the
      // label and explained wherever the chart is captioned -- not
      // presented as if it were the real, named metric.
      complexity: heavyAtoms > 0
        ? ((descriptors.NumRings || 0) + (descriptors.NumAtomStereoCenters || 0) + (descriptors.NumHeteroatoms || 0)) / heavyAtoms
        : null,
      saScore: descriptors.saScore,
    };

    return CC.RADAR_AXES.map(function (axis) {
      const v = raw[axis.key];
      const available = typeof v === 'number' && !isNaN(v);
      let t = 0;
      if (available) {
        t = (v - axis.min) / (axis.max - axis.min);
        t = Math.max(0, Math.min(1, t));
        if (axis.invert) t = 1 - t;
      }
      return { axis: axis, raw: available ? v : null, t: t, available: available };
    });
  };

  /**
   * Renders the radar chart into `container` (a plain DOM element, not
   * an SVG -- this creates its own <svg>). Replaces any previously
   * rendered chart in the same container.
   */
  CC.renderRadarChart = function (container, radarData) {
    container.innerHTML = '';
    if (!radarData) {
      const note = document.createElement('p');
      note.className = 'side-panel-note';
      note.textContent = 'Draw a valid structure to see the property radar.';
      container.appendChild(note);
      return;
    }

    const size = 320;
    const center = size / 2;
    const maxRadius = size / 2 - 56; // leave room for axis labels
    const n = radarData.length;

    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + size + ' ' + size);
    svg.setAttribute('class', 'radar-chart-svg');

    function pointAt(index, t) {
      const angle = (Math.PI * 2 * index) / n - Math.PI / 2;
      return {
        x: center + Math.cos(angle) * maxRadius * t,
        y: center + Math.sin(angle) * maxRadius * t,
      };
    }

    // Gridlines (concentric polygons at 25/50/75/100%).
    [0.25, 0.5, 0.75, 1.0].forEach(function (ringT) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const p = pointAt(i, ringT);
        pts.push(p.x + ',' + p.y);
      }
      const poly = document.createElementNS(SVG_NS, 'polygon');
      poly.setAttribute('points', pts.join(' '));
      poly.setAttribute('class', 'radar-grid-ring');
      svg.appendChild(poly);
    });

    // Axis spokes + labels.
    radarData.forEach(function (d, i) {
      const outer = pointAt(i, 1);
      const spoke = document.createElementNS(SVG_NS, 'line');
      spoke.setAttribute('x1', center); spoke.setAttribute('y1', center);
      spoke.setAttribute('x2', outer.x); spoke.setAttribute('y2', outer.y);
      spoke.setAttribute('class', 'radar-spoke');
      svg.appendChild(spoke);

      const labelPos = pointAt(i, 1.14);
      const label = document.createElementNS(SVG_NS, 'text');
      label.setAttribute('x', labelPos.x);
      label.setAttribute('y', labelPos.y);
      label.setAttribute('class', 'radar-axis-label');
      label.setAttribute('text-anchor', Math.abs(labelPos.x - center) < 4 ? 'middle' : (labelPos.x > center ? 'start' : 'end'));
      label.textContent = d.axis.label;
      const title = document.createElementNS(SVG_NS, 'title');
      title.textContent = d.available
        ? d.axis.label + ': ' + d.raw.toFixed(2) + (d.axis.unit ? ' ' + d.axis.unit : '')
        : d.axis.label + ': not available';
      label.appendChild(title);
      svg.appendChild(label);
    });

    // Data polygon -- only drawn through axes that actually have data;
    // an unavailable axis gets a small "no data" ring marker instead of
    // being silently plotted at the center (0 looks like "worst value,"
    // not "unknown," and those aren't the same thing).
    const dataPts = radarData.map(function (d, i) { return pointAt(i, d.available ? d.t : 0); });
    const polyPoints = dataPts.map(function (p) { return p.x + ',' + p.y; }).join(' ');
    const dataPoly = document.createElementNS(SVG_NS, 'polygon');
    dataPoly.setAttribute('points', polyPoints);
    dataPoly.setAttribute('class', 'radar-data-polygon');
    svg.appendChild(dataPoly);

    radarData.forEach(function (d, i) {
      const p = dataPts[i];
      const dot = document.createElementNS(SVG_NS, 'circle');
      dot.setAttribute('cx', p.x);
      dot.setAttribute('cy', p.y);
      dot.setAttribute('r', d.available ? 3 : 4);
      dot.setAttribute('class', d.available ? 'radar-data-point' : 'radar-data-point-missing');
      const title = document.createElementNS(SVG_NS, 'title');
      title.textContent = d.available
        ? d.axis.label + ': ' + d.raw.toFixed(2) + (d.axis.unit ? ' ' + d.axis.unit : '')
        : d.axis.label + ': not available (load the model and run a prediction)';
      dot.appendChild(title);
      svg.appendChild(dot);
    });

    container.appendChild(svg);

    const hasComplexity = radarData.some(function (d) { return d.axis.key === 'complexity'; });
    const hasMissing = radarData.some(function (d) { return !d.available; });
    if (hasComplexity || hasMissing) {
      const caption = document.createElement('p');
      caption.className = 'side-panel-note radar-caption';
      const parts = [];
      if (hasComplexity) parts.push('*Complexity is a simple proxy (rings + stereocenters + heteroatoms per heavy atom), not RDKit\u2019s BertzCT.');
      if (hasMissing) parts.push('Hollow points: load the corresponding model (melting point / logP / logS) and run a prediction to fill in.');
      caption.textContent = parts.join(' ');
      container.appendChild(caption);
    }
  };
})();
