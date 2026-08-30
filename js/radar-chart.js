/**
 * radar-chart.js
 *
 * A 15-axis property radar/spider chart. Pure data-in, SVG-out module --
 * gathering the actual property values (RDKit descriptors, QED, GNN
 * model predictions) is app.js's job, same separation this project uses
 * elsewhere (atom-heatmap.js is the same kind of pure rendering module).
 *
 * Each axis has its own reference range used only for *visual*
 * normalization (mapping a raw value to a 0-1 radius). A value outside
 * its range is clamped, not distorting the rest of the chart.
 *
 * 13 of the 15 ranges below are REAL, derived from the same ChEMBL
 * max_phase==4 (approved) reference population druglikeness.js's own
 * percentile-rank comparison uses (n=3311, largest-fragment-desalted --
 * see that file's header and scripts/compute_druglikeness_distributions.py
 * for the full provenance/caveats, which apply here identically) --
 * specifically the [5th, 95th] percentile band of each descriptor's real
 * distribution across that set, computed by
 * scripts/compute_radar_reference_ranges.py using real RDKit (matched
 * field-for-field against what js/chemistry.js's own get_descriptors()
 * call and js/sascorer.js actually compute -- see that script's own
 * comments for the exact correspondence, e.g. NumHBD/NumHBA are RDKit's
 * plain definition, NOT the separate Lipinski-specific variant
 * druglikeness.js deliberately uses for its own Rule-of-Five filter).
 * Trimmed to the 5th-95th band (not literal min/max) so the small
 * fraction of real approved-drug outliers at either extreme (e.g. a
 * handful under 150 Da or over 1000 Da) don't compress the visual scale
 * for the typical case -- this means a real molecule CAN legitimately
 * clamp to an axis's edge without being an error.
 *
 * The remaining two (MP, logS) are NOT derived from that population --
 * ChEMBL's small-molecule pull has no reliable per-molecule melting-
 * point/solubility data, and this project's own melting-point-v1/
 * logs-aqsoldb-v1 GNN checkpoints aren't practical to batch-run outside
 * the browser (they're this project's own hand-rolled JS D-MPNN format,
 * not a retained standard ONNX/PyTorch artifact). Those two keep their
 * original hand-picked, honestly-approximate ranges. Both MP and logP
 * also depend on GNN models that may not be loaded/run -- those (plus
 * logS, GNN-only) are drawn as a distinct "no data" marker (a small ring
 * at the axis, not silently plotted at 0) rather than pretending
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
    // Not derived (see file header) -- original hand-picked heuristic ranges.
    { key: 'mp', label: 'Melt. pt.', unit: 'K', min: 250, max: 600, source: 'gnn' },
    // Derived (5th-95th pctile, n=3311 approved drugs): min=-1.47 max=6.301.
    { key: 'logP', label: 'logP', unit: '', min: -1.47, max: 6.3, source: 'gnn-or-rdkit' },
    // Not derived (see file header) -- original hand-picked heuristic range.
    { key: 'logS', label: 'logS', unit: '', min: -8, max: 2, source: 'gnn' },
    { key: 'mw', label: 'MW', unit: 'g/mol', min: 135.12, max: 670.87, source: 'rdkit' },
    { key: 'tpsa', label: 'TPSA', unit: '\u00c5\u00b2', min: 12.03, max: 193.91, source: 'rdkit' },
    { key: 'hbd', label: 'HBD', unit: '', min: 0, max: 5, source: 'rdkit' },
    { key: 'hba', label: 'HBA', unit: '', min: 1, max: 11, source: 'rdkit' },
    { key: 'fsp3', label: 'Fsp3', unit: '', min: 0, max: 0.93, source: 'rdkit' },
    { key: 'rotb', label: 'Rot. bonds', unit: '', min: 0, max: 12, source: 'rdkit' },
    { key: 'qed', label: 'QED', unit: '', min: 0.12, max: 0.85, source: 'rdkit' },
    { key: 'rings', label: 'Rings', unit: '', min: 0, max: 6, source: 'rdkit' },
    { key: 'aromaticFraction', label: 'Aromatic frac.', unit: '', min: 0, max: 0.68, source: 'rdkit' },
    { key: 'heteroFraction', label: 'Heteroatom frac.', unit: '', min: 0.1, max: 0.56, source: 'rdkit' },
    { key: 'complexity', label: 'Complexity*', unit: '', min: 0.21, max: 0.65, source: 'rdkit' },
    { key: 'saScore', label: 'Synth. access.', unit: '', min: 2.03, max: 5.58, invert: true, source: 'rdkit' },
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

    // Data polygon -- connects ONLY axes that actually have data. A real
    // bug this replaces: using t=0 (dead center) as a stand-in for
    // "missing" made the shape's own OUTLINE dip to the center at that
    // axis, which reads as "this property's value is at its worst/zero"
    // to anyone looking at the shape -- indistinguishable from a real
    // measurement, even though the individual point at that axis was
    // separately styled as a hollow "no data" marker (see below). All or
    // nothing per axis: an available point is a real vertex in the
    // connected shape; a missing one is skipped entirely from the
    // polygon's own path (the gap is bridged by a straight line between
    // its two real neighbors) and only shown as its own separate marker.
    // A missing axis's own marker is placed at a fixed, neutral radius on
    // ITS OWN spoke (not at 0 = every missing axis's angle collapses to
    // the exact same shared center point, so 2+ missing properties become
    // one indistinguishable overlapping ring -- confirmed a real problem,
    // not hypothetical, for any molecule missing more than one axis) --
    // and not at a "real-looking" radius either, since the hollow/dashed
    // style already marks it as not-a-value; this is just where to draw
    // that marker, never fed into the polygon above.
    const NO_DATA_MARKER_T = 0.5;
    const availableIdx = radarData.map(function (d, i) { return i; }).filter(function (i) { return radarData[i].available; });
    const dataPts = radarData.map(function (d, i) { return pointAt(i, d.available ? d.t : NO_DATA_MARKER_T); });
    if (availableIdx.length >= 3) {
      const polyPoints = availableIdx.map(function (i) { return dataPts[i].x + ',' + dataPts[i].y; }).join(' ');
      const dataPoly = document.createElementNS(SVG_NS, 'polygon');
      dataPoly.setAttribute('points', polyPoints);
      dataPoly.setAttribute('class', 'radar-data-polygon');
      svg.appendChild(dataPoly);
    }

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

    const hasMissing = radarData.some(function (d) { return !d.available; });
    if (hasMissing) {
      const caption = document.createElement('p');
      caption.className = 'side-panel-note radar-caption';
      caption.textContent = 'Hollow points: load the corresponding model (melting point / logP / logS) and run a prediction to fill in.';
      container.appendChild(caption);
    }
  };
})();
