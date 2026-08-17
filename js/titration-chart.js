/**
 * titration-chart.js
 *
 * Pure data-in, SVG-out rendering of a titration curve (average net
 * charge vs. pH) from js/pka-titration.js's CC.PKATitration.computeCurve
 * output -- same separation of concerns this project already uses for
 * radar-chart.js/atom-heatmap.js (gathering the data is the caller's
 * job; this file only draws it).
 */

window.CC = window.CC || {};

(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const WIDTH = 420;
  const HEIGHT = 260;
  const MARGIN = { top: 16, right: 16, bottom: 32, left: 40 };

  function el(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const key in attrs) node.setAttribute(key, attrs[key]);
    }
    return node;
  }

  /**
   * container: a plain DOM element (not the SVG itself).
   * curve: { pH: [...], avgCharge: [...] } from CC.PKATitration.computeCurve.
   * Replaces any previously-rendered chart in `container`.
   */
  CC.renderTitrationChart = function (container, curve) {
    container.innerHTML = '';
    if (!curve || !curve.pH || curve.pH.length === 0) return;

    const plotW = WIDTH - MARGIN.left - MARGIN.right;
    const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;

    const pHMin = curve.pH[0], pHMax = curve.pH[curve.pH.length - 1];
    let qMin = Math.min.apply(null, curve.avgCharge);
    let qMax = Math.max.apply(null, curve.avgCharge);
    // Symmetric-ish padding around whatever range the curve actually
    // spans, always including 0 so the zero-charge reference line is
    // never clipped off the plot.
    qMin = Math.min(qMin, 0) - 0.3;
    qMax = Math.max(qMax, 0) + 0.3;

    function xOf(pH) { return MARGIN.left + ((pH - pHMin) / (pHMax - pHMin || 1)) * plotW; }
    function yOf(q) { return MARGIN.top + (1 - (q - qMin) / (qMax - qMin || 1)) * plotH; }

    const svg = el('svg', { viewBox: '0 0 ' + WIDTH + ' ' + HEIGHT, class: 'titration-chart-svg' });

    // Gridlines + axis ticks every 2 pH units.
    for (let pH = Math.ceil(pHMin / 2) * 2; pH <= pHMax; pH += 2) {
      const x = xOf(pH);
      svg.appendChild(el('line', { x1: x, y1: MARGIN.top, x2: x, y2: MARGIN.top + plotH, class: 'titration-gridline' }));
      const label = el('text', { x: x, y: MARGIN.top + plotH + 14, class: 'titration-axis-label', 'text-anchor': 'middle' });
      label.textContent = String(pH);
      svg.appendChild(label);
    }
    const xTitle = el('text', { x: MARGIN.left + plotW / 2, y: HEIGHT - 4, class: 'titration-axis-title', 'text-anchor': 'middle' });
    xTitle.textContent = 'pH';
    svg.appendChild(xTitle);

    for (let q = Math.ceil(qMin); q <= qMax; q++) {
      const y = yOf(q);
      const label = el('text', { x: MARGIN.left - 6, y: y + 3, class: 'titration-axis-label', 'text-anchor': 'end' });
      label.textContent = String(q);
      svg.appendChild(label);
    }
    const yTitle = el('text', {
      x: -(MARGIN.top + plotH / 2), y: 12, class: 'titration-axis-title', 'text-anchor': 'middle',
      transform: 'rotate(-90)',
    });
    yTitle.textContent = 'Net charge';
    svg.appendChild(yTitle);

    // Zero-charge reference line.
    svg.appendChild(el('line', {
      x1: MARGIN.left, y1: yOf(0), x2: MARGIN.left + plotW, y2: yOf(0), class: 'titration-zero-line',
    }));

    // Physiological pH reference line (7.4).
    if (pHMin <= 7.4 && pHMax >= 7.4) {
      svg.appendChild(el('line', {
        x1: xOf(7.4), y1: MARGIN.top, x2: xOf(7.4), y2: MARGIN.top + plotH, class: 'titration-physio-line',
      }));
      const physioLabel = el('text', { x: xOf(7.4), y: MARGIN.top - 4, class: 'titration-physio-label', 'text-anchor': 'middle' });
      physioLabel.textContent = 'pH 7.4';
      svg.appendChild(physioLabel);
    }

    // Isoelectric point marker, if the curve crosses zero.
    const pI = CC.PKATitration.isoelectricPoint(curve);
    if (pI !== null) {
      svg.appendChild(el('circle', { cx: xOf(pI), cy: yOf(0), r: 3.5, class: 'titration-pi-marker' }));
    }

    // The curve itself.
    const points = curve.pH.map(function (pH, i) { return xOf(pH) + ',' + yOf(curve.avgCharge[i]); }).join(' ');
    svg.appendChild(el('polyline', { points: points, class: 'titration-curve-line' }));

    container.appendChild(svg);
  };
})();
