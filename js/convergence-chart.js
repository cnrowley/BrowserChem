/**
 * convergence-chart.js
 *
 * Pure data-in, SVG-out line chart of gradient-norm convergence history
 * (iteration -> gradNorm) for CC.ANI.optimizeGeometry's live progress --
 * same separation of concerns as titration-chart.js/radar-chart.js
 * (gathering the data is the caller's job; this file only draws it).
 *
 * Log-scale Y axis: gradNorm typically spans several orders of
 * magnitude over one optimization run (e.g. ~1 down toward the 1e-5
 * "gradient-converged" threshold), where a linear axis would flatten
 * the whole trajectory into an indistinguishable line near zero.
 */

window.CC = window.CC || {};

(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const WIDTH = 420;
  const HEIGHT = 150;
  const MARGIN = { top: 10, right: 12, bottom: 24, left: 46 };
  const GRAD_FLOOR = 1e-8; // avoids log10(0) = -Infinity for a genuinely-zero gradient sample

  function el(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const key in attrs) node.setAttribute(key, attrs[key]);
    }
    return node;
  }

  /**
   * container: a plain DOM element (not the SVG itself).
   * history: [{iteration, gradNorm}, ...], ascending by iteration --
   * CC.ANI.optimizeGeometry's onProgress payload accumulated by the
   * caller across possibly several resumed ("optimize further") calls,
   * which is why iteration numbers aren't assumed to start at 0.
   * convergedThreshold (optional): the gradNorm value the optimizer
   * itself treats as converged (1e-5 today) -- drawn as a reference
   * line so it's visually obvious how much further there is to go.
   * Replaces any previously-rendered chart in `container`.
   */
  CC.renderConvergenceChart = function (container, history, convergedThreshold) {
    container.innerHTML = '';
    if (!history || history.length === 0) return;

    const plotW = WIDTH - MARGIN.left - MARGIN.right;
    const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;

    const iterMin = history[0].iteration;
    const iterMax = history[history.length - 1].iteration || 1;

    let logMin = Infinity, logMax = -Infinity;
    history.forEach(function (h) {
      const g = Math.log10(Math.max(h.gradNorm, GRAD_FLOOR));
      if (g < logMin) logMin = g;
      if (g > logMax) logMax = g;
    });
    if (convergedThreshold) {
      logMin = Math.min(logMin, Math.log10(convergedThreshold));
      logMax = Math.max(logMax, Math.log10(convergedThreshold));
    }
    logMin = Math.floor(logMin - 0.15);
    logMax = Math.ceil(logMax + 0.15);

    function xOf(iter) { return MARGIN.left + ((iter - iterMin) / ((iterMax - iterMin) || 1)) * plotW; }
    function yOf(grad) {
      const g = Math.log10(Math.max(grad, GRAD_FLOOR));
      return MARGIN.top + (1 - (g - logMin) / ((logMax - logMin) || 1)) * plotH;
    }

    const svg = el('svg', { viewBox: '0 0 ' + WIDTH + ' ' + HEIGHT, class: 'convergence-chart-svg' });

    // Y gridlines + labels at each whole log10 decade in range.
    for (let p = Math.ceil(logMin); p <= Math.floor(logMax); p++) {
      const y = yOf(Math.pow(10, p));
      svg.appendChild(el('line', { x1: MARGIN.left, y1: y, x2: MARGIN.left + plotW, y2: y, class: 'convergence-gridline' }));
      const label = el('text', { x: MARGIN.left - 6, y: y + 3, class: 'convergence-axis-label', 'text-anchor': 'end' });
      label.textContent = '1e' + p;
      svg.appendChild(label);
    }
    const xTitle = el('text', { x: MARGIN.left + plotW / 2, y: HEIGHT - 4, class: 'convergence-axis-title', 'text-anchor': 'middle' });
    xTitle.textContent = 'iteration';
    svg.appendChild(xTitle);
    const yTitle = el('text', {
      x: -(MARGIN.top + plotH / 2), y: 12, class: 'convergence-axis-title', 'text-anchor': 'middle',
      transform: 'rotate(-90)',
    });
    yTitle.textContent = 'gradient norm';
    svg.appendChild(yTitle);

    // Convergence-target reference line, if given.
    if (convergedThreshold) {
      svg.appendChild(el('line', {
        x1: MARGIN.left, y1: yOf(convergedThreshold), x2: MARGIN.left + plotW, y2: yOf(convergedThreshold),
        class: 'convergence-target-line',
      }));
      const targetLabel = el('text', { x: MARGIN.left + plotW - 2, y: yOf(convergedThreshold) - 3, class: 'convergence-target-label', 'text-anchor': 'end' });
      targetLabel.textContent = 'converged';
      svg.appendChild(targetLabel);
    }

    // The convergence trajectory itself.
    const points = history.map(function (h) { return xOf(h.iteration) + ',' + yOf(h.gradNorm); }).join(' ');
    svg.appendChild(el('polyline', { points: points, class: 'convergence-curve-line' }));

    // Current (most recent) point, so it's obvious where the trajectory
    // currently stands, not just its overall shape.
    const last = history[history.length - 1];
    svg.appendChild(el('circle', { cx: xOf(last.iteration), cy: yOf(last.gradNorm), r: 3, class: 'convergence-current-marker' }));

    container.appendChild(svg);
  };
})();
