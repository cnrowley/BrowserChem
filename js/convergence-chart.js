/**
 * convergence-chart.js
 *
 * Pure data-in, SVG-out line chart of optimization progress (iteration ->
 * gradNorm AND iteration -> energy, on one shared x-axis) for the 3D
 * panel's Optimize button -- same separation of concerns as
 * titration-chart.js/radar-chart.js (gathering the data is the caller's
 * job; this file only draws it).
 *
 * Two Y axes, not one: gradient norm and energy have unrelated scales and
 * units (a gradient norm near a real force-field's convergence threshold
 * is tiny; energy can be a large positive or negative number depending on
 * the model), so they get their own axis each -- left (log, gradient
 * norm: it typically spans several orders of magnitude over one run, a
 * linear axis would flatten most of the trajectory into an
 * indistinguishable line near zero) and right (linear, energy). Each
 * axis's title is colored to match its own curve so the mapping is
 * self-explanatory without a separate legend box.
 */

window.CC = window.CC || {};

(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const WIDTH = 420;
  const HEIGHT = 150;
  const MARGIN = { top: 10, right: 46, bottom: 24, left: 46 };
  const GRAD_FLOOR = 1e-8; // avoids log10(0) = -Infinity for a genuinely-zero gradient sample
  const ENERGY_COLOR = '#5b8fd6';

  function el(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const key in attrs) node.setAttribute(key, attrs[key]);
    }
    return node;
  }

  /**
   * container: a plain DOM element (not the SVG itself).
   * history: [{iteration, gradNorm, energy}, ...], ascending by iteration
   * -- an optimizer's own onProgress payload (CC.ANI.optimizeGeometry,
   * embed3d.js's minimizeStaged, openff-forcefield.js's
   * minimizeStagedSMIRNOFF -- all three now report energy alongside
   * gradNorm) accumulated by the caller across possibly several resumed
   * ("optimize further") calls, which is why iteration numbers aren't
   * assumed to start at 0. An entry missing `energy` just skips the
   * energy curve for that one point (gradNorm still draws) rather than
   * breaking the whole chart.
   * opts.convergedThreshold (optional): the gradNorm value the optimizer
   * itself treats as converged (1e-5 today) -- drawn as a reference
   * line so it's visually obvious how much further there is to go.
   * opts.converged (optional): colors the current-point gradNorm marker
   * to match whether the run actually settled (accent) or was cut off
   * mid-flight (danger) -- same red/default convention the rest of the
   * 3D panel's status text already uses.
   * opts.energyUnit (optional): string for the right-axis title (e.g.
   * 'kcal/mol', 'arb. units', 'Hartree') -- the caller already knows
   * this per active energy model, same convention app.js's
   * "Δ (arb. units)" conformer-list header switch already uses.
   * Replaces any previously-rendered chart in `container`.
   */
  CC.renderConvergenceChart = function (container, history, opts) {
    opts = opts || {};
    const convergedThreshold = opts.convergedThreshold;
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

    const energyHistory = history.filter(function (h) { return typeof h.energy === 'number'; });
    let energyMin = Infinity, energyMax = -Infinity;
    energyHistory.forEach(function (h) {
      if (h.energy < energyMin) energyMin = h.energy;
      if (h.energy > energyMax) energyMax = h.energy;
    });
    const hasEnergy = energyHistory.length > 0;
    if (hasEnergy) {
      const pad = (energyMax - energyMin) * 0.1 || Math.max(Math.abs(energyMax), 1) * 0.1;
      energyMin -= pad; energyMax += pad;
    }

    function xOf(iter) { return MARGIN.left + ((iter - iterMin) / ((iterMax - iterMin) || 1)) * plotW; }
    function yOf(grad) {
      const g = Math.log10(Math.max(grad, GRAD_FLOOR));
      return MARGIN.top + (1 - (g - logMin) / ((logMax - logMin) || 1)) * plotH;
    }
    function yOfEnergy(e) {
      return MARGIN.top + (1 - (e - energyMin) / ((energyMax - energyMin) || 1)) * plotH;
    }

    const svg = el('svg', { viewBox: '0 0 ' + WIDTH + ' ' + HEIGHT, class: 'convergence-chart-svg' });

    // Y gridlines + labels at each whole log10 decade in range (left axis
    // only -- a second set of gridlines for the right axis would clutter
    // the plot without adding real information, a standard dual-axis
    // convention).
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
      x: -(MARGIN.top + plotH / 2), y: 12, class: 'convergence-axis-title convergence-gradnorm-title', 'text-anchor': 'middle',
      transform: 'rotate(-90)',
    });
    yTitle.textContent = 'gradient norm';
    svg.appendChild(yTitle);

    // Right axis: energy, linear, only drawn when the history actually
    // has energy samples (older callers / a mid-refactor gap shouldn't
    // crash the chart, just fall back to gradient-norm-only).
    if (hasEnergy) {
      const rightX = MARGIN.left + plotW;
      const ticks = 4;
      for (let t = 0; t <= ticks; t++) {
        const e = energyMin + (t / ticks) * (energyMax - energyMin);
        const y = yOfEnergy(e);
        const label = el('text', { x: rightX + 6, y: y + 3, class: 'convergence-axis-label convergence-energy-label', 'text-anchor': 'start' });
        label.textContent = e.toFixed(Math.abs(e) >= 100 ? 0 : 1);
        svg.appendChild(label);
      }
      const yTitleRight = el('text', {
        x: MARGIN.top + plotH / 2, y: WIDTH - 10, class: 'convergence-axis-title convergence-energy-title', 'text-anchor': 'middle',
        transform: 'rotate(90)',
      });
      yTitleRight.textContent = 'energy' + (opts.energyUnit ? ' (' + opts.energyUnit + ')' : '');
      svg.appendChild(yTitleRight);
    }

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

    // Gradient-norm trajectory (left axis).
    const points = history.map(function (h) { return xOf(h.iteration) + ',' + yOf(h.gradNorm); }).join(' ');
    svg.appendChild(el('polyline', { points: points, class: 'convergence-curve-line' }));

    // Current (most recent) point, so it's obvious where the trajectory
    // currently stands, not just its overall shape. Color-coded by
    // whether the run actually converged -- see opts.converged above.
    const last = history[history.length - 1];
    const markerClass = 'convergence-current-marker' + (opts.converged === false ? ' is-not-converged' : '');
    svg.appendChild(el('circle', { cx: xOf(last.iteration), cy: yOf(last.gradNorm), r: 3, class: markerClass }));

    // Energy trajectory (right axis), drawn on top so both curves stay
    // visible where they'd otherwise overlap.
    if (hasEnergy) {
      const energyPoints = energyHistory.map(function (h) { return xOf(h.iteration) + ',' + yOfEnergy(h.energy); }).join(' ');
      svg.appendChild(el('polyline', { points: energyPoints, class: 'convergence-energy-line' }));
      const lastE = energyHistory[energyHistory.length - 1];
      svg.appendChild(el('circle', { cx: xOf(lastE.iteration), cy: yOfEnergy(lastE.energy), r: 3, class: 'convergence-energy-marker' }));
    }

    container.appendChild(svg);
  };
})();
