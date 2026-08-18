/**
 * solvation.js
 *
 * Pure data-in, DOM-out module for the solvation panel -- same separation
 * as radar-chart.js/atom-heatmap.js: this file never touches RDKit or the
 * GNN pipeline directly, it only turns { propertyKey -> predicted dGsolv }
 * (already-computed molecular properties, matched by registry propertyKey
 * -- see app.js's refreshSolvationPanel) into a ranked table and a
 * solvent x solvent transfer-energy matrix.
 *
 * Each of the 21 solvents below is its own FIXED-solvent single-molecule
 * Chemprop checkpoint (solv{name}-v1 in model/registry.json) -- solute
 * SMILES in, dGsolv (kcal/mol) in that one solvent out -- not a true
 * two-molecule solute+solvent model (this project's JS D-MPNN only
 * supports one molecular graph per checkpoint; see
 * scripts/prepare_solvation_training_data.py's header for why). A cell
 * is only ever computed from two ALREADY-LOADED-AND-RUN models' results;
 * an unloaded solvent renders as a distinct "no data" pattern, never a
 * fake 0 -- same principle radar-chart.js uses for its own missing axes.
 */

window.CC = window.CC || {};

(function () {
  // Ordered low -> high polarity (approximate dielectric constant, CRC
  // Handbook-range values -- for a visually coherent matrix diagonal
  // gradient only, not a modeled property in their own right) so
  // physically-similar solvents sit near each other in both the table's
  // natural reading order and the matrix's row/column order.
  CC.SOLVENTS = [
    { id: 'hexane', name: 'Hexane', propertyKey: 'solvHexane', smiles: 'CCCCCC', epsApprox: 1.9 },
    { id: 'heptane', name: 'Heptane', propertyKey: 'solvHeptane', smiles: 'CCCCCCC', epsApprox: 1.9 },
    { id: 'cyclohexane', name: 'Cyclohexane', propertyKey: 'solvCyclohexane', smiles: 'C1CCCCC1', epsApprox: 2.0 },
    { id: 'dioxane', name: '1,4-Dioxane', propertyKey: 'solvDioxane', smiles: 'C1COCCO1', epsApprox: 2.2 },
    { id: 'toluene', name: 'Toluene', propertyKey: 'solvToluene', smiles: 'Cc1ccccc1', epsApprox: 2.4 },
    { id: 'diethyl-ether', name: 'Diethyl ether', propertyKey: 'solvDiethylEther', smiles: 'CCOCC', epsApprox: 4.3 },
    { id: 'chloroform', name: 'Chloroform', propertyKey: 'solvChloroform', smiles: 'ClC(Cl)Cl', epsApprox: 4.8 },
    { id: 'ethyl-acetate', name: 'Ethyl acetate', propertyKey: 'solvEthylAcetate', smiles: 'CCOC(C)=O', epsApprox: 6.0 },
    { id: 'acetic-acid', name: 'Acetic acid', propertyKey: 'solvAceticAcid', smiles: 'CC(=O)O', epsApprox: 6.2 },
    { id: 'thf', name: 'THF', propertyKey: 'solvThf', smiles: 'C1CCOC1', epsApprox: 7.6 },
    { id: 'dcm', name: 'DCM', propertyKey: 'solvDcm', smiles: 'ClCCl', epsApprox: 8.9 },
    { id: '1-octanol', name: '1-Octanol', propertyKey: 'solv1Octanol', smiles: 'CCCCCCCCO', epsApprox: 10.3 },
    { id: 'pyridine', name: 'Pyridine', propertyKey: 'solvPyridine', smiles: 'c1ccncc1', epsApprox: 12.3 },
    { id: '2-propanol', name: '2-Propanol', propertyKey: 'solv2Propanol', smiles: 'CC(C)O', epsApprox: 18.3 },
    { id: 'acetone', name: 'Acetone', propertyKey: 'solvAcetone', smiles: 'CC(C)=O', epsApprox: 20.7 },
    { id: 'ethanol', name: 'Ethanol', propertyKey: 'solvEthanol', smiles: 'CCO', epsApprox: 24.3 },
    { id: 'methanol', name: 'Methanol', propertyKey: 'solvMethanol', smiles: 'CO', epsApprox: 32.7 },
    { id: 'dmf', name: 'DMF', propertyKey: 'solvDmf', smiles: 'CN(C)C=O', epsApprox: 36.7 },
    { id: 'acetonitrile', name: 'Acetonitrile', propertyKey: 'solvAcetonitrile', smiles: 'CC#N', epsApprox: 37.5 },
    { id: 'dmso', name: 'DMSO', propertyKey: 'solvDmso', smiles: 'CS(C)=O', epsApprox: 46.7 },
    { id: 'water', name: 'Water', propertyKey: 'solvWater', smiles: 'O', epsApprox: 80.1 },
  ];

  /**
   * [{ solvent, value, available }], sorted most-favorable (most
   * negative dGsolv) first, unavailable (model not loaded/run) solvents
   * pushed to the end rather than interleaved at a fake 0.
   */
  CC.buildSolvationRows = function (values) {
    values = values || {};
    const rows = CC.SOLVENTS.map(function (solvent) {
      const v = values[solvent.propertyKey];
      const available = typeof v === 'number' && isFinite(v);
      return { solvent: solvent, value: available ? v : null, available: available };
    });
    rows.sort(function (a, b) {
      if (a.available && b.available) return a.value - b.value;
      return (a.available ? 0 : 1) - (b.available ? 0 : 1);
    });
    return rows;
  };

  /**
   * { solvents: CC.SOLVENTS (fixed row/column order), cells: cells[i][j]
   * = transfer free energy moving the solute FROM solvents[i] TO
   * solvents[j] = dGsolv(to) - dGsolv(from) -- negative means the
   * transfer is spontaneous (favorable) in that direction. Diagonal is
   * always 0/available (transfer to the same solvent). A cell is
   * available only when BOTH its solvents' models are loaded and run.
   */
  CC.buildTransferMatrix = function (values) {
    values = values || {};
    const n = CC.SOLVENTS.length;
    const cells = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) {
        if (i === j) {
          row.push({ from: CC.SOLVENTS[i], to: CC.SOLVENTS[j], value: 0, available: true });
          continue;
        }
        const vFrom = values[CC.SOLVENTS[i].propertyKey];
        const vTo = values[CC.SOLVENTS[j].propertyKey];
        const available = typeof vFrom === 'number' && isFinite(vFrom) && typeof vTo === 'number' && isFinite(vTo);
        row.push({ from: CC.SOLVENTS[i], to: CC.SOLVENTS[j], value: available ? vTo - vFrom : null, available: available });
      }
      cells.push(row);
    }
    return { solvents: CC.SOLVENTS, cells: cells };
  };

  function fmt(v) {
    return (v > 0 ? '+' : '') + v.toFixed(2);
  }

  CC.renderSolvationTable = function (container, rows) {
    container.innerHTML = '';
    if (!rows || rows.every(function (r) { return !r.available; })) {
      const p = document.createElement('p');
      p.className = 'side-panel-note';
      p.textContent = 'Load one or more solvation models above, then run prediction, to see a ranked table here.';
      container.appendChild(p);
      return;
    }

    const table = document.createElement('table');
    table.className = 'descriptor-table solvation-table';
    const thead = document.createElement('thead');
    thead.innerHTML = '<tr><th>Solvent</th><th style="text-align:right">ΔG<sub>solv</sub> (kcal/mol)</th></tr>';
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    rows.forEach(function (r) {
      const tr = document.createElement('tr');
      const nameTd = document.createElement('td');
      nameTd.textContent = r.solvent.name;
      const valTd = document.createElement('td');
      if (r.available) {
        valTd.textContent = r.value.toFixed(2);
      } else {
        valTd.textContent = '—';
        valTd.style.color = 'var(--text-dark-muted)';
      }
      tr.appendChild(nameTd);
      tr.appendChild(valTd);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    container.appendChild(table);
  };

  /**
   * Diverging color for one cell: mixes toward --accent (teal, favorable/
   * spontaneous, negative transfer energy) or --danger (red-orange,
   * unfavorable/non-spontaneous, positive) from a neutral gray midpoint,
   * scaled by |value| against `scaleMax` (clamped to 1) -- CSS
   * color-mix(in oklch, ...) does the actual perceptual interpolation
   * rather than a hand-rolled RGB lerp, matching the diverging-scale
   * structure (two hues + neutral midpoint, equal treatment per arm)
   * without needing a full precomputed ramp table for a non-brand hue.
   */
  function cellColor(value, scaleMax) {
    const t = Math.max(0, Math.min(1, Math.abs(value) / scaleMax));
    const pct = Math.round(t * 100);
    const pole = value < 0 ? 'var(--accent)' : 'var(--danger)';
    return 'color-mix(in oklch, ' + pole + ' ' + pct + '%, var(--border-light))';
  }

  /**
   * Renders the full solvent x solvent matrix into `container` (expected
   * to be a modal body or similarly wide element -- 21 columns needs
   * real horizontal room, not the narrow side panel). Includes a
   * diverging legend and a live text readout that updates on cell
   * hover/focus (in addition to each cell's native title tooltip) --
   * exact values always available as real text, never color-only.
   */
  CC.renderTransferMatrix = function (container, matrixData) {
    container.innerHTML = '';

    const availableValues = [];
    matrixData.cells.forEach(function (row) {
      row.forEach(function (c) {
        if (c.available && c.from.id !== c.to.id) availableValues.push(Math.abs(c.value));
      });
    });

    if (availableValues.length === 0) {
      const p = document.createElement('p');
      p.className = 'side-panel-note';
      p.textContent = 'Load at least two solvation models above and run prediction to see the transfer matrix.';
      container.appendChild(p);
      return;
    }

    // Auto-scaled to this molecule's own data (floor 1 kcal/mol so a
    // near-uniform matrix doesn't get amplified into a false-looking
    // rainbow from sub-kcal noise), rounded up to the nearest 0.5 for a
    // legend that reads as a clean number rather than an arbitrary max.
    const rawMax = Math.max(1, Math.max.apply(null, availableValues));
    const scaleMax = Math.ceil(rawMax * 2) / 2;

    const readout = document.createElement('div');
    readout.className = 'transfer-matrix-readout';
    readout.textContent = 'Hover a cell for the transfer free energy between two solvents.';
    container.appendChild(readout);

    function showReadout(cell) {
      if (cell.from.id === cell.to.id) {
        readout.textContent = cell.from.name + ' → ' + cell.to.name + ': same solvent.';
        return;
      }
      if (!cell.available) {
        readout.textContent = cell.from.name + ' → ' + cell.to.name + ': load both solvation models and run prediction.';
        return;
      }
      const verdict = cell.value < 0 ? 'spontaneous (favorable)' : 'non-spontaneous (unfavorable)';
      readout.textContent = cell.from.name + ' → ' + cell.to.name + ':  ΔΔG = ' + fmt(cell.value) + ' kcal/mol — ' + verdict + '.';
    }

    const scrollWrap = document.createElement('div');
    scrollWrap.className = 'transfer-matrix-scroll';

    const grid = document.createElement('div');
    grid.className = 'transfer-matrix-grid';
    const n = matrixData.solvents.length;
    grid.style.gridTemplateColumns = '110px repeat(' + n + ', 22px)';

    // Top-left blank corner.
    grid.appendChild(document.createElement('div'));

    // Column headers, rotated to fit 21 narrow columns.
    matrixData.solvents.forEach(function (s) {
      const h = document.createElement('div');
      h.className = 'transfer-matrix-colhead';
      const span = document.createElement('span');
      span.textContent = s.name;
      h.appendChild(span);
      grid.appendChild(h);
    });

    matrixData.cells.forEach(function (row, i) {
      const rowHead = document.createElement('div');
      rowHead.className = 'transfer-matrix-rowhead';
      rowHead.textContent = matrixData.solvents[i].name;
      grid.appendChild(rowHead);

      row.forEach(function (cell) {
        const div = document.createElement('div');
        div.className = 'transfer-matrix-cell';
        if (cell.from.id === cell.to.id) {
          div.classList.add('is-diagonal');
          div.title = cell.from.name + ' → ' + cell.to.name + ': same solvent';
        } else if (!cell.available) {
          div.classList.add('is-unavailable');
          div.title = cell.from.name + ' → ' + cell.to.name + ': not loaded';
        } else {
          div.style.background = cellColor(cell.value, scaleMax);
          div.title = cell.from.name + ' → ' + cell.to.name + ': ' + fmt(cell.value) + ' kcal/mol';
        }
        div.addEventListener('mouseenter', function () { showReadout(cell); });
        div.addEventListener('focus', function () { showReadout(cell); });
        div.tabIndex = 0;
        grid.appendChild(div);
      });
    });

    scrollWrap.appendChild(grid);
    container.appendChild(scrollWrap);

    // Diverging legend: same color-mix stops the cells use, so it's a
    // real read of what's on screen rather than a decorative gradient.
    const legend = document.createElement('div');
    legend.className = 'transfer-matrix-legend';
    legend.innerHTML =
      '<span class="transfer-matrix-legend-label">−' + scaleMax.toFixed(1) + ' (spontaneous)</span>' +
      '<span class="transfer-matrix-legend-bar"></span>' +
      '<span class="transfer-matrix-legend-label">+' + scaleMax.toFixed(1) + ' (non-spontaneous)</span>';
    const bar = legend.querySelector('.transfer-matrix-legend-bar');
    bar.style.background =
      'linear-gradient(to right, ' + cellColor(-scaleMax, scaleMax) + ', var(--border-light), ' + cellColor(scaleMax, scaleMax) + ')';
    container.appendChild(legend);

    const note = document.createElement('p');
    note.className = 'side-panel-note transfer-matrix-note';
    note.textContent = 'Cell = ΔG_transfer moving the solute from the ROW solvent to the COLUMN solvent (ΔGsolv[column] − ΔGsolv[row]). Each value comes from two independently-predicted, FIXED-solvent models, not a real two-molecule solute+solvent calculation -- see the model info popup for what each checkpoint was trained on.';
    container.appendChild(note);
  };
})();
