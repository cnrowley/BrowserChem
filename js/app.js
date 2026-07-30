/**
 * app.js
 *
 * Wires the DOM (tool rail, element picker, undo/redo buttons, canvas) to
 * the molecule model, history stack, controller, and renderer. This is the
 * only file that touches all the pieces together — molecule.js, tools.js,
 * and render.js each stay independently testable.
 */

(function () {
  let molecule = new CC.Molecule();
  let controller = null;
  let svg = null;
  let canvasStats = null;
  let undoBtn = null;
  let redoBtn = null;
  let openBtn = null;
  let saveBtn = null;
  let fileInput = null;
  let validityDot = null;
  let validityText = null;
  let smilesOutput = null;
  let copySmilesBtn = null;
  let currentSmiles = '';
  let currentDescriptors = null;
  let currentValidityText = 'no structure yet';
  let validationTimer = null;
  let generate3dBtn = null;
  let viewer3dNote = null;
  let viewer3d = null;
  let reapplyHeatmap = function () {};
  let invalidateGNNResults = function () {};
  let invalidate3DView = function () {};

  const history = new CC.History(onHistoryChange);
  history.commit(molecule.toJSON());

  function renderNow(meta) {
    CC.render(svg, controller.molecule, { selectedAtomIds: controller.selection });
    if (meta && meta.preview) {
      CC.renderPreview(svg, controller.molecule, meta.preview);
    }
    updateStats();
    reapplyHeatmap();
  }

  function updateStats() {
    if (!canvasStats) return;
    const m = controller.molecule;
    const atomCount = m.atoms.size;
    const bondCount = m.bonds.size;
    canvasStats.textContent = atomCount + (atomCount === 1 ? ' atom' : ' atoms') +
      ' \u00b7 ' + bondCount + (bondCount === 1 ? ' bond' : ' bonds');
  }

  function onHistoryChange() {
    updateUndoRedoButtons();
    scheduleValidation();
  }

  function updateUndoRedoButtons() {
    if (undoBtn) undoBtn.disabled = !history.canUndo();
    if (redoBtn) redoBtn.disabled = !history.canRedo();
  }

  function loadHistoryState(state) {
    if (!state) return;
    molecule = CC.Molecule.fromJSON(state);
    controller.setMolecule(molecule);
    renderNow();
  }

  // ---------- RDKit validation + SMILES ----------

  function scheduleValidation() {
    clearTimeout(validationTimer);
    validationTimer = setTimeout(runValidation, 120);
  }

  function runValidation() {
    if (!validityDot) return;

    const mol = controller.molecule;
    updateModelCompatibilityWarning();
    invalidateGNNResults();
    invalidate3DView();

    if (mol.isEmpty()) {
      setValidityState('idle', 'no structure yet');
      setSmiles('');
      renderDescriptorTable(null);
      updateExportButtons();
      refreshRadarChart();
      updateHRMSButton();
      updateSmartsFiltersButton();
      return;
    }

    const molblock = CC.moleculeToMolblock(mol);
    const result = CC.analyzeMolblock(molblock);

    if (!result.available) {
      setValidityState('loading', 'RDKit.js still loading\u2026');
      setSmiles('');
      renderDescriptorTable(null);
      updateExportButtons();
      refreshRadarChart();
      updateHRMSButton();
      updateSmartsFiltersButton();
      return;
    }

    if (!result.valid) {
      setValidityState('error', result.message || 'invalid structure');
      setSmiles('');
      renderDescriptorTable(null);
      updateExportButtons();
      refreshRadarChart();
      updateHRMSButton();
      updateSmartsFiltersButton();
      return;
    }

    setValidityState('ready', 'valid structure');
    setSmiles(result.smiles || '');
    renderDescriptorTable(withSAScore(result.descriptors));
    updateExportButtons();
    refreshRadarChart();
    updateHRMSButton();
    updateSmartsFiltersButton();
  }

  /**
   * Pulls MP/logP/logS out of whatever GNN molecular properties are
   * currently available (matched by the registry's own propertyKey for
   * each of those three models -- "mp", "logP", "Solubility" -- not by
   * guessing at a model id, since a user could rename/reload models).
   * Falls back to RDKit's own Crippen logP when no logP model is loaded
   * (CC.buildRadarData does that fallback itself); MP and logS have no
   * RDKit equivalent, so those two stay unavailable until the
   * corresponding model has actually been run.
   */
  function refreshRadarChart() {
    const container = document.getElementById('radar-chart-container');
    if (!container) return;
    const gnnValues = {};
    if (lastMolecularProperties) {
      const values = lastMolecularProperties.values;
      if (typeof values.mp === 'number') gnnValues.mp = values.mp;
      if (typeof values.logP === 'number') gnnValues.logP = values.logP;
      if (typeof values.Solubility === 'number') gnnValues.logS = values.Solubility;
    }
    const radarData = CC.buildRadarData(currentDescriptors, gnnValues);
    CC.renderRadarChart(container, radarData);
  }

  /**
   * Proactive warning, not just a reactive failure after clicking Run:
   * checks the current molecule against the property-predictor model
   * families' known vocabulary limits (NAGL-MBIS's hard, non-padding
   * element/degree vocabulary; Chemprop's softer "outside the training
   * vocabulary" case, since it pads gracefully rather than erroring but
   * a prediction touching that pad bucket is still less trustworthy).
   * Checked against the model *families* in general, not just whichever
   * specific checkpoint happens to be loaded right now -- the
   * vocabulary itself doesn't depend on which weights are loaded.
   */
  function updateModelCompatibilityWarning() {
    const warningEl = document.getElementById('compatibility-warning');
    if (!warningEl) return;

    const mol = controller.molecule;
    if (mol.isEmpty()) { warningEl.style.display = 'none'; return; }

    const issues = [];
    if (window.CC && CC.NAGL && CC.NAGL.checkCompatibility) {
      const naglCheck = CC.NAGL.checkCompatibility(mol);
      naglCheck.issues.forEach(function (msg) { issues.push('MBIS charge model: ' + msg); });
    }
    if (window.CC && CC.GNN && CC.GNN.checkChempropCompatibility) {
      const chempropCheck = CC.GNN.checkChempropCompatibility(mol);
      chempropCheck.issues.forEach(function (msg) { issues.push('Chemprop models: ' + msg); });
    }

    if (issues.length === 0) {
      warningEl.style.display = 'none';
      return;
    }
    warningEl.style.display = '';
    warningEl.textContent = '\u26a0 ' + issues.join('; ') + '.';
  }

  // SA Score isn't part of RDKit's own get_descriptors() output -- it's
  // computed separately (sascorer.js) from the app's own Molecule model,
  // not the RDKit JSMol, so it's merged in here rather than in
  // chemistry.js's analyzeMolblock(). Omitted (not shown as 0 or "—")
  // until the fragment-score table has actually loaded, since a score
  // computed without it would silently use RDKit's "unseen fragment"
  // fallback for every single fragment and be meaningless.
  function withSAScore(descriptors) {
    if (!descriptors || !CC.hasSAScoreTable()) return descriptors;
    const sa = CC.calculateSAScore(controller.molecule);
    if (sa === null) return descriptors;
    return Object.assign({}, descriptors, { saScore: sa });
  }

  function renderDescriptorTable(descriptors) {
    currentDescriptors = descriptors;
    const tbody = document.getElementById('descriptor-table');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!descriptors) {
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 2;
      cell.textContent = '\u2014';
      cell.style.color = 'var(--text-dark-muted)';
      row.appendChild(cell);
      tbody.appendChild(row);
      return;
    }

    CC.DESCRIPTOR_FIELDS.forEach(function (field) {
      const value = descriptors[field.key];
      if (typeof value !== 'number') return;
      const row = document.createElement('tr');
      const labelCell = document.createElement('td');
      labelCell.textContent = field.label;
      const valueCell = document.createElement('td');
      valueCell.textContent = value.toFixed(field.decimals) + (field.unit ? ' ' + field.unit : '');
      row.appendChild(labelCell);
      row.appendChild(valueCell);
      tbody.appendChild(row);
    });
  }

  function setValidityState(state, text) {
    currentValidityText = text;
    validityDot.classList.remove('is-loading', 'is-ready', 'is-error');
    if (state === 'ready') validityDot.classList.add('is-ready');
    else if (state === 'error') validityDot.classList.add('is-error');
    else if (state === 'loading') validityDot.classList.add('is-loading');
    validityText.textContent = text;
  }

  function setSmiles(smiles) {
    currentSmiles = smiles;
    smilesOutput.textContent = smiles || '\u2014';
    copySmilesBtn.disabled = !smiles;
  }

  function updateExportButtons() {
    const hasData = !!currentDescriptors;
    ['copy-properties-btn', 'download-csv-btn', 'download-xlsx-btn', 'download-pdf-btn'].forEach(function (id) {
      const btn = document.getElementById(id);
      if (btn) btn.disabled = !hasData;
    });
  }

  function updateHRMSButton() {
    const btn = document.getElementById('hrms-btn');
    if (btn) btn.disabled = !currentDescriptors || typeof currentDescriptors.exactmw !== 'number';
  }

  function updateSmartsFiltersButton() {
    const btn = document.getElementById('smarts-filters-btn');
    if (btn) btn.disabled = !currentDescriptors || !CC.isSmartsFiltersLoaded();
  }

  function setupSmartsFiltersModal() {
    const btn = document.getElementById('smarts-filters-btn');
    const modal = document.getElementById('smarts-filters-modal');
    const closeBtn = document.getElementById('smarts-filters-modal-close');
    const body = document.getElementById('smarts-filters-modal-body');
    if (!btn || !modal) return;

    function close() {
      modal.style.display = 'none';
      CC.clearSmartsHighlight(svg);
    }

    btn.addEventListener('click', function () {
      let results;
      try {
        results = CC.checkSmartsFilters(controller.molecule);
      } catch (err) {
        body.innerHTML = '';
        const p = document.createElement('p');
        p.className = 'side-panel-note';
        p.textContent = 'Could not run structural alert check: ' + err.message;
        body.appendChild(p);
        modal.style.display = '';
        return;
      }

      body.innerHTML = '';
      const meta = CC.getSmartsFilterMeta();

      const summary = document.createElement('p');
      summary.className = 'smarts-filters-summary';
      summary.textContent = results.length === 0
        ? 'No matches against ' + meta.patterns.length + ' patterns across 8 medchem filter sets (PAINS, Glaxo, Dundee, BMS, SureChEMBL, MLSMR, Inpharmatica, LINT).'
        : results.length + ' match' + (results.length === 1 ? '' : 'es') + ' out of ' + meta.patterns.length + ' patterns checked. A match flags a known substructure pattern, not a verdict on the molecule as a whole \u2014 many real, useful drugs match at least one filter.';
      body.appendChild(summary);

      // Group by rule set, in a stable, consistent display order.
      const RULE_SET_ORDER = ['PAINS', 'Glaxo', 'Dundee', 'BMS', 'SureChEMBL', 'MLSMR', 'Inpharmatica', 'LINT'];
      const byRuleSet = {};
      results.forEach(function (r) {
        (byRuleSet[r.ruleSet] = byRuleSet[r.ruleSet] || []).push(r);
      });

      RULE_SET_ORDER.filter(function (rs) { return byRuleSet[rs]; }).forEach(function (ruleSet) {
        const group = document.createElement('div');
        group.className = 'smarts-ruleset-group';

        const info = meta.ruleSets[ruleSet] || {};
        const heading = document.createElement('div');
        heading.className = 'smarts-ruleset-heading';
        heading.textContent = (info.fullName || ruleSet) + ' (' + byRuleSet[ruleSet].length + ')';
        group.appendChild(heading);

        const desc = document.createElement('p');
        desc.className = 'smarts-ruleset-description';
        let descText = info.description || '';
        if (info.citation) descText += ' ' + info.citation;
        desc.textContent = descText;
        group.appendChild(desc);

        byRuleSet[ruleSet].forEach(function (r) {
          const row = document.createElement('div');
          row.className = 'smarts-match-row';

          const left = document.createElement('div');
          const titleSpan = document.createElement('span');
          titleSpan.className = 'smarts-match-title';
          titleSpan.textContent = r.title + (r.matchCount > 1 ? ' (\u00d7' + r.matchCount + ')' : '');
          left.appendChild(titleSpan);
          const smartsSpan = document.createElement('span');
          smartsSpan.className = 'smarts-match-smarts';
          smartsSpan.textContent = r.smarts;
          left.appendChild(smartsSpan);
          row.appendChild(left);

          if (r.atomIds.length > 0) {
            const highlightBtn = document.createElement('button');
            highlightBtn.type = 'button';
            highlightBtn.className = 'smarts-highlight-btn';
            highlightBtn.textContent = 'Highlight';
            highlightBtn.addEventListener('click', function () {
              CC.highlightSmartsMatch(svg, controller.molecule, r.atomIds);
            });
            row.appendChild(highlightBtn);
          }

          group.appendChild(row);
        });

        body.appendChild(group);
      });

      const sourceNote = document.createElement('p');
      sourceNote.className = 'side-panel-note';
      sourceNote.style.marginTop = '10px';
      const sourceLink = document.createElement('a');
      sourceLink.href = meta.sourceUrl;
      sourceLink.textContent = meta.sourceUrl;
      sourceLink.target = '_blank';
      sourceLink.rel = 'noopener noreferrer';
      sourceNote.textContent = 'Patterns from ';
      sourceNote.appendChild(sourceLink);
      body.appendChild(sourceNote);

      modal.style.display = '';
    });

    closeBtn.addEventListener('click', close);
    // Deliberately no "click outside to close" here, unlike the generic
    // modal pattern (see setupHRMSModal) -- the overlay has
    // pointer-events:none specifically so clicks pass through to the
    // canvas underneath, and staying open while panning/zooming to get
    // a better look at a highlighted match is exactly the point.
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && modal.style.display !== 'none') close();
    });
  }

  function setupHRMSModal() {
    const btn = document.getElementById('hrms-btn');
    const modal = document.getElementById('hrms-modal');
    const closeBtn = document.getElementById('hrms-modal-close');
    const body = document.getElementById('hrms-modal-body');
    if (!btn || !modal) return;

    function close() { modal.style.display = 'none'; }

    btn.addEventListener('click', function () {
      const hrms = CC.computeHRMS(controller.molecule, currentDescriptors);
      if (!hrms) return;

      body.innerHTML = '';
      const formula = document.createElement('p');
      formula.className = 'hrms-formula';
      formula.textContent = hrms.formula;
      body.appendChild(formula);

      const mass = document.createElement('p');
      mass.className = 'hrms-exact-mass';
      mass.textContent = 'Exact (monoisotopic) mass: ' + hrms.exactMass.toFixed(4) + ' Da';
      body.appendChild(mass);

      const table = document.createElement('table');
      table.className = 'hrms-adduct-table';
      const tbody = document.createElement('tbody');
      hrms.adducts.forEach(function (a) {
        const row = document.createElement('tr');
        const labelCell = document.createElement('td');
        labelCell.textContent = a.label;
        const mzCell = document.createElement('td');
        mzCell.textContent = a.mz.toFixed(4);
        row.appendChild(labelCell);
        row.appendChild(mzCell);
        tbody.appendChild(row);
      });
      table.appendChild(tbody);
      body.appendChild(table);

      const note = document.createElement('p');
      note.className = 'side-panel-note';
      note.style.marginTop = '10px';
      note.textContent = 'm/z values from CODATA physical constants and RDKit\u2019s own exact mass calculation \u2014 not experimentally measured.';
      body.appendChild(note);

      modal.style.display = '';
    });

    closeBtn.addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && modal.style.display !== 'none') close();
    });
  }

  function setupExportPanel() {
    const copyBtn = document.getElementById('copy-properties-btn');
    const csvBtn = document.getElementById('download-csv-btn');
    const xlsxBtn = document.getElementById('download-xlsx-btn');
    const pdfBtn = document.getElementById('download-pdf-btn');
    const status = document.getElementById('export-status');

    function currentRows() {
      return CC.Export.buildRows({
        validityText: currentValidityText,
        smiles: currentSmiles,
        descriptors: currentDescriptors,
      });
    }

    function flashStatus(text, isError) {
      status.textContent = text;
      status.style.color = isError ? 'var(--danger)' : 'var(--text-dark-muted)';
      setTimeout(function () { status.textContent = ''; }, 2500);
    }

    copyBtn.addEventListener('click', function () {
      CC.Export.copyToClipboard(currentRows())
        .then(function () { flashStatus('Copied ' + currentRows().length + ' properties to clipboard.'); })
        .catch(function (err) { flashStatus('Copy failed: ' + err.message, true); });
    });

    csvBtn.addEventListener('click', function () {
      CC.Export.downloadCSV(currentRows(), 'chemcanvas-properties.csv');
      flashStatus('Downloaded CSV.');
    });

    xlsxBtn.addEventListener('click', function () {
      const result = CC.Export.downloadXLSX(currentRows(), 'chemcanvas-properties.xlsx');
      flashStatus(result.ok ? 'Downloaded XLSX.' : result.message, !result.ok);
    });

    pdfBtn.addEventListener('click', function () {
      const result = CC.Export.downloadPDF(currentRows(), 'chemcanvas-properties.pdf', currentSmiles || 'ChemCanvas structure');
      flashStatus(result.ok ? 'Downloaded PDF.' : result.message, !result.ok);
    });
  }

  function setupPropertiesNav() {
    const btn = document.getElementById('show-properties-btn');
    btn.addEventListener('click', function () {
      const propertiesTab = document.querySelector('.side-tab[data-panel="properties"]');
      if (propertiesTab) propertiesTab.click();
    });
  }

  function setupSidePanelTabs() {
    const tabs = document.querySelectorAll('.side-tab');
    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        tabs.forEach(function (t) {
          t.classList.remove('is-active');
          t.setAttribute('aria-selected', 'false');
        });
        tab.classList.add('is-active');
        tab.setAttribute('aria-selected', 'true');

        document.querySelectorAll('.side-panel-content').forEach(function (panel) {
          panel.classList.add('is-hidden');
        });
        const targetPanel = document.getElementById('panel-' + tab.dataset.panel);
        if (targetPanel) targetPanel.classList.remove('is-hidden');
      });
    });
  }

  function setupSmilesInput() {
    const input = document.getElementById('smiles-input');
    const btn = document.getElementById('load-smiles-btn');
    const status = document.getElementById('smiles-input-status');
    if (!input || !btn) return;

    function loadFromSmiles() {
      const smiles = input.value.trim();
      status.textContent = '';
      if (!smiles) return;

      const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
      if (!RDKit) {
        status.textContent = 'RDKit.js is still loading \u2014 try again in a moment.';
        return;
      }

      let mol = null;
      try {
        // RDKit.js's get_mol() auto-generates 2D coordinates (and wedge
        // bonds for any defined stereocenters) for a bare SMILES, so the
        // resulting molblock is ready to draw as-is -- no separate
        // "compute coords" step needed.
        mol = RDKit.get_mol(smiles);
        if (!mol || !mol.is_valid()) {
          status.textContent = 'Not a valid SMILES string.';
          return;
        }
        const molblock = mol.get_molblock();
        const loaded = CC.molblockToMolecule(molblock);
        loadNewMolecule(loaded);
      } catch (err) {
        status.textContent = 'Could not parse that SMILES: ' + err.message;
        console.error('[ChemCanvas] SMILES load failed', err);
      } finally {
        if (mol) mol.delete();
      }
    }

    btn.addEventListener('click', loadFromSmiles);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        loadFromSmiles();
      }
    });
  }

  /**
   * Replace the current molecule and fit the viewport to it. Used by
   * every "load a whole new structure" path (SMILES, Open file) --
   * NOT by incremental edits, which should never yank the view out from
   * under someone mid-drawing. RDKit-generated coordinates (SMILES) and
   * an arbitrary molfile's own coordinates (Open) both have no
   * relationship to this app's default viewBox, unlike a structure
   * built by hand on-screen.
   */
  function loadNewMolecule(loaded) {
    molecule = loaded;
    controller.setMolecule(molecule);
    history.commit(molecule.toJSON());
    renderNow();
    CC.fitViewToContent(svg, molecule);
  }

  /**
   * "Clean up" (bound to the L key): redraws the whole structure with
   * standard bond lengths/angles via RDKit's own coordinate generator
   * (the same one used for "Load SMILES"), then centers the result.
   * Connectivity/bond orders/charges are untouched -- only positions
   * change, same as ChemDraw's own "Clean Up Structure."
   *
   * Centers the actual stored atom coordinates (not just the viewport)
   * so the underlying data itself is normalized -- useful if the
   * structure gets exported afterward, not just for how it looks right
   * now. loadNewMolecule's own fitViewToContent call handles the
   * viewport framing on top of that.
   */
  function cleanupStructure() {
    if (controller.molecule.isEmpty()) return;
    const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
    if (!RDKit) return; // not loaded yet -- nothing to clean up with

    let mol = null;
    try {
      const molblock = CC.moleculeToMolblock(controller.molecule);
      mol = RDKit.get_mol(molblock);
      if (!mol || !mol.is_valid()) return;

      const cleanedMolblock = mol.get_new_coords();
      const cleaned = CC.molblockToMolecule(cleanedMolblock);

      const atoms = Array.from(cleaned.atoms.values());
      if (atoms.length > 0) {
        const xs = atoms.map(function (a) { return a.x; });
        const ys = atoms.map(function (a) { return a.y; });
        const centroidX = (Math.min.apply(null, xs) + Math.max.apply(null, xs)) / 2;
        const centroidY = (Math.min.apply(null, ys) + Math.max.apply(null, ys)) / 2;
        const targetX = CC.DEFAULT_VIEWBOX.width / 2;
        const targetY = CC.DEFAULT_VIEWBOX.height / 2;
        const dx = targetX - centroidX, dy = targetY - centroidY;
        atoms.forEach(function (a) { a.x += dx; a.y += dy; });
      }

      loadNewMolecule(cleaned);
    } catch (err) {
      console.error('[ChemCanvas] structure cleanup failed', err);
    } finally {
      if (mol) mol.delete();
    }
  }

  function setupFileIO() {
    openBtn = document.getElementById('open-btn');
    saveBtn = document.getElementById('save-btn');
    fileInput = document.getElementById('file-input');

    openBtn.addEventListener('click', function () { fileInput.click(); });

    fileInput.addEventListener('change', function () {
      const file = fileInput.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function () {
        try {
          const loaded = CC.molblockToMolecule(String(reader.result));
          loadNewMolecule(loaded);
        } catch (err) {
          window.alert('Could not read that file as a molfile: ' + err.message);
        }
        fileInput.value = '';
      };
      reader.readAsText(file);
    });

    saveBtn.addEventListener('click', function () {
      const molblock = CC.moleculeToMolblock(controller.molecule, 'ChemCanvas structure');
      const blob = new Blob([molblock], { type: 'chemical/x-mdl-molfile' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'structure.mol';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }

  function setupPropertiesPanel() {
    validityDot = document.getElementById('validity-dot');
    validityText = document.getElementById('validity-text');
    smilesOutput = document.getElementById('smiles-output');
    copySmilesBtn = document.getElementById('copy-smiles-btn');

    copySmilesBtn.addEventListener('click', function () {
      if (!currentSmiles) return;
      navigator.clipboard.writeText(currentSmiles).then(function () {
        const original = copySmilesBtn.textContent;
        copySmilesBtn.textContent = 'Copied';
        setTimeout(function () { copySmilesBtn.textContent = original; }, 1200);
      });
    });
  }

  // ---------- 2D -> 3D generation ----------

  function setup3DPanel() {
    generate3dBtn = document.getElementById('generate-3d-btn');
    viewer3dNote = document.getElementById('viewer3d-note');
    const optimizeBtn = document.getElementById('optimize-3d-btn');
    const progressWrap = document.getElementById('viewer3d-progress');
    const progressFill = document.getElementById('viewer3d-progress-fill');
    const progressNote = document.getElementById('viewer3d-progress-note');

    let lastInitial = null; // CC.buildInitial3D() result, reused by the Optimize button

    function renderResult(result) {
      viewer3d = window.chemCanvasLibs && window.chemCanvasLibs.viewer3d;
      if (viewer3d) CC.render3D(viewer3d, result);
    }

    function describeResult(result, opts) {
      opts = opts || {};
      const convergedNote = opts.optimized
        ? (result.converged
          ? ' \u2014 converged (settled on its own, not just cut off by the time budget)'
          : ' \u2014 stopped before fully converging (hit the time budget; try Optimize again for another pass, or accept this as a reasonable approximate structure)')
        : '';
      return result.atoms.length + ' atoms (incl. implicit H)' +
        (result.energy !== null && result.energy !== undefined
          ? ', force-field energy ' + result.energy.toFixed(2) + ' (arbitrary units \u2014 not a real force field, lower is better)' + convergedNote
          : ', not yet optimized \u2014 click "Optimize geometry\u2026" to relax it');
    }

    generate3dBtn.addEventListener('click', function () {
      if (controller.molecule.isEmpty()) {
        viewer3dNote.textContent = 'Nothing to generate yet \u2014 draw a structure first.';
        optimizeBtn.disabled = true;
        return;
      }

      // buildInitial3D is fast (just implicit-H addition + 2D-seeded
      // coordinates, no force-field optimization) -- no progress UI needed.
      lastInitial = CC.buildInitial3D(controller.molecule);
      renderResult(lastInitial);
      viewer3dNote.textContent = describeResult(lastInitial, { optimized: false });
      optimizeBtn.disabled = false;
    });

    // Called on every 2D edit (see runValidation) -- a 3D structure or
    // optimized geometry for whatever the molecule *used* to be is worse
    // than no 3D view at all, since nothing on screen would indicate it's
    // now out of sync with the 2D structure you're actually looking at.
    invalidate3DView = function () {
      if (!lastInitial) return; // nothing generated yet -- nothing to invalidate
      lastInitial = null;
      if (viewer3d) CC.render3D(viewer3d, { atoms: [], bonds: [] });
      viewer3dNote.textContent = 'Structure has changed \u2014 click "Generate 3D structure" to update.';
      optimizeBtn.disabled = true;
    };

    optimizeBtn.addEventListener('click', async function () {
      if (!lastInitial || lastInitial.atoms.length === 0) {
        viewer3dNote.textContent = 'Click "Generate 3D structure" first.';
        return;
      }

      generate3dBtn.disabled = true;
      optimizeBtn.disabled = true;
      progressWrap.style.display = '';
      progressFill.style.width = '0%';
      progressNote.textContent = 'Starting conformer search\u2026';

      try {
        const result = await CC.optimize3D(lastInitial, {
          onProgress: function (info) {
            // No single well-defined "percent done" for a multi-attempt,
            // multi-stage, time-budgeted search -- attempt progress is
            // the honest, legible proxy rather than a fake-precise number.
            const pct = Math.round(((info.attempt - 1) / info.totalAttempts) * 100);
            progressFill.style.width = pct + '%';
            progressNote.textContent = 'Attempt ' + info.attempt + '/' + info.totalAttempts +
              ' \u2014 ' + info.stage +
              (info.bestEnergySoFar !== null ? ' (best so far: ' + info.bestEnergySoFar.toFixed(2) + ')' : '');
          },
        });
        progressFill.style.width = '100%';
        renderResult(result);
        viewer3dNote.textContent = describeResult(result, { optimized: true });
      } catch (err) {
        viewer3dNote.textContent = 'Optimization failed: ' + err.message;
        console.error('[ChemCanvas] 3D optimization failed', err);
      } finally {
        generate3dBtn.disabled = false;
        optimizeBtn.disabled = false;
        progressWrap.style.display = 'none';
      }
    });
  }

  // ---------- GNN prediction (demo D-MPNN + optional ONNX model) ----------

  let lastAtomProperties = null; // atomId -> {propName: value}, for heatmap re-render
  let lastMolecularProperties = null; // {propName: value} + propertyMeta, for the radar chart

  function setupGNNPanel() {
    const runDemoBtn = document.getElementById('run-demo-gnn-btn');
    const loadChempropBtn = document.getElementById('load-chemprop-btn');
    const chempropFileInput = document.getElementById('chemprop-file-input');
    const chempropStatus = document.getElementById('chemprop-model-status');
    const loadOnnxBtn = document.getElementById('load-onnx-btn');
    const onnxFileInput = document.getElementById('onnx-file-input');
    const onnxStatus = document.getElementById('onnx-model-status');
    const molOutput = document.getElementById('gnn-molecular-output'); // <tbody>
    const gnnTable = document.getElementById('gnn-molecular-output-table');
    const gnnEmptyNote = document.getElementById('gnn-empty-note');
    const heatmapRow = document.getElementById('gnn-atom-heatmap-row');
    const heatmapSelect = document.getElementById('heatmap-property-select');
    const clearHeatmapBtn = document.getElementById('clear-heatmap-btn');

    // molOutput is a <tbody> now (folded into the same table styling as
    // the RDKit descriptors above it) -- a <tbody> can only contain
    // <tr> elements, so "no results yet" / error states show as a
    // sibling note with the table hidden, rather than injected markup.
    function showGnnMessage(text) {
      gnnTable.style.display = 'none';
      gnnEmptyNote.textContent = text;
      gnnEmptyNote.style.display = '';
    }

    function runPrediction() {
      if (controller.molecule.isEmpty()) {
        showGnnMessage('Draw a structure first.');
        return;
      }
      runDemoBtn.disabled = true;
      CC.GNN.predictMolecule(controller.molecule)
        .then(function (result) {
          renderGNNOutput(result);
        })
        .catch(function (err) {
          showGnnMessage('Prediction failed: ' + err.message);
          console.error('[ChemCanvas] GNN prediction failed', err);
        })
        .finally(function () {
          runDemoBtn.disabled = false;
        });
    }

    /**
     * An expandable <tr> (hidden by default, toggled by the property's
     * info button) showing everything the registry already knows about
     * that model: definition, dataset provenance, architecture/training,
     * metrics, and any honest caveats -- all sourced directly from
     * registry.json, which is the same data validate_registry.py checks
     * and the model list already summarizes, just shown in full here
     * instead of the one-line summary.
     */
    function buildPropertyInfoRow(entry) {
      const tr = document.createElement('tr');
      tr.className = 'property-info-row';
      tr.style.display = 'none';
      const td = document.createElement('td');
      td.colSpan = 2;

      const box = document.createElement('div');
      box.className = 'property-info-box';

      function section(title, contentEl) {
        if (!contentEl) return;
        const h = document.createElement('div');
        h.className = 'property-info-heading';
        h.textContent = title;
        box.appendChild(h);
        box.appendChild(contentEl);
      }

      function textBlock(text) {
        if (!text) return null;
        const p = document.createElement('p');
        p.textContent = text;
        return p;
      }

      function defList(pairs) {
        const rows = pairs.filter(function (p) { return p[1] !== null && p[1] !== undefined && p[1] !== ''; });
        if (rows.length === 0) return null;
        const dl = document.createElement('dl');
        dl.className = 'property-info-deflist';
        rows.forEach(function (pair) {
          const dt = document.createElement('dt');
          dt.textContent = pair[0];
          const dd = document.createElement('dd');
          if (pair[2] === 'link') {
            const a = document.createElement('a');
            a.href = pair[1]; a.textContent = pair[1]; a.target = '_blank'; a.rel = 'noopener noreferrer';
            dd.appendChild(a);
          } else {
            dd.textContent = pair[1];
          }
          dl.appendChild(dt);
          dl.appendChild(dd);
        });
        return dl;
      }

      section('What this predicts', textBlock(entry.description));

      const ds = entry.dataset || {};
      section('Dataset', defList([
        ['Name', ds.name],
        ['Size', ds.size ? ds.size + ' molecules' : null],
        ['Split strategy', ds.splitStrategy],
        ['Source', ds.sourceUrl, 'link'],
      ]));

      const training = entry.training || {};
      const hp = training.hyperparameters;
      section('Model', defList([
        ['Architecture', entry.engine === 'nagl' ? 'GraphSAGE + electronegativity equalization' : 'Chemprop D-MPNN'],
        ['Hyperparameters', hp ? (typeof hp === 'string' ? hp : JSON.stringify(hp)) : null],
        ['Date trained', training.dateTrained],
      ]));

      if (entry.metrics) {
        const m = entry.metrics;
        const rows = [];
        if (m.primary) rows.push([m.primary.name, m.primary.value + (m.primary.units ? ' ' + m.primary.units : '')]);
        if (m.testSetSize) rows.push(['Test set size', m.testSetSize]);
        section('Reported metrics', defList(rows));
        if (m.note) section('', textBlock(m.note));
      }

      section('Notes & known limitations', textBlock(entry.notes));

      td.appendChild(box);
      tr.appendChild(td);
      return tr;
    }

    function renderGNNOutput(result) {
      molOutput.innerHTML = '';
      lastMolecularProperties = { values: result.molecularProperties || {}, meta: result.propertyMeta || {} };
      refreshRadarChart();

      const warningsEl = document.getElementById('gnn-prediction-warnings');
      if (warningsEl) {
        if (result.warnings && result.warnings.length > 0) {
          warningsEl.style.display = '';
          warningsEl.textContent = '\u26a0 ' + result.warnings.join('; ') + '.';
        } else {
          warningsEl.style.display = 'none';
        }
      }

      if (result.raw) {
        // ONNX backend: output shape is whatever your model produces —
        // show raw named tensors rather than guessing structure.
        Object.keys(result.raw).forEach(function (name) {
          const row = document.createElement('tr');
          const labelCell = document.createElement('td');
          labelCell.textContent = name;
          const valueCell = document.createElement('td');
          const vals = result.raw[name];
          valueCell.textContent = vals.length > 4
            ? '[' + vals.slice(0, 4).map(function (v) { return v.toFixed(2); }).join(', ') + ', \u2026]'
            : '[' + vals.map(function (v) { return v.toFixed(2); }).join(', ') + ']';
          row.appendChild(labelCell);
          row.appendChild(valueCell);
          molOutput.appendChild(row);
        });
        gnnEmptyNote.style.display = 'none';
        gnnTable.style.display = '';
        heatmapRow.style.display = 'none';
        lastAtomProperties = null;
        return;
      }

      const propertyMeta = result.propertyMeta || {};
      Object.keys(result.molecularProperties || {}).forEach(function (name) {
        const row = document.createElement('tr');
        const labelCell = document.createElement('td');
        const meta = propertyMeta[name];
        // Chemprop's own task/target-column name (e.g. a generic "label"
        // if that's what the training CSV's column was literally called)
        // makes a poor row label -- prefer the registry's displayName
        // when this property came from a registry-loaded model. Falls
        // back to the raw key for ad-hoc "load from files" models, which
        // have no registry entry to look a nicer name up from.
        const registryEntry = meta && meta.modelId ? CC.GNN.getRegistryEntry(meta.modelId) : null;
        labelCell.textContent = registryEntry ? registryEntry.displayName : name;

        // Info toggle: only shown for registry-loaded models, since an
        // ad-hoc "load from files" model has no dataset/training/notes
        // metadata to show in the first place -- nothing to click into.
        if (registryEntry) {
          const infoBtn = document.createElement('button');
          infoBtn.type = 'button';
          infoBtn.className = 'property-info-btn';
          infoBtn.setAttribute('aria-label', 'About this property');
          infoBtn.setAttribute('aria-expanded', 'false');
          infoBtn.textContent = '\u24d8'; // circled small "i"
          labelCell.appendChild(infoBtn);
        }

        const valueCell = document.createElement('td');
        const value = result.molecularProperties[name];

        if (meta && meta.taskType === 'classification') {
          // Chemprop's own convention: 0.5 is the positive/negative
          // decision boundary (same threshold used for its own
          // train/val accuracy metrics unless a model was specifically
          // calibrated otherwise).
          const isPositive = value >= 0.5;
          const badge = document.createElement('span');
          badge.className = 'classification-badge ' + (isPositive ? 'is-positive' : 'is-negative');
          badge.textContent = isPositive ? 'Positive' : 'Negative';
          valueCell.appendChild(badge);
          const scoreSpan = document.createElement('span');
          scoreSpan.className = 'classification-score';
          scoreSpan.textContent = ' (' + value.toFixed(2) + ')';
          valueCell.appendChild(scoreSpan);
        } else {
          valueCell.textContent = value.toFixed(2);
        }
        if (registryEntry && registryEntry.units) {
          const unitSpan = document.createElement('span');
          unitSpan.className = 'property-units';
          unitSpan.textContent = ' ' + registryEntry.units;
          valueCell.appendChild(unitSpan);
        }

        row.appendChild(labelCell);
        row.appendChild(valueCell);
        molOutput.appendChild(row);

        if (registryEntry) {
          const infoRow = buildPropertyInfoRow(registryEntry);
          molOutput.appendChild(infoRow);
          const infoBtn = labelCell.querySelector('.property-info-btn');
          infoBtn.addEventListener('click', function () {
            const isOpen = infoRow.style.display !== 'none';
            infoRow.style.display = isOpen ? 'none' : '';
            infoBtn.setAttribute('aria-expanded', String(!isOpen));
          });
        }
      });

      const hasAtomProperties = result.atomProperties && result.atomProperties.length > 0;
      if (molOutput.children.length === 0) {
        showGnnMessage(hasAtomProperties
          ? 'No molecule-level predictions for this structure — see the atom heatmap below for per-atom results.'
          : 'No predictions returned for this structure.');
      } else {
        gnnEmptyNote.style.display = 'none';
        gnnTable.style.display = '';
      }

      // Build atomId -> {propName: value} map for the heatmap, and
      // populate the property dropdown.
      if (hasAtomProperties && result.atomIds) {
        lastAtomProperties = {};
        result.atomIds.forEach(function (atomId, i) {
          lastAtomProperties[atomId] = result.atomProperties[i];
        });
        const propNames = Object.keys(result.atomProperties[0]);
        heatmapSelect.innerHTML = '';
        propNames.forEach(function (name) {
          const opt = document.createElement('option');
          opt.value = name;
          opt.textContent = name;
          heatmapSelect.appendChild(opt);
        });
        heatmapRow.style.display = '';
        applyHeatmap();
      } else {
        heatmapRow.style.display = 'none';
        lastAtomProperties = null;
      }
    }

    function applyHeatmap() {
      if (!lastAtomProperties || heatmapRow.style.display === 'none') return;
      const propName = heatmapSelect.value;
      const values = {};
      Object.keys(lastAtomProperties).forEach(function (atomId) {
        values[atomId] = lastAtomProperties[atomId][propName];
      });
      const scale = CC.renderAtomHeatmap(svg, controller.molecule, values, propName);
      const legendEl = document.getElementById('heatmap-legend-container');
      if (legendEl) CC.renderHeatmapLegend(legendEl, scale);

      const showTextCheckbox = document.getElementById('heatmap-show-text');
      if (showTextCheckbox && showTextCheckbox.checked) {
        CC.renderAtomValueLabels(svg, controller.molecule, values);
      } else {
        CC.clearAtomValueLabels(svg);
      }
    }
    reapplyHeatmap = applyHeatmap;

    // Called on every 2D edit (see runValidation) -- same reasoning as
    // invalidate3DView: a prediction table or atom heatmap for whatever
    // the molecule *used* to be is actively misleading once the
    // structure has changed, not just outdated.
    invalidateGNNResults = function () {
      if (!lastAtomProperties && gnnTable.style.display === 'none') return; // nothing to invalidate
      lastAtomProperties = null;
      lastMolecularProperties = null;
      molOutput.innerHTML = '';
      showGnnMessage('Structure has changed \u2014 click "Run prediction" to update.');
      heatmapRow.style.display = 'none';
      CC.clearAtomHeatmap(svg);
      CC.clearAtomValueLabels(svg);
      const legendEl = document.getElementById('heatmap-legend-container');
      if (legendEl) legendEl.innerHTML = '';
      refreshRadarChart();
    };

    runDemoBtn.addEventListener('click', runPrediction);
    heatmapSelect.addEventListener('change', applyHeatmap);
    const showTextCheckbox = document.getElementById('heatmap-show-text');
    if (showTextCheckbox) showTextCheckbox.addEventListener('change', applyHeatmap);
    clearHeatmapBtn.addEventListener('click', function () {
      CC.clearAtomHeatmap(svg);
      CC.clearAtomValueLabels(svg);
      const legendEl = document.getElementById('heatmap-legend-container');
      if (legendEl) legendEl.innerHTML = '';
    });

    function updateRunButtonLabel() {
      const n = CC.GNN.getLoadedChempropModelIds().length;
      runDemoBtn.textContent = n === 0
        ? 'Run prediction (demo weights)'
        : 'Run prediction (' + n + ' model' + (n === 1 ? '' : 's') + ' loaded)';
    }
    updateRunButtonLabel();

    function formatMetric(entry) {
      if (!entry.metrics || !entry.metrics.primary) return '';
      const p = entry.metrics.primary;
      return p.name + ' ' + (typeof p.value === 'number' ? p.value.toFixed(3) : p.value);
    }

    function renderRegistryList(entries) {
      const container = document.getElementById('model-registry-list');
      container.innerHTML = '';
      if (entries.length === 0) {
        container.innerHTML = '<p class="side-panel-note">No models in the registry yet.</p>';
        return;
      }

      entries.forEach(function (entry) {
        const row = document.createElement('div');
        row.className = 'model-registry-row';

        const info = document.createElement('div');
        info.className = 'model-registry-info';
        const nameEl = document.createElement('div');
        nameEl.className = 'model-registry-name';
        nameEl.textContent = entry.displayName;
        info.appendChild(nameEl);

        const subEl = document.createElement('div');
        subEl.className = 'model-registry-sub';
        const datasetName = entry.dataset && entry.dataset.name;
        const metricText = formatMetric(entry);
        const bits = [entry.taskType];
        if (datasetName) bits.push(datasetName);
        if (metricText) bits.push(metricText);
        subEl.textContent = bits.join(' \u00b7 ');
        info.appendChild(subEl);

        row.appendChild(info);

        const btn = document.createElement('button');
        btn.className = 'btn btn-ghost btn-small model-registry-load-btn';
        btn.textContent = CC.GNN.isRegistryModelLoaded(entry.id) ? 'Loaded' : 'Load';
        btn.disabled = CC.GNN.isRegistryModelLoaded(entry.id);
        btn.addEventListener('click', function () {
          btn.disabled = true;
          btn.textContent = 'Loading\u2026';
          CC.GNN.loadRegistryModel(entry.id)
            .then(function () {
              btn.textContent = 'Loaded';
              updateRunButtonLabel();
            })
            .catch(function (err) {
              btn.textContent = 'Load';
              btn.disabled = false;
              console.error('[ChemCanvas] Failed to load registry model "' + entry.id + '"', err);
              const errEl = document.createElement('div');
              errEl.className = 'model-registry-error';
              errEl.textContent = 'Failed: ' + err.message;
              row.appendChild(errEl);
            });
        });
        row.appendChild(btn);

        container.appendChild(row);
      });
    }

    // The registry (model/registry.json by default -- see model-config.js)
    // is fetched once at startup; individual models' weights are only
    // fetched when their row's Load button is clicked (load-on-demand,
    // per this project's earlier load-all-vs-on-demand decision).
    const registryStatus = document.getElementById('registry-status');
    CC.GNN.loadModelRegistry((CC.CONFIG && CC.CONFIG.registryUrl) || 'model/registry.json')
      .then(function (entries) {
        registryStatus.textContent = '';
        renderRegistryList(entries);
      })
      .catch(function (err) {
        registryStatus.textContent = 'Could not load model registry: ' + err.message;
        console.warn('[ChemCanvas] Model registry failed to load', err);
      });

    function applyLoadedChempropModel(id, loadPromise) {
      chempropStatus.textContent = 'Loading model\u2026';
      return loadPromise.then(function (info) {
        chempropStatus.textContent = 'Loaded "' + info.task + '" (' + info.taskType + ', d_h=' + info.dims.d_h + ', depth=' + info.dims.depth + ').';
        updateRunButtonLabel();
        return info;
      }).catch(function (err) {
        chempropStatus.textContent = 'Failed to load model: ' + err.message;
        console.error('[ChemCanvas] Chemprop model load failed', err);
        throw err;
      });
    }

    loadChempropBtn.addEventListener('click', function () { chempropFileInput.click(); });
    chempropFileInput.addEventListener('change', function () {
      const files = Array.from(chempropFileInput.files || []);
      chempropFileInput.value = '';
      if (files.length === 0) return;

      const manifestFile = files.find(function (f) { return /\.json$/i.test(f.name); });
      const binFile = files.find(function (f) { return /\.bin$/i.test(f.name); });
      if (!manifestFile || !binFile) {
        chempropStatus.textContent = 'Select both the manifest .json and weights .bin file together.';
        return;
      }

      // Ad-hoc models loaded this way (not yet in the registry) get a
      // synthetic id derived from the manifest filename plus a
      // timestamp, so loading the same file twice in one session doesn't
      // collide with the first load.
      const adHocId = 'adhoc:' + manifestFile.name.replace(/\.json$/i, '') + ':' + Date.now();

      applyLoadedChempropModel(
        adHocId,
        Promise.all([manifestFile.text().then(JSON.parse), binFile.arrayBuffer()])
          .then(function (results) { return CC.GNN.loadChempropModelFromBuffers(adHocId, results[0], results[1]); })
      );
    });

    loadOnnxBtn.addEventListener('click', function () { onnxFileInput.click(); });
    onnxFileInput.addEventListener('change', function () {
      const file = onnxFileInput.files[0];
      if (!file) return;
      onnxStatus.textContent = 'Loading model\u2026';
      const reader = new FileReader();
      reader.onload = function () {
        CC.GNN.loadOnnxModel(reader.result)
          .then(function () {
            const info = CC.GNN.getOnnxModelInfo();
            onnxStatus.textContent = 'Model loaded. Inputs: ' + info.inputNames.join(', ') +
              ' \u2014 Outputs: ' + info.outputNames.join(', ');
            runDemoBtn.textContent = 'Run demo D-MPNN (model loaded \u2014 using ONNX now)';
          })
          .catch(function (err) {
            onnxStatus.textContent = 'Failed to load model: ' + err.message;
            console.error('[ChemCanvas] ONNX model load failed', err);
          });
      };
      reader.readAsArrayBuffer(file);
      onnxFileInput.value = '';
    });
  }

  function activateTool(toolName) {
    const btn = document.querySelector('.tool-btn[data-tool="' + toolName + '"]');
    if (!btn) return;
    document.querySelectorAll('.tool-btn').forEach(function (b) { b.classList.remove('is-active'); });
    btn.classList.add('is-active');
    controller.setTool(toolName);
    svg.classList.toggle('tool-select', toolName === 'select');
    renderNow();
  }

  function setupToolRail() {
    const toolButtons = document.querySelectorAll('.tool-btn');
    toolButtons.forEach(function (btn) {
      btn.addEventListener('click', function () { activateTool(btn.dataset.tool); });
    });
  }

  function setupElementPicker() {
    const chips = document.querySelectorAll('.element-chip');
    const atomToolBtn = document.querySelector('.tool-btn[data-tool="atom"]');
    chips.forEach(function (chip) {
      chip.addEventListener('click', function () {
        chips.forEach(function (c) { c.classList.remove('is-active'); });
        chip.classList.add('is-active');
        controller.setElement(chip.dataset.element);

        document.querySelectorAll('.tool-btn').forEach(function (b) { b.classList.remove('is-active'); });
        atomToolBtn.classList.add('is-active');
        controller.setTool('atom');
        svg.classList.remove('tool-select');
        renderNow();
      });
    });
  }

  function setupBondStylePicker() {
    const chips = document.querySelectorAll('.bond-style-chip');
    const bondToolBtn = document.querySelector('.tool-btn[data-tool="bond"]');
    chips.forEach(function (chip) {
      chip.addEventListener('click', function () {
        chips.forEach(function (c) { c.classList.remove('is-active'); });
        chip.classList.add('is-active');
        controller.setBondStyle(chip.dataset.bondStyle);

        document.querySelectorAll('.tool-btn').forEach(function (b) { b.classList.remove('is-active'); });
        bondToolBtn.classList.add('is-active');
        controller.setTool('bond');
        svg.classList.remove('tool-select');
        renderNow();
      });
    });
  }

  function setupRingSizePicker() {
    const chips = document.querySelectorAll('.ring-size-chip');
    const ringToolBtn = document.querySelector('.tool-btn[data-tool="ring"]');
    chips.forEach(function (chip) {
      chip.addEventListener('click', function () {
        chips.forEach(function (c) { c.classList.remove('is-active'); });
        chip.classList.add('is-active');
        controller.setRingSize(parseInt(chip.dataset.ringSize, 10));
        controller.setRingAromatic(chip.dataset.ringAromatic === 'true');

        document.querySelectorAll('.tool-btn').forEach(function (b) { b.classList.remove('is-active'); });
        ringToolBtn.classList.add('is-active');
        controller.setTool('ring');
        svg.classList.remove('tool-select');
        renderNow();
      });
    });
  }

  function setupUndoRedo() {
    undoBtn = document.getElementById('undo-btn');
    redoBtn = document.getElementById('redo-btn');

    undoBtn.addEventListener('click', function () { loadHistoryState(history.undo()); });
    redoBtn.addEventListener('click', function () { loadHistoryState(history.redo()); });

    document.addEventListener('keydown', function (e) {
      const meta = e.ctrlKey || e.metaKey;
      if (meta && e.key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault();
        loadHistoryState(history.undo());
      } else if (meta && ((e.key.toLowerCase() === 'z' && e.shiftKey) || e.key.toLowerCase() === 'y')) {
        e.preventDefault();
        loadHistoryState(history.redo());
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && controller.currentTool === 'select') {
        if (controller.deleteSelection()) {
          e.preventDefault();
          renderNow();
        }
      }
    });
  }

  /**
   * Scroll-to-zoom and a manual "Fit to view" button. Deliberately doesn't
   * touch tools.js's pointer-drag handling for drawing -- CC.svgPoint()
   * re-derives model coordinates from whatever the viewBox currently is,
   * so mutating the viewBox here (via wheel or Fit to view) can't
   * conflict with the existing draw/select/move interactions at all.
   */
  function setupCanvasNavigation() {
    svg.addEventListener('wheel', function (e) {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
      CC.zoomViewBoxAt(svg, e.clientX, e.clientY, factor);
    }, { passive: false });

    const fitBtn = document.getElementById('fit-view-btn');
    if (fitBtn) {
      fitBtn.addEventListener('click', function () {
        CC.fitViewToContent(svg, controller.molecule);
      });
    }

    const cleanupBtn = document.getElementById('cleanup-btn');
    if (cleanupBtn) {
      cleanupBtn.addEventListener('click', cleanupStructure);
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    svg = document.getElementById('structure-svg');
    canvasStats = document.getElementById('canvas-stats');

    controller = new CC.Controller(svg, molecule, history, {
      onMoleculeChanged: renderNow,
      onSelectionChanged: function () { renderNow(); },
      onToolShortcut: activateTool,
      onCleanupShortcut: cleanupStructure,
    });

    setupToolRail();
    setupElementPicker();
    setupBondStylePicker();
    setupRingSizePicker();
    setupUndoRedo();
    setupFileIO();
    setupSmilesInput();
    setupPropertiesPanel();
    setupExportPanel();
    setupHRMSModal();
    setupSmartsFiltersModal();

    // Loaded once at startup, same as the model registry -- data-only
    // (no weights to fetch on demand), so there's no reason to defer it.
    CC.loadSmartsFilters()
      .then(function () {
        updateSmartsFiltersButton();
      })
      .catch(function (err) {
        console.warn('[ChemCanvas] Structural alert filter data failed to load', err);
      });
    setupPropertiesNav();
    setupSidePanelTabs();
    setup3DPanel();
    setupGNNPanel();
    setupCanvasNavigation();
    updateUndoRedoButtons();
    renderNow();

    // RDKit.js initializes asynchronously (see lib-loader.js); if the user
    // starts drawing before it's ready, retry validation once it reports in.
    waitForRDKitThenValidate();

    // The SA Score fragment table (sascorer.js) is a ~5.4MB static asset
    // shipped with the app, not something the user loads manually like a
    // Chemprop model -- fetch it once at startup and just re-render the
    // descriptor table when it lands, however long that takes.
    CC.loadSAScoreTable('model/sa-fragment-scores-manifest.json', 'model/sa-fragment-scores.bin')
      .then(function () { runValidation(); })
      .catch(function (err) {
        console.warn('[ChemCanvas] SA Score table failed to load; synthetic accessibility will be omitted.', err);
      });
  });

  function waitForRDKitThenValidate() {
    if (window.chemCanvasLibs && window.chemCanvasLibs.RDKit) {
      runValidation();
      return;
    }
    setTimeout(waitForRDKitThenValidate, 300);
  }
})();
