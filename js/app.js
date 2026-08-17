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
  let smilesToggleBtn = null;
  let copySmilesBtn = null;
  let currentSmiles = '';
  let smilesExpanded = false;
  const SMILES_TRUNCATE_AT = 40;
  let currentDescriptors = null;
  let currentValidityText = 'no structure yet';
  let validationTimer = null;
  let generate3dBtn = null;
  let viewer3dNote = null;
  let viewer3d = null;
  let reapplyHeatmap = function () {};
  let invalidateGNNResults = function () {};
  let invalidate3DView = function () {};
  let invalidateTitration = function () {};
  let notifyAni2xModelsChanged = function () {};
  let refreshRegistryList = function () {};
  let openDrugLikenessModal = function () {};
  let openMicrostateModal = function () {};
  let openPropertyInfoModal = function () {};
  let getCurrent3DGeometry = function () { return null; }; // set by setup3DPanel() -- lets SASA reuse an already-generated structure instead of building its own
  let refreshValidationPanel = function () {}; // set by setupValidationPanel() -- called from runValidation() on every 2D edit
  let lastStructureReport = null; // most recent CC.Validate.checkStructure() result, for other panels (e.g. gating a Run/Load button) to read without recomputing

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
    lastStructureReport = window.CC.Validate ? CC.Validate.checkStructure(mol) : null;
    refreshValidationPanel();
    refreshRegistryList(); // re-renders model-list tier badges against the new structure report
    updateModelCompatibilityWarning();
    invalidateGNNResults();
    invalidate3DView();
    invalidateTitration();

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
      renderDrugLikeness(null);
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

    renderDrugLikeness(descriptors);
  }

  function ordinalSuffix(n) {
    const j = n % 10, k = n % 100;
    if (j === 1 && k !== 11) return 'st';
    if (j === 2 && k !== 12) return 'nd';
    if (j === 3 && k !== 13) return 'rd';
    return 'th';
  }

  // Five real rule-based drug-likeness filters (Lipinski/Ghose/Veber/
  // Egan/Muegge — see druglikeness.js's own header for exact thresholds
  // and the disclosed Crippen-LogP substitution) plus, once the
  // reference distribution has loaded, each underlying property's
  // percentile rank against ~3300 real FDA-approved-drug-proxy
  // molecules (data/druglikeness_reference.json).
  function renderDrugLikeness(descriptors) {
    const filtersTable = document.getElementById('druglikeness-filters-table');
    const filtersOutput = document.getElementById('druglikeness-filters-output');
    const percentilesTable = document.getElementById('druglikeness-percentiles-table');
    const percentilesOutput = document.getElementById('druglikeness-percentiles-output');
    const note = document.getElementById('druglikeness-note');
    if (!filtersTable || !percentilesTable) return; // panel not set up yet

    if (!descriptors) {
      filtersTable.style.display = 'none';
      percentilesTable.style.display = 'none';
      note.style.display = '';
      note.textContent = 'Draw a structure to see drug-likeness filters.';
      return;
    }

    const result = CC.DrugLikeness.evaluate(descriptors, controller.molecule);

    filtersOutput.innerHTML = '';
    Object.keys(result.filters).forEach(function (name) {
      const f = result.filters[name];
      const row = document.createElement('tr');
      const labelCell = document.createElement('td');
      labelCell.textContent = CC.DrugLikeness.FILTER_LABELS[name];
      const infoBtn = document.createElement('button');
      infoBtn.className = 'filter-info-btn';
      infoBtn.type = 'button';
      infoBtn.textContent = 'ⓘ';
      infoBtn.title = 'Show ' + CC.DrugLikeness.FILTER_LABELS[name] + '’s criteria';
      infoBtn.addEventListener('click', function () {
        openDrugLikenessModal(name, f);
      });
      labelCell.appendChild(infoBtn);
      const valueCell = document.createElement('td');
      const badge = document.createElement('span');
      badge.className = 'classification-badge ' + (f.pass ? 'is-positive' : 'is-negative');
      badge.textContent = f.pass ? 'Pass' : (f.violations + ' violation' + (f.violations === 1 ? '' : 's'));
      valueCell.appendChild(badge);
      if (f.referencePassRate !== null) {
        const refSpan = document.createElement('span');
        refSpan.className = 'property-units';
        refSpan.textContent = ' (' + Math.round(f.referencePassRate * 100) + '% of approved drugs pass)';
        valueCell.appendChild(refSpan);
      }
      row.appendChild(labelCell);
      row.appendChild(valueCell);
      filtersOutput.appendChild(row);
    });

    const integerProps = { hbd: true, hba: true, rotatableBonds: true, heavyAtoms: true };
    percentilesOutput.innerHTML = '';
    let anyPercentile = false;
    Object.keys(CC.DrugLikeness.PROPERTY_LABELS).forEach(function (key) {
      const value = result.inputs[key];
      if (typeof value !== 'number' || isNaN(value)) return;
      const pct = result.percentiles[key];
      const row = document.createElement('tr');
      const labelCell = document.createElement('td');
      labelCell.textContent = CC.DrugLikeness.PROPERTY_LABELS[key];
      const valueCell = document.createElement('td');
      let text = value.toFixed(integerProps[key] ? 0 : 2);
      if (pct !== null) {
        anyPercentile = true;
        const rounded = Math.round(pct);
        text += ' (' + rounded + ordinalSuffix(rounded) + ' percentile)';
      }
      valueCell.textContent = text;
      row.appendChild(labelCell);
      row.appendChild(valueCell);
      percentilesOutput.appendChild(row);
    });

    filtersTable.style.display = '';
    percentilesTable.style.display = anyPercentile ? '' : 'none';
    note.style.display = anyPercentile ? 'none' : '';
    note.textContent = anyPercentile ? '' : 'Loading FDA-approved-drug reference distribution…';
  }

  function setValidityState(state, text) {
    currentValidityText = text;
    validityDot.classList.remove('is-loading', 'is-ready', 'is-error');
    if (state === 'ready') validityDot.classList.add('is-ready');
    else if (state === 'error') validityDot.classList.add('is-error');
    else if (state === 'loading') validityDot.classList.add('is-loading');
    validityText.textContent = text;
  }

  // Long canonical SMILES (a real problem for anything beyond a small
  // molecule -- the 50-atom example that motivated this is ~140
  // characters) otherwise dominate the side panel's width or wrap
  // across many lines just sitting there unread. Collapsed to a short
  // preview by default with a [show]/[hide] toggle; Copy always copies
  // the FULL string regardless of which state is currently displayed.
  function renderSmilesDisplay() {
    const full = currentSmiles || '';
    if (!full) {
      smilesOutput.textContent = '\u2014';
      smilesToggleBtn.style.display = 'none';
      return;
    }
    const needsTruncation = full.length > SMILES_TRUNCATE_AT;
    if (needsTruncation && !smilesExpanded) {
      smilesOutput.textContent = full.slice(0, SMILES_TRUNCATE_AT) + '\u2026';
      smilesToggleBtn.textContent = '[show]';
    } else {
      smilesOutput.textContent = full;
      smilesToggleBtn.textContent = '[hide]';
    }
    smilesToggleBtn.style.display = needsTruncation ? '' : 'none';
  }

  function setSmiles(smiles) {
    currentSmiles = smiles;
    smilesExpanded = false; // every new molecule/edit starts collapsed again
    renderSmilesDisplay();
    copySmilesBtn.disabled = !smiles;
  }

  function updateExportButtons() {
    const hasData = !!currentDescriptors;
    ['copy-properties-btn', 'download-csv-btn', 'download-xlsx-btn', 'download-pdf-btn', 'download-sdf-btn'].forEach(function (id) {
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

  // Per-criterion detail behind each Lipinski/Ghose/Veber/Egan/Muegge
  // filter row's \u24d8 button -- the row itself only ever shows an aggregate
  // "N violations", which doesn't say WHICH criteria those are; this is
  // literally CC.DrugLikeness.evaluate()'s own f.criteria for that one
  // filter, just laid out as a pass/fail list instead of a count.
  function setupDrugLikenessModal() {
    const modal = document.getElementById('druglikeness-modal');
    const title = document.getElementById('druglikeness-modal-title');
    const closeBtn = document.getElementById('druglikeness-modal-close');
    const body = document.getElementById('druglikeness-modal-body');
    if (!modal) return;

    function close() { modal.style.display = 'none'; }

    openDrugLikenessModal = function (name, f) {
      title.textContent = CC.DrugLikeness.FILTER_LABELS[name] + ' \u2014 ' + (f.pass ? 'passes every criterion' : f.violations + ' violation' + (f.violations === 1 ? '' : 's'));
      body.innerHTML = '';

      const list = document.createElement('ul');
      list.className = 'filter-criteria-list';
      f.criteria.forEach(function (c) {
        const item = document.createElement('li');
        item.className = 'filter-criterion ' + (c.violated ? 'is-violated' : 'is-met');
        const mark = document.createElement('span');
        mark.className = 'filter-criterion-mark';
        mark.textContent = c.violated ? '\u2715' : '\u2713';
        const label = document.createElement('span');
        label.className = 'filter-criterion-label';
        label.textContent = c.label;
        const value = document.createElement('span');
        value.className = 'filter-criterion-value';
        value.textContent = typeof c.value === 'number' ? c.value.toFixed(2) + (c.unit ? ' ' + c.unit : '') : '\u2014';
        item.appendChild(mark);
        item.appendChild(label);
        item.appendChild(value);
        list.appendChild(item);
      });
      body.appendChild(list);

      if (f.referencePassRate !== null) {
        const note = document.createElement('p');
        note.className = 'side-panel-note';
        note.style.marginTop = '10px';
        note.textContent = Math.round(f.referencePassRate * 100) + '% of the FDA-approved-drug reference set passes this filter overall.';
        body.appendChild(note);
      }

      modal.style.display = '';
    };

    closeBtn.addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && modal.style.display !== 'none') close();
    });
  }

  // Behind each titration microstate row's "View" button -- a real 2D
  // depiction of that exact species (protonation state, formal charges
  // and all), built on demand rather than pre-rendered for every row
  // (see buildMicrostateViewButton's own comment for why). Reuses the
  // SAME atom coordinates already on the main canvas
  // (buildMicrostateStructure only ever touches .charge on the
  // ionizable sites, never position), so this is chemically faithful,
  // not a generic/schematic icon -- CC.render already draws charge
  // labels on top.
  function setupMicrostateModal() {
    const modal = document.getElementById('microstate-modal');
    const title = document.getElementById('microstate-modal-title');
    const closeBtn = document.getElementById('microstate-modal-close');
    const body = document.getElementById('microstate-modal-body');
    if (!modal) return;

    function close() { modal.style.display = 'none'; }

    openMicrostateModal = function (sitesForCurve, microstate, region) {
      body.innerHTML = '';
      title.textContent = 'pH ' + region.pHStart.toFixed(1) + '–' + region.pHEnd.toFixed(1) +
        ' · net charge ' + (microstate.netCharge > 0 ? '+' : '') + microstate.netCharge;

      const built = CC.PKAMicrostates.buildMicrostateStructure(controller.molecule, sitesForCurve, microstate);
      if (!built) {
        const note = document.createElement('p');
        note.className = 'side-panel-note';
        note.textContent = 'Structure unavailable (RDKit.js not ready, or this microstate didn’t sanitize).';
        body.appendChild(note);
        modal.style.display = '';
        return;
      }

      const structureSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      structureSvg.setAttribute('class', 'microstate-modal-structure');
      body.appendChild(structureSvg);
      CC.render(structureSvg, built.molecule, {});
      CC.setViewBox(structureSvg, CC.computeFitViewBox(structureSvg, built.molecule, 30));

      const desc = document.createElement('p');
      desc.className = 'side-panel-note';
      desc.style.marginTop = '10px';
      desc.textContent = describeMicrostate(sitesForCurve, microstate);
      body.appendChild(desc);

      modal.style.display = '';
    };

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
    const sdfBtn = document.getElementById('download-sdf-btn');
    const titleInput = document.getElementById('export-title-input');
    const status = document.getElementById('export-status');

    function currentRows() {
      return CC.Export.buildRows({
        validityText: currentValidityText,
        smiles: currentSmiles,
        descriptors: currentDescriptors,
      });
    }

    // Default (pre-title-entry) title is the molecular formula -- short,
    // unique-ish, and (unlike a raw SMILES) already filename-safe, so
    // both reportTitle() and filenameBase() below can share it directly.
    function defaultTitle() {
      if (controller.molecule && !controller.molecule.isEmpty()) {
        const formula = CC.computeMolecularFormula(controller.molecule);
        if (formula) return formula;
      }
      return 'ChemCanvas structure';
    }

    // User-entered title, used as: the PDF report heading, the SDF
    // molblock's own title line (conventionally a molecule name), and
    // the download filename (sanitized) for every export format -- one
    // field ties all of them together instead of each hardcoding
    // "chemcanvas-properties".
    function reportTitle() {
      return titleInput.value.trim() || defaultTitle();
    }

    function filenameBase() {
      const safe = reportTitle().replace(/[^a-z0-9_\- ]+/gi, '').trim().replace(/\s+/g, '-');
      return safe || 'chemcanvas-properties';
    }

    function flashStatus(text, isError) {
      status.textContent = text;
      status.style.color = isError ? 'var(--danger)' : 'var(--text-dark-muted)';
      setTimeout(function () { status.textContent = ''; }, 2500);
      CC.Logger[isError ? 'error' : 'success'](text);
    }

    copyBtn.addEventListener('click', function () {
      CC.Export.copyToClipboard(currentRows())
        .then(function () { flashStatus('Copied ' + currentRows().length + ' properties to clipboard.'); })
        .catch(function (err) { flashStatus('Copy failed: ' + err.message, true); });
    });

    csvBtn.addEventListener('click', function () {
      CC.Export.downloadCSV(currentRows(), filenameBase() + '.csv');
      flashStatus('Downloaded CSV.');
    });

    xlsxBtn.addEventListener('click', function () {
      const result = CC.Export.downloadXLSX(currentRows(), filenameBase() + '.xlsx');
      flashStatus(result.ok ? 'Downloaded XLSX.' : result.message, !result.ok);
    });

    pdfBtn.addEventListener('click', function () {
      const atomTable = CC.Export.buildAtomTable(controller.molecule, lastAtomProperties);
      const bondTable = CC.Export.buildBondTable(controller.molecule, lastBondProperties);
      const result = CC.Export.downloadPDF(
        currentRows(), filenameBase() + '.pdf', reportTitle(),
        atomTable, bondTable
      );
      flashStatus(result.ok ? 'Downloaded PDF.' : result.message, !result.ok);
    });

    sdfBtn.addEventListener('click', function () {
      const result = CC.Export.downloadSDF(
        controller.molecule, reportTitle(), currentRows(),
        lastAtomProperties, lastBondProperties, filenameBase() + '.sdf'
      );
      flashStatus(result.ok ? 'Downloaded SDF.' : result.message, !result.ok);
    });
  }

  // One short "site: protonated/deprotonated" clause per site, e.g.
  // "carboxylic acid: deprotonated, aliphatic amine: protonated" --
  // "protonated" always means literally bears the extra H here,
  // regardless of whether that's the site's neutral or charged form (an
  // acid site is neutral when protonated; a base site is charged when
  // protonated) -- matches a net-charge column/label rather than
  // requiring the reader to also track site class. Top-level (not
  // nested in setupTitrationPanel) since both the titration table rows
  // and the microstate structure modal need it.
  function describeMicrostate(sitesForCurve, microstate) {
    return sitesForCurve.map(function (site, i) {
      return site.name.replace(/_/g, ' ') + ': ' + (microstate.protonation[i] ? 'protonated' : 'deprotonated');
    }).join(', ');
  }

  function formatMetric(entry) {
    if (!entry.metrics || !entry.metrics.primary) return '';
    const p = entry.metrics.primary;
    return p.name + ' ' + (typeof p.value === 'number' ? p.value.toFixed(3) : p.value);
  }

  // Builds the content shown behind every predictive model's "[?]" popup
  // (see setupPropertyInfoModal): what it predicts, its training dataset,
  // reported metrics/expected accuracy, and any documented domain-of-
  // applicability caveats -- all sourced directly from registry.json,
  // which is the same data validate_registry.py checks and the model
  // list's one-line summary is drawn from, just shown in full here. Top-
  // level (not nested in setupPropertiesPanel) so both the Properties
  // panel's model list/results table AND the Titration tab's own "[?]"
  // can build the same content for the same registry entry.
  function buildPropertyInfoBox(entry) {
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
      section('Reported metrics (expected accuracy)', defList(rows));
      if (m.note) section('', textBlock(m.note));
    }

    section('Notes, known limitations & domain of application', textBlock(entry.notes));

    return box;
  }

  // Generic "[?]" popup used everywhere a registry entry (predictive
  // model) is shown -- the GNN results table, the model "Load" list, and
  // section headings like the Titration tab's -- so there's exactly one
  // popup implementation and one place to change its look, no matter
  // which panel triggered it.
  function setupPropertyInfoModal() {
    const modal = document.getElementById('property-info-modal');
    const title = document.getElementById('property-info-modal-title');
    const closeBtn = document.getElementById('property-info-modal-close');
    const body = document.getElementById('property-info-modal-body');
    if (!modal) return;

    function close() { modal.style.display = 'none'; }

    openPropertyInfoModal = function (displayName, contentEl) {
      title.textContent = displayName;
      body.innerHTML = '';
      body.appendChild(contentEl);
      modal.style.display = '';
    };

    closeBtn.addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && modal.style.display !== 'none') close();
    });
  }

  // Aqueous pKa / titration curve panel -- its own tab (per the user's
  // explicit request for a separate panel, not folded into the GNN
  // atom-heatmap system every other atom-level property uses). Own
  // "Compute" button rather than auto-running on every edit: like SASA,
  // this loads a model (~1.2MB, once) and runs real chemistry
  // (SMARTS site detection + a titration-curve computation), not the
  // instant kind of thing that belongs behind an always-live button.
  function setupTitrationPanel() {
    const computeBtn = document.getElementById('compute-titration-btn');
    const status = document.getElementById('titration-status');
    const sitesSection = document.getElementById('titration-sites-section');
    const sitesOutput = document.getElementById('titration-sites-output');
    const curveSection = document.getElementById('titration-curve-section');
    const chartContainer = document.getElementById('titration-chart-container');
    const piNote = document.getElementById('titration-pi-note');
    const microstatesSection = document.getElementById('titration-microstates-section');
    const microstatesOutput = document.getElementById('titration-microstates-output');
    const infoBtn = document.getElementById('titration-info-btn');
    if (infoBtn) {
      infoBtn.addEventListener('click', function () {
        const entry = CC.GNN.getRegistryEntry('aqueous-pka');
        if (entry) openPropertyInfoModal(entry.displayName, buildPropertyInfoBox(entry));
      });
    }

    function reset() {
      status.textContent = '';
      sitesSection.style.display = 'none';
      curveSection.style.display = 'none';
      microstatesSection.style.display = 'none';
      sitesOutput.innerHTML = '';
      chartContainer.innerHTML = '';
      piNote.textContent = '';
      microstatesOutput.innerHTML = '';
    }
    invalidateTitration = reset;

    // "View structure" button for one microstate row -- deliberately
    // on-demand (opens the shared modal, see setupMicrostateModal) rather
    // than a small thumbnail embedded directly in every row: rendering a
    // full 2D depiction per row unconditionally means a molecule with
    // several ionizable sites (2^N microstates, though only the ones
    // that actually dominate somewhere get a row at all -- still often
    // 4-6+ rows for a real drug-like molecule) pays for several structure
    // renders on every curve computation whether anyone looks at them or
    // not, and small inline sketches of anything but a tiny molecule are
    // too cramped to read anyway.
    function buildMicrostateViewButton(sitesForCurve, microstate, region) {
      const btn = document.createElement('button');
      btn.className = 'btn btn-ghost btn-small';
      btn.type = 'button';
      btn.textContent = 'View';
      btn.addEventListener('click', function () {
        openMicrostateModal(sitesForCurve, microstate, region);
      });
      return btn;
    }

    computeBtn.addEventListener('click', async function () {
      if (controller.molecule.isEmpty()) {
        status.textContent = 'Draw a structure first.';
        return;
      }

      const sites = CC.PKAMicrostates.findIonizableSites(controller.molecule);
      if (sites.length === 0) {
        reset();
        status.textContent = 'No ionizable groups detected in this structure.';
        return;
      }

      computeBtn.disabled = true;
      status.textContent = 'Loading pKa model…';

      try {
        if (!CC.GNN.hasChempropModel('aqueous-pka')) {
          const entry = CC.GNN.getRegistryEntries().find(function (e) { return e.id === 'aqueous-pka'; });
          if (!entry) throw new Error('aqueous-pka model not found in the registry');
          await CC.GNN.loadRegistryModel(entry.id);
          refreshRegistryList();
        }

        status.textContent = 'Predicting pKa values…';
        const result = CC.GNN.predictChemprop(controller.molecule, 'aqueous-pka');
        const pkaByAtomId = {};
        result.atomIds.forEach(function (atomId, i) {
          const props = result.atomProperties[i];
          if (props && typeof props.pka === 'number') pkaByAtomId[atomId] = props.pka;
        });

        const validSites = [];
        const validPKa = [];
        sites.forEach(function (site) {
          if (typeof pkaByAtomId[site.atomId] === 'number') {
            validSites.push(site);
            validPKa.push(pkaByAtomId[site.atomId]);
          }
        });

        sitesOutput.innerHTML = '';
        sites.forEach(function (site) {
          const row = document.createElement('tr');
          const nameCell = document.createElement('td');
          nameCell.textContent = site.name.replace(/_/g, ' ') + ' (' + site.element + ')';
          const classCell = document.createElement('td');
          classCell.textContent = site.cls === 'acid' ? 'acidic' : 'basic';
          const pkaCell = document.createElement('td');
          const pkaValue = pkaByAtomId[site.atomId];
          pkaCell.textContent = typeof pkaValue === 'number' ? pkaValue.toFixed(1) : '—';
          row.appendChild(nameCell);
          row.appendChild(classCell);
          row.appendChild(pkaCell);
          sitesOutput.appendChild(row);
        });
        sitesSection.style.display = '';

        const missingCount = sites.length - validSites.length;
        status.textContent = missingCount > 0
          ? missingCount + ' of ' + sites.length + ' detected site(s) had no predicted pKa — omitted from the curve.'
          : '';

        if (validSites.length > 0) {
          const curve = CC.PKATitration.computeCurve(validSites, validPKa);
          CC.renderTitrationChart(chartContainer, curve);
          const pI = CC.PKATitration.isoelectricPoint(curve);
          piNote.textContent = pI !== null
            ? 'Isoelectric point (average net charge = 0): pH ' + pI.toFixed(1)
            : 'Net charge never crosses zero across pH 0–14 (' +
              (curve.avgCharge[0] > 0 ? 'stays positive' : 'stays negative') + ' throughout).';
          curveSection.style.display = '';

          const regions = CC.PKATitration.dominantMicrostateRegions(curve);
          microstatesOutput.innerHTML = '';
          regions.forEach(function (region, i) {
            const microstate = curve.microstates[region.microstateIndex];
            const row = document.createElement('tr');

            const structureCell = document.createElement('td');
            structureCell.appendChild(buildMicrostateViewButton(validSites, microstate, region));

            const rangeCell = document.createElement('td');
            rangeCell.textContent = 'pH ' + region.pHStart.toFixed(1) + '–' + region.pHEnd.toFixed(1);

            const chargeCell = document.createElement('td');
            chargeCell.textContent = (microstate.netCharge > 0 ? '+' : '') + microstate.netCharge;

            const stateCell = document.createElement('td');
            stateCell.textContent = describeMicrostate(validSites, microstate);

            row.appendChild(structureCell);
            row.appendChild(rangeCell);
            row.appendChild(chargeCell);
            row.appendChild(stateCell);
            microstatesOutput.appendChild(row);
          });
          microstatesSection.style.display = regions.length > 0 ? '' : 'none';
          CC.Logger.success('Computed titration curve: ' + validSites.length + ' site(s), ' + regions.length + ' dominant microstate region(s)');
        } else {
          curveSection.style.display = 'none';
          microstatesSection.style.display = 'none';
        }
      } catch (err) {
        status.textContent = 'Failed: ' + err.message;
        console.error('[ChemCanvas] Titration curve computation failed', err);
        CC.Logger.error('Titration curve computation failed: ' + err.message);
      } finally {
        computeBtn.disabled = false;
      }
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
          CC.Logger.warning('SMILES load failed (invalid): ' + smiles);
          return;
        }
        const molblock = mol.get_molblock();
        const loaded = CC.molblockToMolecule(molblock);
        loadNewMolecule(loaded);
        CC.Logger.success('Loaded structure from SMILES: ' + smiles);
      } catch (err) {
        status.textContent = 'Could not parse that SMILES: ' + err.message;
        console.error('[ChemCanvas] SMILES load failed', err);
        CC.Logger.error('SMILES load failed: ' + err.message);
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
          CC.Logger.success('Loaded structure from file: ' + file.name);
        } catch (err) {
          window.alert('Could not read that file as a molfile: ' + err.message);
          CC.Logger.error('Failed to load file "' + file.name + '": ' + err.message);
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
      CC.Logger.info('Saved structure.mol');
    });
  }

  function setupPropertiesPanel() {
    validityDot = document.getElementById('validity-dot');
    validityText = document.getElementById('validity-text');
    smilesOutput = document.getElementById('smiles-output');
    smilesToggleBtn = document.getElementById('smiles-toggle-btn');
    copySmilesBtn = document.getElementById('copy-smiles-btn');

    smilesToggleBtn.addEventListener('click', function () {
      smilesExpanded = !smilesExpanded;
      renderSmilesDisplay();
    });

    copySmilesBtn.addEventListener('click', function () {
      if (!currentSmiles) return;
      navigator.clipboard.writeText(currentSmiles).then(function () {
        const original = copySmilesBtn.textContent;
        copySmilesBtn.textContent = 'Copied';
        setTimeout(function () { copySmilesBtn.textContent = original; }, 1200);
      });
    });
  }

  // ---------- structure validation panel ----------

  const SEVERITY_RANK = { error: 3, warning: 2, info: 1 };
  const TIER_LABEL = { compatible: 'Compatible', warning: 'Caution', blocked: 'Blocked' };

  function setupValidationPanel() {
    const banner = document.getElementById('validation-status-banner');
    const issuesList = document.getElementById('validation-issues-list');
    const issuesNote = document.getElementById('validation-issues-note');
    const compatTable = document.getElementById('validation-compat-table');
    const compatBody = document.getElementById('validation-compat-body');
    const compatNote = document.getElementById('validation-compat-note');
    if (!banner) return;

    function renderIssue(issue) {
      const row = document.createElement('div');
      row.className = 'validation-issue-row';

      const badge = document.createElement('div');
      badge.className = 'validation-issue-badge sev-' + issue.severity;
      row.appendChild(badge);

      const body = document.createElement('div');
      body.className = 'validation-issue-body';
      const label = document.createElement('div');
      label.className = 'validation-issue-label';
      label.textContent = issue.label;
      body.appendChild(label);
      const message = document.createElement('div');
      message.className = 'validation-issue-message';
      message.textContent = issue.message;
      body.appendChild(message);
      row.appendChild(body);

      if (issue.atomIds && issue.atomIds.length > 0) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'validation-highlight-btn';
        btn.textContent = 'Highlight';
        btn.addEventListener('click', function () {
          CC.highlightSmartsMatch(svg, controller.molecule, issue.atomIds);
        });
        row.appendChild(btn);
      }

      return row;
    }

    function renderCompatRow(entry) {
      const tr = document.createElement('tr');
      const nameCell = document.createElement('td');
      nameCell.textContent = entry.displayName;
      const statusCell = document.createElement('td');
      const tierBadge = document.createElement('span');
      tierBadge.className = 'tier-badge tier-' + entry.tier;
      tierBadge.textContent = TIER_LABEL[entry.tier] || entry.tier;
      statusCell.appendChild(tierBadge);
      if (entry.reasons && entry.reasons.length > 0) {
        const reasons = document.createElement('div');
        reasons.className = 'validation-compat-reasons';
        reasons.textContent = entry.reasons.join('; ');
        statusCell.appendChild(reasons);
      }
      tr.appendChild(nameCell);
      tr.appendChild(statusCell);
      return tr;
    }

    refreshValidationPanel = function () {
      const mol = controller.molecule;
      issuesList.innerHTML = '';
      CC.clearSmartsHighlight(svg);

      if (mol.isEmpty()) {
        banner.className = 'validation-status-banner status-idle';
        banner.textContent = 'Draw a structure to check it.';
        issuesNote.style.display = '';
        issuesNote.textContent = 'Nothing drawn yet.';
        compatTable.style.display = 'none';
        compatNote.style.display = '';
        compatNote.textContent = 'Draw a structure to see per-model compatibility.';
        return;
      }

      const report = lastStructureReport;
      if (!report || !report.available) {
        banner.className = 'validation-status-banner status-idle';
        banner.textContent = 'RDKit.js still loading…';
        issuesNote.style.display = '';
        issuesNote.textContent = '';
        compatTable.style.display = 'none';
        compatNote.style.display = '';
        compatNote.textContent = '';
        return;
      }

      // Banner: worst severity present, or a clean bill of health.
      if (report.counts.error > 0) {
        banner.className = 'validation-status-banner status-error';
        banner.textContent = '✗ Structure incompatible — predictions blocked until fixed.';
      } else if (report.counts.warning > 0) {
        banner.className = 'validation-status-banner status-warning';
        banner.textContent = '⚠ Structure valid, ' + report.counts.warning + ' issue(s) worth a look — some models may be outside their applicability domain.';
      } else if (report.counts.info > 0) {
        banner.className = 'validation-status-banner status-ok';
        banner.textContent = '✓ Structure valid, ' + report.counts.info + ' informational note(s).';
      } else {
        banner.className = 'validation-status-banner status-ok';
        banner.textContent = '✓ Structure valid — no issues found.';
      }

      if (report.issues.length === 0) {
        issuesNote.style.display = '';
        issuesNote.textContent = 'No structural issues found.';
      } else {
        issuesNote.style.display = 'none';
        report.issues
          .slice()
          .sort(function (a, b) { return SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity]; })
          .forEach(function (issue) { issuesList.appendChild(renderIssue(issue)); });
      }

      // Model compatibility table -- needs the registry loaded, which may
      // still be in flight on first paint (registryStatus fetch is async).
      if (!window.CC.GNN || !CC.GNN.getRegistryEntries || CC.GNN.getRegistryEntries().length === 0) {
        compatTable.style.display = 'none';
        compatNote.style.display = '';
        compatNote.textContent = 'Model registry not loaded yet.';
        return;
      }

      const compat = CC.Validate.checkModelCompatibility(mol, report);
      compatBody.innerHTML = '';
      compat
        .slice()
        .sort(function (a, b) {
          const tierRank = { blocked: 3, warning: 2, compatible: 1 };
          return tierRank[b.tier] - tierRank[a.tier];
        })
        .forEach(function (entry) { compatBody.appendChild(renderCompatRow(entry)); });
      compatTable.style.display = '';
      compatNote.style.display = 'none';
    };
  }

  // ---------- 2D -> 3D generation ----------

  function setup3DPanel() {
    generate3dBtn = document.getElementById('quick-preview-btn');
    viewer3dNote = document.getElementById('viewer3d-note');
    const quickPreviewSelect = document.getElementById('quick-preview-select');
    const conformerModelSelect = document.getElementById('conformer-model-select');
    const conformerSearchBtn = document.getElementById('conformer-search-btn');
    const conformerSearchNote = document.getElementById('conformer-search-note');
    const conformerListTable = document.getElementById('conformer-list-table');
    const conformerListBody = document.getElementById('conformer-list-body');
    const progressWrap = document.getElementById('viewer3d-progress');
    const progressFill = document.getElementById('viewer3d-progress-fill');
    const progressNote = document.getElementById('viewer3d-progress-note');
    const measureDistanceBtn = document.getElementById('measure-distance-btn');
    const clearMeasurementsBtn = document.getElementById('clear-measurements-btn');
    const measureDistanceStatus = document.getElementById('measure-distance-status');
    const shape3dTable = document.getElementById('shape3d-table');
    const shape3dOutput = document.getElementById('shape3d-output');
    const shape3dNote = document.getElementById('shape3d-note');

    // Non-planarity / 3D shape descriptors -- "unlocked" (shown) only
    // once a real 3D structure exists, since PBF/NPR are meaningless on
    // the 2D-seeded z=0-ish placeholder buildInitial3D alone produces
    // before any relaxation. Uses ALL atoms including implicit H
    // (matching RDKit's own CalcPBF convention -- see molecular-shape.js).
    function renderShapeMetrics(atoms) {
      if (!atoms || atoms.length < 3) {
        shape3dTable.style.display = 'none';
        shape3dNote.style.display = '';
        shape3dNote.textContent = 'Generate a 3D structure to see plane-of-best-fit and shape descriptors.';
        return;
      }
      const pbf = CC.Shape.planeOfBestFit(atoms);
      const npr = CC.Shape.principalMomentRatios(atoms);
      shape3dOutput.innerHTML = '';
      [
        ['Plane of best fit (PBF)', pbf.toFixed(3) + ' Å'],
        ['NPR1 (rod ↔ disc/sphere)', npr.npr1.toFixed(3)],
        ['NPR2 (disc ↔ sphere)', npr.npr2.toFixed(3)],
      ].forEach(function (pair) {
        const tr = document.createElement('tr');
        const td1 = document.createElement('td'); td1.textContent = pair[0];
        const td2 = document.createElement('td'); td2.textContent = pair[1];
        tr.appendChild(td1); tr.appendChild(td2);
        shape3dOutput.appendChild(tr);
      });
      shape3dTable.style.display = '';
      shape3dNote.style.display = 'none';
    }

    let lastInitial = null; // CC.buildInitial3D() result for the quick-preview "Classical seed" path -- CC.ConformerSearch.run builds its own internally, so this is only needed here for the quick-preview button.
    let currentGeometry = null; // {atoms, bonds} of whatever conformer is currently rendered/selected.
    let currentGeometryOptimized = false; // whether currentGeometry has been through a real energy-model optimization (vs. just a raw seed) -- SASA and similar panels check this via getCurrent3DGeometry.
    let currentGeometryConverged = false; // whether that optimization pass actually settled (vs. just hit its time/iteration budget).

    function renderResult(result) {
      viewer3d = window.chemCanvasLibs && window.chemCanvasLibs.viewer3d;
      if (viewer3d) CC.render3D(viewer3d, result);
      currentGeometry = { atoms: result.atoms, bonds: result.bonds };
      renderShapeMetrics(result.atoms);
      // CC.render3D just cleared every drawn measurement (the old ones
      // referred to atom positions that no longer exist) -- measure mode
      // itself stays on if it was on (re-applied inside CC.render3D), but
      // any distance shown as text here is now stale, so clear it too.
      if (measureDistanceStatus) measureDistanceStatus.textContent = measureModeOn ? 'Click two atoms in the 3D view…' : '';
    }

    function resetConformerList() {
      conformerListTable.style.display = 'none';
      conformerListBody.innerHTML = '';
    }

    // Renders the ensemble table and selects (views) `idx` within it --
    // one function so both the initial "auto-view the top conformer" call
    // and a later "View" button click go through the same selection/
    // button-relabeling logic.
    function selectConformer(idx, result) {
      const c = result.conformers[idx];
      renderResult(c);
      currentGeometryOptimized = true;
      currentGeometryConverged = c.converged;
      Array.from(conformerListBody.children).forEach(function (tr, i) {
        const btn = tr.children[3].firstChild;
        btn.textContent = i === idx ? 'Viewing' : 'View';
        btn.disabled = i === idx;
      });
    }

    function renderConformerList(result) {
      conformerListBody.innerHTML = '';
      result.conformers.forEach(function (c, idx) {
        const tr = document.createElement('tr');
        const rankCell = document.createElement('td'); rankCell.textContent = c.rank;
        const energyCell = document.createElement('td'); energyCell.textContent = (c.relativeEnergyKcal >= 0 ? '+' : '') + c.relativeEnergyKcal.toFixed(2);
        const convergedCell = document.createElement('td'); convergedCell.textContent = c.converged ? 'yes' : 'no';
        const viewCell = document.createElement('td');
        const viewBtn = document.createElement('button');
        viewBtn.className = 'btn btn-ghost btn-small';
        viewBtn.type = 'button';
        viewBtn.addEventListener('click', function () { selectConformer(idx, result); });
        viewCell.appendChild(viewBtn);
        tr.appendChild(rankCell); tr.appendChild(energyCell); tr.appendChild(convergedCell); tr.appendChild(viewCell);
        conformerListBody.appendChild(tr);
      });
      conformerListTable.style.display = '';
      selectConformer(0, result);
    }

    // Distance measurement and the quick-preview/conformer-search buttons
    // just need real 3D coordinates on screen to be usable, not an
    // optimized structure.
    function updateButtonState() {
      const hasGeometry = !!(currentGeometry && currentGeometry.atoms && currentGeometry.atoms.length > 0);
      measureDistanceBtn.disabled = !hasGeometry;
      clearMeasurementsBtn.disabled = !hasGeometry;
    }
    notifyAni2xModelsChanged = updateButtonState; // kept as the same hook name other code already calls after any model finishes loading elsewhere

    let measureModeOn = false;

    function setMeasureButtonVisualState() {
      measureDistanceBtn.className = measureModeOn ? 'btn btn-accent btn-small' : 'btn btn-ghost btn-small';
      measureDistanceBtn.textContent = measureModeOn ? 'Measuring… (click two atoms)' : 'Measure distance';
    }

    // Called when the underlying 3D structure goes away entirely
    // (invalidate3DView, on a 2D edit) -- unlike a mere regenerate/
    // reoptimize (handled inside renderResult above, which keeps measure
    // mode on across a re-render), there's nothing left to click on here,
    // so this actually turns measure mode off rather than just clearing
    // stale text. Explicitly tells viewer3d.js too (not just this
    // panel's own toggle boolean) so the two stay in sync -- otherwise a
    // later "Preview" click would silently come back up with
    // atoms still clickable even though this button shows "off".
    function resetMeasureUI() {
      measureModeOn = false;
      setMeasureButtonVisualState();
      measureDistanceStatus.textContent = '';
      CC.Viewer3D.setMeasureMode(viewer3d, false);
    }

    measureDistanceBtn.addEventListener('click', function () {
      measureModeOn = !measureModeOn;
      setMeasureButtonVisualState();
      CC.Viewer3D.setMeasureMode(viewer3d, measureModeOn, function (result) {
        measureDistanceStatus.textContent = result.element1 + '–' + result.element2 + ': ' +
          result.distanceAngstrom.toFixed(2) + ' Å';
      });
      measureDistanceStatus.textContent = measureModeOn ? 'Click two atoms in the 3D view…' : '';
    });

    clearMeasurementsBtn.addEventListener('click', function () {
      CC.Viewer3D.clearMeasurements(viewer3d);
      measureDistanceStatus.textContent = measureModeOn ? 'Click two atoms in the 3D view…' : '';
    });

    generate3dBtn.addEventListener('click', async function () {
      if (controller.molecule.isEmpty()) {
        viewer3dNote.textContent = 'Nothing to preview yet \u2014 draw a structure first.';
        return;
      }

      resetConformerList();
      lastInitial = CC.buildInitial3D(controller.molecule);
      currentGeometryOptimized = false;
      currentGeometryConverged = false;

      if (quickPreviewSelect.value === 'geomol') {
        const compatibility = CC.GeoMol.checkCompatibility(controller.molecule);
        if (!compatibility.compatible) {
          viewer3dNote.textContent = 'Cannot use GeoMol: ' + compatibility.issues.join('; ') + '. Showing the classical seed instead.';
          renderResult(lastInitial);
          updateButtonState();
          return;
        }

        generate3dBtn.disabled = true;
        conformerSearchBtn.disabled = true;
        quickPreviewSelect.disabled = true;

        // Load-on-click, same pattern as the conformer-search energy
        // models below: the model (~1MB) loads automatically the first
        // time it's needed instead of requiring a separate trip to the
        // Properties panel first.
        if (CC.GeoMol.getLoadedModelIds().length === 0) {
          const entry = CC.GNN.getRegistryEntries().find(function (e) { return (e.engine || 'chemprop') === 'geomol'; });
          if (!entry) {
            viewer3dNote.textContent = 'No GeoMol model entry found in the model registry. Showing the classical seed instead.';
            renderResult(lastInitial);
            generate3dBtn.disabled = false;
            conformerSearchBtn.disabled = false;
            quickPreviewSelect.disabled = false;
            updateButtonState();
            return;
          }
          progressWrap.style.display = '';
          progressFill.style.width = '0%';
          progressNote.textContent = 'Loading GeoMol model\u2026';
          try {
            await CC.GNN.loadRegistryModel(entry.id);
            refreshRegistryList();
          } catch (err) {
            viewer3dNote.textContent = 'Failed to load GeoMol model: ' + err.message + '. Showing the classical seed instead.';
            console.error('[ChemCanvas] GeoMol model load failed', err);
            renderResult(lastInitial);
            progressWrap.style.display = 'none';
            generate3dBtn.disabled = false;
            conformerSearchBtn.disabled = false;
            quickPreviewSelect.disabled = false;
            updateButtonState();
            return;
          }
        }

        progressWrap.style.display = '';
        progressFill.style.width = '100%';
        progressNote.textContent = 'Predicting conformer with GeoMol\u2026';
        try {
          const modelId = CC.GeoMol.getLoadedModelIds()[0];
          const result = CC.GeoMol.generateConformer(controller.molecule, modelId);
          renderResult(result);
          viewer3dNote.textContent = result.atoms.length + ' atoms (incl. implicit H), generated with GeoMol \u2014 a single learned prediction, not an energy-minimized structure. Run a conformer search below to relax/rank real conformers, or click "Preview" again for a different sampled conformer.';
          CC.Logger.success('Quick preview: generated 3D structure with GeoMol (' + result.atoms.length + ' atoms incl. implicit H)');
        } catch (err) {
          viewer3dNote.textContent = 'GeoMol prediction failed: ' + err.message + '. Showing the classical seed instead.';
          console.error('[ChemCanvas] GeoMol conformer generation failed', err);
          CC.Logger.error('GeoMol conformer generation failed: ' + err.message);
          renderResult(lastInitial);
        } finally {
          progressWrap.style.display = 'none';
          generate3dBtn.disabled = false;
          conformerSearchBtn.disabled = false;
          quickPreviewSelect.disabled = false;
          updateButtonState();
        }
        return;
      }

      renderResult(lastInitial);
      viewer3dNote.textContent = lastInitial.atoms.length + ' atoms (incl. implicit H), classical seed \u2014 not optimized. Run a conformer search below to relax/rank real conformers.';
      CC.Logger.success('Quick preview (classical seed, ' + lastInitial.atoms.length + ' atoms incl. implicit H)');
      updateButtonState();
    });

    // Called on every 2D edit (see runValidation) -- a 3D structure or
    // optimized geometry for whatever the molecule *used* to be is worse
    // than no 3D view at all, since nothing on screen would indicate it's
    // now out of sync with the 2D structure you're actually looking at.
    invalidate3DView = function () {
      if (!lastInitial && !currentGeometry) return; // nothing generated yet -- nothing to invalidate
      lastInitial = null;
      currentGeometry = null;
      currentGeometryOptimized = false;
      currentGeometryConverged = false;
      if (viewer3d) CC.render3D(viewer3d, { atoms: [], bonds: [] });
      viewer3dNote.textContent = 'Structure has changed \u2014 click "Preview" or run a conformer search to update.';
      conformerSearchNote.textContent = '';
      resetConformerList();
      resetMeasureUI();
      renderShapeMetrics(null); // re-locks the 3D shape section until a fresh structure exists
      updateButtonState();
    };

    // Runs the selected energy model's conformer search (see
    // js/conformer-search.js) -- loads whatever model-specific weights it
    // needs (NAGL-MBIS for smirnoff, ANI-2x for ani2x) on demand first,
    // same load-on-click pattern this panel's GeoMol quick-preview above
    // already uses, rather than requiring a separate trip to the
    // Properties panel first.
    conformerSearchBtn.addEventListener('click', async function () {
      if (controller.molecule.isEmpty()) {
        conformerSearchNote.textContent = 'Nothing to search yet \u2014 draw a structure first.';
        return;
      }

      const model = conformerModelSelect.value;
      const modelLabelForLog = conformerModelSelect.options[conformerModelSelect.selectedIndex].textContent;

      generate3dBtn.disabled = true;
      conformerSearchBtn.disabled = true;
      conformerModelSelect.disabled = true;
      progressWrap.style.display = '';
      progressFill.style.width = '0%';
      progressNote.textContent = 'Preparing ' + modelLabelForLog + '\u2026';
      resetConformerList();
      conformerSearchNote.textContent = '';

      try {
        const opts = {
          energyModel: model,
          onProgress: function (info) {
            const pct = Math.round(((info.seed - 1) / info.totalSeeds) * 100);
            progressFill.style.width = pct + '%';
            if (info.phase === 'seed-done') {
              progressNote.textContent = 'Seed ' + info.seed + '/' + info.totalSeeds + ' done (energy ' + info.energyKcal.toFixed(2) + ' kcal/mol)';
            } else {
              progressNote.textContent = 'Seed ' + info.seed + '/' + info.totalSeeds + ' \u2014 ' + info.stage;
            }
          },
        };

        if (model === 'smirnoff') {
          // A missing/failed NAGL load is NOT fatal here: CC.ConformerSearch
          // just omits electrostatics rather than refusing to run (same
          // honest fallback OPENFF_INTEGRATION.md documents).
          if (!CC.OpenFF.isForceFieldLoaded()) {
            progressNote.textContent = 'Loading OpenFF Sage force field\u2026';
            await CC.OpenFF.loadForceField();
          }
          let naglModelId = CC.NAGL.getLoadedModelIds()[0];
          if (!naglModelId) {
            const entry = CC.GNN.getRegistryEntries().find(function (e) { return e.engine === 'nagl'; });
            if (entry) {
              progressNote.textContent = 'Loading NAGL-MBIS charge model\u2026';
              try {
                await CC.GNN.loadRegistryModel(entry.id);
                refreshRegistryList();
                naglModelId = CC.NAGL.getLoadedModelIds()[0];
              } catch (err) {
                console.error('[ChemCanvas] NAGL model load failed before conformer search \u2014 continuing without electrostatics', err);
              }
            }
          }
          opts.naglModelId = naglModelId;
        } else if (model === 'ani2x') {
          const compatibility = CC.ANI.checkCompatibility(controller.molecule);
          if (!compatibility.compatible) {
            throw new Error('Cannot use ANI-2x: ' + compatibility.issues.join('; '));
          }
          let aniModelId = CC.ANI.getLoadedModelIds()[0];
          if (!aniModelId) {
            const entry = CC.GNN.getRegistryEntries().find(function (e) { return (e.engine || 'chemprop') === 'ani2x'; });
            if (!entry) throw new Error('No ANI-2x model entry found in the model registry.');
            progressNote.textContent = 'Loading ANI-2x model\u2026';
            await CC.GNN.loadRegistryModel(entry.id);
            refreshRegistryList();
            aniModelId = CC.ANI.getLoadedModelIds()[0];
          }
          opts.aniModelId = aniModelId;
        }

        progressNote.textContent = 'Generating seed geometries\u2026';
        const result = await CC.ConformerSearch.run(controller.molecule, opts);
        progressFill.style.width = '100%';

        if (result.conformers.length === 0) {
          conformerSearchNote.textContent = 'No conformers found (' + result.seedsGenerated + ' seed(s) generated, ' +
            result.seedsOptimized + ' optimized) \u2014 try again, or a different energy model.';
          CC.Logger.warning('Conformer search (' + modelLabelForLog + '): no conformers found');
        } else {
          renderConformerList(result);
          const chargeNote = (model === 'smirnoff' && !result.chargesAvailable)
            ? ' (no NAGL-MBIS charges loaded \u2014 electrostatics omitted)' : '';
          conformerSearchNote.textContent = result.modelLabel + ': ' + result.seedsGenerated + ' seed(s) generated, ' +
            result.seedsOptimized + ' optimized, ' + result.conformersWithinWindow + ' within the 6 kcal/mol energy window, ' +
            result.conformers.length + ' distinct conformer(s) kept' + chargeNote + '. Energies in ' + result.energyUnit + '.';
          CC.Logger.success('Conformer search (' + result.modelLabel + '): ' + result.conformers.length +
            ' conformer(s), best ' + result.conformers[0].energy.toFixed(2) + ' ' + result.energyUnit);
        }
      } catch (err) {
        conformerSearchNote.textContent = 'Conformer search failed: ' + err.message;
        console.error('[ChemCanvas] conformer search failed', err);
        CC.Logger.error('Conformer search failed: ' + err.message);
      } finally {
        generate3dBtn.disabled = false;
        conformerSearchBtn.disabled = false;
        conformerModelSelect.disabled = false;
        progressWrap.style.display = 'none';
        updateButtonState();
      }
    });

    updateButtonState();

    // currentGeometry is reliably null'd out by invalidate3DView() on every
    // 2D edit (see above), so any non-null value here is guaranteed to
    // still match the molecule currently on the canvas -- safe for another
    // panel (SASA) to reuse without re-deriving/re-checking that itself.
    getCurrent3DGeometry = function () {
      if (!currentGeometry || !currentGeometry.atoms || currentGeometry.atoms.length === 0) return null;
      return {
        atoms: currentGeometry.atoms,
        bonds: currentGeometry.bonds,
        optimized: currentGeometryOptimized,
        converged: currentGeometryConverged,
      };
    };
  }

  // ---------- GNN prediction (demo D-MPNN + optional ONNX model) ----------

  let lastAtomProperties = null; // atomId -> {propName: value}, for heatmap re-render
  let lastBondProperties = null; // bondId -> {propName: value}, for bond heatmap re-render
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
    const bondHeatmapRow = document.getElementById('gnn-bond-heatmap-row');
    const bondHeatmapSelect = document.getElementById('bond-heatmap-property-select');
    const clearBondHeatmapBtn = document.getElementById('clear-bond-heatmap-btn');
    const computeSasaBtn = document.getElementById('compute-sasa-btn');
    const sasaStatus = document.getElementById('sasa-status');

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
          const nMol = Object.keys(result.molecularProperties || {}).length;
          const nAtom = (result.atomProperties || []).length;
          const nBond = (result.bondProperties || []).length;
          let msg = 'Ran prediction: ' + nMol + ' molecular, ' + nAtom + ' atom-level, ' + nBond + ' bond-level propert' + (nMol + nAtom + nBond === 1 ? 'y' : 'ies');
          if (result.warnings && result.warnings.length) {
            CC.Logger.warning(msg + ' — ' + result.warnings.length + ' model warning(s): ' + result.warnings.join('; '));
          } else {
            CC.Logger.success(msg);
          }
        })
        .catch(function (err) {
          showGnnMessage('Prediction failed: ' + err.message);
          console.error('[ChemCanvas] GNN prediction failed', err);
          CC.Logger.error('GNN prediction failed: ' + err.message);
        })
        .finally(function () {
          runDemoBtn.disabled = false;
        });
    }

    // Merges a batch of per-atom properties (GNN/NAGL/pKa atom-level
    // results, or the SASA button's own results) into the SAME
    // atomId -> {propName: value} map the atom heatmap reads, ADDING to
    // whatever's already there for each atom rather than replacing it --
    // so e.g. NAGL charges computed via "Run prediction" and steric
    // accessibility computed via the separate "Compute (SASA)" button
    // both stay selectable in the same dropdown regardless of which was
    // run first. Rebuilds the dropdown from the union of every key
    // across every atom (not just atom 0 -- a sparse property, e.g.
    // shift_19f present only on F atoms, can easily be absent from
    // whichever atom happens to be first), preserving the current
    // selection if it's still a valid option.
    function mergeAtomHeatmapProperties(atomIds, atomProperties) {
      if (!lastAtomProperties) lastAtomProperties = {};
      atomIds.forEach(function (atomId, i) {
        lastAtomProperties[atomId] = Object.assign(lastAtomProperties[atomId] || {}, atomProperties[i]);
      });

      const propNameSet = {};
      Object.keys(lastAtomProperties).forEach(function (atomId) {
        Object.keys(lastAtomProperties[atomId]).forEach(function (name) { propNameSet[name] = true; });
      });
      // aqueous-pka predicts a value at EVERY heavy atom (per its own
      // registry description), but most of those aren't chemically
      // meaningful -- only the specific atoms js/pka-microstates.js's
      // SMARTS detector identifies as real ionizable sites are, and
      // those already get a correct, filtered display in the Titration
      // tab. Showing the raw per-atom "pka" property here as a
      // selectable heatmap just surfaces noise (a real report: a user
      // couldn't tell what it meant, since most colored atoms have no
      // real pKa). Excluded from the dropdown only -- still kept in
      // lastAtomProperties itself for CSV/PDF/SDF export, where a power
      // user exporting raw data can reasonably want it with context.
      // pka-ch (js/pka-model.js) is NOT excluded: it already restricts
      // itself to candidate C-H sites rather than every atom.
      const propNames = Object.keys(propNameSet).filter(function (name) { return name !== 'pka'; });
      const previousSelection = heatmapSelect.value;
      heatmapSelect.innerHTML = '';
      propNames.forEach(function (name) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        heatmapSelect.appendChild(opt);
      });
      if (propNames.indexOf(previousSelection) !== -1) heatmapSelect.value = previousSelection;

      heatmapRow.style.display = '';
      applyHeatmap();
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
        bondHeatmapRow.style.display = 'none';
        lastBondProperties = null;
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

        // Info popup: only shown for registry-loaded models, since an
        // ad-hoc "load from files" model has no dataset/training/notes
        // metadata to show in the first place -- nothing to click into.
        if (registryEntry) {
          const infoBtn = document.createElement('button');
          infoBtn.type = 'button';
          infoBtn.className = 'property-info-btn';
          infoBtn.setAttribute('aria-label', 'About this property');
          infoBtn.textContent = '[?]';
          infoBtn.addEventListener('click', function () {
            openPropertyInfoModal(registryEntry.displayName, buildPropertyInfoBox(registryEntry));
          });
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
        // HOMO-LUMO gap: also show the photon-energy-equivalent
        // wavelength (E(eV) = 1239.84198 / lambda(nm), the standard
        // hc constant in eV·nm -- a real physical relation, not a
        // fitted one) alongside the eV value the model directly
        // predicts. This is the wavelength of a photon whose ENERGY
        // matches the frontier-orbital gap -- a useful, standard way to
        // express a HOMO-LUMO gap, but not the same thing as a computed
        // UV-Vis absorption maximum (which includes excited-state
        // relaxation/electron-correlation effects this simple gap
        // doesn't capture) -- the info panel's notes say this too.
        if (registryEntry && registryEntry.propertyKey === 'gap' && value > 0) {
          const nm = 1239.84198 / value;
          const nmSpan = document.createElement('span');
          nmSpan.className = 'property-units';
          nmSpan.title = 'Wavelength of a photon whose energy equals this gap (E[eV] = 1239.84 / λ[nm]) -- not a computed UV-Vis absorption maximum.';
          nmSpan.textContent = ' (≈ ' + nm.toFixed(0) + ' nm)';
          valueCell.appendChild(nmSpan);
        }

        row.appendChild(labelCell);
        row.appendChild(valueCell);
        molOutput.appendChild(row);
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

      // Merge into the SAME atomId -> {propName: value} map the heatmap
      // reads, rather than replacing it outright -- a prior SASA
      // computation (see mergeAtomHeatmapProperties below) is still
      // valid for the current structure even if this particular
      // "Run prediction" click had no atom-level GNN/NAGL/pKa models
      // loaded, so a run with zero atom properties of its own leaves
      // whatever's already there alone instead of hiding it.
      if (hasAtomProperties && result.atomIds) {
        mergeAtomHeatmapProperties(result.atomIds, result.atomProperties);
      }

      // Same idea as the atom heatmap above, but keyed by bondId (e.g.
      // bond dissociation enthalpy) -- this app's first bond-level
      // property, so a molecule with none loaded just hides the row.
      const hasBondProperties = result.bondProperties && result.bondProperties.length > 0;
      if (hasBondProperties && result.bondIds) {
        lastBondProperties = {};
        result.bondIds.forEach(function (bondId, i) {
          lastBondProperties[bondId] = result.bondProperties[i];
        });
        const bondPropNameSet = {};
        result.bondProperties.forEach(function (props) {
          Object.keys(props).forEach(function (name) { bondPropNameSet[name] = true; });
        });
        bondHeatmapSelect.innerHTML = '';
        Object.keys(bondPropNameSet).forEach(function (name) {
          const opt = document.createElement('option');
          opt.value = name;
          opt.textContent = name;
          bondHeatmapSelect.appendChild(opt);
        });
        bondHeatmapRow.style.display = '';
        applyBondHeatmap();
      } else {
        bondHeatmapRow.style.display = 'none';
        lastBondProperties = null;
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
        CC.renderAtomValueLabels(svg, controller.molecule, values, propName);
      } else {
        CC.clearAtomValueLabels(svg);
      }
    }

    function applyBondHeatmap() {
      if (!lastBondProperties || bondHeatmapRow.style.display === 'none') return;
      const propName = bondHeatmapSelect.value;
      const values = {};
      Object.keys(lastBondProperties).forEach(function (bondId) {
        values[bondId] = lastBondProperties[bondId][propName];
      });
      const scale = CC.renderBondHeatmap(svg, controller.molecule, values, propName);
      const legendEl = document.getElementById('bond-heatmap-legend-container');
      if (legendEl) CC.renderHeatmapLegend(legendEl, scale);

      const showTextCheckbox = document.getElementById('bond-heatmap-show-text');
      if (showTextCheckbox && showTextCheckbox.checked) {
        CC.renderBondValueLabels(svg, controller.molecule, values, propName);
      } else {
        CC.clearBondValueLabels(svg);
      }
    }
    reapplyHeatmap = function () { applyHeatmap(); applyBondHeatmap(); };

    // Called on every 2D edit (see runValidation) -- same reasoning as
    // invalidate3DView: a prediction table or atom heatmap for whatever
    // the molecule *used* to be is actively misleading once the
    // structure has changed, not just outdated.
    invalidateGNNResults = function () {
      if (!lastAtomProperties && !lastBondProperties && gnnTable.style.display === 'none') return; // nothing to invalidate
      lastAtomProperties = null;
      lastBondProperties = null;
      lastMolecularProperties = null;
      molOutput.innerHTML = '';
      showGnnMessage('Structure has changed \u2014 click "Run prediction" to update.');
      heatmapRow.style.display = 'none';
      CC.clearAtomHeatmap(svg);
      CC.clearAtomValueLabels(svg);
      const legendEl = document.getElementById('heatmap-legend-container');
      if (legendEl) legendEl.innerHTML = '';
      bondHeatmapRow.style.display = 'none';
      CC.clearBondHeatmap(svg);
      CC.clearBondValueLabels(svg);
      const bondLegendEl = document.getElementById('bond-heatmap-legend-container');
      if (bondLegendEl) bondLegendEl.innerHTML = '';
      // Steric accessibility was computed from a 3D conformer built off
      // the OLD 2D structure -- no longer valid once that's changed
      // (same reasoning invalidate3DView already applies to the 3D tab).
      sasaStatus.textContent = '';
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
    bondHeatmapSelect.addEventListener('change', applyBondHeatmap);
    const bondShowTextCheckbox = document.getElementById('bond-heatmap-show-text');
    if (bondShowTextCheckbox) bondShowTextCheckbox.addEventListener('change', applyBondHeatmap);
    clearBondHeatmapBtn.addEventListener('click', function () {
      CC.clearBondHeatmap(svg);
      CC.clearBondValueLabels(svg);
      const bondLegendEl = document.getElementById('bond-heatmap-legend-container');
      if (bondLegendEl) bondLegendEl.innerHTML = '';
    });

    // Steric accessibility (SASA) -- deliberately its own button, not
    // folded into "Run prediction". Deliberately does NOT build its own
    // 3D structure either (it used to: a from-scratch conformer search
    // via CC.SASA.predict, real find from earlier in this project's own
    // history -- that path was slow (several seconds to tens of
    // seconds) AND its own time budget routinely still wasn't enough to
    // reach real convergence, silently producing unreliable numbers).
    // SASA only ever makes sense on top of a REAL, already-relaxed
    // conformer, and building/refining that is the 3D tab's whole job
    // (with its own convergence chart and "Optimize further" control) --
    // this button now only ever measures whatever's already there,
    // pointing the user to the 3D tab instead of quietly doing that work
    // (badly) itself.
    computeSasaBtn.addEventListener('click', function () {
      if (controller.molecule.isEmpty()) {
        sasaStatus.textContent = 'Draw a structure first.';
        return;
      }

      const existing = getCurrent3DGeometry();
      if (!existing || !existing.optimized) {
        sasaStatus.style.color = 'var(--danger)';
        sasaStatus.textContent = 'No optimized 3D structure yet — go to the 3D view tab and run a conformer search.';
        CC.Logger.warning('SASA: no optimized 3D structure available yet');
        return;
      }

      sasaStatus.textContent = '';
      sasaStatus.style.color = '';
      try {
        const result = CC.SASA.predictFromAtoms(controller.molecule, existing.atoms, {});
        result.converged = existing.converged;
        if (result.atomProperties.length > 0) {
          mergeAtomHeatmapProperties(result.atomIds, result.atomProperties);
        }
        sasaStatus.style.color = result.converged ? '' : 'var(--danger)';
        const convergedNote = result.converged
          ? 'converged'
          : '⚠ that 3D structure did NOT fully converge — the resulting values may be unreliable, not ' +
            'just imprecise (an unrelaxed structure has spurious close contacts between atoms that a real ' +
            'conformer wouldn’t, which artificially lowers exposure everywhere); try "Optimize further…" ' +
            'in the 3D tab first';
        sasaStatus.textContent = 'Computed steric accessibility from the existing ' + result.numAtomsIncludingH +
          '-atom (incl. implicit H) conformer — ' + convergedNote + '. See "steric_accessibility" ' +
          '(0–1, fraction exposed) and "sasa" (Å²) in the atom heatmap above.';
        CC.Logger[result.converged ? 'success' : 'warning']('Computed SASA (' + result.numAtomsIncludingH + ' atoms, ' + (result.converged ? 'converged' : 'did not converge') + ')');
      } catch (err) {
        sasaStatus.textContent = 'Steric accessibility computation failed: ' + err.message;
        console.error('[ChemCanvas] SASA computation failed', err);
        CC.Logger.error('SASA computation failed: ' + err.message);
      }
    });

    function updateRunButtonLabel() {
      const n = CC.GNN.getLoadedChempropModelIds().length;
      runDemoBtn.textContent = n === 0
        ? 'Run prediction (demo weights)'
        : 'Run prediction (' + n + ' model' + (n === 1 ? '' : 's') + ' loaded)';
    }
    updateRunButtonLabel();

    // Purely a UI grouping -- see registry.json's "categories" field (and
    // model-registry.js's schema doc) for what each bucket means. An
    // entry lists ONE OR MORE categories and appears in every matching
    // section -- e.g. melting point shows under both "Environmental &
    // analytical" and "Characterization" on purpose (the user explicitly
    // wants that redundancy, not a single home per model). An entry with
    // no recognized category at all falls back to "general" rather than
    // silently vanishing from the panel entirely (mirrors
    // CC.GNN.loadModelRegistry's own "skip a malformed entry, don't
    // break everything else" philosophy, just applied to a cosmetic
    // field instead of a structural one).
    const REGISTRY_CATEGORIES = [
      { key: 'general', containerId: 'model-registry-list-general' },
      { key: 'characterization', containerId: 'model-registry-list-characterization' },
      { key: 'environmental-analytical', containerId: 'model-registry-list-environmental-analytical' },
      { key: 'medicinal', containerId: 'model-registry-list-medicinal' },
      { key: 'structure-tools', containerId: 'model-registry-list-structure-tools' },
    ];

    function renderRegistryList(entries) {
      // Computed once per render (not once per entry -- checkModelCompatibility
      // already covers every entry in one pass) so each row can show a
      // compact tier badge without gating the Load button itself: loading
      // weights is harmless regardless of the CURRENT molecule (you might
      // load a model now and draw a compatible structure later) -- the
      // Validation tab is the authoritative place to see WHY.
      let compatByModelId = null;
      if (window.CC.Validate && controller && !controller.molecule.isEmpty() && lastStructureReport) {
        try {
          compatByModelId = {};
          CC.Validate.checkModelCompatibility(controller.molecule, lastStructureReport).forEach(function (c) {
            compatByModelId[c.id] = c;
          });
        } catch (err) {
          compatByModelId = null;
        }
      }

      const byCategory = {};
      REGISTRY_CATEGORIES.forEach(function (c) { byCategory[c.key] = []; });
      entries.forEach(function (entry) {
        const cats = (entry.categories || []).filter(function (c) { return byCategory[c]; });
        (cats.length ? cats : ['general']).forEach(function (key) {
          byCategory[key].push(entry);
        });
      });

      REGISTRY_CATEGORIES.forEach(function (c) {
        const container = document.getElementById(c.containerId);
        if (!container) return;
        container.innerHTML = '';
        const categoryEntries = byCategory[c.key];
        if (categoryEntries.length === 0) {
          container.innerHTML = '<p class="side-panel-note">No models in this category yet.</p>';
          return;
        }

        categoryEntries.forEach(function (entry) {
          const row = document.createElement('div');
          row.className = 'model-registry-row';

          const info = document.createElement('div');
          info.className = 'model-registry-info';
          const nameEl = document.createElement('div');
          nameEl.className = 'model-registry-name';
          nameEl.textContent = entry.displayName;
          const infoBtn = document.createElement('button');
          infoBtn.type = 'button';
          infoBtn.className = 'property-info-btn';
          infoBtn.setAttribute('aria-label', 'About this model');
          infoBtn.textContent = '[?]';
          infoBtn.addEventListener('click', function () {
            openPropertyInfoModal(entry.displayName, buildPropertyInfoBox(entry));
          });
          nameEl.appendChild(infoBtn);
          const compat = compatByModelId && compatByModelId[entry.id];
          if (compat && compat.tier !== 'compatible') {
            const tierBadge = document.createElement('span');
            tierBadge.className = 'tier-badge tier-' + compat.tier;
            tierBadge.textContent = TIER_LABEL[compat.tier] || compat.tier;
            tierBadge.title = compat.reasons.join('; ');
            nameEl.appendChild(tierBadge);
          }
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
            CC.Logger.info('Loading model: ' + entry.displayName + '\u2026');
            CC.GNN.loadRegistryModel(entry.id)
              .then(function () {
                // A full re-render, not just mutating this one button --
                // an entry listed under more than one category (see
                // REGISTRY_CATEGORIES above) has a SEPARATE row/button
                // per section, and every copy needs to flip to "Loaded"
                // together, not just whichever one was actually clicked.
                refreshRegistryList();
                updateRunButtonLabel();
                notifyAni2xModelsChanged();
                CC.Logger.success('Loaded model: ' + entry.displayName);
              })
              .catch(function (err) {
                btn.textContent = 'Load';
                btn.disabled = false;
                console.error('[ChemCanvas] Failed to load registry model "' + entry.id + '"', err);
                CC.Logger.error('Failed to load model "' + entry.displayName + '": ' + err.message);
                const errEl = document.createElement('div');
                errEl.className = 'model-registry-error';
                errEl.textContent = 'Failed: ' + err.message;
                row.appendChild(errEl);
              });
          });
          row.appendChild(btn);

          container.appendChild(row);
        });
      });
    }

    // Lets other panels (the 3D view's conformer-search auto-load,
    // see setup3DPanel) re-render this list after loading a model from
    // outside this panel, so its "Load"/"Loaded" button state stays honest.
    // Also re-renders the Validation tab's compatibility table: every
    // call site that calls refreshRegistryList() does so because a
    // model just finished loading/unloading, which is exactly the other
    // state that table depends on (structural issues alone don't
    // change) -- one shared refresh point instead of adding a second
    // refreshValidationPanel() call at each of those sites individually.
    refreshRegistryList = function () {
      renderRegistryList(CC.GNN.getRegistryEntries());
      refreshValidationPanel();
    };

    // The registry (model/registry.json by default -- see model-config.js)
    // is fetched once at startup; individual models' weights are only
    // fetched when their row's Load button is clicked (load-on-demand,
    // per this project's earlier load-all-vs-on-demand decision).
    const registryStatus = document.getElementById('registry-status');
    CC.GNN.loadModelRegistry((CC.CONFIG && CC.CONFIG.registryUrl) || 'model/registry.json')
      .then(function (entries) {
        registryStatus.textContent = '';
        renderRegistryList(entries);
        refreshValidationPanel(); // registry just arrived -- the model-compatibility table couldn't be filled in until now
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

  // ---------- Implicit solvation (GB/SA) ----------

  // Cross-checks that the current 3D geometry's own implicit-hydrogen
  // bookkeeping (per heavy atom) matches the MBIS model's expanded-graph
  // bookkeeping for the SAME molecule, before zipping their two arrays
  // together position-by-position. They come from independently-derived
  // H-perception paths that usually agree for ordinary neutral organics
  // but aren't guaranteed to: embed3d.js's classical force field uses a
  // simple valence-table heuristic (elements.js), GeoMol uses RDKit's own
  // AddHs, and the MBIS model uses RDKit's own annotations -- a charged,
  // hypervalent, or otherwise unusual atom could give the classical path
  // a different implicit-H count than RDKit's. Silently zipping mismatched
  // arrays together would misassign every charge after the first
  // divergence rather than just failing loudly, so this is checked, not
  // assumed (same "don't silently produce an unreliable number" reasoning
  // steric-accessibility.js's own header documents for SASA).
  function hCountsMatchNaglCharges(numHeavyAtoms, atoms3D, bonds3D, naglResult) {
    if (atoms3D.length !== naglResult.charges.length) return false;

    const from3D = new Array(numHeavyAtoms).fill(0);
    bonds3D.forEach(function (b) {
      if (b.a1 < numHeavyAtoms && b.a2 >= numHeavyAtoms) from3D[b.a1]++;
      else if (b.a2 < numHeavyAtoms && b.a1 >= numHeavyAtoms) from3D[b.a2]++;
    });

    const fromNagl = new Array(numHeavyAtoms).fill(0);
    for (let h = numHeavyAtoms; h < naglResult.elements.length; h++) {
      const neighbors = naglResult.adjacency[h] || [];
      if (neighbors.length === 1 && neighbors[0] < numHeavyAtoms) fromNagl[neighbors[0]]++;
    }

    for (let i = 0; i < numHeavyAtoms; i++) {
      if (from3D[i] !== fromNagl[i]) return false;
    }
    return true;
  }

  function setupSolventPanel() {
    const enableCheckbox = document.getElementById('solvent-enable-checkbox');
    const controls = document.getElementById('solvent-controls');
    const solventSelect = document.getElementById('solvent-select');
    const customRow = document.getElementById('solvent-custom-row');
    const customEpsInput = document.getElementById('solvent-custom-eps');
    const computeBtn = document.getElementById('compute-solvent-btn');
    const statusEl = document.getElementById('solvent-status');
    const outputTable = document.getElementById('solvent-output-table');
    const outputBody = document.getElementById('solvent-output');

    CC.Solvent.SOLVENTS.forEach(function (s) {
      const opt = document.createElement('option');
      opt.value = s.id;
      opt.textContent = s.name + ' (ε=' + s.eps + ')';
      solventSelect.appendChild(opt);
    });
    const customOpt = document.createElement('option');
    customOpt.value = 'custom';
    customOpt.textContent = 'Custom…';
    solventSelect.appendChild(customOpt);

    enableCheckbox.addEventListener('change', function () {
      controls.style.display = enableCheckbox.checked ? '' : 'none';
    });

    solventSelect.addEventListener('change', function () {
      customRow.style.display = solventSelect.value === 'custom' ? '' : 'none';
    });

    function selectedEps() {
      if (solventSelect.value === 'custom') return parseFloat(customEpsInput.value);
      const entry = CC.Solvent.SOLVENTS.find(function (s) { return s.id === solventSelect.value; });
      return entry ? entry.eps : NaN;
    }

    function showError(text) {
      statusEl.style.color = 'var(--danger)';
      statusEl.textContent = text;
      outputTable.style.display = 'none';
    }

    function renderResult(result, converged) {
      statusEl.style.color = converged ? '' : 'var(--danger)';
      statusEl.textContent = converged
        ? ''
        : '⚠ the underlying 3D structure did NOT fully converge — this energy may be unreliable, not just imprecise; try "Optimize further…" in the 3D tab first.';

      outputBody.innerHTML = '';
      const rows = [
        ['Solvent dielectric (ε)', result.epsSolvent.toFixed(2)],
        ['Polar (GB)', result.polar.toFixed(2) + ' kcal/mol'],
        ['Nonpolar (SASA, γ=0.0072 kcal/mol/Å²)', result.nonpolar.toFixed(2) + ' kcal/mol'],
        ['Total ΔG (solvation)', result.total.toFixed(2) + ' kcal/mol'],
        ['Total SASA', result.totalSasa.toFixed(1) + ' Å²'],
      ];
      rows.forEach(function (pair) {
        const row = document.createElement('tr');
        const labelCell = document.createElement('td');
        labelCell.textContent = pair[0];
        const valueCell = document.createElement('td');
        valueCell.textContent = pair[1];
        row.appendChild(labelCell);
        row.appendChild(valueCell);
        outputBody.appendChild(row);
      });
      outputTable.style.display = '';
    }

    computeBtn.addEventListener('click', function () {
      const mol = controller.molecule;
      if (mol.isEmpty()) { showError('Draw a structure first.'); return; }

      const existing = getCurrent3DGeometry();
      if (!existing || !existing.optimized) {
        showError('No optimized 3D structure yet — go to the 3D view tab and run a conformer search.');
        return;
      }

      const naglIds = CC.NAGL.getLoadedModelIds();
      if (naglIds.length === 0) {
        showError('No MBIS partial-charge model loaded — load "MBIS partial charges (per atom)" above first.');
        return;
      }

      const epsSolvent = selectedEps();
      if (!(epsSolvent >= 1)) { showError('Dielectric constant must be a number ≥ 1.'); return; }

      let naglResult;
      try {
        naglResult = CC.NAGL.predictAll(mol, naglIds[0]);
      } catch (err) {
        showError('MBIS charge prediction failed: ' + err.message);
        return;
      }

      if (!hCountsMatchNaglCharges(mol.atoms.size, existing.atoms, existing.bonds, naglResult)) {
        showError('This 3D structure’s implicit hydrogens don’t line up with the MBIS model’s own ' +
          'hydrogen count for the same molecule — try regenerating the 3D structure with the other 3D ' +
          'generation method (3D view tab) and running MBIS again.');
        return;
      }

      try {
        const result = CC.Solvent.predict(existing.atoms, naglResult.charges, epsSolvent);
        renderResult(result, existing.converged);
        CC.Logger.success('Computed implicit solvation energy (ε=' + epsSolvent.toFixed(1) + '): ' +
          'ΔG = ' + result.total.toFixed(2) + ' kcal/mol');
      } catch (err) {
        showError('Solvation energy computation failed: ' + err.message);
        console.error('[ChemCanvas] implicit solvent computation failed', err);
      }
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

  // App-wide event log console -- pure rendering + download here;
  // CC.Logger (js/logger.js) owns the actual entries and is what every
  // other file calls into. Backfills whatever was already logged before
  // this ran (RDKit/ONNX/3Dmol readiness from lib-loader.js fire well
  // before DOMContentLoaded's setup functions get here), then subscribes
  // for everything after.
  function setupLogConsole() {
    const toggleBtn = document.getElementById('log-toggle-btn');
    const toggleDot = toggleBtn.querySelector('.status-dot');
    const toggleSummary = document.getElementById('log-toggle-summary');
    const consolePanel = document.getElementById('log-console');
    const consoleBody = document.getElementById('log-console-body');
    const consoleSummary = document.getElementById('log-console-summary');
    const clearBtn = document.getElementById('log-clear-btn');
    const downloadBtn = document.getElementById('log-download-btn');
    const closeBtn = document.getElementById('log-close-btn');

    function summaryText(entries) {
      const n = entries.length;
      const errors = entries.filter(function (e) { return e.level === 'error'; }).length;
      const warnings = entries.filter(function (e) { return e.level === 'warning'; }).length;
      let text = n + ' ' + (n === 1 ? 'entry' : 'entries');
      if (errors) text += ' · ' + errors + ' error' + (errors === 1 ? '' : 's');
      else if (warnings) text += ' · ' + warnings + ' warning' + (warnings === 1 ? '' : 's');
      return text;
    }

    function updateSummary() {
      const entries = CC.Logger.getEntries();
      const text = summaryText(entries);
      toggleSummary.textContent = text;
      consoleSummary.textContent = text;
      const hasError = entries.some(function (e) { return e.level === 'error'; });
      const hasWarning = entries.some(function (e) { return e.level === 'warning'; });
      toggleDot.classList.remove('is-error', 'is-warning', 'is-ready');
      if (hasError) toggleDot.classList.add('is-error');
      else if (hasWarning) toggleDot.classList.add('is-warning');
      else if (entries.length) toggleDot.classList.add('is-ready');
    }

    function isScrolledNearBottom() {
      return consoleBody.scrollHeight - consoleBody.scrollTop - consoleBody.clientHeight < 40;
    }

    function appendEntryRow(entry) {
      const wasNearBottom = isScrolledNearBottom();
      const row = document.createElement('div');
      row.className = 'log-entry log-entry-' + entry.level;
      const timeEl = document.createElement('span');
      timeEl.className = 'log-entry-time';
      timeEl.textContent = CC.Logger.formatTime(entry.time);
      const levelEl = document.createElement('span');
      levelEl.className = 'log-entry-level';
      levelEl.textContent = entry.level.toUpperCase();
      const messageEl = document.createElement('span');
      messageEl.className = 'log-entry-message';
      messageEl.textContent = entry.message;
      row.appendChild(timeEl);
      row.appendChild(levelEl);
      row.appendChild(messageEl);
      consoleBody.appendChild(row);
      if (wasNearBottom) consoleBody.scrollTop = consoleBody.scrollHeight;
    }

    function renderAll() {
      consoleBody.innerHTML = '';
      const entries = CC.Logger.getEntries();
      if (entries.length === 0) {
        consoleBody.innerHTML = '<p class="log-console-empty">No log entries yet.</p>';
      } else {
        entries.forEach(appendEntryRow);
        consoleBody.scrollTop = consoleBody.scrollHeight;
      }
      updateSummary();
    }

    renderAll();
    CC.Logger.subscribe(function (entry) {
      if (entry === null) {
        renderAll(); // clear()
        return;
      }
      const emptyNote = consoleBody.querySelector('.log-console-empty');
      if (emptyNote) emptyNote.remove();
      appendEntryRow(entry);
      updateSummary();
    });

    function setConsoleOpen(open) {
      consolePanel.style.display = open ? 'flex' : 'none';
      if (open) consoleBody.scrollTop = consoleBody.scrollHeight;
    }

    toggleBtn.addEventListener('click', function () {
      setConsoleOpen(consolePanel.style.display === 'none');
    });
    closeBtn.addEventListener('click', function () { setConsoleOpen(false); });

    clearBtn.addEventListener('click', function () {
      CC.Logger.clear();
    });

    downloadBtn.addEventListener('click', function () {
      const blob = new Blob([CC.Logger.toText()], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      a.download = 'chemcanvas-log-' + stamp + '.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    svg = document.getElementById('structure-svg');
    canvasStats = document.getElementById('canvas-stats');

    controller = new CC.Controller(svg, molecule, history, {
      onMoleculeChanged: renderNow,
      onSelectionChanged: function () { renderNow(); },
      onToolShortcut: activateTool,
      onCleanupShortcut: cleanupStructure,
      onHoverChanged: function (target) { CC.updateHoverHighlight(svg, target); },
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
    setupDrugLikenessModal();
    setupMicrostateModal();
    setupSmartsFiltersModal();
    setupPropertyInfoModal();
    setupTitrationPanel();

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
    setupValidationPanel();
    setup3DPanel();
    setupGNNPanel();
    setupSolventPanel();
    setupCanvasNavigation();
    setupLogConsole();
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

    // Same pattern as the SA Score table above: a static reference
    // dataset (FDA-approved-drug percentile distributions), fetched once
    // at startup, re-rendering the drug-likeness section once it lands
    // so a molecule drawn before it finishes loading gets its
    // percentiles filled in retroactively rather than staying blank.
    CC.DrugLikeness.loadReference()
      .then(function () { renderDrugLikeness(currentDescriptors); })
      .catch(function (err) {
        console.warn('[ChemCanvas] Drug-likeness reference distribution failed to load; percentiles will be omitted.', err);
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
