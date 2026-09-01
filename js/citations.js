/**
 * citations.js
 *
 * A small bibliography engine over data/citations.json (the hand-curated
 * list of real papers/datasets/software behind this app's trained models
 * and physics features). Three jobs:
 *
 *   1. Resolve "which citation keys are actually relevant right now" --
 *      CC.Citations.forLoadedModels() walks the SAME registry state
 *      js/model-registry.js already tracks (CC.GNN.getRegistryEntries() +
 *      isRegistryModelLoaded()), so a model only shows up in the
 *      bibliography once it's actually been loaded in this session, not
 *      just because it exists in the catalog. Each model's own
 *      dataset.citationKey (already in model/registry.json) is combined
 *      with a fixed per-"engine" architecture citation (e.g. every
 *      "chemprop" model also cites the Chemprop package + D-MPNN papers).
 *   2. Format one entry as a human-readable reference line (with a real
 *      DOI link where one exists).
 *   3. Export a set of entries as BibTeX or RIS text -- CC.Citations.toBibTeX
 *      / .toRIS below, consumed by js/export.js's downloadBibTeX/downloadRIS.
 *
 * Same honesty convention as the rest of this app (see CLAUDE.md): an
 * entry with no confirmed DOI says so in its own `note` field rather than
 * inventing one. Never hand-add a citation with a guessed DOI/volume/page
 * -- verify it (a real publisher/DOI-resolver page, not just memory) and
 * update data/citations.json, or leave the field out.
 */

window.CC = window.CC || {};
CC.Citations = {};

(function () {
  let dataPromise = null;
  let byKey = null;

  // Every model in model/registry.json declares an "engine" (see
  // js/model-registry.js's own schema doc) -- this maps each engine to
  // the citation key(s) for the architecture/package it's built on, on
  // top of that model's own dataset.citationKey. A model with no
  // "engine" field defaults to "chemprop" (model-registry.js's own
  // default), so this map's "chemprop" entry is also the fallback.
  const ENGINE_CITATIONS = {
    chemprop: ['Heid2024Chemprop', 'Yang2019DMPNN'],
    nagl: ['nagl-mbis-software'],
    ani2x: ['Devereux2020ANI2x'],
    geomol: ['Ganea2021GeoMol'],
    // pka (pKalculator port) has no separate architecture citation --
    // ree2024pkalculator (pushed via dataset.citationKey below) already
    // covers the one paper behind both the method and its training data.
    pka: [],
  };

  // Always relevant once a structure has been drawn/validated at all --
  // RDKit backs 2D validation/descriptors/canonical SMILES for every
  // molecule, and QED + SA Score are computed as part of the standard
  // 2D properties/drug-likeness output (js/qed.js, js/sascorer.js), not
  // opt-in models.
  const CORE_CITATIONS = ['RDKitSoftware', 'Bickerton2012QED', 'Ertl2009SAScore'];

  // Feature-specific citations for physics/ML components that aren't
  // "models" in model/registry.json at all (so forLoadedModels() has no
  // other way to know they exist) -- surfaced as a separate "other
  // methods available in this app" list rather than auto-detected,
  // since reliably detecting "was this feature actually used this
  // session" for all of these would need new state-tracking wired
  // through several unrelated UI handlers for comparatively little
  // benefit. See js/dsasa.js, js/embed3d.js (CREST-inspired conformer
  // search), js/smarts-filters.js, js/openff-forcefield.js,
  // js/viewer3d.js, js/ocsrglyph-model.js for where each is actually used.
  // NOT listed here: nmrshiftdb2/nmrexp2025/nagl-mbis-software -- those
  // now have real dataset.citationKey entries directly on their
  // registry models (nmr-13c/1h/19f, nagl-mbis-charges), so
  // forLoadedModels() already surfaces them once that model is loaded,
  // same as any other trained model's dataset citation.
  CC.Citations.OTHER_METHOD_CITATIONS = [
    'Cao2024dSASA', 'Hummel2015Thesis', 'Pracht2020CREST',
    'Baell2010PAINS', 'Boothroyd2023Sage', 'Rego2015ThreeDmol',
    'OCSRGlyphSoftware',
  ];

  CC.Citations.load = function (url) {
    if (dataPromise) return dataPromise;
    dataPromise = fetch(url || 'data/citations.json').then(function (r) {
      if (!r.ok) throw new Error('citations.json fetch failed: ' + r.status);
      return r.json();
    }).then(function (data) {
      byKey = {};
      (data.entries || []).forEach(function (e) { byKey[e.key] = e; });
      return byKey;
    });
    return dataPromise;
  };

  CC.Citations.get = function (key) {
    return (byKey && byKey[key]) || null;
  };

  CC.Citations.isLoaded = function () { return !!byKey; };

  /**
   * Citation keys for one model/registry.json entry: its own
   * dataset.citationKey (if any) plus its engine's architecture
   * citation(s). Order: dataset first, architecture after -- the
   * dataset is what's specific to THIS model.
   */
  // A handful of older registry entries (ani2x-v1, geomol-drugs-v1) put
  // the full human-readable citation text into dataset.citationKey
  // itself (e.g. "Devereux et al., J. Chem. Theory Comput. 2020, 16, 7,
  // 4192-4202") instead of a short bibliography key -- there because the
  // model IS the cited paper, already covered by that engine's own
  // ENGINE_CITATIONS entry below, so it's skipped here rather than
  // looked up (it would never match a data/citations.json key anyway).
  // Every real short key in this registry (see model/registry.json) is a
  // single word with no spaces, optionally with digits/hyphens.
  function looksLikeShortKey(s) {
    return /^[A-Za-z][\w-]*$/.test(s);
  }

  CC.Citations.forRegistryEntry = function (entry) {
    const keys = [];
    // dataset.citationKey is usually one short key, but a model whose
    // training data is a real, comparably-weighted combination of
    // multiple sources (e.g. pka-microstate-freeenergy: IUPAC +
    // Baltruschat & Czodrowski + Nevolianis et al. + pKaHub, all with
    // their own real fidelity_weight in that model's own training CSV,
    // not one dominant source with the others as an afterthought) can
    // give an array instead, so every source it actually learned from is
    // individually citable/exportable rather than only the first pick.
    const ck = entry && entry.dataset && entry.dataset.citationKey;
    const ckList = Array.isArray(ck) ? ck : (ck ? [ck] : []);
    ckList.filter(looksLikeShortKey).forEach(function (k) { keys.push(k); });
    const engine = (entry && entry.engine) || 'chemprop';
    (ENGINE_CITATIONS[engine] || []).forEach(function (k) { keys.push(k); });
    return keys;
  };

  /**
   * Every citation key relevant to the CURRENT session: CORE_CITATIONS
   * (always, once `includeCore` isn't false) plus forRegistryEntry() for
   * every registry model CC.GNN reports as currently loaded --
   * deduplicated, in first-seen order. Returns [] (not an error) if
   * CC.GNN isn't available/registry not loaded yet.
   */
  CC.Citations.forLoadedModels = function (includeCore) {
    const seen = {};
    const keys = [];
    function add(k) {
      if (!k || seen[k]) return;
      seen[k] = true;
      keys.push(k);
    }
    if (includeCore !== false) CORE_CITATIONS.forEach(add);
    if (window.CC && CC.GNN && typeof CC.GNN.getRegistryEntries === 'function') {
      CC.GNN.getRegistryEntries().forEach(function (entry) {
        if (!CC.GNN.isRegistryModelLoaded(entry.id)) return;
        CC.Citations.forRegistryEntry(entry).forEach(add);
      });
    }
    return keys;
  };

  // A literal "et al." entry in the authors array (used when only the
  // first author is known/confirmed -- e.g. a paper found via a dataset
  // README that didn't spell out the full author list) is a stand-in
  // abbreviation, not a name -- joined with ", " like the rest rather
  // than " & ", and never given a second trailing period on top of the
  // one "et al." already ends with.
  function authorsInline(authors) {
    if (!authors || authors.length === 0) return '';
    if (authors.length === 1) return authors[0];
    if (authors[authors.length - 1] === 'et al.') return authors.join(', ');
    if (authors.length <= 3) return authors.slice(0, -1).join(', ') + ' & ' + authors[authors.length - 1];
    return authors[0] + ' et al.';
  }

  function trimPeriod(s) { return s.replace(/\.\s*$/, ''); }

  /**
   * One human-readable reference line, e.g.:
   *   "Mansouri, K. et al. OPERA models for predicting physicochemical
   *   properties and environmental fate endpoints. J. Cheminform. 2018,
   *   10, 10. https://doi.org/10.1186/s13321-018-0263-1"
   * Falls back to entry.url if there's no doi, and appends entry.note
   * (e.g. "DOI not confirmed -- verify before citing formally") in
   * parentheses if present.
   */
  CC.Citations.formatLine = function (key) {
    const e = CC.Citations.get(key);
    if (!e) return key + ' (unresolved citation key)';
    const parts = [];
    if (e.authors && e.authors.length) parts.push(trimPeriod(authorsInline(e.authors)) + '.');
    if (e.title) parts.push(trimPeriod(e.title) + '.');
    const venue = e.journal || e.booktitle || e.school || e.publisher || '';
    const venueBits = [venue, e.year, e.volume, e.pages].filter(Boolean).join(', ');
    if (venueBits) parts.push(venueBits + '.');
    const link = e.doi ? 'https://doi.org/' + e.doi : e.url;
    if (link) parts.push(link);
    let line = parts.join(' ');
    if (e.note) line += ' (' + e.note + ')';
    return line;
  };

  // ---------- BibTeX ----------

  function bibtexEscape(s) {
    return String(s == null ? '' : s).replace(/[{}]/g, '');
  }

  const BIBTEX_TYPE = {
    'article-journal': 'article',
    thesis: 'phdthesis',
    software: 'misc',
    dataset: 'misc',
    'paper-conference': 'inproceedings',
    preprint: 'unpublished',
    report: 'techreport',
  };

  function bibtexField(name, value) {
    if (value === undefined || value === null || value === '') return '';
    return '  ' + name + ' = {' + bibtexEscape(value) + '},\n';
  }

  CC.Citations.toBibTeX = function (keys) {
    return keys.map(function (key) {
      const e = CC.Citations.get(key);
      if (!e) return '% unresolved citation key: ' + key + '\n';
      const type = BIBTEX_TYPE[e.type] || 'misc';
      let out = '@' + type + '{' + bibtexEscape(key) + ',\n';
      if (e.authors && e.authors.length) out += bibtexField('author', e.authors.join(' and '));
      out += bibtexField('title', e.title);
      if (e.journal) out += bibtexField('journal', e.journal);
      if (e.booktitle) out += bibtexField('booktitle', e.booktitle);
      if (e.school) out += bibtexField('school', e.school);
      if (e.publisher) out += bibtexField('publisher', e.publisher);
      if (e.institution) out += bibtexField('institution', e.institution);
      out += bibtexField('year', e.year);
      out += bibtexField('volume', e.volume);
      out += bibtexField('number', e.number);
      out += bibtexField('pages', e.pages);
      out += bibtexField('doi', e.doi);
      out += bibtexField('url', e.doi ? 'https://doi.org/' + e.doi : e.url);
      out += bibtexField('note', e.note);
      out = out.replace(/,\n$/, '\n');
      out += '}\n';
      return out;
    }).join('\n');
  };

  // ---------- RIS ----------

  const RIS_TYPE = {
    'article-journal': 'JOUR',
    thesis: 'THES',
    software: 'COMP',
    dataset: 'DATA',
    'paper-conference': 'CPAPER',
    preprint: 'UNPB',
    report: 'RPRT',
  };

  CC.Citations.toRIS = function (keys) {
    return keys.map(function (key) {
      const e = CC.Citations.get(key);
      if (!e) return 'TY  - GEN\nTI  - unresolved citation key: ' + key + '\nER  - \n';
      const lines = [];
      lines.push('TY  - ' + (RIS_TYPE[e.type] || 'GEN'));
      (e.authors || []).forEach(function (a) { lines.push('AU  - ' + a); });
      if (e.title) lines.push('TI  - ' + e.title);
      if (e.journal) lines.push('T2  - ' + e.journal);
      if (e.booktitle) lines.push('T2  - ' + e.booktitle);
      if (e.school) lines.push('PB  - ' + e.school);
      if (e.publisher) lines.push('PB  - ' + e.publisher);
      if (e.year) lines.push('PY  - ' + e.year);
      if (e.volume) lines.push('VL  - ' + e.volume);
      if (e.pages) {
        const m = /^(\S+)[–-](\S+)$/.exec(e.pages);
        if (m) { lines.push('SP  - ' + m[1]); lines.push('EP  - ' + m[2]); }
        else lines.push('SP  - ' + e.pages);
      }
      if (e.doi) lines.push('DO  - ' + e.doi);
      lines.push('UR  - ' + (e.doi ? 'https://doi.org/' + e.doi : (e.url || '')));
      if (e.note) lines.push('N1  - ' + e.note);
      lines.push('ER  - ');
      return lines.join('\n') + '\n';
    }).join('\n');
  };
})();
