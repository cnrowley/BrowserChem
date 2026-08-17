/**
 * export.js
 *
 * Turns the current property set (validity, SMILES, RDKit descriptors)
 * into a flat list of {label, value} rows, then offers it in four forms:
 * clipboard text, CSV, XLSX (via SheetJS), and PDF (via jsPDF + AutoTable).
 *
 * All three download paths degrade gracefully with a clear status message
 * if their CDN library hasn't finished loading yet — same pattern as
 * lib-loader.js uses for RDKit/ONNX/3Dmol.
 *
 * The PDF additionally accepts atom-/bond-level property tables (see
 * buildAtomTable/buildBondTable below) -- built from the SAME
 * atomId/bondId -> {propName: value} maps app.js's atom/bond heatmaps
 * already hold (lastAtomProperties/lastBondProperties), so whatever's
 * currently selectable in either heatmap dropdown is exactly what shows
 * up here, one column per property, one row per atom/bond that has at
 * least one value.
 *
 * A fifth form, SDF (buildSDF/downloadSDF below), is structural rather
 * than tabular: it round-trips through other cheminformatics tools, so
 * unlike the four forms above it carries the actual connection table
 * (via CC.moleculeToMolblock) alongside every computed property, not
 * just the property list on its own.
 */

window.CC = window.CC || {};
CC.Export = {};

(function () {
  CC.Export.buildRows = function (state) {
    const rows = [];
    rows.push({ label: 'Validity', value: state.validityText || '' });
    rows.push({ label: 'Canonical SMILES', value: state.smiles || '' });

    if (state.descriptors) {
      CC.DESCRIPTOR_FIELDS.forEach(function (field) {
        const value = state.descriptors[field.key];
        if (typeof value !== 'number') return;
        rows.push({
          label: field.label + (field.unit ? ' (' + field.unit + ')' : ''),
          value: Number(value.toFixed(field.decimals)),
        });
      });
    }

    return rows;
  };

  // Union of every property key across every atom/bond, not just the
  // first one -- a sparse property (e.g. shift_19f, present only on F
  // atoms) can easily be absent from whichever id happens to be first.
  // Same pattern app.js's heatmap-dropdown population already uses.
  function unionPropertyKeys(propsById) {
    const set = {};
    Object.keys(propsById).forEach(function (id) {
      Object.keys(propsById[id]).forEach(function (name) { set[name] = true; });
    });
    return Object.keys(set);
  }

  function formatCell(v, propName) {
    if (v === undefined || v === null) return '—';
    if (typeof v !== 'number' || isNaN(v)) return String(v);
    if (Number.isInteger(v)) return String(v);
    const decimals = CC.propertyDecimals(propName, 3);
    return v.toFixed(decimals);
  }

  /**
   * { head: [...], body: [[...], ...] } ready for jsPDF-AutoTable (or any
   * other row/column consumer) -- one row per heavy atom that has at
   * least one property, columns are ['Atom #', 'Element', ...every
   * property key currently present]. Empty { head: [], body: [] } if
   * there's no molecule or no atom properties at all yet.
   */
  CC.Export.buildAtomTable = function (molecule, atomProperties) {
    if (!molecule || !atomProperties) return { head: [], body: [] };
    const propNames = unionPropertyKeys(atomProperties);
    if (propNames.length === 0) return { head: [], body: [] };

    const head = ['Atom #', 'Element'].concat(propNames);
    const body = [];
    Array.from(molecule.atoms.values()).forEach(function (atom, i) {
      const props = atomProperties[atom.id];
      if (!props || Object.keys(props).length === 0) return;
      body.push([String(i + 1), atom.element].concat(propNames.map(function (name) {
        return formatCell(props[name], name);
      })));
    });
    return { head: head, body: body };
  };

  /**
   * Same idea as buildAtomTable, for bonds -- columns
   * ['Bond #', 'Atoms', 'Order', ...every property key present]. 'Atoms'
   * labels each bond by its two endpoints' element + 1-based atom number
   * (e.g. "C1-C6"), matching buildAtomTable's own atom numbering so the
   * two tables cross-reference cleanly.
   */
  CC.Export.buildBondTable = function (molecule, bondProperties) {
    if (!molecule || !bondProperties) return { head: [], body: [] };
    const propNames = unionPropertyKeys(bondProperties);
    if (propNames.length === 0) return { head: [], body: [] };

    const atomIndexById = new Map();
    Array.from(molecule.atoms.values()).forEach(function (a, i) { atomIndexById.set(a.id, i + 1); });

    const head = ['Bond #', 'Atoms', 'Order'].concat(propNames);
    const body = [];
    Array.from(molecule.bonds.values()).forEach(function (bond, i) {
      const props = bondProperties[bond.id];
      if (!props || Object.keys(props).length === 0) return;
      const a1 = molecule.atoms.get(bond.a1);
      const a2 = molecule.atoms.get(bond.a2);
      const label = (a1 ? a1.element + atomIndexById.get(bond.a1) : '?') + '–' +
        (a2 ? a2.element + atomIndexById.get(bond.a2) : '?');
      body.push([String(i + 1), label, String(bond.order)].concat(propNames.map(function (name) {
        return formatCell(props[name], name);
      })));
    });
    return { head: head, body: body };
  };

  CC.Export.copyToClipboard = function (rows) {
    const text = rows.map(function (r) { return r.label + '\t' + r.value; }).join('\n');
    return navigator.clipboard.writeText(text);
  };

  function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function csvEscape(value) {
    const s = String(value);
    if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  }

  CC.Export.downloadCSV = function (rows, filename) {
    const lines = ['Property,Value'];
    rows.forEach(function (r) {
      lines.push(csvEscape(r.label) + ',' + csvEscape(r.value));
    });
    const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/csv' });
    triggerDownload(blob, filename || 'properties.csv');
  };

  CC.Export.downloadXLSX = function (rows, filename) {
    const XLSX = window.XLSX;
    if (!XLSX) return { ok: false, message: 'XLSX library still loading \u2014 try again in a moment' };

    const wsData = [['Property', 'Value']].concat(rows.map(function (r) { return [r.label, r.value]; }));
    const worksheet = XLSX.utils.aoa_to_sheet(wsData);
    worksheet['!cols'] = [{ wch: 26 }, { wch: 30 }];
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Properties');
    XLSX.writeFile(workbook, filename || 'properties.xlsx');
    return { ok: true };
  };

  /**
   * atomTable/bondTable (optional): { head, body } from buildAtomTable /
   * buildBondTable above -- each renders as its own labeled table below
   * the main property list, only if it actually has rows.
   */
  CC.Export.downloadPDF = function (rows, filename, title, atomTable, bondTable) {
    const jspdfNs = window.jspdf;
    if (!jspdfNs || !jspdfNs.jsPDF) {
      return { ok: false, message: 'PDF library still loading \u2014 try again in a moment' };
    }
    const doc = new jspdfNs.jsPDF();
    doc.setFontSize(14);
    doc.text(title || 'ChemCanvas structure properties', 14, 16);
    doc.setFontSize(9);
    doc.text(new Date().toLocaleString(), 14, 22);

    if (typeof doc.autoTable !== 'function') {
      return { ok: false, message: 'PDF table plugin still loading \u2014 try again in a moment' };
    }

    const pageHeight = doc.internal.pageSize.getHeight();
    // A fixed width for the label/id columns, with every other column
    // left at AutoTable's own 'auto' sizing -- without this, a single
    // long unbroken string in one column (a canonical SMILES with no
    // spaces to wrap on is the real case that triggered this: an
    // ~85-character SMILES) dominates AutoTable's default content-based
    // width split and squeezes every other column down to a couple of
    // characters wide, which is what "the property column is only 2
    // characters wide" actually was -- confirmed by reproducing it
    // directly (doc.lastAutoTable.columns[0].width came back ~15pt,
    // essentially just the header text's own minimum width) before
    // fixing it this way, not guessed.
    doc.autoTable({
      startY: 28,
      head: [['Property', 'Value']],
      body: rows.map(function (r) { return [r.label, String(r.value)]; }),
      styles: { fontSize: 9, overflow: 'linebreak' },
      headStyles: { fillColor: [79, 176, 165] },
      columnStyles: { 0: { cellWidth: 70 }, 1: { cellWidth: 'auto' } },
    });

    let nextY = doc.lastAutoTable.finalY + 12;

    function drawLabeledTable(label, table, idColWidth, extraColumnStyles) {
      if (!table || !table.body || table.body.length === 0) return;
      if (nextY > pageHeight - 30) { doc.addPage(); nextY = 20; }
      doc.setFontSize(11);
      doc.setTextColor(40);
      doc.text(label, 14, nextY);
      doc.autoTable({
        startY: nextY + 4,
        head: [table.head],
        body: table.body,
        styles: { fontSize: 8, overflow: 'linebreak' },
        headStyles: { fillColor: [79, 176, 165] },
        columnStyles: Object.assign({ 0: { cellWidth: idColWidth } }, extraColumnStyles || {}),
      });
      nextY = doc.lastAutoTable.finalY + 12;
    }

    drawLabeledTable('Atom properties', atomTable, 16, { 1: { cellWidth: 18 } });
    drawLabeledTable('Bond properties', bondTable, 16, { 1: { cellWidth: 22 }, 2: { cellWidth: 16 } });

    doc.save(filename || 'properties.pdf');
    return { ok: true };
  };

  // SDF data-item field names can't contain '<', '>', or whitespace at
  // the edges -- stripped/trimmed rather than rejected outright, so an
  // odd label still produces a valid (if slightly mangled) file instead
  // of silently dropping that property.
  function sdfFieldName(label) {
    return String(label).replace(/[<>]/g, '').trim().replace(/\s+/g, '_') || 'field';
  }

  function pushSdfItem(lines, name, valueLines) {
    lines.push('> <' + sdfFieldName(name) + '>');
    valueLines.forEach(function (v) { lines.push(v === undefined || v === null ? '' : String(v)); });
    lines.push(''); // blank line terminates each SDF data item
  }

  /**
   * A single-record SD file: the molecule's own connection table (same
   * writer "Save as .mol file" uses, so it round-trips identically) plus
   * every computed property as an SDF data item --
   *   - rows (molecule-level: descriptors, GNN molecular predictions):
   *     one item per row, single value line.
   *   - atomProperties/bondProperties (atomId/bondId -> {propName:
   *     value} maps, e.g. lastAtomProperties/lastBondProperties): one
   *     item per property NAME (not per atom), with one value line per
   *     atom/bond in molecule.atoms/bonds order -- the same "list of
   *     per-atom values under one field" convention RDKit's own SDF
   *     round-tripping of atom-level data uses, so a field like
   *     `atom_mbis-charges` has exactly molecule.atoms.size value lines,
   *     line N corresponding to the Nth atom (1-indexed, same numbering
   *     as the connection table itself).
   */
  CC.Export.buildSDF = function (molecule, title, rows, atomProperties, bondProperties) {
    const molblock = CC.moleculeToMolblock(molecule, title || '');
    const lines = molblock.replace(/\n+$/, '').split('\n');

    (rows || []).forEach(function (r) { pushSdfItem(lines, r.label, [r.value]); });

    if (molecule && atomProperties) {
      const atoms = Array.from(molecule.atoms.values());
      unionPropertyKeys(atomProperties).forEach(function (name) {
        pushSdfItem(lines, 'atom_' + name, atoms.map(function (a) {
          const props = atomProperties[a.id];
          return props ? props[name] : undefined;
        }));
      });
    }

    if (molecule && bondProperties) {
      const bonds = Array.from(molecule.bonds.values());
      unionPropertyKeys(bondProperties).forEach(function (name) {
        pushSdfItem(lines, 'bond_' + name, bonds.map(function (b) {
          const props = bondProperties[b.id];
          return props ? props[name] : undefined;
        }));
      });
    }

    lines.push('$$$$');
    return lines.join('\n') + '\n';
  };

  CC.Export.downloadSDF = function (molecule, title, rows, atomProperties, bondProperties, filename) {
    if (!molecule || molecule.isEmpty()) {
      return { ok: false, message: 'Draw a structure first' };
    }
    const sdf = CC.Export.buildSDF(molecule, title, rows, atomProperties, bondProperties);
    const blob = new Blob([sdf], { type: 'chemical/x-mdl-sdfile' });
    triggerDownload(blob, filename || 'molecule.sdf');
    return { ok: true };
  };
})();
