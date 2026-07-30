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

  CC.Export.downloadPDF = function (rows, filename, title) {
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
    doc.autoTable({
      startY: 28,
      head: [['Property', 'Value']],
      body: rows.map(function (r) { return [r.label, String(r.value)]; }),
      styles: { fontSize: 9 },
      headStyles: { fillColor: [79, 176, 165] },
    });
    doc.save(filename || 'properties.pdf');
    return { ok: true };
  };
})();
