/**
 * compute_ch_bde_atom_features.js
 *
 * Per-ATOM (not molecule-wide min) [bde_ch, has_ch] pairs for every
 * carbon bearing at least one hydrogen, for chemprop's real
 * --atom-features-path (true pre-message-passing atom feature fusion,
 * V_f -- NOT the X_d molecule-level descriptor path every other CYP
 * experiment this session used). One-off experiment testing whether
 * atom-level detail (which atom is reactive) beats the molecule-wide
 * BDE-XH minimum already shown to add nothing as an X_d descriptor.
 *
 * bde_ch = 0, has_ch = 0 for every non-(C-with-H) atom -- chemprop
 * requires a dense feature row for every atom, no gaps.
 *
 * Atom order is `mol.GetAtoms()` iteration order for RDKit-parsed
 * `smiles` -- verified empirically (not just assumed) that RDKit.js and
 * Python RDKit agree on this order for the same canonical SMILES before
 * writing this script.
 *
 * Usage:
 *   CC_BASE_URL=http://localhost:8000/ node compute_ch_bde_atom_features.js \
 *     <in.csv> <out.ndjson>
 *
 * Writes one JSON object per line, SAME ORDER as <in.csv>'s rows:
 *   {"ok": true, "features": [[bde_ch, has_ch], ...]}  (one row per heavy atom)
 *   {"ok": false, "error": "..."}
 * scripts/build_ch_bde_npz.py consumes this + the original CSV to build
 * chemprop's atom_features_0.npz, dropping any {"ok": false} rows from
 * both.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { buildSandbox, loadChempropModel, moleculeFromSmiles, REPO_ROOT } = require('./harness.js');

function parseCsv(text) {
  const lines = text.trim().split(/\r\n|\n/);
  const header = lines[0].split(',');
  return lines.slice(1).map((line) => {
    const vals = line.split(',');
    const row = {};
    header.forEach((h, i) => { row[h] = vals[i]; });
    return row;
  });
}

async function main() {
  const [inPath, outPath] = process.argv.slice(2);
  if (!inPath || !outPath) {
    console.error('usage: node compute_ch_bde_atom_features.js <in.csv> <out.ndjson>');
    process.exit(1);
  }

  const rows = parseCsv(fs.readFileSync(inPath, 'utf8'));
  console.error(`loaded ${rows.length} rows from ${inPath}`);

  const sandbox = await buildSandbox();
  const src = fs.readFileSync(path.join(REPO_ROOT, 'js/chemprop-features-explicit-h.js'), 'utf8');
  vm.runInContext(src, sandbox, { filename: 'js/chemprop-features-explicit-h.js' });
  await loadChempropModel(sandbox, 'bde-chemprop-v1', 'model/bde-chemprop/bde-manifest.json', 'model/bde-chemprop/bde.bin');
  console.error('bde-chemprop-v1 loaded');

  const out = fs.createWriteStream(outPath, { flags: 'w' });
  let ok = 0, failed = 0;
  const t0 = Date.now();
  for (let i = 0; i < rows.length; i++) {
    const smi = rows[i].smiles;
    try {
      const mol = moleculeFromSmiles(sandbox, smi);
      const heavyAtoms = Array.from(mol.atoms.values());
      const bdeResult = sandbox.CC.GNN.predictChemprop(mol, 'bde-chemprop-v1');
      const byAtomId = {};
      bdeResult.atomIds.forEach((atomId, idx) => {
        const props = bdeResult.atomProperties[idx];
        const v = props && Object.values(props)[0];
        if (typeof v === 'number') byAtomId[atomId] = v;
      });
      const features = heavyAtoms.map((a) => {
        if (a.element === 'C' && typeof byAtomId[a.id] === 'number') return [byAtomId[a.id], 1];
        return [0, 0];
      });
      out.write(JSON.stringify({ ok: true, features }) + '\n');
      ok++;
    } catch (err) {
      out.write(JSON.stringify({ ok: false, error: err.message }) + '\n');
      failed++;
    }
    if ((i + 1) % 500 === 0 || i === rows.length - 1) {
      const elapsed = (Date.now() - t0) / 1000;
      console.error(`[${i + 1}/${rows.length}] ok=${ok} failed=${failed} elapsed=${elapsed.toFixed(0)}s`);
    }
  }
  out.end();
  console.error(`done. ok=${ok} failed=${failed}`);
}

main().catch((err) => { console.error('FATAL', err); process.exit(1); });
