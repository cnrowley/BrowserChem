/**
 * compute_electrophile_reactivity_feature.js
 *
 * Real electrophile-reactivity-v1 predicted probability (protein-
 * reactive / covalent-warhead likelihood) for every unique molecule in
 * a training CSV -- offline data prep for testing whether Ames
 * mutagenicity prediction improves from this feature (real chemical
 * rationale: many Ames-positive mutagens -- aromatic amines after
 * bioactivation, epoxides, Michael acceptors, alkylating agents -- ARE
 * electrophiles, the same reactivity class electrophile-reactivity-v1
 * was trained to detect, on a MUCH larger dataset -- 62678 vs Ames's
 * 6505 -- so this is a real transfer-learning-adjacent idea even in its
 * cheapest X_d-feature form).
 *
 * Usage:
 *   CC_BASE_URL=http://localhost:8000/ node compute_electrophile_reactivity_feature.js \
 *     <in.csv> <out.csv>
 *
 * Writes <out.csv>: smiles,electrophileReactivity (RDKit-canonical SMILES).
 */
const fs = require('fs');
const { buildSandbox, loadChempropModel, moleculeFromSmiles } = require('./harness.js');

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
    console.error('usage: node compute_electrophile_reactivity_feature.js <in.csv> <out.csv>');
    process.exit(1);
  }

  const rows = parseCsv(fs.readFileSync(inPath, 'utf8'));
  console.error(`loaded ${rows.length} rows from ${inPath}`);

  const sandbox = await buildSandbox();
  await loadChempropModel(sandbox, 'electrophile-reactivity-v1', 'model/electrophile-reactivity/manifest.json', 'model/electrophile-reactivity/weights.bin');
  console.error('electrophile-reactivity-v1 loaded');

  const out = fs.createWriteStream(outPath, { flags: 'w' });
  out.write('smiles,electrophileReactivity\n');

  let ok = 0, failed = 0;
  const t0 = Date.now();
  for (let i = 0; i < rows.length; i++) {
    const rawSmiles = rows[i].smiles;
    try {
      const mol = moleculeFromSmiles(sandbox, rawSmiles);
      const rdmol = sandbox.chemCanvasLibs.RDKit.get_mol(rawSmiles);
      const canonicalSmiles = rdmol.get_smiles();
      rdmol.delete();

      const result = sandbox.CC.GNN.predictChemprop(mol, 'electrophile-reactivity-v1');
      const score = result.molecularProperties.label;
      if (typeof score !== 'number') throw new Error('electrophile-reactivity-v1 produced no usable prediction');

      out.write([canonicalSmiles, score].join(',') + '\n');
      ok++;
    } catch (err) {
      failed++;
      console.error(`row ${i} FAILED (${rawSmiles}): ${err.message}`);
    }
    if ((i + 1) % 1000 === 0 || i === rows.length - 1) {
      const elapsed = (Date.now() - t0) / 1000;
      console.error(`[${i + 1}/${rows.length}] ok=${ok} failed=${failed} elapsed=${elapsed.toFixed(0)}s`);
    }
  }
  out.end();
  console.error(`done. ok=${ok} failed=${failed}`);
}

main().catch((err) => { console.error('FATAL', err); process.exit(1); });
