/**
 * compute_pka_embeddings.js
 *
 * Real pooled D-MPNN embeddings for the pka-microstate-freeenergy
 * checkpoint's own training microstates -- the data half of building
 * model/pka-microstate-freeenergy/applicability-domain.json that
 * scripts/compute_applicability_domain.py's generic pure-numpy pipeline
 * can't produce for this model: that script assumes one SMILES column
 * and no X_d feature fusion, neither of which fits this model (paired
 * smiles_protonated/smiles_deprotonated columns, numExtraDescriptors=3).
 *
 * Uses CC.GNN.getPooledEmbedding (js/chemprop-model.js) directly -- the
 * REAL trained D-MPNN forward pass this app runs at inference time, not
 * a second parallel reimplementation. Deliberately does NOT load NAGL or
 * logp-v1 or compute X_d descriptors at all: getPooledEmbedding returns
 * the vector BEFORE X_d fusion (see its own header comment in
 * chemprop-model.js) -- the same space CC.AD.tierForEmbedding compares
 * against at real prediction time, and extra descriptors never affect it.
 *
 * Each of this model's own 21,407 training ROWS is a (protonated,
 * deprotonated) SMILES PAIR, but a real query at inference time embeds
 * one MICROSTATE at a time (js/pka-freeenergy-predict.js's
 * microstateFreeEnergy calls CC.GNN.predictChemprop once per side) -- so
 * this script embeds both sides of every sampled row as independent
 * points, not one embedding per row, matching what actually gets
 * embedded at runtime.
 *
 * Usage:
 *   CC_BASE_URL=http://localhost:8000/ node compute_pka_embeddings.js \
 *     <in.csv> <out.ndjson> [--max-rows 4000] [--seed 0]
 *
 * Writes one JSON object per line to <out.ndjson>: {"smiles", "embedding"}.
 * scripts/compute_pka_applicability_domain.py consumes this file.
 */
const fs = require('fs');
const { buildSandbox, loadChempropModel, moleculeFromSmiles } = require('./harness.js');

function parseCsv(text) {
  // Python's csv module writes '\r\n' row terminators -- see
  // score_batch.js's own parseCsv comment for why splitting on '\n' alone
  // corrupts the header's last column name.
  const lines = text.trim().split(/\r\n|\n/);
  const header = lines[0].split(',');
  return lines.slice(1).map((line) => {
    const vals = line.split(',');
    const row = {};
    header.forEach((h, i) => { row[h] = vals[i]; });
    return row;
  });
}

function parseArgs(argv) {
  const positional = [];
  const opts = { maxRows: 4000, seed: 0 };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--max-rows') opts.maxRows = parseInt(argv[++i], 10);
    else if (argv[i] === '--seed') opts.seed = parseInt(argv[++i], 10);
    else positional.push(argv[i]);
  }
  return { positional, opts };
}

// Deterministic seeded shuffle (mulberry32) -- good enough for a
// subsample, no need for a real Mersenne Twister here.
function seededShuffle(arr, seed) {
  let s = seed >>> 0;
  function rand() {
    s |= 0; s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

async function main() {
  const { positional, opts } = parseArgs(process.argv.slice(2));
  const [inPath, outPath] = positional;
  if (!inPath || !outPath) {
    console.error('usage: node compute_pka_embeddings.js <in.csv> <out.ndjson> [--max-rows 4000] [--seed 0]');
    process.exit(1);
  }

  const rows = parseCsv(fs.readFileSync(inPath, 'utf8'));
  console.error(`loaded ${rows.length} rows from ${inPath}`);
  const sampled = opts.maxRows && rows.length > opts.maxRows
    ? seededShuffle(rows, opts.seed).slice(0, opts.maxRows)
    : rows;
  console.error(`embedding ${sampled.length} rows (${sampled.length * 2} microstates) -- maxRows=${opts.maxRows}`);

  const sandbox = await buildSandbox();
  await loadChempropModel(
    sandbox,
    'pka-microstate-freeenergy',
    'model/pka-microstate-freeenergy/pka-microstate-freeenergy-manifest.json',
    'model/pka-microstate-freeenergy/pka-microstate-freeenergy.bin'
  );
  console.error('checkpoint loaded');

  const out = fs.createWriteStream(outPath, { flags: 'w' });
  let ok = 0, failed = 0;
  const t0 = Date.now();
  for (let i = 0; i < sampled.length; i++) {
    const r = sampled[i];
    for (const side of ['smiles_protonated', 'smiles_deprotonated']) {
      const smi = r[side];
      try {
        const mol = moleculeFromSmiles(sandbox, smi);
        const embedding = sandbox.CC.GNN.getPooledEmbedding(mol, 'pka-microstate-freeenergy');
        out.write(JSON.stringify({ smiles: smi, embedding: embedding.map((v) => Math.round(v * 1e6) / 1e6) }) + '\n');
        ok++;
      } catch (err) {
        failed++;
        console.error(`row ${i} [${side}] FAILED (${smi}): ${err.message}`);
      }
    }
    if ((i + 1) % 500 === 0 || i === sampled.length - 1) {
      const elapsed = (Date.now() - t0) / 1000;
      console.error(`[${i + 1}/${sampled.length}] ok=${ok} failed=${failed} elapsed=${elapsed.toFixed(0)}s`);
    }
  }
  out.end();
  console.error(`done. ok=${ok} failed=${failed}`);
}

main().catch((err) => { console.error('FATAL', err); process.exit(1); });
