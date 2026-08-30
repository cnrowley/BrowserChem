/**
 * score_batch.js
 *
 * Batch-computes physical_energy_protonated/physical_energy_deprotonated
 * for a microstate-pair CSV via harness.js's real, unmodified
 * CC.PKAPhysicalBaseline.compute(). Writes output incrementally so a long
 * run can be inspected mid-flight.
 *
 * Two input schemas, auto-detected from the header:
 *   - single-pair (scripts/prepare_pka_microstate_training_data.py,
 *     scripts/prepare_pka_qm_pretrain_data.py): smiles_protonated,
 *     smiles_deprotonated -- one structure per side. A failing structure
 *     (thrown error -- an unusable 3D structure, not a placeholder) drops
 *     the WHOLE row.
 *   - ensemble (scripts/prepare_pkahub_data.py): smiles_protonated_list,
 *     smiles_deprotonated_list -- semicolon-joined, one or more structures
 *     per side (the real Uni-pKa macro-pKa ensemble treatment, see that
 *     script's own header for why). A failing INDIVIDUAL structure is
 *     dropped from its own list, not the whole row -- the ensemble
 *     formula only needs at least one survivor per side; a row is only
 *     dropped entirely if a whole SIDE ends up with zero survivors.
 * Both are disclosed-not-silent, matching every other prepare_*.py
 * script's own convention.
 *
 * Real cost, not a rounding error: with GB/SA solvent genuinely active
 * (see harness.js's own header for how a missing-dependency bug can make
 * this look deceptively FAST by silently skipping the whole solvation
 * term) this runs at roughly 0.15-0.2 rows/sec single-threaded --
 * js/pka-physical-baseline.js's own header already discloses this cost
 * ("60 iterations alone took ~10s on a ~30-heavy-atom molecule"). Use
 * run_sharded.sh to parallelize across CPU cores for anything more than
 * a few hundred rows -- each molecule is scored completely independently,
 * so this embarrassingly parallelizes with zero shared state.
 *
 * Usage (single process, optionally row-range-limited):
 *   CC_BASE_URL=http://localhost:8000/ node score_batch.js <in.csv> <out.csv> [startRow] [endRow]
 *
 * Requires a real static server running at CC_BASE_URL serving this
 * repo's root (`python3 -m http.server 8000` from the repo root, per
 * CLAUDE.md) -- fetch() can't read data/openff-sage-2.1.0.json or
 * model/nagl-mbis-charges/* over file://.
 */
const fs = require('fs');
const { buildSandbox, loadEngines, moleculeFromSmiles } = require('./harness.js');

function parseCsv(text) {
  // Real bug, caught by hand (not by any thrown error): Python's csv
  // module writes '\r\n' row terminators by DEFAULT regardless of
  // platform (RFC 4180), and every one of this project's prepare_*.py
  // scripts writes CSVs that way. Splitting on '\n' alone leaves a
  // trailing '\r' glued onto the last field of every line -- for the
  // HEADER line specifically, that corrupts the LAST column name (e.g.
  // 'sources' becomes 'sources\r'), so any lookup by clean column name
  // silently returns undefined instead of throwing, which propagated all
  // the way into a written-out CSV missing its `sources` column's values
  // entirely before this was caught. Splitting on /\r\n|\n/ handles both
  // Python's '\r\n' and this script's own plain '\n' output uniformly.
  const lines = text.trim().split(/\r\n|\n/);
  const header = lines[0].split(',');
  const rows = lines.slice(1).map((line) => {
    // Plain split is safe here -- this project's own microstate CSVs
    // never quote fields (SMILES/inchikey/site_name/site_cls/sources have
    // no commas), confirmed against every prepare_*.py script's output.
    const vals = line.split(',');
    const row = {};
    header.forEach((h, i) => { row[h] = vals[i]; });
    return row;
  });
  return { header, rows };
}

async function main() {
  const inPath = process.argv[2];
  const outPath = process.argv[3];
  // Optional row range [start, end) for sharding a big CSV across
  // parallel processes (see run_sharded.sh) -- each shard writes its own
  // header, merged/stripped by the caller afterward. Omit both for
  // "whole file, one process".
  const start = process.argv[4] ? parseInt(process.argv[4], 10) : 0;
  const end = process.argv[5] ? parseInt(process.argv[5], 10) : Infinity;

  const { header, rows: allRows } = parseCsv(fs.readFileSync(inPath, 'utf8'));
  const rows = allRows.slice(start, Math.min(end, allRows.length));
  console.error(`loaded ${rows.length} rows (range [${start},${Math.min(end, allRows.length)})) from ${inPath}`);

  const sandbox = await buildSandbox();
  const naglModelId = await loadEngines(sandbox);
  console.error('engines ready, naglModelId=' + naglModelId);

  const isEnsemble = header.includes('smiles_protonated_list');
  console.error('schema: ' + (isEnsemble ? 'ensemble (list per side)' : 'single-pair'));

  // Preserve every input column (e.g. `fidelity_weight`) -- this script
  // only ever APPENDS the physical_energy_* column(s), never assumes a
  // fixed input schema beyond the one(s) it reads by name below.
  const newCols = isEnsemble
    ? ['physical_energy_protonated_list', 'physical_energy_deprotonated_list']
    : ['physical_energy_protonated', 'physical_energy_deprotonated'];
  const out = fs.createWriteStream(outPath, { flags: 'w' });
  out.write(header.concat(newCols).join(',') + '\n');

  async function scoreOne(smiles) {
    const mol = moleculeFromSmiles(sandbox, smiles);
    return sandbox.CC.PKAPhysicalBaseline.compute(mol, { naglModelId });
  }

  // Scores every SMILES in a semicolon-joined list, dropping individual
  // failures rather than the whole list (see file header). Returns null if
  // EVERY structure in the list failed (this side has no survivors at
  // all, so the row itself can't be scored) -- otherwise returns the
  // SURVIVING smiles alongside their energies, same length and order, so
  // the caller can overwrite the smiles_*_list column to match (a real
  // bug caught before it shipped: passing through the ORIGINAL smiles
  // list unchanged while writing a shorter energies list would silently
  // misalign index i in one column with index i in the other).
  async function scoreList(rowIndex, sideLabel, smilesList) {
    const smilesArr = smilesList.split(';');
    const survivingSmiles = [];
    const energies = [];
    for (const smi of smilesArr) {
      try {
        energies.push(await scoreOne(smi));
        survivingSmiles.push(smi);
      } catch (err) {
        console.error(`row ${rowIndex} [${sideLabel}] dropped microstate (${smi}): ${err.message}`);
      }
    }
    return energies.length ? { smiles: survivingSmiles, energies } : null;
  }

  let kept = 0, failed = 0;
  const t0 = Date.now();
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    try {
      if (isEnsemble) {
        const p = await scoreList(i, 'protonated', r.smiles_protonated_list);
        const d = await scoreList(i, 'deprotonated', r.smiles_deprotonated_list);
        if (!p || !d) throw new Error('one whole side has zero scoreable microstates');
        const rOut = Object.assign({}, r, {
          smiles_protonated_list: p.smiles.join(';'),
          smiles_deprotonated_list: d.smiles.join(';'),
        });
        out.write(header.map((h) => rOut[h]).concat([p.energies.join(';'), d.energies.join(';')]).join(',') + '\n');
      } else {
        const eP = await scoreOne(r.smiles_protonated);
        const eD = await scoreOne(r.smiles_deprotonated);
        out.write(header.map((h) => r[h]).concat([eP, eD]).join(',') + '\n');
      }
      kept++;
    } catch (err) {
      failed++;
      console.error(`row ${i} FAILED: ${err.message}`);
    }
    if ((i + 1) % 10 === 0 || i === rows.length - 1) {
      const elapsed = (Date.now() - t0) / 1000;
      const rate = (i + 1) / elapsed;
      const eta = (rows.length - i - 1) / rate;
      console.error(`[${i + 1}/${rows.length}] kept=${kept} failed=${failed} elapsed=${elapsed.toFixed(0)}s rate=${rate.toFixed(2)}/s eta=${(eta / 60).toFixed(1)}min`);
    }
  }
  out.end();
  console.error(`done. kept=${kept} failed=${failed}`);
}

main().catch((err) => { console.error('FATAL', err); process.exit(1); });
