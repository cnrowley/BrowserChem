/**
 * add_extra_features.js
 *
 * Adds real per-microstate extra descriptors to an already-physics-scored
 * microstate CSV, for the feature-fusion (X_d) variant of
 * scripts/train_pka_microstate_freeenergy.py. Same two schemas
 * score_batch.js handles (auto-detected from the header), single-pair or
 * ensemble (semicolon-joined lists per side, one or more microstates --
 * see scripts/prepare_pkahub_data.py's own header for why): per
 * microstate,
 *   - `naglChargeMin`/`naglChargeMax`: the minimum/maximum NAGL-MBIS
 *     partial charge among that microstate's own heavy atoms (CC.NAGL.
 *     predict, already-loaded model). Deliberately NOT "the charge at
 *     the one ionizable atom" -- that needs a real atom-to-atom mapping
 *     between the protonated/deprotonated SMILES (their RDKit atom
 *     indices don't correspond 1:1 after two independent canonical-SMILES
 *     round-trips), which is a real, nontrivial graph-matching problem
 *     this script deliberately avoids getting wrong silently. Min/max
 *     across ALL heavy atoms needs no atom mapping at all and still
 *     captures real chemistry a single collapsed SMIRNOFF energy scalar
 *     can't: an anion's charge concentrated on one atom (localized, less
 *     stabilized) vs. delocalized across a conjugated system (more
 *     stabilized, generally a lower/more-favorable pKa) show up as very
 *     different min values for chemically similar-looking structures.
 *   - `logP`: this project's own existing logp-v1 Chemprop checkpoint's
 *     prediction (the "existing model prediction" the user asked to
 *     fold in) -- lipophilicity is a real, independent correlate of
 *     ionization favorability (desolvation cost) the current physical
 *     baseline (aqueous-only SMIRNOFF+GBSA) doesn't separately expose to
 *     the correction network as its own signal.
 *
 * Cheap relative to score_batch.js's own physical-baseline computation --
 * a NAGL forward pass and one Chemprop forward pass per microstate, no
 * iterative geometry optimization -- so this is deliberately a SEPARATE,
 * fast follow-up pass over an already-scored CSV rather than folded into
 * score_batch.js itself (which stays focused on the genuinely expensive
 * physics-optimization step).
 *
 * Usage:
 *   CC_BASE_URL=http://localhost:8000/ node add_extra_features.js <in.csv> <out.csv> [startRow] [endRow]
 */
const fs = require('fs');
const { buildSandbox, loadEngines, loadChempropModel, moleculeFromSmiles } = require('./harness.js');

function parseCsv(text) {
  // See score_batch.js's own parseCsv comment -- Python's csv module
  // writes '\r\n' row terminators by default; splitting on '\n' alone
  // corrupts the header's last column name with a trailing '\r'.
  const lines = text.trim().split(/\r\n|\n/);
  const header = lines[0].split(',');
  const rows = lines.slice(1).map((line) => {
    const vals = line.split(',');
    const row = {};
    header.forEach((h, i) => { row[h] = vals[i]; });
    return row;
  });
  return { header, rows };
}

function chargeMinMax(sandbox, naglModelId, molecule) {
  const result = sandbox.CC.NAGL.predict(molecule, naglModelId);
  const charges = result.atomProperties.map((p) => Object.values(p)[0]);
  return { min: Math.min(...charges), max: Math.max(...charges) };
}

function netFormalCharge(molecule) {
  let net = 0;
  molecule.atoms.forEach((a) => { net += a.charge; });
  return net;
}

// logp-v1 has its own real applicability-domain vocabulary gate (loaded
// from model/logp/applicability-domain.json in the BROWSER, via
// js/applicability-domain.js -- deliberately NOT loaded into this
// harness's own sandbox, see harness.js's JS_FILES) that refuses to run
// on a net-charged molecule (confirmed directly: "net molecular charge -1
// never appeared in training (trained on net charges: 0, 1)"). Computing
// logP separately per microstate (an earlier version of this script did)
// would train on values the deployed browser can never reproduce for the
// charged side of every single site -- a real offline/online mismatch,
// not just a style choice. Fixed by computing logP ONCE per row, from
// whichever of the pair is net-charge-0 (every row surviving
// scripts/filter_zwitterion_pairs.py has exactly one such structure by
// construction), and sharing that one value into both microstates' own
// feature vectors -- also more physically sensible than "logP of a bare
// anion" in the first place.
function logP(sandbox, molecule) {
  const result = sandbox.CC.GNN.predictChemprop(molecule, 'logp-v1');
  return result.molecularProperties.logP;
}

function chargeFeaturesFor(sandbox, naglModelId, smiles) {
  const mol = moleculeFromSmiles(sandbox, smiles);
  const cm = chargeMinMax(sandbox, naglModelId, mol);
  return { mol, naglChargeMin: cm.min, naglChargeMax: cm.max };
}

// Charge-feature version of score_batch.js's own scoreList: computes
// per-microstate NAGL charge min/max for every SMILES in a semicolon-
// joined list, dropping individual failures rather than the whole list
// (same reasoning as score_batch.js). Returns null if every structure
// failed. The RETURNED smiles list is the SURVIVING subset, same length/
// order as the returned feature arrays -- callers must overwrite the
// smiles_*_list column with it, not pass the original list through
// unchanged (same index-misalignment bug already caught and fixed once
// in score_batch.js).
function chargeFeatureListFor(sandbox, naglModelId, rowIndex, sideLabel, smilesList) {
  const smilesArr = smilesList.split(';');
  const survivingSmiles = [];
  const mols = [];
  const naglMin = [];
  const naglMax = [];
  for (const smi of smilesArr) {
    try {
      const f = chargeFeaturesFor(sandbox, naglModelId, smi);
      survivingSmiles.push(smi);
      mols.push(f.mol);
      naglMin.push(f.naglChargeMin);
      naglMax.push(f.naglChargeMax);
    } catch (err) {
      console.error(`row ${rowIndex} [${sideLabel}] dropped microstate (${smi}): ${err.message}`);
    }
  }
  return survivingSmiles.length ? { smiles: survivingSmiles, mols, naglMin, naglMax } : null;
}

async function main() {
  const inPath = process.argv[2];
  const outPath = process.argv[3];
  const start = process.argv[4] ? parseInt(process.argv[4], 10) : 0;
  const end = process.argv[5] ? parseInt(process.argv[5], 10) : Infinity;

  const { header, rows: allRows } = parseCsv(fs.readFileSync(inPath, 'utf8'));
  const rows = allRows.slice(start, Math.min(end, allRows.length));
  console.error(`loaded ${rows.length} rows (range [${start},${Math.min(end, allRows.length)})) from ${inPath}`);

  const sandbox = await buildSandbox();
  const naglModelId = await loadEngines(sandbox);
  await loadChempropModel(sandbox, 'logp-v1', 'model/logp/manifest.json', 'model/logp/weights.bin');
  console.error('engines ready, naglModelId=' + naglModelId + ', logp-v1 loaded');

  const isEnsemble = header.includes('smiles_protonated_list');
  console.error('schema: ' + (isEnsemble ? 'ensemble (list per side)' : 'single-pair'));

  const newCols = isEnsemble
    ? ['naglChargeMin_protonated_list', 'naglChargeMax_protonated_list',
       'naglChargeMin_deprotonated_list', 'naglChargeMax_deprotonated_list', 'logP']
    : ['naglChargeMin_protonated', 'naglChargeMax_protonated',
       'naglChargeMin_deprotonated', 'naglChargeMax_deprotonated', 'logP'];
  const out = fs.createWriteStream(outPath, { flags: 'w' });
  out.write(header.concat(newCols).join(',') + '\n');

  // The one shared logP value for a whole site: found from whichever
  // microstate, ANYWHERE across both (possibly multi-element) lists, is
  // net-charge-0 -- generalizes the single-pair version's "whichever of
  // the pair is neutral" to "whichever of the whole ensemble is neutral"
  // (every ensemble row has at least one such structure by construction:
  // scripts/prepare_pkahub_data.py only ever groups microspecies within
  // one real formal-charge state at a time, and every transition this
  // project trains on has a net-charge-0 side).
  function findNeutralMol(...molLists) {
    for (const mols of molLists) {
      for (const mol of mols) {
        if (netFormalCharge(mol) === 0) return mol;
      }
    }
    return null;
  }

  let kept = 0, failed = 0;
  const t0 = Date.now();
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    try {
      if (isEnsemble) {
        const p = chargeFeatureListFor(sandbox, naglModelId, i, 'protonated', r.smiles_protonated_list);
        const d = chargeFeatureListFor(sandbox, naglModelId, i, 'deprotonated', r.smiles_deprotonated_list);
        if (!p || !d) throw new Error('one whole side has zero scoreable microstates');
        const neutralMol = findNeutralMol(p.mols, d.mols);
        if (!neutralMol) throw new Error('no microstate in this ensemble is net-charge-0 -- cannot compute a logp-v1-compatible logP');
        const sharedLogP = logP(sandbox, neutralMol);
        const rOut = Object.assign({}, r, {
          smiles_protonated_list: p.smiles.join(';'),
          smiles_deprotonated_list: d.smiles.join(';'),
        });
        const extra = [p.naglMin.join(';'), p.naglMax.join(';'), d.naglMin.join(';'), d.naglMax.join(';'), sharedLogP];
        out.write(header.map((h) => rOut[h]).concat(extra).join(',') + '\n');
      } else {
        const fp = chargeFeaturesFor(sandbox, naglModelId, r.smiles_protonated);
        const fd = chargeFeaturesFor(sandbox, naglModelId, r.smiles_deprotonated);
        const neutralMol = netFormalCharge(fp.mol) === 0 ? fp.mol : netFormalCharge(fd.mol) === 0 ? fd.mol : null;
        if (!neutralMol) throw new Error('neither microstate is net-charge-0 -- cannot compute a logp-v1-compatible logP');
        const sharedLogP = logP(sandbox, neutralMol);
        const extra = [fp.naglChargeMin, fp.naglChargeMax, fd.naglChargeMin, fd.naglChargeMax, sharedLogP];
        out.write(header.map((h) => r[h]).concat(extra).join(',') + '\n');
      }
      kept++;
    } catch (err) {
      failed++;
      console.error(`row ${i} FAILED: ${err.message}`);
    }
    if ((i + 1) % 200 === 0 || i === rows.length - 1) {
      const elapsed = (Date.now() - t0) / 1000;
      const rate = (i + 1) / elapsed;
      console.error(`[${i + 1}/${rows.length}] kept=${kept} failed=${failed} elapsed=${elapsed.toFixed(0)}s rate=${rate.toFixed(2)}/s`);
    }
  }
  out.end();
  console.error(`done. kept=${kept} failed=${failed}`);
}

main().catch((err) => { console.error('FATAL', err); process.exit(1); });
