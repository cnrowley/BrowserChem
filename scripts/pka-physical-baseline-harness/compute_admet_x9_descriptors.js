/**
 * compute_admet_x9_descriptors.js
 *
 * Computes the real, model-derived 9-descriptor X_d feature-fusion set
 * (logP, LogD at pH 7, most-acidic/most-basic site pKa + has-site flags,
 * NAGL-MBIS partial-charge min/max/mean) for every unique molecule
 * across whichever training CSV directories are passed in -- offline
 * data prep for "does adding these as extra Chemprop descriptors
 * improve this classifier" experiments (see scripts/join_admet_x9_descriptors.py
 * for the next step). Originally written for the CYP450 panel (renamed
 * from compute_admet_x9_descriptors.js once the same recipe proved worth
 * reusing for Ames mutagenicity and BBBP -- the descriptor formulas
 * themselves were never CYP-specific, only the directory list was).
 *
 * Also computed BDE (weakest C-H bond) as a 10th/11th/12th column in an
 * earlier version -- dropped here after a real experiment (see
 * model/registry.json's cyp{isoform}-substrate-v1 entries) showed it
 * added nothing on top of these 9, in every representation tried
 * (molecule-level X_d summary, alongside SASA, and as a true per-atom
 * Chemprop V_f feature via scripts/build_ch_bde_npz.py). Not worth the
 * extra compute time (the bond-level model needs an explicit-H graph,
 * the slowest part of the original script) for a feature that's closed.
 *
 * Deliberately uses `aqueous-pka` (fast, single-shot atom-level Chemprop
 * regression -- js/chemprop-model.js's generic path) instead of
 * `pka-microstate-freeenergy` for the pKa signal: the latter's
 * CC.PKAFreeEnergy.predictAllSites runs a real SMIRNOFF+GB/SA geometry
 * optimization per microstate (js/pka-physical-baseline.js), several
 * seconds PER SITE -- completely impractical over the tens of thousands
 * of unique molecules here. aqueous-pka is one D-MPNN forward pass, the
 * same order of cost as compute_pka_embeddings.js's own rate.
 *
 * Every descriptor here reuses this app's own REAL, already-deployed
 * inference code, unmodified, loaded via harness.js's `vm` sandbox --
 * not a second parallel reimplementation:
 *   - logP: logp-v1 (CC.GNN.predictChemprop -> molecularProperties.logP)
 *   - ionizable sites: CC.PKAMicrostates.findIonizableSites (SMARTS)
 *   - per-site pKa: aqueous-pka (CC.GNN.predictChemprop, atom-level)
 *   - LogD(pH7): CC.PKATitration.fractionNeutral(sites, pKaValues, 7.0),
 *     logD = logP + log10(fractionNeutral) -- literally js/app.js's own
 *     renderLogD formula.
 *   - NAGL-MBIS charges: CC.NAGL.predict, reduced to min/max/mean over
 *     heavy atoms.
 *
 * Usage:
 *   CC_BASE_URL=http://localhost:8000/ node compute_admet_x9_descriptors.js \
 *     <out.csv> [--dirs data/cyp,data/cyp_substrate,data/ames,data/bbbp] [--limit N]
 *
 * Writes <out.csv> keyed by RDKit-canonical SMILES: smiles,logp,logd,
 * pka_acidic,has_acidic,pka_basic,has_basic,nagl_min,nagl_max,nagl_mean
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { buildSandbox, loadChempropModel, moleculeFromSmiles, REPO_ROOT, BASE_URL } = require('./harness.js');

const EXTRA_JS_FILES = ['js/pka-microstates.js', 'js/pka-titration.js'];
const DEFAULT_DIRS = ['data/cyp', 'data/cyp_substrate', 'data/ames', 'data/bbbp'];

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

function collectUniqueSmiles(dirs) {
  const seen = new Set();
  for (const dir of dirs) {
    const full = path.join(REPO_ROOT, dir);
    if (!fs.existsSync(full)) continue;
    for (const file of fs.readdirSync(full)) {
      if (!file.endsWith('.csv') || file.includes('descriptors') || file.includes('with_descriptors')) continue;
      const rows = parseCsv(fs.readFileSync(path.join(full, file), 'utf8'));
      rows.forEach((r) => { if (r.smiles) seen.add(r.smiles); });
    }
  }
  return Array.from(seen);
}

function parseArgs(argv) {
  const positional = [];
  const opts = { limit: null, dirs: DEFAULT_DIRS };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--limit') opts.limit = parseInt(argv[++i], 10);
    else if (argv[i] === '--dirs') opts.dirs = argv[++i].split(',');
    else positional.push(argv[i]);
  }
  return { positional, opts };
}

async function main() {
  const { positional, opts } = parseArgs(process.argv.slice(2));
  const [outPath] = positional;
  if (!outPath) {
    console.error('usage: node compute_admet_x9_descriptors.js <out.csv> [--dirs d1,d2,...] [--limit N]');
    process.exit(1);
  }

  let smilesList = collectUniqueSmiles(opts.dirs);
  console.error(`found ${smilesList.length} unique SMILES across ${opts.dirs.join(', ')}`);
  if (opts.limit) smilesList = smilesList.slice(0, opts.limit);

  const sandbox = await buildSandbox();
  for (const rel of EXTRA_JS_FILES) {
    const src = fs.readFileSync(path.join(REPO_ROOT, rel), 'utf8');
    vm.runInContext(src, sandbox, { filename: rel });
  }

  await sandbox.CC.NAGL.loadModel(
    'nagl-mbis-charges',
    BASE_URL + 'model/nagl-mbis-charges/manifest.json',
    BASE_URL + 'model/nagl-mbis-charges/weights.bin'
  );
  await loadChempropModel(sandbox, 'logp-v1', 'model/logp/manifest.json', 'model/logp/weights.bin');
  await loadChempropModel(sandbox, 'aqueous-pka', 'model/aqueous-pka/pka-manifest.json', 'model/aqueous-pka/pka.bin');
  console.error('all engines loaded');

  const out = fs.createWriteStream(outPath, { flags: 'w' });
  out.write('smiles,logp,logd,pka_acidic,has_acidic,pka_basic,has_basic,nagl_min,nagl_max,nagl_mean\n');

  let ok = 0, failed = 0;
  const t0 = Date.now();
  for (let i = 0; i < smilesList.length; i++) {
    const rawSmiles = smilesList[i];
    try {
      const mol = moleculeFromSmiles(sandbox, rawSmiles);
      const rdmol = sandbox.chemCanvasLibs.RDKit.get_mol(rawSmiles);
      const canonicalSmiles = rdmol.get_smiles();
      rdmol.delete();

      const logpResult = sandbox.CC.GNN.predictChemprop(mol, 'logp-v1');
      const logP = logpResult.molecularProperties.logP;
      if (typeof logP !== 'number') throw new Error('logp-v1 produced no usable prediction');

      const sites = sandbox.CC.PKAMicrostates.findIonizableSites(mol);
      let pkaAcidic = 7, hasAcidic = 0, pkaBasic = 7, hasBasic = 0;
      let fractionNeutral = 1;
      if (sites.length > 0) {
        const pkaResult = sandbox.CC.GNN.predictChemprop(mol, 'aqueous-pka');
        const pkaByAtomId = {};
        pkaResult.atomIds.forEach((atomId, idx) => {
          const props = pkaResult.atomProperties[idx];
          if (props && typeof props.pka === 'number') pkaByAtomId[atomId] = props.pka;
        });
        const validSites = [], validPKa = [];
        sites.forEach((site) => {
          if (typeof pkaByAtomId[site.atomId] === 'number') {
            validSites.push(site);
            validPKa.push(pkaByAtomId[site.atomId]);
          }
        });
        const acidPkas = validSites.map((s, idx) => (s.cls === 'acid' ? validPKa[idx] : null)).filter((v) => v !== null);
        const basePkas = validSites.map((s, idx) => (s.cls === 'base' ? validPKa[idx] : null)).filter((v) => v !== null);
        if (acidPkas.length) { pkaAcidic = Math.min.apply(null, acidPkas); hasAcidic = 1; }
        if (basePkas.length) { pkaBasic = Math.max.apply(null, basePkas); hasBasic = 1; }
        if (validSites.length) fractionNeutral = sandbox.CC.PKATitration.fractionNeutral(validSites, validPKa, 7.0);
      }
      const logD = logP + Math.log10(fractionNeutral);

      const naglResult = sandbox.CC.NAGL.predict(mol, 'nagl-mbis-charges');
      const charges = naglResult.atomProperties.map((p) => Object.values(p)[0]).filter((v) => typeof v === 'number');
      if (!charges.length) throw new Error('no NAGL charges available');
      const naglMin = Math.min.apply(null, charges);
      const naglMax = Math.max.apply(null, charges);
      const naglMean = charges.reduce((a, b) => a + b, 0) / charges.length;

      out.write([canonicalSmiles, logP, logD, pkaAcidic, hasAcidic, pkaBasic, hasBasic, naglMin, naglMax, naglMean].join(',') + '\n');
      ok++;
    } catch (err) {
      failed++;
      console.error(`row ${i} FAILED (${rawSmiles}): ${err.message}`);
    }
    if ((i + 1) % 500 === 0 || i === smilesList.length - 1) {
      const elapsed = (Date.now() - t0) / 1000;
      console.error(`[${i + 1}/${smilesList.length}] ok=${ok} failed=${failed} elapsed=${elapsed.toFixed(0)}s`);
    }
  }
  out.end();
  console.error(`done. ok=${ok} failed=${failed}`);
}

main().catch((err) => { console.error('FATAL', err); process.exit(1); });
