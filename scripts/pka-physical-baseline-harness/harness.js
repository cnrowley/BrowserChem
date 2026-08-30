/**
 * harness.js
 *
 * The offline headless-Node counterpart js/pka-physical-baseline.js's own
 * header describes ("a headless-Node batch harness reusing these same
 * three engines unmodified") -- previously referenced but not actually
 * present in this repo checkout; this is that harness. Loads this
 * project's own browser JS files (RDKit.js, this project's from-scratch
 * NAGL-MBIS/SMIRNOFF/GB-SA-solvent engines) completely UNMODIFIED into a
 * Node `vm` sandbox, so `CC.PKAPhysicalBaseline.compute()` runs exactly
 * the same code the browser Titration tab runs -- no reimplementation, no
 * risk of the offline training-data computation silently drifting from
 * what the deployed model actually sees at inference time.
 *
 * --- Why a `vm` sandbox instead of jsdom or a bundler ---
 *
 * These files were never written as CommonJS/ES modules -- every one is a
 * plain `window.CC.X = ...` browser global attachment, with `<script>`
 * load ORDER as the real dependency graph (see CLAUDE.md). `vm.
 * createContext(sandbox)` with `sandbox.window = sandbox` (a self-
 * reference) reproduces that exactly: each file, run via `vm.Script(...).
 * runInContext(sandbox)` in the same order index.html loads them in,
 * attaches to the same shared global object real `window.CC.X` code
 * expects, with zero source changes. No jsdom needed -- confirmed
 * directly (see file-by-file dependency audit below) that none of the
 * physical-baseline code path touches the DOM; the only browser-only
 * APIs it needs (`fetch`, `performance.now()`) are already real Node 18+
 * globals.
 *
 * --- The real (not guessed) dependency list ---
 *
 * Confirmed by tracing actual runtime errors, not just reading headers:
 * an earlier version of this harness omitted js/graph-builder.js (NAGL's
 * `CC.NAGL.predict` calls `CC.GNN.getRDKitAnnotations` unconditionally)
 * and js/implicit-solvent.js + js/dsasa.js + js/steric-accessibility.js
 * (embed3d.js's classical-force-field fallback path -- used whenever
 * SMIRNOFF SMARTS typing fails, e.g. nitro-group-heavy structures --
 * calls `CC.Solvent.predict`, a completely different module from the
 * DOM-only js/solvation.js despite the similar name). Both gaps were
 * caught by getChargesForAtoms3D's own documented graceful-degradation
 * behavior (js/openff-forcefield.js: a failed NAGL call silently returns
 * zero charges rather than throwing) -- the missing-graph-builder.js gap
 * produced NO errors at all, just silently charge-free (physically
 * meaningless) energies for every single molecule, caught only by
 * manually inspecting a known charged NAGL prediction rather than
 * trusting the absence of thrown errors. Real lesson for anyone touching
 * this file: a clean run with zero logged failures is NOT sufficient
 * evidence the physics is real -- spot-check that NAGL charges are
 * actually nonzero (see the harness self-check in score_batch.js's own
 * comments) before trusting a batch's output.
 *
 * `js/model-adapters.js` is needed only because js/nagl-model.js calls
 * `CC.ModelAdapters.register(...)` at load time (throws immediately
 * otherwise). js/solvation.js (the DOM-only solvation-panel UI module,
 * NOT js/implicit-solvent.js's `CC.Solvent`) is deliberately excluded --
 * unrelated to physical-baseline scoring, and it does touch `document`.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const initRDKitModule = require('@rdkit/rdkit');
const mlMatrixLib = require('ml-matrix');

const REPO_ROOT = process.env.CC_REPO_ROOT || path.resolve(__dirname, '../..');
const BASE_URL = process.env.CC_BASE_URL || 'http://localhost:8000/';

const JS_FILES = [
  'js/model-adapters.js',
  'js/molecule.js',
  'js/geometry.js',
  'js/history.js',
  'js/molfile.js',
  'js/chemistry.js',
  'js/vec3.js',
  'js/elements.js',
  'js/embed3d.js',
  'js/internal-coords.js',
  'js/steric-accessibility.js',
  'js/dsasa.js',
  'js/implicit-solvent.js',
  'js/graph-builder.js',
  'js/nagl-features.js',
  'js/nagl-model.js',
  'js/openff-forcefield.js',
  'js/pka-physical-baseline.js',
  // Only needed by callers that also want to run a Chemprop molecule-level
  // model (e.g. logp-v1) via CC.GNN.predictChemprop -- harmless to always
  // load (pure `window.CC.X = ...` attachments, same as everything else
  // here, no DOM/fetch side effects at load time).
  'js/atom-features.js',
  'js/bond-features.js',
  'js/chemprop-features.js',
  'js/dmpnn.js',
  'js/pooling.js',
  'js/gnn-heads.js',
  'js/chemprop-model.js',
];

async function buildSandbox() {
  const sandbox = {};
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.console = console;
  sandbox.fetch = fetch;
  sandbox.performance = performance;
  sandbox.URL = URL;
  sandbox.Math = Math;
  sandbox.setTimeout = setTimeout;
  sandbox.clearTimeout = clearTimeout;
  sandbox.mlMatrix = mlMatrixLib;
  sandbox.location = { href: BASE_URL };
  vm.createContext(sandbox);

  for (const rel of JS_FILES) {
    const src = fs.readFileSync(path.join(REPO_ROOT, rel), 'utf8');
    new vm.Script(src, { filename: rel }).runInContext(sandbox);
  }

  const RDKit = await initRDKitModule({
    locateFile: (file) => path.join(__dirname, 'node_modules/@rdkit/rdkit/dist', file),
  });
  sandbox.chemCanvasLibs = { RDKit, ort: null, $3Dmol: null };

  return sandbox;
}

/**
 * Loads the SMIRNOFF force field and one NAGL-MBIS charge model, fetched
 * over HTTP from a real running static server (`python3 -m http.server`
 * at the repo root, per CLAUDE.md -- `fetch()` can't read `file://` URLs
 * in Node any more than RDKit.js's WASM loader can in the browser, same
 * underlying reason).
 */
async function loadEngines(sandbox, opts) {
  opts = opts || {};
  const naglModelId = opts.naglModelId || 'nagl-mbis-charges';
  await sandbox.CC.OpenFF.loadForceField(BASE_URL + 'data/openff-sage-2.1.0.json');
  await sandbox.CC.NAGL.loadModel(
    naglModelId,
    BASE_URL + 'model/nagl-mbis-charges/manifest.json',
    BASE_URL + 'model/nagl-mbis-charges/weights.bin'
  );
  return naglModelId;
}

function moleculeFromSmiles(sandbox, smiles) {
  const RDKit = sandbox.chemCanvasLibs.RDKit;
  const mol = RDKit.get_mol(smiles);
  if (!mol) throw new Error('RDKit failed to parse SMILES: ' + smiles);
  const molblock = mol.get_molblock();
  mol.delete();
  return sandbox.CC.molblockToMolecule(molblock);
}

/** Loads one molecule-level Chemprop checkpoint (e.g. logp-v1) by its
 * model/registry.json id -- looks up files/task the same way
 * js/model-registry.js's own CC.GNN.loadRegistryModel does, but directly
 * (this harness doesn't load model-registry.js itself, which has its own
 * multi-engine dispatch this single-purpose helper doesn't need). */
async function loadChempropModel(sandbox, registryId, manifestRelPath, binRelPath) {
  await sandbox.CC.GNN.loadChempropModel(registryId, BASE_URL + manifestRelPath, BASE_URL + binRelPath);
  return registryId;
}

module.exports = { buildSandbox, loadEngines, loadChempropModel, moleculeFromSmiles, REPO_ROOT, BASE_URL };
