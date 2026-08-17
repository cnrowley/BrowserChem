/**
 * lib-loader.js
 *
 * Initializes the third-party engines (RDKit.js, ONNX Runtime Web, 3Dmol.js)
 * that were loaded as UMD globals via <script> tags in index.html, and
 * reports their readiness in the status bar.
 *
 * This file does NOT touch molecule data yet — it only proves the engines
 * load and initialize correctly. Structure logic gets wired up in a later
 * stage.
 *
 * Once all three are ready, window.chemCanvasLibs holds live references:
 *   window.chemCanvasLibs.RDKit   -> RDKit module instance
 *   window.chemCanvasLibs.ort     -> onnxruntime-web namespace
 *   window.chemCanvasLibs.$3Dmol  -> 3Dmol.js namespace
 */

window.chemCanvasLibs = {
  RDKit: null,
  ort: null,
  $3Dmol: null,
};

function setStatus(itemId, state, text) {
  const item = document.getElementById(itemId);
  if (!item) return;
  const dot = item.querySelector('.status-dot');
  const stateEl = item.querySelector('.status-state');
  dot.classList.remove('is-loading', 'is-ready', 'is-error');
  dot.classList.add(state === 'ready' ? 'is-ready' : state === 'error' ? 'is-error' : 'is-loading');
  stateEl.textContent = text;
}

function initRDKit() {
  setStatus('status-rdkit', 'loading', 'initializing\u2026');
  if (typeof window.initRDKitModule !== 'function') {
    setStatus('status-rdkit', 'error', 'script failed to load');
    CC.Logger.error('RDKit.js script failed to load (network/CDN issue?)');
    return;
  }
  window.initRDKitModule()
    .then(function (RDKit) {
      window.chemCanvasLibs.RDKit = RDKit;
      setStatus('status-rdkit', 'ready', RDKit.version());
      console.log('[ChemCanvas] RDKit.js ready, version', RDKit.version());
      CC.Logger.success('RDKit.js ready (v' + RDKit.version() + ')');
    })
    .catch(function (err) {
      setStatus('status-rdkit', 'error', 'init failed');
      console.error('[ChemCanvas] RDKit.js failed to initialize', err);
      CC.Logger.error('RDKit.js failed to initialize: ' + err.message);
    });
}

function initOrt() {
  setStatus('status-ort', 'loading', 'checking\u2026');
  if (typeof window.ort === 'undefined') {
    setStatus('status-ort', 'error', 'script failed to load');
    CC.Logger.error('onnxruntime-web script failed to load (network/CDN issue?)');
    return;
  }
  // ort.min.js executes synchronously and exposes the global immediately —
  // no async init step is required until we actually create a session.
  window.chemCanvasLibs.ort = window.ort;
  setStatus('status-ort', 'ready', 'ready (no model loaded)');
  console.log('[ChemCanvas] onnxruntime-web ready');
  CC.Logger.success('onnxruntime-web ready');
}

function init3Dmol() {
  setStatus('status-3dmol', 'loading', 'checking\u2026');
  if (typeof window.$3Dmol === 'undefined') {
    setStatus('status-3dmol', 'error', 'script failed to load');
    CC.Logger.error('3Dmol.js script failed to load (network/CDN issue?)');
    return;
  }
  window.chemCanvasLibs.$3Dmol = window.$3Dmol;
  setStatus('status-3dmol', 'ready', 'ready (viewer idle)');
  console.log('[ChemCanvas] 3Dmol.js ready');
  CC.Logger.success('3Dmol.js ready');

  // Instantiate an empty viewer in the 3D panel so the container is proven
  // to work, but don't load any molecule into it yet.
  const container = document.getElementById('viewer3d');
  if (container) {
    const viewer = window.$3Dmol.createViewer(container, {
      backgroundColor: 'white',
    });
    viewer.render();
    window.chemCanvasLibs.viewer3d = viewer;
  }
}

function checkWebGPU() {
  const el = document.getElementById('status-webgpu');
  if (!el) return;
  if (navigator.gpu) {
    el.textContent = 'available';
  } else {
    el.textContent = 'unavailable (falls back to WASM)';
  }
}

document.addEventListener('DOMContentLoaded', function () {
  initRDKit();
  initOrt();
  init3Dmol();
  checkWebGPU();
});
