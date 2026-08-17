/**
 * model-adapters.js
 *
 * A small registration table so the model-registry/prediction/validation
 * layers can look up "how do I load, check, and run model engine X"
 * without each of them independently hard-coding an if/else chain over
 * engine name strings. Before this file existed, that same 5-way
 * if/else (chemprop/nagl/ani2x/geomol/pka) was written out separately in
 * model-registry.js (load + unload), gnn-inference.js (predict), and
 * structure-validation.js (compatibility) -- adding a new engine meant
 * touching all three, and one of them (structure-validation.js) had
 * silently drifted out of sync: it had no 'pka' branch at all, so a pKa
 * model's real compatibility requirement (its own NAGL charge model must
 * already be loaded -- see pka-model.js) was never actually checked
 * there, just silently defaulted to the generic chemprop vocabulary
 * check instead.
 *
 * Two adapter shapes are registered here, not one, because two genuinely
 * different kinds of engine exist in this app:
 *   - kind: 'property'  (chemprop, nagl, pka) -- predict(molecule, id)
 *     returns atom/molecular-level property values, merged into the
 *     Properties panel's results table by gnn-inference.js.
 *   - kind: 'geometry'  (ani2x, geomol) -- generate/optimize a 3D
 *     structure, called directly from the 3D panel and
 *     conformer-search.js, never through the property-prediction merge.
 * Forcing both into one identically-shaped "predict(input)" would hide
 * that real difference rather than simplify anything. Every adapter,
 * regardless of kind, always provides load/unload/hasModel/
 * getLoadedModelIds/validate -- genuinely the same operation (fetch
 * weights, forget weights, is anything loaded, is this molecule okay to
 * run) no matter which kind of work the model actually does.
 *
 * Each engine file (chemprop-model.js, nagl-model.js, pka-model.js,
 * ani2x-model.js, geomol-assembly.js) registers itself once, from the
 * bottom of its own existing IIFE, using functions that already existed
 * before this file did -- this is a wiring/consolidation layer, not new
 * model logic. Load order between this file and the engine files
 * doesn't matter except that this file must load first (it only defines
 * the table; engine files call .register() into it at their own load
 * time), which is why it's near the very top of index.html's script
 * list.
 */

window.CC = window.CC || {};
CC.ModelAdapters = window.CC.ModelAdapters || {};

(function () {
  const registry = new Map(); // engine name -> adapter object

  CC.ModelAdapters.register = function (engineName, adapter) {
    registry.set(engineName, adapter);
  };

  CC.ModelAdapters.get = function (engineName) {
    return registry.get(engineName);
  };

  CC.ModelAdapters.list = function () {
    return Array.from(registry.keys());
  };
})();
