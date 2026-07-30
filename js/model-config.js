/**
 * model-config.js
 *
 * Points at the model registry (model-registry.js) -- the catalog of
 * available Chemprop models, listing dataset/metrics/hyperparameter
 * metadata for each. Defaults to the registry.json bundled with this
 * app (model/registry.json), which is enough for everything to work
 * out of the box with no configuration.
 *
 * Override registryUrl if you'd rather host your models' registry.json
 * (and their manifest.json/weights.bin files) somewhere else entirely --
 * a GitHub repo via its raw.githubusercontent.com URL works well (see
 * CHEMPROP_INTEGRATION.md for why: it serves files with
 * Access-Control-Allow-Origin: *, so cross-origin fetch() just works,
 * no server-side CORS config needed). File paths inside registry.json
 * are resolved relative to wherever registry.json itself was fetched
 * from, so a repo laid out the same way as this project's model/
 * directory will work unmodified once you swap in its raw URL here.
 */

window.CC = window.CC || {};

CC.CONFIG = {
  registryUrl: 'model/registry.json',
};
