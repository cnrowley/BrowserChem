/**
 * conformer-search.js
 *
 * A real, from-scratch, client-side conformer search: generate a diverse
 * set of starting geometries, locally optimize every one with a chosen
 * energy model (classical force field / OpenFF Sage SMIRNOFF / ANI-2x),
 * then prune the resulting ensemble down to distinct, low-energy
 * conformers -- an energy value AND a 3D structure per conformer, not
 * just the single best structure CC.optimize3D/CC.OpenFF.optimize3D/
 * CC.ANI.optimizeGeometry each already return on their own.
 *
 * ---------------------------------------------------------------------
 * Why "CREST-inspired" and not literally CREST
 * ---------------------------------------------------------------------
 * CREST (github.com/crest-lab/crest) is a real, widely-used conformer
 * search tool -- but it's a compiled Fortran/C binary built around
 * semiempirical tight-binding QM (GFN-xTB) forces and metadynamics
 * biasing, run as a separate offline process. Neither piece is available
 * here: this project is a static browser page with no server and no
 * native binary execution (see CLAUDE.md), and there is no WASM build of
 * CREST or xtb to load instead. Bundling/calling real CREST is not
 * feasible from a browser tab.
 *
 * What IS reused from CREST, directly from its own source (cloned from
 * github.com/crest-lab/crest to check, not from memory) rather than
 * guessed:
 *   - The overall shape of its pipeline: generate many candidate
 *     geometries -> locally optimize each -> prune to a distinct,
 *     low-energy ensemble. This module follows that same shape, using
 *     systematic/randomized rotatable-bond sampling in place of CREST's
 *     metadynamics step (which needs real QM forces to bias against;
 *     this project has no QM engine at all, semiempirical or otherwise).
 *   - CREST's own CREGEN post-processing defaults, confirmed directly
 *     against src/confparse.f90 in a real CREST checkout:
 *       ewin = 6.0 kcal/mol   -- final energy window above the global minimum
 *       rthr = 0.125 Angstrom -- RMSD threshold (after best-fit alignment)
 *                                below which two conformers count as duplicates
 *     CREGEN's actual comparison also uses a tight energy pre-filter
 *     (ethr=0.05 kcal/mol, only bothering to compute RMSD for pairs
 *     already that close in energy) purely as a performance optimization
 *     for CREST's much larger typical ensembles (thousands of candidate
 *     structures) -- skipped here since this app's conformer counts (tens,
 *     not thousands) make the full O(n^2) all-pairs RMSD comparison cheap
 *     regardless. CREGEN's secondary "RMSD < 2*rthr AND matching
 *     rotational constants" fallback duplicate check is also not
 *     replicated -- the primary rthr check above is what's implemented.
 *   - all-atom (not heavy-atom-only) RMSD as the default, matching
 *     CREGEN's own default (heavyrmsd defaults to false in CREST's
 *     source; heavy-atom-only is an opt-in CREST flag, and an opt-in
 *     option here too via opts.heavyOnlyRmsd).
 *
 * ---------------------------------------------------------------------
 * Energy models
 * ---------------------------------------------------------------------
 * All three of this app's existing 3D optimizers are pluggable here via
 * a small adapter (MODELS below), each reusing that engine's own
 * already-validated single-seed optimizer rather than a new one:
 *   - classical: CC.Embed3DShared.optimizeSeedClassical (embed3d.js)
 *   - smirnoff:  CC.OpenFF.optimizeSeed (openff-forcefield.js), with
 *                electrostatics from NAGL-MBIS if opts.naglModelId names
 *                a loaded model, omitted otherwise -- same substitution
 *                OPENFF_INTEGRATION.md documents for single-structure use.
 *   - ani2x:     CC.ANI.optimizeGeometry (ani2x-model.js), preceded by a
 *                quick classical pre-relax per each seed -- load-bearing,
 *                not optional (see app.js's ANI-2x button handler for the
 *                real structure-collapse case this avoids). Runs far
 *                fewer iterations per seed than a dedicated single-
 *                structure ANI-2x optimize (ANI-2x is documented
 *                elsewhere as genuinely slow, ~3-5 iterations/second) --
 *                conformer *ranking* across many seeds, not one fully
 *                converged structure, is the goal here.
 *
 * Energies from different models are NOT on the same absolute scale
 * (classical: arbitrary hand-tuned units; SMIRNOFF: real kcal/mol; ANI-2x:
 * Hartree) -- MODELS[x].toKcalMol() converts each model's own energy
 * value to kcal/mol *for ranking/energy-window purposes only* (the
 * classical model's "kcal/mol" after this conversion is still not a real
 * physical energy, just its own arbitrary units used as-is, since there's
 * no real unit to convert from). The model's real native energy value is
 * still what's returned per conformer, not silently replaced.
 *
 * ---------------------------------------------------------------------
 * Known limitation, inherited from the existing infrastructure
 * ---------------------------------------------------------------------
 * Seed diversity comes entirely from acyclic rotatable-bond torsion
 * states (embed3d.js's own findRotatableBonds, which already excludes
 * ring bonds -- same definition CC.optimize3D's own multi-attempt search
 * already relies on). Ring-puckering conformers (e.g. cyclohexane
 * chair/boat/twist-boat) are NOT sampled by this search -- this was
 * already true of every other conformer-generating path in this project,
 * not a new gap introduced here, but worth stating plainly rather than
 * implying broader coverage than what's real.
 */

window.CC = window.CC || {};
CC.ConformerSearch = window.CC.ConformerSearch || {};

(function () {
  const HARTREE_TO_KCAL = 627.5094740631;
  const STAGGERED_ANGLES = [60 * Math.PI / 180, 180 * Math.PI / 180, 300 * Math.PI / 180];

  const MODELS = {
    classical: {
      id: 'classical',
      label: 'Classical force field',
      energyUnit: 'arbitrary units (not real kcal/mol -- see embed3d.js)',
      toKcalMol: function (e) { return e; },
      // Solvation needs per-atom partial charges even for this engine
      // (the classical force field itself has none of its own) -- reuses
      // the same NAGL-MBIS charge model the smirnoff engine's
      // electrostatics already relies on, computed once here rather than
      // once per seed. Charges are only fetched at all if solvent is
      // actually enabled -- no reason to require a loaded NAGL model
      // just to run the plain classical force field.
      prepare: async function (molecule, opts) {
        if (!opts.solvent || !opts.solvent.enabled) return {};
        const shared = CC.Embed3DShared;
        const seed = shared.withImplicitHydrogens(molecule, opts.aromaticSet || new Set());
        const chargesResult = CC.OpenFF.getChargesForAtoms3D(molecule, seed.atoms3d, seed.bonds3d, opts.naglModelId);
        return { chargesResult: chargesResult };
      },
      optimizeSeed: async function (atoms3d, bonds3d, aromaticSet, ctx, iterations, deadline, onProgress, solventOpts) {
        const solvent = solventOpts && solventOpts.enabled && ctx.chargesResult && ctx.chargesResult.available
          ? { enabled: true, epsSolvent: solventOpts.epsSolvent, charges: ctx.chargesResult.charges }
          : null;
        return CC.Embed3DShared.optimizeSeedClassical(atoms3d, bonds3d, aromaticSet, iterations, deadline, onProgress, solvent);
      },
    },

    smirnoff: {
      id: 'smirnoff',
      label: 'OpenFF Sage (SMIRNOFF)',
      energyUnit: 'kcal/mol',
      toKcalMol: function (e) { return e; },
      prepare: async function (molecule, opts) {
        if (!CC.OpenFF.isForceFieldLoaded()) await CC.OpenFF.loadForceField();
        const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
        if (!RDKit) throw new Error('RDKit not available yet');
        CC.OpenFF.compileAll(RDKit);
        const shared = CC.Embed3DShared;
        const seed = shared.withImplicitHydrogens(molecule, opts.aromaticSet || new Set());
        const chargesResult = CC.OpenFF.getChargesForAtoms3D(molecule, seed.atoms3d, seed.bonds3d, opts.naglModelId);
        return { RDKit: RDKit, chargesResult: chargesResult };
      },
      optimizeSeed: async function (atoms3d, bonds3d, aromaticSet, ctx, iterations, deadline, onProgress, solventOpts) {
        return CC.OpenFF.optimizeSeed(ctx.RDKit, atoms3d, bonds3d, ctx.chargesResult, iterations, deadline, onProgress, solventOpts);
      },
    },

    ani2x: {
      id: 'ani2x',
      label: 'ANI-2x',
      energyUnit: 'Hartree (converted to kcal/mol for ranking/window)',
      toKcalMol: function (e) { return e * HARTREE_TO_KCAL; },
      prepare: async function (molecule, opts) {
        const compat = CC.ANI.checkCompatibility(molecule);
        if (!compat.compatible) throw new Error('Cannot use ANI-2x: ' + compat.issues.join('; '));
        if (!opts.aniModelId) throw new Error('No ANI-2x model loaded -- pass opts.aniModelId');
        return { modelId: opts.aniModelId };
      },
      // Classical pre-relax first, THEN ANI-2x refinement -- see file
      // header: skipping the pre-relax is a real, previously-observed
      // structure-collapse failure mode for ANI-2x on raw seeds, not a
      // theoretical concern.
      optimizeSeed: async function (atoms3d, bonds3d, aromaticSet, ctx, iterations, deadline, onProgress) {
        const preRelaxIterations = Math.max(60, Math.round(iterations * 0.4));
        const preRelaxDeadline = Math.min(deadline, performance.now() + Math.max(3000, (deadline - performance.now()) * 0.4));
        const relaxed = await CC.Embed3DShared.optimizeSeedClassical(atoms3d, bonds3d, aromaticSet, preRelaxIterations, preRelaxDeadline, function () {
          if (onProgress) onProgress('classical pre-relax');
        });

        const aniIterations = Math.max(30, iterations - preRelaxIterations);
        const aniResult = await CC.ANI.optimizeGeometry(relaxed.atoms, relaxed.bonds, ctx.modelId, {
          maxIterations: aniIterations,
          deadline: deadline,
          onProgress: onProgress ? function () { onProgress('ANI-2x refinement'); } : undefined,
        });
        return {
          energy: aniResult.energy,
          converged: aniResult.converged,
          gradNorm: aniResult.gradNorm,
          atoms: aniResult.atoms,
          bonds: aniResult.bonds,
        };
      },
    },
  };

  CC.ConformerSearch.getModelOptions = function () {
    return Object.keys(MODELS).map(function (id) { return { id: id, label: MODELS[id].label, energyUnit: MODELS[id].energyUnit }; });
  };

  // ---------- seed generation: systematic (small n) or randomized (large n) rotatable-bond torsion sampling ----------

  function comboFromIndex(idx, n, angleChoices) {
    const combo = [];
    let rem = idx;
    for (let b = 0; b < n; b++) { combo.push(angleChoices[rem % angleChoices.length]); rem = Math.floor(rem / angleChoices.length); }
    return combo;
  }

  function applyTorsionCombo(shared, atoms3d, bonds3d, rotatableBonds, combo) {
    rotatableBonds.forEach(function (rb, i) {
      const moving = shared.sideAtoms(bonds3d, atoms3d.length, rb.j, rb.k);
      const origin = atoms3d[rb.j];
      const axis = CC.vec3.sub(atoms3d[rb.k], atoms3d[rb.j]);
      const angle = combo[i];
      moving.forEach(function (idx) {
        const rotated = CC.vec3.rotateAroundAxis(atoms3d[idx], origin, axis, angle);
        atoms3d[idx].x = rotated.x; atoms3d[idx].y = rotated.y; atoms3d[idx].z = rotated.z;
      });
    });
  }

  // Returns an array of { atoms3d, bonds3d } seed geometries -- always
  // includes the plain as-drawn seed first, then either a full
  // combinatorial enumeration of the 3 staggered rotamer states per
  // rotatable bond (when that fits within maxSeeds) or randomized
  // rotamer draws otherwise (same STAGGERED_ANGLES-drawing mechanic
  // CC.optimize3D's own multi-attempt search already uses, via
  // CC.Embed3DShared.randomizeRotatableBonds).
  function buildSeedGeometries(molecule, aromaticSet, rotatableBonds, maxSeeds) {
    const shared = CC.Embed3DShared;
    const seeds = [shared.withImplicitHydrogens(molecule, aromaticSet)];
    const n = rotatableBonds.length;
    if (n === 0 || maxSeeds <= 1) return seeds;

    const remainingSlots = maxSeeds - 1;
    const totalSystematic = Math.pow(3, n);

    if (totalSystematic <= remainingSlots) {
      for (let idx = 0; idx < totalSystematic; idx++) {
        const built = shared.withImplicitHydrogens(molecule, aromaticSet);
        applyTorsionCombo(shared, built.atoms3d, built.bonds3d, rotatableBonds, comboFromIndex(idx, n, STAGGERED_ANGLES));
        seeds.push(built);
      }
    } else {
      for (let s = 0; s < remainingSlots; s++) {
        const built = shared.withImplicitHydrogens(molecule, aromaticSet);
        shared.randomizeRotatableBonds(built.atoms3d, built.bonds3d, rotatableBonds);
        seeds.push(built);
      }
    }
    return seeds;
  }

  // ---------- entry point ----------

  /**
   * opts:
   *   energyModel      'classical' | 'smirnoff' | 'ani2x' (required)
   *   naglModelId      loaded NAGL-MBIS model id -- smirnoff electrostatics only
   *   aniModelId       loaded ANI-2x model id -- required for ani2x
   *   maxSeeds         starting geometries to generate (default scales with
   *                    rotatable-bond count, capped at 30)
   *   maxConformers    cap on the final ensemble size after pruning (default 20)
   *   energyWindowKcal CREST's ewin (default 6.0)
   *   rmsdThreshold    CREST's rthr, Angstrom (default 0.125)
   *   heavyOnlyRmsd    default false -- see file header
   *   iterationsPerSeed, timeBudgetMs -- both have real defaults below
   *   onProgress({ seed, totalSeeds, stage, phase })
   *
   * Returns { conformers: [{rank, energy, relativeEnergyKcal, converged,
   * atoms, bonds}, ...] (sorted ascending by energy), model, modelLabel,
   * energyUnit, seedsGenerated, seedsOptimized, conformersWithinWindow }.
   * `conformers` can be empty (e.g. every seed failed) without throwing --
   * only real preconditions (missing model, incompatible molecule) throw.
   */
  CC.ConformerSearch.run = async function (molecule, opts) {
    opts = opts || {};
    const empty = { conformers: [], model: opts.energyModel, seedsGenerated: 0, seedsOptimized: 0, conformersWithinWindow: 0 };
    if (!molecule || molecule.isEmpty()) return empty;

    const modelAdapter = MODELS[opts.energyModel];
    if (!modelAdapter) throw new Error('Unknown energy model "' + opts.energyModel + '" -- expected classical, smirnoff, or ani2x');

    const initial = CC.buildInitial3D(molecule);
    if (initial.atoms.length === 0) return empty;

    const rotatableBonds = initial.rotatableBonds || [];
    const aromaticSet = initial.aromaticSet || new Set();

    const ctx = await modelAdapter.prepare(molecule, Object.assign({ aromaticSet: aromaticSet }, opts));

    const maxSeeds = opts.maxSeeds || Math.max(4, Math.min(30, 4 + 4 * rotatableBonds.length));
    const seeds = buildSeedGeometries(molecule, aromaticSet, rotatableBonds, maxSeeds);

    const heavyAtomCount = molecule.atoms.size;
    const isAni = opts.energyModel === 'ani2x';
    const perSeedIterations = opts.iterationsPerSeed || (isAni ? 150 : 300);
    const totalBudgetMs = opts.timeBudgetMs ||
      Math.min(180000, Math.max(15000, seeds.length * heavyAtomCount * (isAni ? 700 : 250)));
    const perSeedBudgetMs = Math.max(2500, totalBudgetMs / seeds.length);
    const overallDeadline = performance.now() + totalBudgetMs;

    const optimized = [];
    for (let i = 0; i < seeds.length; i++) {
      if (performance.now() > overallDeadline) break;
      const seed = seeds[i];
      const seedDeadline = Math.min(overallDeadline, performance.now() + perSeedBudgetMs);
      let result;
      try {
        result = await modelAdapter.optimizeSeed(seed.atoms3d, seed.bonds3d, aromaticSet, ctx, perSeedIterations, seedDeadline, function (stage) {
          if (opts.onProgress) opts.onProgress({ seed: i + 1, totalSeeds: seeds.length, stage: stage, phase: 'optimizing' });
        }, opts.solvent);
      } catch (err) {
        // One bad seed (e.g. a pathological starting geometry the
        // optimizer can't recover from) shouldn't sink the whole search
        // -- same graceful-degradation posture as gnn-inference.js's
        // per-model failure handling.
        console.error('[ChemCanvas] conformer search: seed ' + (i + 1) + ' failed', err);
        continue;
      }
      optimized.push(result);
      if (opts.onProgress) {
        opts.onProgress({ seed: i + 1, totalSeeds: seeds.length, phase: 'seed-done', energyKcal: modelAdapter.toKcalMol(result.energy) });
      }
    }

    if (optimized.length === 0) {
      return { conformers: [], model: opts.energyModel, modelLabel: modelAdapter.label, energyUnit: modelAdapter.energyUnit, seedsGenerated: seeds.length, seedsOptimized: 0, conformersWithinWindow: 0 };
    }

    // ---- CREST CREGEN-style pruning (see file header for the real
    // sourced defaults and what's simplified relative to CREST's own
    // implementation).
    const rmsdThreshold = opts.rmsdThreshold != null ? opts.rmsdThreshold : 0.125;
    const energyWindowKcal = opts.energyWindowKcal != null ? opts.energyWindowKcal : 6.0;
    const heavyOnlyRmsd = !!opts.heavyOnlyRmsd;
    const maxConformers = opts.maxConformers || 20;

    const sorted = optimized.slice().sort(function (a, b) { return a.energy - b.energy; });
    const lowestKcal = modelAdapter.toKcalMol(sorted[0].energy);

    let withinWindowCount = 0;
    const kept = [];
    for (let i = 0; i < sorted.length; i++) {
      const c = sorted[i];
      const relKcal = modelAdapter.toKcalMol(c.energy) - lowestKcal;
      if (relKcal > energyWindowKcal) break; // sorted ascending -- nothing after this can be closer
      withinWindowCount++;
      const isDuplicate = kept.some(function (k) {
        return CC.Shape.rmsd(k.atoms, c.atoms, { heavyOnly: heavyOnlyRmsd }) < rmsdThreshold;
      });
      if (isDuplicate) continue;
      kept.push(c);
      if (kept.length >= maxConformers) break;
    }

    const conformers = kept.map(function (c, idx) {
      return {
        rank: idx + 1,
        energy: c.energy,
        relativeEnergyKcal: modelAdapter.toKcalMol(c.energy) - lowestKcal,
        converged: c.converged,
        atoms: c.atoms,
        bonds: c.bonds,
        unmatched: c.unmatched, // SMIRNOFF only -- undefined for other models
      };
    });

    return {
      conformers: conformers,
      model: opts.energyModel,
      modelLabel: modelAdapter.label,
      energyUnit: modelAdapter.energyUnit,
      seedsGenerated: seeds.length,
      seedsOptimized: optimized.length,
      conformersWithinWindow: withinWindowCount,
      chargesAvailable: ctx.chargesResult ? ctx.chargesResult.available : undefined,
    };
  };
})();
