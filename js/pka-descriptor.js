/**
 * pka-descriptor.js
 *
 * Builds the "graph charge shell" descriptor jensengroup/pKalculator's
 * LightGBM C-H pKa model uses (Ree et al., ChemRxiv 2024,
 * 10.26434/chemrxiv-2024-56h5h; MIT-licensed source at
 * github.com/jensengroup/pKalculator, smi2gcs/DescriptorCreator/
 * GraphChargeShell.py) -- a from-scratch JS port of that algorithm, not
 * a guess at it: for a target atom, walk outward shell by shell
 * (n_shells=6), at each shell collecting each already-visited atom's
 * not-yet-visited neighbors into a fixed-size "block" (4 slots for the
 * target atom's own first shell, 3 thereafter -- one neighbor direction
 * is already accounted for past shell 1), padding short blocks with
 * charge-0 dummy slots, sorting each block by CIP-like priority
 * (aggregate neighborhood atomic number, expanding outward until
 * unambiguous) with charge as the tiebreak, then appending every
 * (real-or-dummy) block member's charge to the descriptor vector in
 * that sorted order. Output length is fixed for a given n_shells
 * (1457 for n_shells=6) regardless of the target atom's actual local
 * connectivity, which is what lets a fixed-input-size LightGBM model
 * consume it.
 *
 * Unlike the original, this reads charges from THIS APP's own NAGL-MBIS
 * model (CC.NAGL.predictAll(), model/nagl-mbis-charges/) instead of
 * CM5/GFN1-xTB charges (which need a real xtb binary subprocess -- no
 * WASM build exists, so it can't run client-side; see
 * PKA_INTEGRATION.md). The pka-ch-nagl LightGBM model
 * (model/pka-ch-nagl/) was retrained from scratch on this charge source
 * with the same descriptor shape and hyperparameters as the original
 * paper -- it is NOT the original paper's model, and its accuracy is
 * honestly reported as somewhat worse (see model/registry.json's notes)
 * since NAGL charges are an approximation of real QM-derived CM5
 * charges, not identical to them.
 *
 * One intentional, documented deviation from the original Python: for a
 * target atom whose real neighbor count is less than a shell's fixed
 * block size, the upstream code's CIP tie-break can index a Python list
 * with a stray dummy-atom sentinel (-1) in a way that crashes under
 * every currently-installable RDKit version tested (2021.09-2025.09;
 * confirmed by literally running their unmodified code) -- apparently a
 * latent bug never hit by their own pinned environment. Dummy atoms
 * always contribute a charge of exactly 0 to the descriptor regardless
 * of their exact sorted position among OTHER dummies, so this port
 * gives every dummy slot a fixed, harmless priority (never crashes, and
 * has no effect on real atoms' sort order or on the descriptor's
 * values) instead of reproducing the crash. See
 * scripts/convert_pka_lightgbm.py's training pipeline notes for the
 * empirical check that motivated this.
 */

window.CC = window.CC || {};
CC.PKA = window.CC.PKA || {};

(function () {
  const DUMMY = -1;
  const DUMMY_CHARGE = 0.0;
  const DUMMY_ATOMIC_NUM = -10; // lower than any real element -- see file header

  function atomicNumOf(elements, idx) {
    return idx === DUMMY ? DUMMY_ATOMIC_NUM : (CC.ELEMENT_TO_ATOMIC_NUMBER[elements[idx]] || 0);
  }
  function chargeOf(charges, idx) {
    return idx === DUMMY ? DUMMY_CHARGE : charges[idx];
  }
  function neighborsOf(adjacency, idx) {
    return idx === DUMMY ? [] : adjacency[idx];
  }

  CC.PKA.descriptorLength = function (nShells) {
    let result = 5;
    let previousShell = 4;
    let m = 1;
    while (m < nShells) {
      result += 3 * previousShell;
      previousShell = 3 * previousShell;
      m += 1;
    }
    return result;
  };

  function fillBlock(block, length) {
    while (block.length < length) block.push(DUMMY);
  }

  function sortBlockCharge(block, charges) {
    block.sort(function (a, b) { return chargeOf(charges, a) - chargeOf(charges, b); });
  }

  // CIP-priority sort with a charge tiebreak -- see file header for the
  // one intentional deviation from upstream (dummy atoms never crash the
  // priority lookup, and always compare as lowest/tied-lowest priority).
  function sortBlockCip(block, elements, charges, adjacency) {
    let priorities = block.map(function (idx) { return atomicNumOf(elements, idx); });
    const neighborSets = block.map(function (idx) { return new Set([idx]); });

    function hasDuplicates(arr) { return new Set(arr).size !== arr.length; }

    while (hasDuplicates(priorities)) {
      const oldCardinalities = neighborSets.map(function (s) { return s.size; });
      block.forEach(function (idx, num) {
        neighborsOf(adjacency, idx).forEach(function (nbr) { neighborSets[num].add(nbr); });
      });
      priorities = neighborSets.map(function (set) {
        let sum = 0;
        set.forEach(function (idx) { sum += atomicNumOf(elements, idx); });
        return sum;
      });
      const newCardinalities = neighborSets.map(function (s) { return s.size; });
      let unchanged = true;
      for (let i = 0; i < oldCardinalities.length; i++) {
        if (oldCardinalities[i] !== newCardinalities[i]) { unchanged = false; break; }
      }
      if (unchanged) break;
    }

    // Stable sort block (and priorities in lockstep) ascending by priority.
    const order = block.map(function (_, i) { return i; });
    order.sort(function (i, j) { return priorities[i] - priorities[j]; });
    const sortedBlock = order.map(function (i) { return block[i]; });
    const sortedPriorities = order.map(function (i) { return priorities[i]; });
    block.length = 0;
    Array.prototype.push.apply(block, sortedBlock);

    if (hasDuplicates(sortedPriorities)) {
      // Within each run of tied priorities, re-sort by charge.
      let start = 0;
      while (start < sortedPriorities.length) {
        let end = start + 1;
        while (end < sortedPriorities.length && sortedPriorities[end] === sortedPriorities[start]) end++;
        if (end - start > 1) {
          const slice = block.slice(start, end);
          slice.sort(function (a, b) { return chargeOf(charges, a) - chargeOf(charges, b); });
          for (let k = start; k < end; k++) block[k] = slice[k - start];
        }
        start = end;
      }
    }
  }

  /**
   * elements, charges, adjacency: CC.NAGL.predictAll()'s output (full
   * heavy+H node set). targetIdx: node index of the target (carbon)
   * atom. Returns a plain array of floats, length
   * CC.PKA.descriptorLength(nShells).
   */
  CC.PKA.buildDescriptor = function (elements, charges, adjacency, targetIdx, nShells, useCipSort) {
    let lastShell = [targetIdx];
    let currentShell = [];
    const contained = new Set([targetIdx]);

    const descriptor = [chargeOf(charges, targetIdx)];
    let length = 4;

    for (let shell = 0; shell < nShells; shell++) {
      lastShell.forEach(function (lsIdx) {
        let block = neighborsOf(adjacency, lsIdx).filter(function (nbr) { return !contained.has(nbr); });
        fillBlock(block, length);
        if (useCipSort) sortBlockCip(block, elements, charges, adjacency);
        else sortBlockCharge(block, charges);
        currentShell = currentShell.concat(block);
        currentShell.forEach(function (idx) { if (idx !== DUMMY) contained.add(idx); });
        length = 3;
      });

      currentShell.forEach(function (idx) { descriptor.push(chargeOf(charges, idx)); });
      lastShell = currentShell;
      currentShell = [];
    }

    return descriptor;
  };
})();
