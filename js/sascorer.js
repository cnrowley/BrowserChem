/**
 * sascorer.js
 *
 * Synthetic Accessibility Score, a bit-exact port of RDKit's
 * Contrib/SA_Score/sascorer.py (Ertl & Schuffenhauer, J. Cheminf. 1:8
 * (2009); implementation by Peter Ertl & Greg Landrum). Runs fully
 * client-side, no server, no model to load beyond a static ~5.4MB
 * fragment-score lookup table (converted from RDKit's own
 * fpscores.pkl.gz — see export_sa_fragment_scores.py).
 *
 * WHY THIS ISN'T A SMALL PORT — read before trusting/modifying this file:
 * SA_Score's "fragment score" half needs the *unfolded* Morgan/ECFP
 * fingerprint identifiers RDKit's fpscores.pkl.gz table is keyed on —
 * the raw 32-bit hash IDs from `GetSparseCountFingerprint`, not a folded
 * bit vector. RDKit.js's exposed `get_morgan_fp()` only returns a folded
 * fixed-size bit vector, so those raw IDs aren't available through any
 * documented JS API. This file instead reimplements RDKit's actual C++
 * Morgan-fingerprint hashing algorithm directly (atom connectivity
 * invariants + the iterative circular-environment growth loop), reverse
 * engineered from RDKit's own source
 * (Code/GraphMol/Fingerprints/{FingerprintUtil,MorganGenerator}.cpp and
 * Code/RDGeneral/hash/hash.hpp) rather than guessed at. It has been
 * validated against real RDKit's `rdFingerprintGenerator.GetMorganGenerator`
 * fragment IDs and against `sascorer.calculateScore()` end-to-end on a
 * large, chemically diverse test set (see the validation harness this
 * project ships with) — but if you ever see this file's score disagree
 * with real RDKit's, trust RDKit and treat it as a bug here, not there.
 *
 * The hash function is RDKit's own vendored (frozen, pre-Boost-1.81)
 * `gboost::hash_combine`, and critically operates in *32-bit* space
 * (`hash_result_t` is `std::uint32_t`, not `size_t`) — every intermediate
 * value here is a plain JS number kept in range via `>>> 0` after each
 * combine, not BigInt/64-bit arithmetic.
 *
 * What's exact vs. approximated:
 *   - Fragment score (the hard half): exact, modulo one deliberate
 *     simplification — the "delta mass" invariant component is always 0
 *     here, since this app has no isotope-labeling feature (RDKit's own
 *     value is 0 for any atom without an explicit isotope anyway, so
 *     this only diverges for isotope-labeled input, which this editor
 *     can't produce).
 *   - Chiral center / spiro-atom / bridgehead-atom counts (feature
 *     score): exact — read directly from RDKit.js's own
 *     `get_descriptors()` (NumAtomStereoCenters, NumSpiroAtoms,
 *     NumBridgeheadAtoms) rather than reimplemented.
 *   - Macrocycle count, ring membership: exact — from the same RDKit
 *     ring-perception pass chemprop-features.js already uses.
 */

window.CC = window.CC || {};

(function () {
  let fragmentIds = null;   // Uint32Array, sorted ascending
  let fragmentScores = null; // Float32Array, same order/index as fragmentIds

  /**
   * Fetch and parse the fragment-score table (sa-fragment-scores.bin +
   * its manifest). Call once at startup; calculateSAScore() works
   * without it too (fragment score just falls back to -4 for every
   * fragment, per RDKit's own default-for-unseen-fragment behavior —
   * see fragmentScore() below), but loading it gives real scores.
   */
  CC.loadSAScoreTable = function (manifestUrl, binUrl) {
    return Promise.all([
      fetch(manifestUrl).then(function (r) {
        if (!r.ok) throw new Error('failed to fetch SA score manifest: ' + r.status);
        return r.json();
      }),
      fetch(binUrl).then(function (r) {
        if (!r.ok) throw new Error('failed to fetch SA score table: ' + r.status);
        return r.arrayBuffer();
      }),
    ]).then(function (results) {
      const manifest = results[0];
      const buf = results[1];
      fragmentIds = new Uint32Array(buf, 0, manifest.count);
      fragmentScores = new Float32Array(buf, manifest.idBytes, manifest.count);
      return { count: manifest.count };
    });
  };

  CC.hasSAScoreTable = function () { return !!fragmentIds; };

  // Binary search fragmentIds (sorted ascending) for `id`; returns the
  // matching score, or -4 (RDKit's own fallback for an unseen fragment —
  // see sascorer.py's `_fscores.get(id, -4)`) if not found or no table
  // is loaded yet.
  function fragmentScore(id) {
    if (!fragmentIds) return -4;
    let lo = 0, hi = fragmentIds.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >>> 1;
      const v = fragmentIds[mid];
      if (v === id) return fragmentScores[mid];
      if (v < id) lo = mid + 1; else hi = mid - 1;
    }
    return -4;
  }

  // ---- RDKit's frozen (pre-Boost-1.81) 32-bit hash_combine ----
  // seed ^= value + 0x9e3779b9 + (seed<<6) + (seed>>2), all mod 2^32.
  function hashCombine32(seed, value) {
    const term = (value >>> 0) + 0x9e3779b9 + (seed << 6) + (seed >>> 2);
    return (seed ^ term) >>> 0;
  }

  function hashVector32(values) {
    let seed = 0;
    for (let i = 0; i < values.length; i++) seed = hashCombine32(seed, values[i]);
    return seed;
  }

  // hash_value(std::pair<A,B>): a fresh seed=0, combine first then second.
  function hashPair32(a, b) {
    let seed = hashCombine32(0, a >>> 0);
    seed = hashCombine32(seed, b >>> 0);
    return seed;
  }

  const BOND_TYPE_SINGLE = 1, BOND_TYPE_DOUBLE = 2, BOND_TYPE_TRIPLE = 3, BOND_TYPE_AROMATIC = 12;

  function bondTypeInvariant(bond, isAromatic) {
    if (isAromatic) return BOND_TYPE_AROMATIC;
    if (bond.order === 3) return BOND_TYPE_TRIPLE;
    if (bond.order === 2) return BOND_TYPE_DOUBLE;
    return BOND_TYPE_SINGLE;
  }

  /**
   * RDKit's getConnectivityInvariants(): the round-0 (radius-0) Morgan
   * atom invariant, per atom, matching MorganAtomInvGenerator's default
   * (includeRingMembership=true).
   */
  function connectivityInvariant(atomicNum, totalDegree, totalNumHs, formalCharge, inRing) {
    const components = [
      atomicNum >>> 0,
      totalDegree >>> 0,
      totalNumHs >>> 0,
      formalCharge >>> 0, // two's-complement wrap for negative charges, matching C++ static_cast<uint32_t>(int)
      0, // delta mass: always 0 -- see file header
    ];
    if (inRing) components.push(1);
    return hashVector32(components);
  }

  /**
   * The Morgan/ECFP circular-environment growth loop (MorganGenerator.cpp's
   * getEnvironments), radius fixed at 2 to match sascorer.py's
   * `GetMorganGenerator(radius=2)`. Returns a Map<fragmentId, count> —
   * equivalent to `GetSparseCountFingerprint().GetNonzeroElements()`.
   *
   * atoms: [{ atomicNum, totalDegree, totalNumHs, formalCharge, inRing }]
   * bonds: [{ a, b, order }] (a/b are atom indices; order is 1/2/3; pass
   *        isAromatic per-bond via bonds[i].isAromatic)
   */
  function morganFragmentCounts(atoms, bonds, radius) {
    const numAtoms = atoms.length;
    const atomInvariants = atoms.map(function (a) {
      return connectivityInvariant(a.atomicNum, a.totalDegree, a.totalNumHs, a.formalCharge, a.inRing);
    });

    const atomBonds = atoms.map(function () { return []; }); // [{bondId, other}]
    bonds.forEach(function (b, bondId) {
      atomBonds[b.a].push({ bondId: bondId, other: b.b });
      atomBonds[b.b].push({ bondId: bondId, other: b.a });
    });
    const bondInvariant = bonds.map(function (b) { return bondTypeInvariant(b, b.isAromatic); });

    const counts = new Map();
    function record(id) { counts.set(id, (counts.get(id) || 0) + 1); }

    let currentInvariants = atomInvariants.slice();
    for (let i = 0; i < numAtoms; i++) record(currentInvariants[i] >>> 0);

    let atomNeighborhoods = atoms.map(function () { return new Set(); }); // bond ids in this atom's environment so far
    const deadAtoms = new Array(numAtoms).fill(false);
    const seenNeighborhoods = new Set(); // serialized bond-id-set keys, global across all atoms/rounds

    for (let layer = 0; layer < radius; layer++) {
      const roundNeighborhoods = atomNeighborhoods.map(function (s) { return new Set(s); });
      const nextInvariants = new Array(numAtoms).fill(0);
      const roundResults = []; // {key, invar, atomIdx}

      for (let atomIdx = 0; atomIdx < numAtoms; atomIdx++) {
        if (deadAtoms[atomIdx]) continue;
        if (atomBonds[atomIdx].length === 0) { deadAtoms[atomIdx] = true; continue; }

        const neighborhoodInvariants = []; // [bondTypeInv, neighborInvariant]
        atomBonds[atomIdx].forEach(function (nb) {
          roundNeighborhoods[atomIdx].add(nb.bondId);
          atomNeighborhoods[nb.other].forEach(function (bid) { roundNeighborhoods[atomIdx].add(bid); });
          neighborhoodInvariants.push([bondInvariant[nb.bondId], currentInvariants[nb.other]]);
        });
        neighborhoodInvariants.sort(function (x, y) { return x[0] - y[0] || x[1] - y[1]; });

        let invar = layer >>> 0;
        invar = hashCombine32(invar, currentInvariants[atomIdx] >>> 0);
        neighborhoodInvariants.forEach(function (pair) {
          invar = hashCombine32(invar, hashPair32(pair[0], pair[1]));
        });
        nextInvariants[atomIdx] = invar >>> 0;

        const key = Array.from(roundNeighborhoods[atomIdx]).sort(function (a, b) { return a - b; }).join(',');
        roundResults.push({ key: key, invar: invar >>> 0, atomIdx: atomIdx });
      }

      // Deterministic ordering only (mirrors the C++ sort of AccumTuple);
      // the resulting count multiset doesn't depend on tie-break order.
      roundResults.sort(function (x, y) {
        if (x.key !== y.key) return x.key < y.key ? -1 : 1;
        return x.invar - y.invar || x.atomIdx - y.atomIdx;
      });

      roundResults.forEach(function (r) {
        if (!seenNeighborhoods.has(r.key)) {
          seenNeighborhoods.add(r.key);
          record(r.invar);
        } else {
          deadAtoms[r.atomIdx] = true;
        }
      });

      currentInvariants = nextInvariants;
      atomNeighborhoods = roundNeighborhoods;
    }

    return counts;
  }

  /**
   * Compute the SA Score (1 = easy to synthesize, 10 = hard) for a
   * molecule. Returns null for an empty molecule. Needs
   * CC.GNN.getRDKitAnnotations (graph-builder.js) and a molblock built
   * the same way chemprop-features.js does.
   */
  CC.calculateSAScore = function (molecule) {
    const atomList = Array.from(molecule.atoms.values());
    const nAtoms = atomList.length;
    if (nAtoms === 0) return null;

    const atomIdToIndex = new Map();
    atomList.forEach(function (a, i) { atomIdToIndex.set(a.id, i); });

    const molblock = CC.moleculeToMolblock(molecule);
    const annotations = CC.GNN.getRDKitAnnotations(molblock);
    const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;

    // Descriptor-derived counts (chiral centers, spiro/bridgehead atoms) —
    // read straight from RDKit rather than reimplemented; see file header.
    let nChiralCenters = 0, nSpiro = 0, nBridgeheads = 0;
    let ringSizes = [];
    if (RDKit) {
      let mol = null;
      try {
        mol = RDKit.get_mol(molblock);
        if (mol && mol.is_valid()) {
          const descriptors = JSON.parse(mol.get_descriptors());
          nChiralCenters = descriptors.NumAtomStereoCenters || 0;
          nSpiro = descriptors.NumSpiroAtoms || 0;
          nBridgeheads = descriptors.NumBridgeheadAtoms || 0;
          const json = JSON.parse(mol.get_json());
          const molData = json.molecules && json.molecules[0];
          const ext = molData && (molData.extensions || []).find(function (e) { return e.name === 'rdkitRepresentation'; });
          ringSizes = ((ext && ext.atomRings) || []).map(function (r) { return r.length; });
        }
      } catch (err) {
        // fall through with zeros -- annotations already degrade gracefully elsewhere
      } finally {
        if (mol) mol.delete();
      }
    }

    const atomsForFp = atomList.map(function (atom, index) {
      const atomicNum = CC.ELEMENT_TO_ATOMIC_NUMBER[atom.element] || 0;
      const heavyDegree = molecule.getDegree(atom.id);
      const numH = annotations.numHByAtomIndex.has(index) ? annotations.numHByAtomIndex.get(index) : 0;
      return {
        atomicNum: atomicNum,
        totalDegree: heavyDegree + numH,
        totalNumHs: numH,
        formalCharge: atom.charge,
        inRing: annotations.ringAtomIndices.has(index),
      };
    });

    const bondsForFp = Array.from(molecule.bonds.values()).map(function (b) {
      const i = atomIdToIndex.get(b.a1), j = atomIdToIndex.get(b.a2);
      const key = i < j ? i + '_' + j : j + '_' + i;
      return { a: i, b: j, order: b.order, isAromatic: annotations.aromaticBondPairs.has(key) };
    });

    const counts = morganFragmentCounts(atomsForFp, bondsForFp, 2);

    // ---- fragment score ----
    let score1 = 0, nf = 0;
    counts.forEach(function (count, id) {
      nf += count;
      score1 += fragmentScore(id) * count;
    });
    score1 /= nf;

    // ---- feature score ----
    const nMacrocycles = ringSizes.filter(function (s) { return s > 8; }).length;
    const sizePenalty = Math.pow(nAtoms, 1.005) - nAtoms;
    const stereoPenalty = Math.log10(nChiralCenters + 1);
    const spiroPenalty = Math.log10(nSpiro + 1);
    const bridgePenalty = Math.log10(nBridgeheads + 1);
    const macrocyclePenalty = nMacrocycles > 0 ? Math.log10(2) : 0;
    const score2 = -sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty;

    // ---- fingerprint density correction ----
    let score3 = 0;
    const numBits = counts.size;
    if (nAtoms > numBits) score3 = Math.log(nAtoms / numBits) * 0.5;

    let sascore = score1 + score2 + score3;

    const min = -4.0, max = 2.5;
    sascore = 11 - ((sascore - min + 1) / (max - min)) * 9;
    if (sascore > 8) sascore = 8 + Math.log(sascore + 1 - 9);
    if (sascore > 10) sascore = 10;
    else if (sascore < 1) sascore = 1;

    return sascore;
  };
})();
