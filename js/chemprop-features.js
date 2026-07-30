/**
 * chemprop-features.js
 *
 * Bit-exact re-implementation of Chemprop's default featurizers —
 * `MultiHotAtomFeaturizer.v2()` (72-dim) and the default
 * `MultiHotBondFeaturizer` (14-dim) — for use with real trained Chemprop
 * checkpoints (chemprop-model.js). These are deliberately separate from
 * atom-features.js / bond-features.js, which stay simplified and are
 * still what the untrained "demo" D-MPNN uses; mixing the two up would
 * silently feed a real checkpoint the wrong-shaped, wrong-ordered input.
 *
 * Where this can and can't be exact, honestly:
 *   - Element, degree (heavy-atom bonds + H count), formal charge,
 *     aromaticity, and ring membership: exact — either tracked directly
 *     by this app or read from RDKit's own annotation pass.
 *   - Implicit H count: exact — read from RDKit's parsed molblock
 *     (`impHs`), not guessed from a valence table.
 *   - Chiral tag (CW/CCW): exact when RDKit could assign one — it's
 *     perceived from the molblock's wedge/hash bonds + 2D coordinates
 *     during parsing, same as real Chemprop training data prepared from
 *     2D structures. We only ever emit "unspecified", "cw", or "ccw"
 *     (never CHI_OTHER, which real Chemprop essentially never sees either).
 *   - Double-bond stereo (cis/trans): best-effort. RDKit's CommonChem
 *     export only reports a coarse "cis"/"trans" label, not the full
 *     STEREOZ/STEREOE-vs-STEREOCIS/STEREOTRANS distinction Chemprop's
 *     enum index encodes — we map cis->Z(2), trans->E(3), which is the
 *     common case once stereo is CIP-perceived (the usual case for a
 *     structure with real 2D coordinates), but isn't guaranteed for
 *     every edge case.
 *   - Hybridization: NOT exposed by RDKit.js's get_json() at all (no
 *     per-atom hybridization field in CommonChem). Approximated here
 *     from bond orders/aromaticity/degree — see guessHybridization().
 *     This is the one feature block where a real RDKit-perceived value
 *     could disagree with ours.
 *   - Conjugation: same approximation atom/bond-features.js already
 *     uses (aromatic or double bond) — RDKit.js doesn't expose per-bond
 *     conjugation either.
 */

window.CC = window.CC || {};
CC.GNN = window.CC.GNN || {};

(function () {
  // ---- exact choice lists, in Chemprop's own order (index = position) ----
  const ATOMIC_NUM_CHOICES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 53,
  ];
  const DEGREE_CHOICES = [0, 1, 2, 3, 4, 5];
  const FORMAL_CHARGE_CHOICES = [-1, -2, 1, 2, 0];
  const CHIRAL_TAG_CHOICES = [0, 1, 2, 3]; // unspecified, CW, CCW, other
  const NUM_H_CHOICES = [0, 1, 2, 3, 4];
  const HYBRIDIZATION_CHOICES = ['S', 'SP', 'SP2', 'SP2D', 'SP3', 'SP3D', 'SP3D2'];
  const BOND_TYPE_CHOICES = ['single', 'double', 'triple', 'aromatic'];
  const STEREO_CHOICES = [0, 1, 2, 3, 4, 5]; // RDKit BondStereo enum values

  // RDKit's periodic-table average atomic weights for the elements v2
  // covers (H..Kr, plus I) — used exactly as Chemprop does: mass * 0.01.
  const ATOMIC_WEIGHT_BY_Z = {
    1: 1.008, 2: 4.003, 3: 6.941, 4: 9.012, 5: 10.812, 6: 12.011, 7: 14.007,
    8: 15.999, 9: 18.998, 10: 20.18, 11: 22.99, 12: 24.305, 13: 26.982,
    14: 28.086, 15: 30.974, 16: 32.067, 17: 35.453, 18: 39.948, 19: 39.098,
    20: 40.078, 21: 44.956, 22: 47.867, 23: 50.944, 24: 51.996, 25: 54.938,
    26: 55.845, 27: 58.933, 28: 58.693, 29: 63.546, 30: 65.39, 31: 69.723,
    32: 72.61, 33: 74.922, 34: 78.96, 35: 79.904, 36: 83.8, 53: 126.904,
  };

  const ELEMENT_TO_Z = {
    H: 1, He: 2, Li: 3, Be: 4, B: 5, C: 6, N: 7, O: 8, F: 9, Ne: 10,
    Na: 11, Mg: 12, Al: 13, Si: 14, P: 15, S: 16, Cl: 17, Ar: 18, K: 19,
    Ca: 20, Sc: 21, Ti: 22, V: 23, Cr: 24, Mn: 25, Fe: 26, Co: 27, Ni: 28,
    Cu: 29, Zn: 30, Ga: 31, Ge: 32, As: 33, Se: 34, Br: 35, Kr: 36, I: 53,
  };

  // A one-hot block of length choices.length + 1: the last slot is the
  // "not in this list" pad, exactly matching Chemprop's
  // `choices.get(value, len(choices))` fallback.
  function oneHotBlock(value, choices) {
    const vec = new Array(choices.length + 1).fill(0);
    const idx = choices.indexOf(value);
    vec[idx === -1 ? choices.length : idx] = 1;
    return vec;
  }

  function usedValence(molecule, atomId) {
    return molecule.getBondsForAtom(atomId).reduce(function (sum, b) { return sum + b.order; }, 0);
  }

  // Fallback only used when RDKit annotations aren't available (RDKit not
  // ready yet / invalid structure) — prefer annotations.numHByAtomIndex,
  // which reflects RDKit's actual valence model rather than this guess.
  function implicitHCountFallback(molecule, atomId, element) {
    const data = CC.elementData(element);
    return Math.max(0, data.valence - usedValence(molecule, atomId));
  }

  // Best-effort hybridization (see file header: RDKit.js doesn't expose
  // this per-atom, so this is inferred from bond orders/aromaticity/degree
  // rather than read from RDKit directly).
  //
  // One case needs more than bond orders on the atom itself: a lone-pair
  // heteroatom (N/O/S) whose own bonds are all formally single but that
  // sits directly on a conjugated system — ester/carboxylic-acid O, amide
  // N, phenol O, aniline N — donates that lone pair into the adjacent pi
  // system and is sp2 in RDKit's actual perception, not sp3. Detect this
  // by checking whether any neighbor is itself aromatic or multiply-bonded
  // to a third atom.
  function hasConjugatedNeighbor(molecule, atomId, atomIdToIndex, aromaticSet) {
    const bonds = molecule.getBondsForAtom(atomId);
    return bonds.some(function (b) {
      const neighborId = b.a1 === atomId ? b.a2 : b.a1;
      if (aromaticSet.has(atomIdToIndex.get(neighborId))) return true;
      const neighborBonds = molecule.getBondsForAtom(neighborId);
      // A hypervalent neighbor's formal double bond (phosphate P=O,
      // sulfonyl S=O — anything with 4+ heavy-atom neighbors) isn't true
      // pi-conjugation the way a carbonyl's is, so it shouldn't promote
      // *this* atom to sp2 the way an ester/amide carbonyl does.
      if (neighborBonds.length >= 4) return false;
      return neighborBonds.some(function (nb) {
        const third = nb.a1 === neighborId ? nb.a2 : nb.a1;
        return third !== atomId && nb.order >= 2;
      });
    });
  }

  // Bare (unbonded) monatomic ions default differently depending on
  // element family: RDKit calls a naked alkali/alkaline-earth cation
  // (Na+, K+, Mg2+...) hybridization "S" (no valence lone pairs to
  // hybridize), but a naked main-group anion/atom (Cl-, O2-...) "SP3"
  // (its lone pairs are still there). Rare in practice — a 2D sketcher
  // mostly draws bonded structures — but cheap to get right.
  const S_HYBRIDIZED_METALS = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba'];

  function guessHybridization(molecule, atomId, isAromatic, totalDegree, element, atomIdToIndex, aromaticSet) {
    if (totalDegree >= 6) return 'SP3D2';
    if (totalDegree === 5) return 'SP3D';
    if (totalDegree === 4) {
      // Four sigma-bond directions is tetrahedral geometry regardless of
      // any hypervalent double-bond character on top of it — a phosphate
      // P=O or sulfonyl S=O center is still SP3 in RDKit's own
      // perception, not SP2. (A true SP2 double bond, e.g. a carbonyl,
      // can only coexist with at most 3 neighbors in the first place, so
      // this branch only ever fires for genuinely hypervalent atoms.)
      return 'SP3';
    }
    if (isAromatic) return 'SP2';
    // Neutral trivalent boron (boronic acids, boranes, boronate esters) is
    // sp2 regardless of what it's bonded to -- unlike carbon/nitrogen sp2,
    // this isn't from pi-bonding or conjugation, it's just boron's own
    // valence-3 electron count leaving an empty p-orbital. Degree-4 boron
    // (anionic borohydride/borate) is correctly sp3 and already handled
    // by the totalDegree===4 branch above.
    if (element === 'B' && totalDegree === 3) return 'SP2';
    const orders = molecule.getBondsForAtom(atomId).map(function (b) { return b.order; });
    if (orders.indexOf(3) !== -1) return 'SP';
    if (orders.indexOf(2) !== -1) return 'SP2';
    if (totalDegree === 0) return S_HYBRIDIZED_METALS.indexOf(element) !== -1 ? 'S' : 'SP3';
    // Lone-pair conjugation into an adjacent pi system (ester O, amide N,
    // phenol O, aniline N) promotes N/O to sp2 even with only single
    // bonds of their own -- but NOT sulfur: RDKit keeps a single-bonded
    // exocyclic S (thiophenol-type) at SP3 even when attached straight to
    // an aromatic ring, unlike O/N in the same position. (Sulfur still
    // reaches SP2 through its own multiple bond, e.g. a thioamide C=S, or
    // through the hypervalent branch above for sulfonyl/sulfonamide S --
    // both already handled before this point.)
    if (
      (element === 'N' || element === 'O') &&
      hasConjugatedNeighbor(molecule, atomId, atomIdToIndex, aromaticSet)
    ) {
      return 'SP2';
    }
    return 'SP3';
  }

  /**
   * Chemprop's MultiHotAtomFeaturizer.v2(), 72 dims per atom:
   *   atomic number (38) | degree (7) | formal charge (6) | chiral tag (5)
   *   | num Hs (6) | hybridization (8) | aromatic (1) | mass/100 (1)
   */
  CC.GNN.buildChempropAtomFeatures = function (molecule, annotations) {
    const atoms = Array.from(molecule.atoms.values());
    const atomIdToIndex = new Map();
    atoms.forEach(function (a, i) { atomIdToIndex.set(a.id, i); });
    const aromaticSet = (annotations && annotations.aromaticAtomIndices) || new Set();
    const chiralByIdx = (annotations && annotations.chiralTagByAtomIndex) || new Map();
    const numHByIdx = (annotations && annotations.numHByAtomIndex) || new Map();

    // Exposed for buildChempropBondFeatures: which atoms are sp/sp2/aromatic
    // (including lone-pair-conjugated heteroatoms) vs. sp3/hypervalent —
    // "is this atom part of a pi system at all", used to derive bond
    // conjugation below. Filled in alongside the feature rows so
    // hybridization is only computed once per atom.
    const sp2ByAtomId = new Map();

    const rows = atoms.map(function (atom, index) {
      const z = ELEMENT_TO_Z[atom.element] || 0;
      const isAromatic = aromaticSet.has(index);
      const heavyDegree = molecule.getDegree(atom.id);
      const numH = numHByIdx.has(index)
        ? numHByIdx.get(index)
        : implicitHCountFallback(molecule, atom.id, atom.element);
      const totalDegree = heavyDegree + numH;
      const hybridization = guessHybridization(
        molecule, atom.id, isAromatic, totalDegree, atom.element, atomIdToIndex, aromaticSet
      );
      sp2ByAtomId.set(atom.id, hybridization === 'SP' || hybridization === 'SP2' || hybridization === 'SP2D');
      const mass = (ATOMIC_WEIGHT_BY_Z[z] || 12.011) * 0.01;
      const chiralStr = chiralByIdx.get(index); // 'cw' | 'ccw' | undefined
      const chiralTag = chiralStr === 'cw' ? 1 : chiralStr === 'ccw' ? 2 : 0;

      return [].concat(
        oneHotBlock(z, ATOMIC_NUM_CHOICES),
        oneHotBlock(Math.min(totalDegree, 6), DEGREE_CHOICES), // 6 -> forced pad, matches "out of range"
        oneHotBlock(atom.charge, FORMAL_CHARGE_CHOICES),
        oneHotBlock(chiralTag, CHIRAL_TAG_CHOICES),
        oneHotBlock(Math.min(numH, 5), NUM_H_CHOICES),
        oneHotBlock(hybridization, HYBRIDIZATION_CHOICES),
        [isAromatic ? 1 : 0],
        [mass]
      );
    });

    return {
      rows: rows,
      dim: rows.length > 0 ? rows[0].length : 72,
      atomIds: atoms.map(function (a) { return a.id; }),
      sp2ByAtomId: sp2ByAtomId,
    };
  };

  /**
   * Chemprop's default MultiHotBondFeaturizer, 14 dims per bond:
   *   null (1) | bond type (4) | conjugated (1) | in ring (1) | stereo (7)
   */
  /**
   * `sp2ByAtomId` comes from buildChempropAtomFeatures's return value —
   * build atom features for this molecule first and pass its
   * `sp2ByAtomId` map through, so hybridization is only computed once.
   */
  CC.GNN.buildChempropBondFeatures = function (molecule, atomIdToIndex, annotations, sp2ByAtomId) {
    const bonds = Array.from(molecule.bonds.values());
    const aromaticBondPairs = (annotations && annotations.aromaticBondPairs) || new Set();
    const ringBondPairs = (annotations && annotations.ringBondPairs) || new Set();
    const bondStereoByPair = (annotations && annotations.bondStereoByPair) || new Map();
    sp2ByAtomId = sp2ByAtomId || new Map();

    // A bond is conjugated (RDKit's GetIsConjugated) when both of its
    // atoms are part of a pi system (sp/sp2/aromatic, including
    // lone-pair-donating heteroatoms like an ester O or amide N) *and*
    // that pi system extends beyond just the two atoms of this one bond —
    // i.e. at least one endpoint has some other bond into a multiple-bond
    // or lone-pair-conjugated neighbor. Both conditions matter: without
    // the first, a plain single bond between two accidental sp2 atoms
    // wouldn't be excluded; without the second, an isolated alkene like
    // the C=C in 2-butene would wrongly count as conjugated (both its
    // carbons are sp2, but purely *because of* that one bond — nothing
    // extends the system further).
    function isSp2ish(atomId) { return !!sp2ByAtomId.get(atomId); }
    function hasAdjacentConjugation(atomId, excludeBondId) {
      return molecule.getBondsForAtom(atomId).some(function (nb) {
        if (nb.id === excludeBondId) return false;
        const thirdId = nb.a1 === atomId ? nb.a2 : nb.a1;
        return isSp2ish(thirdId);
      });
    }

    const rows = bonds.map(function (b) {
      const i = atomIdToIndex.get(b.a1);
      const j = atomIdToIndex.get(b.a2);
      const key = i < j ? i + '_' + j : j + '_' + i;
      const isAromatic = aromaticBondPairs.has(key);
      const inRing = ringBondPairs.has(key);
      const conjugated = isAromatic || (
        isSp2ish(b.a1) && isSp2ish(b.a2) &&
        (hasAdjacentConjugation(b.a1, b.id) || hasAdjacentConjugation(b.a2, b.id))
      );

      const bondType = isAromatic ? 'aromatic' : b.order === 3 ? 'triple' : b.order === 2 ? 'double' : 'single';

      // 'cis'/'trans' from RDKit -> RDKit's BondStereo enum int (see file
      // header for the STEREOZ/STEREOCIS caveat); anything else -> STEREONONE.
      const stereoStr = bondStereoByPair.get(key);
      const stereoValue = stereoStr === 'cis' ? 2 : stereoStr === 'trans' ? 3 : 0;

      return {
        i: i,
        j: j,
        features: [].concat(
          [0], // "is this bond null?" — always 0, we only featurize real bonds
          oneHotBlock(bondType, BOND_TYPE_CHOICES).slice(0, BOND_TYPE_CHOICES.length), // no pad slot used here
          [conjugated ? 1 : 0],
          [inRing ? 1 : 0],
          oneHotBlock(stereoValue, STEREO_CHOICES)
        ),
      };
    });

    return {
      rows: rows,
      dim: rows.length > 0 ? rows[0].features.length : 1 + BOND_TYPE_CHOICES.length + 2 + (STEREO_CHOICES.length + 1),
    };
  };

  /**
   * Chemprop's V2 featurizer doesn't hard-fail on an out-of-vocabulary
   * element or formal charge the way nagl-features.js does -- oneHotBlock
   * pads gracefully to an "unknown" bucket instead. That's not the same
   * as "still reliable," though: a model was never trained on examples
   * that hit that pad bucket, so a prediction involving one is real
   * information a user should get, just as a softer warning than NAGL's
   * "this genuinely can't run" -- framed as "outside the training
   * vocabulary" rather than blocked outright.
   *
   * Returns { compatible: boolean, issues: string[] }. Deliberately
   * covers the whole (broad) V2 vocabulary, not just what this app's own
   * element picker offers -- relevant for structures loaded via SMILES,
   * which can contain elements the picker itself doesn't.
   */
  CC.GNN.checkChempropCompatibility = function (molecule) {
    if (!molecule || molecule.atoms.size === 0) return { compatible: true, issues: [] };
    const issues = [];
    Array.from(molecule.atoms.values()).forEach(function (atom) {
      const z = ELEMENT_TO_Z[atom.element];
      if (!z || ATOMIC_NUM_CHOICES.indexOf(z) === -1) {
        issues.push('element "' + atom.element + '" is outside the training vocabulary');
      }
      if (FORMAL_CHARGE_CHOICES.indexOf(atom.charge) === -1) {
        issues.push('formal charge ' + atom.charge + ' on a ' + atom.element + ' atom is outside the training vocabulary');
      }
    });
    return { compatible: issues.length === 0, issues: issues };
  };
})();
