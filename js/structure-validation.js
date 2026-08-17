/**
 * structure-validation.js
 *
 * A dedicated structural-validation layer that runs BEFORE any property
 * model, rather than letting each model independently discover a problem
 * (or worse, silently produce a number for a structure it was never
 * meant to see). Two entry points:
 *
 *   CC.Validate.checkStructure(molecule)
 *     Real structural checks -- valence, charges, radicals, fragments/
 *     salts, duplicates, stereochemistry, aromaticity, explicit
 *     hydrogens, metals, hypervalent atoms, unusual isotopes. Runs once
 *     per molecule (see the Validation side-panel / runValidation in
 *     app.js), independent of which models are even loaded.
 *
 *   CC.Validate.checkModelCompatibility(molecule, structureReport)
 *     For every model this app knows about (the registry + the built-in
 *     GeoMol/ANI-2x/OpenFF/NAGL engines), combines that structural report
 *     with each engine's own hard compatibility rule (CC.NAGL.
 *     checkCompatibility, CC.ANI.checkCompatibility, CC.GeoMol.
 *     checkCompatibility, CC.GNN.checkChempropCompatibility -- all
 *     pre-existing, reused here rather than duplicated) into ONE
 *     three-tier verdict per model:
 *       'blocked'    -- structure incompatible; prediction refused
 *       'warning'    -- structure valid, but likely outside the model's
 *                       real training/applicability domain; prediction
 *                       permitted, with a reason attached
 *       'compatible' -- no known issue
 *
 * ---------------------------------------------------------------------
 * Method (what's really checked vs. what's a documented heuristic)
 * ---------------------------------------------------------------------
 * Every structural check below reads directly off RDKit.js's own
 * CommonChem get_json() export (or get_stereo_tags()/get_frags()), not a
 * reimplementation of RDKit's own sanitization/perception -- verified
 * directly against a live RDKit.js instance before writing this file
 * (field names like `chg`/`nRad`/`isotope`/`bo`/`stereo`, get_frags()'s
 * MolList iterator shape, get_stereo_tags()'s "(?)" undefined-marker
 * convention), not assumed from memory or from a different RDKit build.
 * Two checks don't have a direct RDKit API and are computed here instead,
 * both using real, well-established methods, not guesses:
 *   - "Undefined E/Z": RDKit's get_stereo_tags() only lists bonds it
 *     already assigned a CIP descriptor to -- a potentially-stereogenic
 *     double bond with NO defined stereo simply doesn't appear there at
 *     all, so it can't be told apart from "not stereogenic" that way.
 *     This uses RDKit's own per-atom CIP-like canonical ranks
 *     (get_json()'s `cipRanks` extension) to determine whether each
 *     double-bond terminus has two distinguishable substituents (the
 *     same substitution-symmetry test real stereogenic-bond perception
 *     uses), then flags the bond only if both ends are asymmetric, it's
 *     not in a ring, and RDKit assigned it no stereo.
 *   - "Aromaticity inconsistency": flags an SSSR ring whose bonds
 *     alternate single/double (the pattern a chemist would read as
 *     "meant to be aromatic") that RDKit's own aromaticity model does
 *     NOT mark aromatic (e.g. an 8-membered ring -- 4n+2 not satisfied).
 *     Deliberately NOT flagged the other direction (an aromatic ring
 *     RDKit accepts without full alternation, e.g. pyrrole/furan) --
 *     that's completely normal heteroaromatic chemistry, not a mistake.
 *
 * The applicability-domain WARNING tier in checkModelCompatibility is an
 * honest heuristic, not a per-model verified domain-of-applicability
 * study: it reflects general, well-known QSAR/cheminformatics practice
 * (most public ADMET/property training sets are curated to single,
 * neutral-ish, standard-isotope, non-radical organic structures) applied
 * per model *engine family*, the same granularity this app's existing
 * CC.GNN.checkChempropCompatibility already uses ("checked against the
 * model families in general, not just whichever specific checkpoint
 * happens to be loaded"). It is not a claim that any specific loaded
 * checkpoint's real training set has been inspected structure-by-structure.
 */

window.CC = window.CC || {};
CC.Validate = window.CC.Validate || {};

(function () {
  // Common metals relevant to organometallic/coordination-complex drug
  // chemistry -- not an exhaustive periodic table classification (no
  // lanthanides/actinides beyond La/Ce, no superheavy elements).
  const METAL_ELEMENTS = new Set([
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
    'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po',
    'La', 'Ce',
  ]);

  // Lowest/"default" valence per element, for hypervalency detection --
  // exceeding (standard valence + formal charge) while still being a
  // real, RDKit-sanitizable structure is exactly what "hypervalent"
  // means here (e.g. neutral pentavalent P, hexavalent S) -- distinct
  // from a genuine valence violation (RDKit refuses to sanitize at all).
  // Elements not listed here (metals, etc.) have no well-defined "normal"
  // valence to compare against and are skipped by this specific check.
  const STANDARD_VALENCE = { H: 1, B: 3, C: 4, N: 3, O: 2, F: 1, Si: 4, P: 3, S: 2, Cl: 1, Br: 1, I: 1 };

  // Canonical SMILES (RDKit's own output, confirmed live) -> friendly
  // name, for recognizing common small counterions among disconnected
  // fragments. Deliberately small/curated, not exhaustive -- an
  // unrecognized small ionic fragment still gets flagged generically as
  // a probable counterion by size/charge heuristic (see classifyFragment).
  const KNOWN_COUNTERIONS = {
    '[Na+]': 'sodium', '[K+]': 'potassium', '[Li+]': 'lithium', '[Cs+]': 'cesium', '[Rb+]': 'rubidium',
    '[Ca+2]': 'calcium', '[Mg+2]': 'magnesium', '[Zn+2]': 'zinc', '[NH4+]': 'ammonium', '[H+]': 'proton',
    '[Cl-]': 'chloride', '[Br-]': 'bromide', '[I-]': 'iodide', '[F-]': 'fluoride', '[OH-]': 'hydroxide',
    'O=C(O)C(=O)O': 'oxalic acid', 'CC(=O)O': 'acetic acid', 'O=S(=O)(O)O': 'sulfuric acid',
    'O=C(O)C(F)(F)F': 'trifluoroacetic acid', 'O': 'water (hydrate)',
  };

  function heavyAtomsInOrder(molecule) {
    return Array.from(molecule.atoms.values());
  }

  // ---------- checks that don't need RDKit ----------

  function checkExplicitHydrogens(heavyAtoms) {
    const hAtoms = heavyAtoms.filter(function (a) { return a.element === 'H'; });
    if (hAtoms.length === 0) return null;
    return {
      id: 'explicit-hydrogen', severity: 'info', label: 'Explicit hydrogen atoms',
      message: hAtoms.length + ' hydrogen atom(s) drawn explicitly rather than left implicit -- harmless, but unusual style; every model in this app already adds implicit hydrogens itself where needed.',
      atomIds: hAtoms.map(function (a) { return a.id; }),
    };
  }

  function checkMetals(heavyAtoms) {
    const metalAtoms = heavyAtoms.filter(function (a) { return METAL_ELEMENTS.has(a.element); });
    if (metalAtoms.length === 0) return null;
    const elements = Array.from(new Set(metalAtoms.map(function (a) { return a.element; })));
    return {
      id: 'metal', severity: 'warning', label: 'Metal atom(s)',
      message: 'Contains ' + elements.join(', ') + ' -- essentially no model in this app was trained on organometallic/coordination-complex structures.',
      atomIds: metalAtoms.map(function (a) { return a.id; }),
    };
  }

  // Two atoms drawn on (near enough) the same 2D spot -- almost always an
  // accidental duplicate-paste/duplicate-place, not intentional chemistry.
  function checkCoincidentAtoms(heavyAtoms) {
    const EPS = (window.CC.BOND_LENGTH || 40) * 0.15;
    const flagged = new Set();
    for (let i = 0; i < heavyAtoms.length; i++) {
      for (let j = i + 1; j < heavyAtoms.length; j++) {
        const a = heavyAtoms[i], b = heavyAtoms[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        if (Math.sqrt(dx * dx + dy * dy) < EPS) { flagged.add(a.id); flagged.add(b.id); }
      }
    }
    if (flagged.size === 0) return null;
    return {
      id: 'duplicate-atoms', severity: 'warning', label: 'Coincident atoms',
      message: flagged.size + ' atom(s) sit on (or nearly on) top of another atom -- likely an accidental duplicate rather than intended chemistry.',
      atomIds: Array.from(flagged),
    };
  }

  // ---------- RDKit-backed checks ----------

  function parseCommonChem(mol) {
    const json = JSON.parse(mol.get_json());
    const molData = json.molecules[0];
    const defaults = json.defaults.atom;
    const atoms = molData.atoms.map(function (a) {
      return {
        z: a.z !== undefined ? a.z : defaults.z,
        chg: a.chg !== undefined ? a.chg : defaults.chg,
        nRad: a.nRad !== undefined ? a.nRad : defaults.nRad,
        isotope: a.isotope !== undefined ? a.isotope : defaults.isotope,
      };
    });
    const bonds = molData.bonds.map(function (b) {
      return { atoms: b.atoms, bo: b.bo !== undefined ? b.bo : 1, stereo: b.stereo !== undefined ? b.stereo : 'unspecified' };
    });
    const ext = (molData.extensions || []).find(function (e) { return e.name === 'rdkitRepresentation'; }) || {};
    return { atoms: atoms, bonds: bonds, atomRings: ext.atomRings || [], aromaticBondIdx: new Set(ext.aromaticBonds || []), cipRanks: ext.cipRanks || null };
  }

  const Z_TO_ELEMENT = { 1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I' };

  function checkChargesAndRadicalsAndIsotopes(heavyAtoms, parsed) {
    const issues = [];
    const unusualChargeAtoms = [];
    const radicalAtoms = [];
    const isotopeAtoms = [];
    let netCharge = 0;

    parsed.atoms.forEach(function (a, i) {
      netCharge += a.chg;
      const appAtom = heavyAtoms[i];
      if (!appAtom) return; // implicit/added H, not one of the app's own drawn atoms
      if (Math.abs(a.chg) >= 2) unusualChargeAtoms.push(appAtom);
      if (a.nRad > 0) radicalAtoms.push(appAtom);
      if (a.isotope !== 0) isotopeAtoms.push(appAtom);
    });

    if (unusualChargeAtoms.length > 0 || Math.abs(netCharge) >= 2) {
      issues.push({
        id: 'unusual-formal-charge', severity: 'warning', label: 'Unusual formal charge',
        message: (unusualChargeAtoms.length > 0
          ? unusualChargeAtoms.length + ' atom(s) with |formal charge| ≥ 2'
          : '') +
          (Math.abs(netCharge) >= 2 ? (unusualChargeAtoms.length > 0 ? '; ' : '') + 'net molecular charge is ' + (netCharge > 0 ? '+' : '') + netCharge : '') +
          ' -- uncommon in most training data.',
        atomIds: unusualChargeAtoms.map(function (a) { return a.id; }),
      });
    }
    if (radicalAtoms.length > 0) {
      issues.push({
        id: 'radical', severity: 'warning', label: 'Radical / open-shell atom(s)',
        message: radicalAtoms.length + ' atom(s) with unpaired electron(s) -- essentially no property model in this app was trained on open-shell species.',
        atomIds: radicalAtoms.map(function (a) { return a.id; }),
      });
    }
    if (isotopeAtoms.length > 0) {
      issues.push({
        id: 'unusual-isotope', severity: 'info', label: 'Explicit isotope label(s)',
        message: isotopeAtoms.length + ' atom(s) with an explicit (non-natural-abundance) isotope label -- harmless for most 2D-graph-based models, but relevant if exact mass matters (see HRMS panel).',
        atomIds: isotopeAtoms.map(function (a) { return a.id; }),
      });
    }
    return issues;
  }

  function checkHypervalent(heavyAtoms, parsed) {
    const bondOrderSum = new Array(parsed.atoms.length).fill(0);
    parsed.bonds.forEach(function (b) {
      bondOrderSum[b.atoms[0]] += b.bo;
      bondOrderSum[b.atoms[1]] += b.bo;
    });
    const hypervalentAtoms = [];
    parsed.atoms.forEach(function (a, i) {
      const appAtom = heavyAtoms[i];
      if (!appAtom) return;
      const el = Z_TO_ELEMENT[a.z];
      const standard = STANDARD_VALENCE[el];
      if (standard === undefined) return; // no well-defined "normal" valence for this element (e.g. a metal)
      const expected = standard + a.chg;
      if (expected > 0 && bondOrderSum[i] > expected) hypervalentAtoms.push(appAtom);
    });
    if (hypervalentAtoms.length === 0) return null;
    return {
      id: 'hypervalent', severity: 'info', label: 'Hypervalent atom(s)',
      message: hypervalentAtoms.length + ' atom(s) exceed their element’s standard valence while still being chemically valid (e.g. pentavalent P, hexavalent S) -- common in real chemistry (sulfones, phosphates), but some force-field bond-length/angle tables (see embed3d.js) cover these less precisely than ordinary valences.',
      atomIds: hypervalentAtoms.map(function (a) { return a.id; }),
    };
  }

  // Real substitution-symmetry test for double-bond stereogenicity: a
  // terminus is "distinguishable" if it has a heavy-atom substituent
  // whose canonical CIP rank differs from any other substituent at that
  // same terminus (a lone substituent is always distinguishable from the
  // implicit H filling the rest of that terminus's valence). See file
  // header for why this can't just be read off get_stereo_tags().
  function checkUndefinedEZ(heavyAtoms, parsed, ringBondPairs) {
    if (!parsed.cipRanks) return null;
    const neighborsOf = function (idx) {
      const result = [];
      parsed.bonds.forEach(function (b) {
        if (b.atoms[0] === idx) result.push(b.atoms[1]);
        else if (b.atoms[1] === idx) result.push(b.atoms[0]);
      });
      return result;
    };
    const flaggedAtoms = [];
    parsed.bonds.forEach(function (b) {
      if (b.bo !== 2) return;
      const a1 = b.atoms[0], a2 = b.atoms[1];
      const key = a1 < a2 ? a1 + '_' + a2 : a2 + '_' + a1;
      if (ringBondPairs.has(key)) return; // in-ring double bonds excluded -- see file header

      function endIsStereogenic(center, other) {
        const subs = neighborsOf(center).filter(function (n) { return n !== other; });
        if (subs.length === 0) return false; // terminal =CH2 -- both substituents are H, identical
        if (subs.length === 1) return true; // one real substituent vs. implicit H -- always distinguishable
        return parsed.cipRanks[subs[0]] !== parsed.cipRanks[subs[1]];
      }

      if (!endIsStereogenic(a1, a2) || !endIsStereogenic(a2, a1)) return;
      if (b.stereo !== 'unspecified') return;

      [a1, a2].forEach(function (idx) { if (heavyAtoms[idx]) flaggedAtoms.push(heavyAtoms[idx]); });
    });
    if (flaggedAtoms.length === 0) return null;
    return {
      id: 'undefined-e-z', severity: 'info', label: 'Undefined double-bond (E/Z) geometry',
      message: (flaggedAtoms.length / 2) + ' double bond(s) could be E or Z but have no defined geometry drawn -- 3D generation will pick an arbitrary one.',
      atomIds: flaggedAtoms,
    };
  }

  function checkAromaticityInconsistency(heavyAtoms, parsed) {
    const bondOrderByPair = new Map();
    const aromaticPairSet = new Set();
    parsed.bonds.forEach(function (b, idx) {
      const key = b.atoms[0] < b.atoms[1] ? b.atoms[0] + '_' + b.atoms[1] : b.atoms[1] + '_' + b.atoms[0];
      bondOrderByPair.set(key, b.bo);
      if (parsed.aromaticBondIdx.has(idx)) aromaticPairSet.add(key);
    });

    const flaggedAtoms = new Set();
    let ringCount = 0;
    parsed.atomRings.forEach(function (ring) {
      let doubleCount = 0;
      let allRingBondsKnown = true;
      const pairs = [];
      for (let k = 0; k < ring.length; k++) {
        const a = ring[k], b = ring[(k + 1) % ring.length];
        const key = a < b ? a + '_' + b : b + '_' + a;
        pairs.push(key);
        if (!bondOrderByPair.has(key)) { allRingBondsKnown = false; break; }
        if (bondOrderByPair.get(key) === 2) doubleCount++;
      }
      if (!allRingBondsKnown) return;
      const isAromatic = pairs.every(function (p) { return aromaticPairSet.has(p); });
      const fullyAlternating = doubleCount === Math.floor(ring.length / 2) && ring.length % 2 === 0;
      if (fullyAlternating && !isAromatic) {
        ringCount++;
        ring.forEach(function (idx) { if (heavyAtoms[idx]) flaggedAtoms.add(heavyAtoms[idx]); });
      }
    });
    if (ringCount === 0) return null;
    return {
      id: 'aromaticity-inconsistency', severity: 'warning', label: 'Ring drawn with alternating bonds is not aromatic',
      message: ringCount + ' ring(s) have a fully alternating single/double bond pattern (the way an aromatic ring is usually drawn) but RDKit’s own aromaticity model does not perceive them as aromatic -- likely a ring size or geometry that doesn’t satisfy Hückel’s rule (e.g. an 8-membered ring), not a drawing error necessarily, but worth checking against what was intended.',
      atomIds: Array.from(flaggedAtoms).map(function (a) { return a.id; }),
    };
  }

  function checkStereocenters(heavyAtoms, mol) {
    let tags;
    try { tags = JSON.parse(mol.get_stereo_tags()); } catch (err) { return null; }
    const undefinedAtoms = [];
    (tags.CIP_atoms || []).forEach(function (entry) {
      const idx = entry[0], tag = entry[1];
      if (tag === '(?)' && heavyAtoms[idx]) undefinedAtoms.push(heavyAtoms[idx]);
    });
    if (undefinedAtoms.length === 0) return null;
    return {
      id: 'undefined-stereocenter', severity: 'info', label: 'Undefined stereocenter(s)',
      message: undefinedAtoms.length + ' stereocenter(s) have no defined configuration (no wedge/hash bond drawn) -- 3D generation will pick an arbitrary one, and this may not represent the intended stereoisomer.',
      atomIds: undefinedAtoms.map(function (a) { return a.id; }),
    };
  }

  // ---------- fragments: disconnected / salts / counterions / duplicates ----------

  function checkFragments(heavyAtoms, mol) {
    const issues = [];
    const fragMappingRaw = mol.get_frags();
    const mapping = JSON.parse(fragMappingRaw.mappings);
    const molList = fragMappingRaw.molList;

    const fragments = [];
    molList.reset();
    let fi = 0;
    while (!molList.at_end()) {
      const fmol = molList.next();
      let smiles = '';
      try { smiles = fmol.get_smiles(); } catch (err) { smiles = ''; }
      const atomIdxs = mapping.fragsMolAtomMapping[fi] || [];
      const atomIds = atomIdxs.map(function (idx) { return heavyAtoms[idx] ? heavyAtoms[idx].id : null; }).filter(Boolean);
      fragments.push({ smiles: smiles, atomIds: atomIds, heavyAtomCount: atomIdxs.length });
      fmol.delete();
      fi++;
    }
    molList.delete();

    if (fragments.length <= 1) return { issues: issues, fragments: fragments };

    // Largest (by heavy-atom count) fragment is treated as the "parent"
    // structure -- standard convention, not something this app invented.
    const sorted = fragments.slice().sort(function (a, b) { return b.heavyAtomCount - a.heavyAtomCount; });
    const parent = sorted[0];
    const others = fragments.filter(function (f) { return f !== parent; });

    const namedCounterionAtomIds = [];
    const unnamedSmallFragmentAtomIds = [];
    others.forEach(function (f) {
      if (KNOWN_COUNTERIONS[f.smiles]) {
        namedCounterionAtomIds.push({ atomIds: f.atomIds, name: KNOWN_COUNTERIONS[f.smiles] });
      } else if (f.heavyAtomCount <= 6) {
        unnamedSmallFragmentAtomIds.push(f);
      }
    });

    issues.push({
      id: 'disconnected-fragments', severity: 'info', label: 'Disconnected fragments',
      message: fragments.length + ' disconnected components drawn on the same canvas.',
      atomIds: fragments.reduce(function (acc, f) { return acc.concat(f.atomIds); }, []),
    });

    issues.push({
      id: 'salt', severity: 'warning', label: 'Salt / multi-component structure',
      message: 'The structure has ' + others.length + ' non-parent component(s) alongside the largest fragment -- most single-molecule property models were trained on the neutral parent structure alone, not the salt form.',
      atomIds: others.reduce(function (acc, f) { return acc.concat(f.atomIds); }, []),
    });

    if (namedCounterionAtomIds.length > 0) {
      const names = namedCounterionAtomIds.map(function (c) { return c.name; });
      issues.push({
        id: 'counterion', severity: 'info', label: 'Recognized counterion(s)',
        message: 'Recognized: ' + names.join(', ') + '.',
        atomIds: namedCounterionAtomIds.reduce(function (acc, c) { return acc.concat(c.atomIds); }, []),
      });
    }
    if (unnamedSmallFragmentAtomIds.length > 0) {
      issues.push({
        id: 'counterion', severity: 'info', label: 'Probable counterion(s) (unrecognized)',
        message: unnamedSmallFragmentAtomIds.length + ' small fragment(s) alongside the main structure that look like counterions but aren’t in this app’s small recognized-ion list.',
        atomIds: unnamedSmallFragmentAtomIds.reduce(function (acc, f) { return acc.concat(f.atomIds); }, []),
      });
    }

    // Duplicate fragments -- same canonical SMILES appearing more than once.
    const bySmiles = new Map();
    fragments.forEach(function (f) {
      if (!bySmiles.has(f.smiles)) bySmiles.set(f.smiles, []);
      bySmiles.get(f.smiles).push(f);
    });
    const dupGroups = Array.from(bySmiles.values()).filter(function (g) { return g.length > 1; });
    if (dupGroups.length > 0) {
      issues.push({
        id: 'duplicate-fragments', severity: 'info', label: 'Duplicate fragments',
        message: dupGroups.length + ' distinct structure(s) appear more than once among the disconnected components (e.g. two identical counterions) -- may be intentional.',
        atomIds: dupGroups.reduce(function (acc, g) { return acc.concat(g.reduce(function (a2, f) { return a2.concat(f.atomIds); }, [])); }, []),
      });
    }

    return { issues: issues, fragments: fragments };
  }

  // ---------- entry point: full structural report ----------

  CC.Validate.checkStructure = function (molecule) {
    const empty = { available: true, valid: true, issues: [], fragments: [], counts: { error: 0, warning: 0, info: 0 } };
    if (!molecule || molecule.isEmpty()) return empty;

    const heavyAtoms = heavyAtomsInOrder(molecule);
    const issues = [];

    // Checks that don't need RDKit -- run regardless of whether the
    // structure ends up sanitizing, since they're independently useful.
    [checkExplicitHydrogens(heavyAtoms), checkMetals(heavyAtoms), checkCoincidentAtoms(heavyAtoms)]
      .forEach(function (issue) { if (issue) issues.push(issue); });

    const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
    if (!RDKit) {
      const result = finalize(true, issues, []);
      result.available = false;
      return result;
    }

    const molblock = CC.moleculeToMolblock(molecule);
    let mol = null;
    try {
      mol = RDKit.get_mol(molblock);
    } catch (err) {
      mol = null;
    }

    if (!mol || !mol.is_valid()) {
      if (mol) mol.delete();
      issues.unshift({
        id: 'valence-violation', severity: 'error', label: 'Structure failed sanitization',
        message: 'RDKit could not parse this structure -- check atom valence (e.g. too many bonds for an element’s normal/expanded valence).',
        atomIds: [],
      });
      return finalize(false, issues, []);
    }

    let fragments = [];
    try {
      const parsed = parseCommonChem(mol);
      const ringBondPairs = new Set();
      parsed.atomRings.forEach(function (ring) {
        for (let k = 0; k < ring.length; k++) {
          const a = ring[k], b = ring[(k + 1) % ring.length];
          ringBondPairs.add(a < b ? a + '_' + b : b + '_' + a);
        }
      });

      checkChargesAndRadicalsAndIsotopes(heavyAtoms, parsed).forEach(function (i) { issues.push(i); });
      [
        checkHypervalent(heavyAtoms, parsed),
        checkUndefinedEZ(heavyAtoms, parsed, ringBondPairs),
        checkAromaticityInconsistency(heavyAtoms, parsed),
        checkStereocenters(heavyAtoms, mol),
      ].forEach(function (issue) { if (issue) issues.push(issue); });

      const fragResult = checkFragments(heavyAtoms, mol);
      fragResult.issues.forEach(function (i) { issues.push(i); });
      fragments = fragResult.fragments;
    } finally {
      mol.delete();
    }

    return finalize(true, issues, fragments);
  };

  function finalize(valid, issues, fragments) {
    const counts = { error: 0, warning: 0, info: 0 };
    issues.forEach(function (i) { counts[i.severity] = (counts[i.severity] || 0) + 1; });
    return { available: true, valid: valid, issues: issues, fragments: fragments, counts: counts };
  }

  // ---------- model compatibility (three-tier) ----------

  // A curated set of structural issue ids each model ENGINE family is
  // realistically outside the applicability domain for -- see file
  // header's honesty note. Every engine also inherits the hard
  // 'valence-violation' block automatically (checked separately, not
  // listed per-engine below).
  const APPLICABILITY_WARNING_IDS = {
    chemprop: ['radical', 'salt', 'metal', 'unusual-isotope', 'unusual-formal-charge', 'hypervalent'],
    nagl: ['radical', 'salt', 'metal', 'hypervalent'],
    geomol: ['radical', 'metal'],
    ani2x: ['radical', 'metal'],
  };

  function engineOf(entry) { return entry.engine || 'chemprop'; }

  // Looked up via CC.ModelAdapters (model-adapters.js) rather than an
  // if/else chain hard-coded here per engine -- this used to be a THIRD
  // independent copy of the same dispatch model-registry.js and
  // gnn-inference.js each had their own version of, and it had silently
  // drifted: there was no 'pka' branch at all, so a pKa model's real
  // requirement (its own NAGL charge model must already be loaded --
  // see pka-model.js) was never actually checked here, just silently
  // defaulted to the generic chemprop vocabulary check below instead.
  // Falling back to the 'chemprop' adapter's own validate for any
  // engine with no adapter registered preserves that same generic-
  // vocabulary-check-as-default behavior for a genuinely unknown engine.
  function hardCompatibilityFor(engine, molecule) {
    const adapter = CC.ModelAdapters.get(engine) || CC.ModelAdapters.get('chemprop');
    if (adapter && adapter.validate) return adapter.validate(molecule);
    return { compatible: true, issues: [] };
  }

  function verdictFor(engine, molecule, structureReport) {
    if (!structureReport.valid) {
      return { tier: 'blocked', reasons: ['structure failed RDKit sanitization (see Validation panel)'] };
    }
    const hard = hardCompatibilityFor(engine, molecule);
    if (!hard.compatible) {
      return { tier: 'blocked', reasons: hard.issues };
    }
    const warningIds = APPLICABILITY_WARNING_IDS[engine] || APPLICABILITY_WARNING_IDS.chemprop;
    const matched = structureReport.issues.filter(function (i) { return warningIds.indexOf(i.id) !== -1; });
    if (matched.length > 0) {
      return { tier: 'warning', reasons: matched.map(function (i) { return i.label; }) };
    }
    return { tier: 'compatible', reasons: [] };
  }

  /**
   * Returns an array of { id, displayName, engine, tier, reasons } --
   * one entry per registry model (CC.GNN.getRegistryEntries()) plus the
   * built-in engines that aren't in the registry (OpenFF Sage, the
   * classical/ANI-2x-driven conformer search itself doesn't need a
   * separate entry -- ani2x's own registry entry already covers it).
   * `structureReport` should be a fresh CC.Validate.checkStructure()
   * result for the SAME molecule -- not recomputed here, so a caller
   * that already has one (e.g. the Validation panel) doesn't pay for
   * RDKit parsing twice.
   */
  CC.Validate.checkModelCompatibility = function (molecule, structureReport) {
    if (!molecule || molecule.isEmpty()) return [];
    const results = [];

    if (window.CC.GNN && CC.GNN.getRegistryEntries) {
      CC.GNN.getRegistryEntries().forEach(function (entry) {
        const engine = engineOf(entry);
        const v = verdictFor(engine, molecule, structureReport);
        results.push({ id: entry.id, displayName: entry.displayName || entry.id, engine: engine, tier: v.tier, reasons: v.reasons });
      });
    }

    // OpenFF Sage isn't a registry entry (it's a force field, not a
    // trained checkpoint) -- same 'chemprop'-family warning heuristic
    // applies reasonably well (it's a general organic force field, not
    // parameterized for metals/radicals either).
    const openffVerdict = verdictFor('chemprop', molecule, structureReport);
    results.push({ id: 'openff-sage', displayName: 'OpenFF Sage (SMIRNOFF)', engine: 'openff', tier: openffVerdict.tier, reasons: openffVerdict.reasons });

    return results;
  };
})();
