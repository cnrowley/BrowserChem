/**
 * openff-forcefield.js
 *
 * A real OpenFF SMIRNOFF force field ("Sage" -- openff-2.1.0.offxml,
 * converted offline by scripts/convert_openff_forcefield.py into
 * data/openff-sage-2.1.0.json) applied to an actual molecule via SMARTS
 * substructure matching against RDKit.js -- as opposed to embed3d.js's
 * hand-tuned "chemically sane, not quantitatively real" force field.
 * See OPENFF_INTEGRATION.md for the full writeup; summary of what's real
 * vs. substituted here:
 *
 *  - Bonds/Angles/ProperTorsions/ImproperTorsions/vdW parameters: real
 *    Sage 2.1.0 numbers, typed via each parameter's own SMIRKS pattern
 *    (SMARTS with atom-map tags) matched against the actual molecule --
 *    not approximated or hand-tuned.
 *  - Electrostatics: Sage's own protocol calls for AM1-BCC partial
 *    charges, which need semiempirical QM this project doesn't have in a
 *    browser. This uses this project's own NAGL-MBIS charge model
 *    (nagl-model.js) instead -- a real, different-from-upstream
 *    substitution, documented honestly, not silently passed off as the
 *    real Sage protocol. If no NAGL model is loaded, electrostatics is
 *    simply omitted (charges treated as zero) rather than faked.
 *  - Constraints/LibraryCharges/ToolkitAM1BCC sections of the offxml:
 *    not used -- see convert_openff_forcefield.py's header for why.
 *
 * ---------------------------------------------------------------------
 * SMIRKS typing, concretely
 * ---------------------------------------------------------------------
 * Each SMIRNOFF parameter's SMIRKS (e.g. "[#6X4:1]-[#6X4:2]") is a SMARTS
 * pattern with atom-map numbers (:1, :2, ...) marking which matched atom
 * plays which role. RDKit.js's get_qmol()+get_substruct_matches() (the
 * same API smarts-filters.js already uses) does the actual substructure
 * matching; parseSmirksAtomMap() below independently parses the SMIRKS
 * text itself to recover, for each *query* atom position (in the same
 * left-to-right creation order RDKit assigns query atom indices in --
 * standard SMARTS/SMILES parsing behavior, not an assumption specific to
 * this project), which map number (if any) that position carries. That
 * lets a match's `atoms` index array (RDKit's own query-atom-index order)
 * be reinterpreted as "atom at map position 1", "atom at map position 2",
 * etc., without needing every position to be tagged (recursive $(...)
 * sub-patterns and untagged context atoms are common in Sage's own
 * SMIRKS) -- verified by hand against several real Sage patterns
 * including nested recursive SMARTS (e.g. improper torsion i3:
 * "[*:1]~[#7X3$(*~[#15,#16](!-[*])):2](~[*:3])~[*:4]") before trusting
 * it broadly.
 *
 * SMIRNOFF's own typing rule -- "of every parameter whose SMIRKS matches
 * a given real bond/angle/torsion/atom, the LAST one in file order wins"
 * (file order runs general-to-specific by convention, not enforced here)
 * -- is implemented by scanning each parameter list in file order and
 * simply overwriting a `winners` map keyed by the real matched atom
 * indices each time a (possibly-different) pattern matches the same real
 * term; whatever's in the map after the full scan is, by construction,
 * the last (most specific) match. This project's *own* topology (which
 * atoms are bonded, which triples/quadruples form angles/torsions) is
 * enumerated independently from actual connectivity (not from whatever
 * happens to come back from SMARTS matches) so that a real term without
 * any matching SMIRKS parameter is a detectable, honestly-reported gap
 * (see `unmatched` in typeMolecule's result) rather than silently
 * missing.
 *
 * Improper torsions use SMIRNOFF's documented "trefoil" convention: each
 * matched improper center's parameter is applied three times, once per
 * rotation of its three real substituents through the three non-central
 * torsion positions, each instance using k/3 -- this (not the raw k) is
 * what makes the total restraint independent of which neighbor happened
 * to land in which SMARTS-match slot.
 *
 * ---------------------------------------------------------------------
 * Everything else (numeric optimization, conformer search)
 * ---------------------------------------------------------------------
 * Reuses embed3d.js's already-validated implicit-hydrogen placement,
 * rotatable-bond detection/seeding, and generic finite-difference
 * minimization plumbing via CC.Embed3DShared -- only the energy
 * function and its SMIRNOFF-specific parameter typing are new here. The
 * staged-minimization schedule (bonds+angles -> torsions/impropers ->
 * ramp in nonbonded) mirrors embed3d.js's minimizeStaged for the same
 * reason it exists there: a cold, freshly torsion-randomized start is a
 * much better-conditioned optimization problem when the stiff/expensive
 * terms are introduced gradually rather than all at once.
 *
 * VALIDATION STATUS: the parameter *conversion* (offxml -> JSON) is
 * mechanical unit-checked extraction, not something that can silently
 * drift (see convert_openff_forcefield.py). The SMIRKS *typing* logic
 * above has been checked by hand against real Sage patterns but NOT
 * cross-validated end-to-end against real OpenFF toolkit output for
 * whole molecules (no openff-toolkit install available in this
 * environment) -- treat resulting geometries/energies as a real,
 * from-scratch SMIRNOFF implementation with real Sage numbers, not yet
 * bit-exact-confirmed the way this project's Chemprop/NAGL ports are.
 */

window.CC = window.CC || {};
CC.OpenFF = window.CC.OpenFF || {};

(function () {
  // ONE_4PI_EPS0 (OpenMM's own name for this constant): 138.935456
  // kJ*nm/(mol*e^2), converted to kcal*Angstrom/(mol*e^2) -- the standard
  // Coulomb's-law prefactor these units need or Angstrom.
  const COULOMB_CONST = 332.0637128;
  const GRAD_H = 1e-4;

  let ffData = null;
  let compiled = null; // { bonds, angles, properTorsions, improperTorsions, vdw }, each an array of {param, qmol, atomMap}

  CC.OpenFF.loadForceField = function (url) {
    url = url || 'data/openff-sage-2.1.0.json';
    return fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error('failed to fetch ' + url + ': ' + r.status);
        return r.json();
      })
      .then(function (data) {
        ffData = data;
        compiled = null; // force recompilation against the new data next use
        return data;
      });
  };

  CC.OpenFF.isForceFieldLoaded = function () { return !!ffData; };
  CC.OpenFF.getForceFieldMeta = function () {
    if (!ffData) return null;
    return {
      source: ffData.source,
      aromaticityModel: ffData.aromaticityModel,
      counts: {
        bonds: ffData.bonds.length,
        angles: ffData.angles.length,
        properTorsions: ffData.properTorsions.length,
        improperTorsions: ffData.improperTorsions.length,
        vdw: ffData.vdw.length,
      },
    };
  };

  // ---------- SMIRKS atom-map parsing (see file header) ----------

  function skipRecursiveSmarts(smirks, openParenIndex) {
    let depth = 0;
    let j = openParenIndex;
    for (; j < smirks.length; j++) {
      if (smirks[j] === '(') depth++;
      else if (smirks[j] === ')') {
        depth--;
        if (depth === 0) { j++; break; }
      }
    }
    return j;
  }

  function parseSmirksAtomMap(smirks) {
    const mapNumbers = [];
    const n = smirks.length;
    let i = 0;

    while (i < n) {
      const ch = smirks[i];

      if (ch === '$' && smirks[i + 1] === '(') {
        i = skipRecursiveSmarts(smirks, i + 1);
        continue;
      }

      if (ch === '[') {
        let j = i + 1;
        while (j < n && smirks[j] !== ']') {
          if (smirks[j] === '$' && smirks[j + 1] === '(') {
            j = skipRecursiveSmarts(smirks, j + 1);
            continue;
          }
          j++;
        }
        const atomText = smirks.slice(i, j + 1);
        const m = /:(\d+)\]$/.exec(atomText);
        mapNumbers.push(m ? parseInt(m[1], 10) : null);
        i = j + 1;
        continue;
      }

      if (ch === '*') { mapNumbers.push(null); i++; continue; }

      // Bare (non-bracket) organic-subset atoms -- Sage's own SMIRKS
      // always use bracket atoms for anything meaningful, but handled
      // generically for robustness against any pattern that doesn't.
      if (/[A-Za-z]/.test(ch)) {
        const two = smirks.substr(i, 2);
        if (two === 'Cl' || two === 'Br') { mapNumbers.push(null); i += 2; continue; }
        mapNumbers.push(null);
        i += 1;
        continue;
      }

      // Bond symbols (-=#:~@/\!), ring-closure digits, branch parens,
      // ',', ';', '&' (only meaningful inside brackets, already consumed
      // above) -- none of these introduce a new top-level query atom.
      i++;
    }

    return mapNumbers;
  }

  // ---------- compiling SMIRKS patterns against RDKit ----------

  function compileList(RDKit, list) {
    return list.map(function (p) {
      let qmol = null;
      try {
        qmol = RDKit.get_qmol(p.smirks);
        if (!qmol || !qmol.is_valid()) qmol = null;
      } catch (err) {
        qmol = null;
      }
      return { param: p, qmol: qmol, atomMap: parseSmirksAtomMap(p.smirks) };
    });
  }

  function compileAll(RDKit) {
    if (compiled) return compiled;
    compiled = {
      bonds: compileList(RDKit, ffData.bonds),
      angles: compileList(RDKit, ffData.angles),
      properTorsions: compileList(RDKit, ffData.properTorsions),
      improperTorsions: compileList(RDKit, ffData.improperTorsions),
      vdw: compileList(RDKit, ffData.vdw),
    };
    return compiled;
  }

  // ---------- real topology (independent of SMIRKS matching -- see file header) ----------

  function neighborsOf(bonds3d, atomIndex) {
    const result = [];
    bonds3d.forEach(function (b) {
      if (b.a1 === atomIndex) result.push(b.a2);
      else if (b.a2 === atomIndex) result.push(b.a1);
    });
    return result;
  }

  function allAngles(bonds3d, atomCount) {
    const angles = [];
    for (let j = 0; j < atomCount; j++) {
      const nbrs = neighborsOf(bonds3d, j);
      for (let a = 0; a < nbrs.length; a++) {
        for (let b = a + 1; b < nbrs.length; b++) {
          angles.push({ i: nbrs[a], j: j, k: nbrs[b] });
        }
      }
    }
    return angles;
  }

  function allPropers(bonds3d, atomCount) {
    const propers = [];
    bonds3d.forEach(function (centralBond) {
      const j = centralBond.a1, k = centralBond.a2;
      const isAtoms = neighborsOf(bonds3d, j).filter(function (x) { return x !== k; });
      const lsAtoms = neighborsOf(bonds3d, k).filter(function (x) { return x !== j; });
      if (isAtoms.length === 0 || lsAtoms.length === 0) return;
      isAtoms.forEach(function (i) {
        lsAtoms.forEach(function (l) {
          if (i === l) return; // degenerate in a 3-membered ring
          propers.push({ i: i, j: j, k: k, l: l });
        });
      });
    });
    return propers;
  }

  // Every atom with exactly 3 real neighbors is a genuine SMIRNOFF
  // improper-torsion *candidate* -- includes ordinary pyramidal sp3
  // amine nitrogens too (also 3-coordinate), which is correct: SMIRNOFF
  // relies on the SMIRKS patterns themselves (i3-i7's extra recursive
  // conditions, e.g. "attached to C=X" or "in a conjugated ring") to
  // decide which 3-coordinate atoms actually get an improper restraint,
  // not on a hybridization pre-filter the way embed3d.js's hand-tuned
  // model uses.
  function improperCandidates(bonds3d, atomCount) {
    const centers = [];
    for (let i = 0; i < atomCount; i++) {
      if (neighborsOf(bonds3d, i).length === 3) centers.push(i);
    }
    return centers;
  }

  // ---------- canonical keys (so a physical term is the same key regardless of match direction) ----------

  function bondKey(a, b) { return a < b ? a + '_' + b : b + '_' + a; }
  function angleKey(i, j, k) { return j + '_' + (i < k ? i + '_' + k : k + '_' + i); }
  function properKey(i, j, k, l) {
    const fwd = i + '_' + j + '_' + k + '_' + l;
    const rev = l + '_' + k + '_' + j + '_' + i;
    return fwd < rev ? fwd : rev;
  }

  // ---------- SMIRKS matching -> winners maps (last-match-wins) ----------

  function runMatches(mol, entry) {
    if (!entry.qmol) return null;
    try {
      const matches = JSON.parse(mol.get_substruct_matches(entry.qmol));
      return Array.isArray(matches) ? matches : null;
    } catch (err) {
      return null;
    }
  }

  function typeCategory(mol, compiledList, tagCount, keyFn) {
    const winners = new Map();
    compiledList.forEach(function (entry) {
      const matches = runMatches(mol, entry);
      if (!matches) return;
      matches.forEach(function (m) {
        const atoms = m.atoms || [];
        const tagged = new Array(tagCount);
        for (let qi = 0; qi < entry.atomMap.length && qi < atoms.length; qi++) {
          const tag = entry.atomMap[qi];
          if (tag >= 1 && tag <= tagCount) tagged[tag - 1] = atoms[qi];
        }
        for (let t = 0; t < tagCount; t++) if (tagged[t] === undefined) return;
        winners.set(keyFn.apply(null, tagged), entry.param);
      });
    });
    return winners;
  }

  // Impropers/vdW key off a single real atom (the central atom for
  // impropers, the only atom for vdW), not a canonical multi-atom key --
  // the improper trefoil is assembled from real topology afterward (see
  // assignImpropers), not from whichever neighbor slot a given SMARTS
  // match happened to fill.
  function typeByTaggedAtom(mol, compiledList, tagNumber) {
    const winners = new Map();
    compiledList.forEach(function (entry) {
      const matches = runMatches(mol, entry);
      if (!matches) return;
      matches.forEach(function (m) {
        const atoms = m.atoms || [];
        let atomIndex;
        for (let qi = 0; qi < entry.atomMap.length && qi < atoms.length; qi++) {
          if (entry.atomMap[qi] === tagNumber) { atomIndex = atoms[qi]; break; }
        }
        if (atomIndex === undefined) return;
        winners.set(atomIndex, entry.param);
      });
    });
    return winners;
  }

  // ---------- assembling typed energy terms from real topology + winners ----------

  function assignBonds(bonds3d, winners) {
    const terms = []; let unmatched = 0;
    bonds3d.forEach(function (b) {
      const p = winners.get(bondKey(b.a1, b.a2));
      if (!p) { unmatched++; return; }
      terms.push({ i: b.a1, j: b.a2, length: p.length, kBond: p.k });
    });
    return { terms: terms, unmatched: unmatched };
  }

  function assignAngles(candidates, winners) {
    const terms = []; let unmatched = 0;
    candidates.forEach(function (c) {
      const p = winners.get(angleKey(c.i, c.j, c.k));
      if (!p) { unmatched++; return; }
      terms.push({ i: c.i, j: c.j, k: c.k, angle0: p.angle, kAngle: p.k });
    });
    return { terms: terms, unmatched: unmatched };
  }

  function assignPropers(candidates, winners) {
    const terms = []; let unmatched = 0;
    candidates.forEach(function (c) {
      const p = winners.get(properKey(c.i, c.j, c.k, c.l));
      if (!p) { unmatched++; return; }
      terms.push({ i: c.i, j: c.j, k: c.k, l: c.l, terms: p.terms });
    });
    return { terms: terms, unmatched: unmatched };
  }

  // Not every 3-neighbor candidate is expected to match an improper
  // parameter (see improperCandidates' comment) -- an unmatched
  // candidate here is normal SMIRNOFF behavior, not counted as a gap.
  function assignImpropers(centers, bonds3d, winners) {
    const terms = [];
    centers.forEach(function (center) {
      const p = winners.get(center);
      if (!p) return;
      const nbrs = neighborsOf(bonds3d, center); // exactly 3, by construction
      terms.push({ center: center, a: nbrs[0], b: nbrs[1], c: nbrs[2], terms: p.terms });
    });
    return terms;
  }

  function assignVdw(atomCount, winners) {
    const params = new Array(atomCount).fill(null);
    let unmatched = 0;
    for (let i = 0; i < atomCount; i++) {
      const p = winners.get(i);
      if (!p) { unmatched++; continue; }
      params[i] = { epsilon: p.epsilon, rminHalf: p.rminHalf };
    }
    return { params: params, unmatched: unmatched };
  }

  // 1-2/1-3 excluded entirely, 1-4 scaled (vdwScale14/electrostaticsScale14
  // applied separately at energy-evaluation time, since Sage gives them
  // different values), 1-5+ full strength -- built from the real,
  // exhaustive topology candidates (not the typed/matched subsets), since
  // exclusion is purely about graph distance, independent of whether a
  // SMIRKS parameter was actually found for a given angle/torsion.
  function buildPairs(bonds3d, angleCandidates, properCandidates, atomCount) {
    const excluded = new Set();
    bonds3d.forEach(function (b) { excluded.add(bondKey(b.a1, b.a2)); });
    angleCandidates.forEach(function (a) { excluded.add(bondKey(a.i, a.k)); });

    const pairs14 = new Set();
    properCandidates.forEach(function (t) { pairs14.add(bondKey(t.i, t.l)); });

    const pairs = [];
    for (let a = 0; a < atomCount; a++) {
      for (let b = a + 1; b < atomCount; b++) {
        const key = bondKey(a, b);
        if (excluded.has(key)) continue;
        pairs.push({ a: a, b: b, is14: pairs14.has(key) });
      }
    }
    return pairs;
  }

  function typeMolecule(RDKit, atoms3d, bonds3d) {
    const c = compileAll(RDKit);
    const molblock = CC.atoms3DToMolblock(atoms3d, bonds3d, 'openff-typing');
    let mol = null;
    try {
      // removeHs:false is load-bearing -- this project's atom-index
      // correspondence convention (RDKit match index i == atoms3d[i])
      // only holds if RDKit doesn't silently strip the explicit
      // hydrogens atoms3DToMolblock just wrote back out.
      mol = RDKit.get_mol(molblock, JSON.stringify({ removeHs: false }));
      if (!mol || !mol.is_valid()) {
        throw new Error('RDKit could not parse/sanitize the generated 3D structure for SMIRNOFF typing');
      }

      const atomCount = atoms3d.length;
      const angleCandidates = allAngles(bonds3d, atomCount);
      const properCandidates = allPropers(bonds3d, atomCount);
      const improperCenters = improperCandidates(bonds3d, atomCount);

      const bondWinners = typeCategory(mol, c.bonds, 2, function (a1, a2) { return bondKey(a1, a2); });
      const angleWinners = typeCategory(mol, c.angles, 3, function (a1, a2, a3) { return angleKey(a1, a2, a3); });
      const properWinners = typeCategory(mol, c.properTorsions, 4, function (a1, a2, a3, a4) { return properKey(a1, a2, a3, a4); });
      const improperWinners = typeByTaggedAtom(mol, c.improperTorsions, 2);
      const vdwWinners = typeByTaggedAtom(mol, c.vdw, 1);

      const bondsAssigned = assignBonds(bonds3d, bondWinners);
      const anglesAssigned = assignAngles(angleCandidates, angleWinners);
      const propersAssigned = assignPropers(properCandidates, properWinners);
      const improperTerms = assignImpropers(improperCenters, bonds3d, improperWinners);
      const vdwAssigned = assignVdw(atomCount, vdwWinners);
      const pairs = buildPairs(bonds3d, angleCandidates, properCandidates, atomCount);

      return {
        bondTerms: bondsAssigned.terms,
        angleTerms: anglesAssigned.terms,
        properTerms: propersAssigned.terms,
        improperTerms: improperTerms,
        vdwParams: vdwAssigned.params,
        pairs: pairs,
        unmatched: {
          bonds: bondsAssigned.unmatched,
          angles: anglesAssigned.unmatched,
          properTorsions: propersAssigned.unmatched,
          vdw: vdwAssigned.unmatched,
        },
      };
    } finally {
      if (mol) mol.delete();
    }
  }

  // ---------- partial charges (NAGL-MBIS, see file header) ----------

  // Real per-atom parent lookup for a synthetic H node within THIS file's
  // own atoms3d/bonds3d -- a terminal H always has exactly one bond.
  function parentHeavyAtomOf(bonds3d, hIndex, numHeavy) {
    for (let i = 0; i < bonds3d.length; i++) {
      const b = bonds3d[i];
      if (b.a1 === hIndex && b.a2 < numHeavy) return b.a2;
      if (b.a2 === hIndex && b.a1 < numHeavy) return b.a1;
    }
    return undefined;
  }

  function getChargesForAtoms3D(molecule, atoms3d, bonds3d, naglModelId) {
    const numAtoms = atoms3d.length;
    const zero = new Array(numAtoms).fill(0);
    if (!naglModelId || !window.CC.NAGL || !CC.NAGL.hasModel || !CC.NAGL.hasModel(naglModelId)) {
      return { charges: zero, available: false };
    }

    let naglResult;
    try {
      naglResult = CC.NAGL.predictAll(molecule, naglModelId);
    } catch (err) {
      return { charges: zero, available: false, error: err.message };
    }

    const numHeavy = naglResult.numHeavyAtoms;

    // Bucket NAGL's own synthetic-H charges by parent heavy atom. NAGL's
    // node order and this file's atoms3d order are both "heavy atoms in
    // Molecule.atoms iteration order, then each heavy atom's implicit Hs
    // grouped contiguously right after it" (confirmed directly against
    // nagl-model.js's predictAll() header), so in practice these line up
    // index-for-index -- but bucketing by parent and consuming in
    // whatever order each side offers, rather than assuming strict
    // index-for-index identity, stays correct even if the two engines'
    // implicit-H *placement* order ever diverges: same-parent implicit
    // Hs are chemically near-indistinguishable in MBIS charge anyway (the
    // model has no way to tell topologically-symmetric hydrogens apart),
    // so which bucketed H receives which bucketed charge doesn't matter
    // chemically.
    const naglHBucket = new Map(); // parentHeavyIndex -> [charge, ...]
    for (let i = numHeavy; i < naglResult.charges.length; i++) {
      const parent = (naglResult.adjacency[i] || [])[0];
      if (parent === undefined) continue;
      if (!naglHBucket.has(parent)) naglHBucket.set(parent, []);
      naglHBucket.get(parent).push(naglResult.charges[i]);
    }

    const charges = new Array(numAtoms).fill(0);
    for (let i = 0; i < numHeavy && i < numAtoms; i++) charges[i] = naglResult.charges[i];

    const consumed = new Map(); // parent -> next bucket index
    for (let i = numHeavy; i < numAtoms; i++) {
      const parent = parentHeavyAtomOf(bonds3d, i, numHeavy);
      const bucket = parent !== undefined ? naglHBucket.get(parent) : undefined;
      const idx = consumed.get(parent) || 0;
      if (bucket && idx < bucket.length) {
        charges[i] = bucket[idx];
        consumed.set(parent, idx + 1);
      }
      // else: leave at 0 -- would only happen if the two engines
      // disagreed on implicit-H count for some atom, an honest fallback
      // rather than a crash.
    }

    return { charges: charges, available: true };
  }

  // ---------- energy ----------

  function buildEnergyModel(typed, chargesResult) {
    return {
      bonds: typed.bondTerms,
      angles: typed.angleTerms,
      propers: typed.properTerms,
      impropers: typed.improperTerms,
      vdw: typed.vdwParams,
      pairs: typed.pairs,
      vdwScale14: ffData.vdwScale14,
      elecScale14: ffData.electrostaticsScale14,
      charges: chargesResult.available ? chargesResult.charges : null,
    };
  }

  // GB/SA implicit solvation added on top of the vacuum SMIRNOFF energy --
  // see embed3d.js's solvationEnergy, which this mirrors exactly (same
  // CC.Solvent.predict call, same "ramped in with the nonbonded strength,
  // silently omitted without loaded charges" behavior). Kept as its own
  // copy rather than a shared helper since the two files' `ff`/`positions`
  // shapes differ enough (ff.charges here vs. a separate `solvent.charges`
  // array) that sharing would need its own indirection for no real benefit.
  function solvationEnergySMIRNOFF(positions, atoms3d, solvent, strength) {
    // See embed3d.js's solvationEnergy -- strength===0 (bonds/angles and
    // torsion-only stages, before the nonbonded ramp) skips the O(n^2)
    // Born-radii computation entirely rather than computing it just to
    // multiply by zero.
    if (!solvent || !solvent.enabled || !solvent.charges || strength <= 0) return 0;
    const atoms = atoms3d.map(function (a, i) {
      return { element: a.element, x: positions[i].x, y: positions[i].y, z: positions[i].z };
    });
    return strength * CC.Solvent.predict(atoms, solvent.charges, solvent.epsSolvent).total;
  }

  // stage: { torsion: bool, nonbonded: 0..1 } -- see minimizeStagedSMIRNOFF.
  function computeEnergySMIRNOFF(positions, atoms3d, ff, stage, shared, solvent) {
    let energy = 0;

    // Harmonic bonds/angles: E = 1/2 k (x - x0)^2 -- OpenMM's
    // HarmonicBondForce/HarmonicAngleForce convention, which is what
    // Sage's "k" values are fit against (the offxml's own potential="
    // harmonic" doesn't spell out the 1/2, but OpenFF's Bonds/Angles
    // handlers hand k straight to those two OpenMM forces unmodified).
    for (let i = 0; i < ff.bonds.length; i++) {
      const t = ff.bonds[i];
      const r = CC.vec3.distance(positions[t.i], positions[t.j]);
      const dr = r - t.length;
      energy += 0.5 * t.kBond * dr * dr;
    }
    for (let i = 0; i < ff.angles.length; i++) {
      const t = ff.angles[i];
      const theta = shared.angleBetween(positions[t.i], positions[t.j], positions[t.k]);
      const dTheta = theta - t.angle0;
      energy += 0.5 * t.kAngle * dTheta * dTheta;
    }

    if (stage.torsion) {
      // Proper torsions: E = k(1+cos(n*phi-phase)) -- exactly the
      // offxml's own stated potential string, no extra 1/2 factor.
      for (let i = 0; i < ff.propers.length; i++) {
        const t = ff.propers[i];
        const phi = shared.dihedralAngle(positions[t.i], positions[t.j], positions[t.k], positions[t.l]);
        for (let ti = 0; ti < t.terms.length; ti++) {
          const term = t.terms[ti];
          energy += term.k * (1 + Math.cos(term.periodicity * phi - term.phase));
        }
      }
      // Improper "trefoil": same functional form, applied once per
      // rotation of the 3 real substituents through the non-central
      // positions, each at k/3 -- see file header.
      for (let i = 0; i < ff.impropers.length; i++) {
        const t = ff.impropers[i];
        const perms = [[t.a, t.b, t.c], [t.b, t.c, t.a], [t.c, t.a, t.b]];
        for (let pi = 0; pi < perms.length; pi++) {
          const p = perms[pi];
          const phi = shared.dihedralAngle(positions[p[0]], positions[t.center], positions[p[1]], positions[p[2]]);
          for (let ti = 0; ti < t.terms.length; ti++) {
            const term = t.terms[ti];
            energy += (term.k / 3) * (1 + Math.cos(term.periodicity * phi - term.phase));
          }
        }
      }
    }

    if (stage.nonbonded > 0) {
      for (let i = 0; i < ff.pairs.length; i++) {
        const p = ff.pairs[i];
        const r = CC.vec3.distance(positions[p.a], positions[p.b]);

        const vi = ff.vdw[p.a], vj = ff.vdw[p.b];
        if (vi && vj) {
          const rm = vi.rminHalf + vj.rminHalf;
          const eps = Math.sqrt(vi.epsilon * vj.epsilon);
          const scale = p.is14 ? ff.vdwScale14 : 1.0;
          energy += stage.nonbonded * scale * eps * shared.ljShape(r, rm);
        }

        if (ff.charges) {
          const qi = ff.charges[p.a], qj = ff.charges[p.b];
          const scale = p.is14 ? ff.elecScale14 : 1.0;
          energy += stage.nonbonded * scale * COULOMB_CONST * qi * qj / Math.max(r, 1e-3);
        }
      }
    }

    energy += solvationEnergySMIRNOFF(positions, atoms3d, solvent, stage.nonbonded);

    return energy;
  }

  function numericGradientSMIRNOFF(flat, atoms3d, ff, stage, shared, solvent) {
    const grad = new Float64Array(flat.length);
    for (let i = 0; i < flat.length; i++) {
      const original = flat[i];
      flat[i] = original + GRAD_H;
      const ePlus = computeEnergySMIRNOFF(shared.unflatten(flat), atoms3d, ff, stage, shared, solvent);
      flat[i] = original - GRAD_H;
      const eMinus = computeEnergySMIRNOFF(shared.unflatten(flat), atoms3d, ff, stage, shared, solvent);
      flat[i] = original;
      grad[i] = (ePlus - eMinus) / (2 * GRAD_H);
    }
    return grad;
  }

  // Mirrors embed3d.js's minimize() -- see that file for why numeric
  // gradients + this exact convergence/backtracking scheme.
  async function minimizeSMIRNOFF(atoms3d, ff, iterations, deadline, stage, startFlat, onProgress, solvent) {
    const shared = CC.Embed3DShared;
    let flat = startFlat || shared.flatten(atoms3d);
    let energy = computeEnergySMIRNOFF(shared.unflatten(flat), atoms3d, ff, stage, shared, solvent);
    let step = 0.02;
    let lastGradNorm = Infinity;
    let exitReason = 'iteration-limit';

    for (let iter = 0; iter < iterations; iter++) {
      if (deadline && performance.now() > deadline) { exitReason = 'deadline'; break; }
      if (iter > 0 && iter % 15 === 0) {
        await shared.yieldToUI();
        if (onProgress) onProgress();
      }

      const grad = numericGradientSMIRNOFF(flat, atoms3d, ff, stage, shared, solvent);
      const gradNorm = Math.sqrt(grad.reduce(function (s, g) { return s + g * g; }, 0));
      lastGradNorm = gradNorm;
      if (gradNorm < 1e-5) { exitReason = 'gradient-converged'; break; }

      const trial = new Float64Array(flat.length);
      for (let i = 0; i < flat.length; i++) trial[i] = flat[i] - step * grad[i];
      const trialEnergy = computeEnergySMIRNOFF(shared.unflatten(trial), atoms3d, ff, stage, shared, solvent);

      if (trialEnergy < energy) {
        flat = trial;
        const improvement = energy - trialEnergy;
        energy = trialEnergy;
        step *= 1.2;
        if (improvement < 1e-7) { exitReason = 'energy-plateau'; break; }
      } else {
        step *= 0.5;
        if (step < 1e-7) { exitReason = 'step-too-small'; break; }
      }
    }

    if (!isFinite(lastGradNorm)) {
      const grad = numericGradientSMIRNOFF(flat, atoms3d, ff, stage, shared, solvent);
      lastGradNorm = Math.sqrt(grad.reduce(function (s, g) { return s + g * g; }, 0));
    }

    return {
      positions: shared.unflatten(flat), energy: energy, flat: flat, gradNorm: lastGradNorm,
      exitReason: exitReason,
      settled: exitReason === 'gradient-converged' || exitReason === 'energy-plateau',
    };
  }

  async function minimizeStagedSMIRNOFF(atoms3d, ff, totalIterations, deadline, onProgress, solvent) {
    const stage1Iters = Math.round(totalIterations * 0.25);
    const stage2Iters = Math.round(totalIterations * 0.15);
    const rampSteps = 6;
    const rampItersEach = Math.max(3, Math.round(totalIterations * 0.05));
    let remainingIters = totalIterations - stage1Iters - stage2Iters - rampSteps * rampItersEach;
    if (remainingIters < 0) remainingIters = Math.round(totalIterations * 0.1);

    function report(label) { if (onProgress) onProgress(label); }

    report('bonds & angles');
    let result = await minimizeSMIRNOFF(atoms3d, ff, stage1Iters, deadline, { torsion: false, nonbonded: 0 }, undefined, function () { report('bonds & angles'); });

    report('torsions & impropers');
    result = await minimizeSMIRNOFF(atoms3d, ff, stage2Iters, deadline, { torsion: true, nonbonded: 0 }, result.flat, function () { report('torsions & impropers'); });

    for (let r = 1; r <= rampSteps; r++) {
      if (deadline && performance.now() > deadline) break;
      report('vdW + electrostatics');
      const strength = r / rampSteps;
      result = await minimizeSMIRNOFF(atoms3d, ff, rampItersEach, deadline, { torsion: true, nonbonded: strength }, result.flat, function () { report('vdW + electrostatics'); }, solvent);
    }

    report('final polish');
    result = await minimizeSMIRNOFF(atoms3d, ff, remainingIters, deadline, { torsion: true, nonbonded: 1 }, result.flat, function () { report('final polish'); }, solvent);

    result.converged = result.settled;
    return result;
  }

  // ---------- single-seed optimize (shared by CC.OpenFF.optimize3D below and js/conformer-search.js) ----------

  // Types + minimizes ONE already-built seed geometry. Factored out of
  // CC.OpenFF.optimize3D's attempt loop so js/conformer-search.js can run
  // the exact same, already-validated SMIRNOFF typing/energy path against
  // its own seed geometries without duplicating typeMolecule/
  // buildEnergyModel/minimizeStagedSMIRNOFF wiring a second time.
  async function optimizeGivenSeedSMIRNOFF(RDKit, atoms3d, bonds3d, chargesResult, iterations, deadline, onProgress, solventOpts) {
    const typed = typeMolecule(RDKit, atoms3d, bonds3d);
    const ff = buildEnergyModel(typed, chargesResult);
    // Solvation reuses the SAME NAGL-MBIS charges electrostatics above
    // already computed -- one real charge set, two consumers -- rather
    // than the caller supplying a second copy. Omitted (not faked) if
    // charges weren't available in the first place, same honest fallback
    // electrostatics already uses.
    const solvent = solventOpts && solventOpts.enabled && chargesResult.available
      ? { enabled: true, epsSolvent: solventOpts.epsSolvent, charges: chargesResult.charges }
      : null;
    const result = await minimizeStagedSMIRNOFF(atoms3d, ff, iterations, deadline, onProgress, solvent);
    return {
      energy: result.energy,
      converged: result.converged,
      gradNorm: result.gradNorm,
      exitReason: result.exitReason,
      atoms: atoms3d.map(function (a, i) {
        return { element: a.element, x: result.positions[i].x, y: result.positions[i].y, z: result.positions[i].z };
      }),
      bonds: bonds3d,
      unmatched: typed.unmatched,
    };
  }

  // ---------- entry point ----------

  /**
   * Same shape/contract as CC.optimize3D (embed3d.js): takes a
   * CC.buildInitial3D() result, runs a multi-attempt torsion-driven
   * conformer search with SMIRNOFF-parameterized staged minimization,
   * returns the lowest-energy attempt.
   *
   * opts.naglModelId, if given and that NAGL-MBIS model is loaded, is
   * used for electrostatics (see file header) -- otherwise electrostatics
   * is simply omitted (charges null), not faked.
   *
   * Throws if the force field JSON hasn't been loaded (CC.OpenFF.
   * loadForceField()) or if RDKit isn't ready yet -- these are real
   * preconditions, not something to silently fall back from.
   */
  CC.OpenFF.optimize3D = async function (initial, opts) {
    opts = opts || {};
    const molecule = initial.molecule;
    const rotatableBonds = initial.rotatableBonds || [];
    const aromaticSet = initial.aromaticSet || new Set();

    if (!molecule || molecule.isEmpty()) {
      return { atoms: [], bonds: [], energy: 0, converged: true };
    }
    if (!ffData) throw new Error('OpenFF force field not loaded -- call CC.OpenFF.loadForceField() first');

    const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
    if (!RDKit) throw new Error('RDKit not available yet');
    compileAll(RDKit);

    const shared = CC.Embed3DShared;
    const heavyAtomCount = molecule.atoms.size;
    const totalBudgetMs = opts.timeBudgetMs || Math.min(25000, Math.max(5000, heavyAtomCount * 900));
    const attempts = opts.attempts || Math.max(3, Math.min(8, 3 + rotatableBonds.length));
    const perAttemptBudgetMs = Math.max(3500, totalBudgetMs / attempts);
    const iterations = opts.iterations || 400;

    // Connectivity (and therefore atom count/ordering/implicit-H counts)
    // doesn't depend on the random seeding each attempt does below, so
    // charges only need computing once per molecule, not once per attempt.
    const chargeSeed = shared.withImplicitHydrogens(molecule, aromaticSet);
    const chargesResult = getChargesForAtoms3D(molecule, chargeSeed.atoms3d, chargeSeed.bonds3d, opts.naglModelId);

    let best = null;
    let lastUnmatched = null;
    const overallDeadline = performance.now() + totalBudgetMs;

    for (let attempt = 0; attempt < attempts; attempt++) {
      if (performance.now() > overallDeadline) break;

      const built = shared.withImplicitHydrogens(molecule, aromaticSet);
      const atoms3d = built.atoms3d, bonds3d = built.bonds3d;
      if (attempt > 0) shared.randomizeRotatableBonds(atoms3d, bonds3d, rotatableBonds);

      const attemptDeadline = Math.min(overallDeadline, performance.now() + perAttemptBudgetMs);
      let result;
      try {
        result = await optimizeGivenSeedSMIRNOFF(RDKit, atoms3d, bonds3d, chargesResult, iterations, attemptDeadline, function (stage) {
          if (opts.onProgress) {
            opts.onProgress({
              attempt: attempt + 1,
              totalAttempts: attempts,
              stage: stage,
              bestEnergySoFar: best ? best.energy : null,
            });
          }
        }, opts.solvent);
      } catch (err) {
        throw new Error('SMIRNOFF typing failed: ' + err.message);
      }
      lastUnmatched = result.unmatched;

      if (!best || result.energy < best.energy) best = result;
    }

    if (best) {
      best.unmatched = lastUnmatched;
      best.chargesAvailable = chargesResult.available;
    }
    return best;
  };

  // Exposed for js/conformer-search.js -- see optimizeGivenSeedSMIRNOFF's
  // own comment. getChargesForAtoms3D is exposed too since conformer
  // search computes charges once per molecule up front, same as
  // CC.OpenFF.optimize3D does, rather than once per seed.
  CC.OpenFF.optimizeSeed = optimizeGivenSeedSMIRNOFF;
  CC.OpenFF.getChargesForAtoms3D = getChargesForAtoms3D;
  CC.OpenFF.compileAll = compileAll;
})();
