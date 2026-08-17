/**
 * pka-microstates.js
 *
 * Foundation for an aqueous-solution pKa predictor with a titration
 * curve as its eventual output (separate panel, not built yet — this
 * file is step one: finding ionizable groups and enumerating their
 * protonation microstates). Deliberately a NEW, separate namespace from
 * `CC.PKA` (js/pka-model.js) — that system predicts pKa for C-H bonds
 * via a trained LightGBM model, a different chemistry problem
 * (carbanion stability) from this one (heteroatom acid/base
 * equilibria). Nothing here touches that file.
 *
 * --- What a "microstate" is ---
 *
 * A molecule with N ionizable groups has, in principle, 2^N distinct
 * protonation states ("microstates") -- every combination of each site
 * being protonated or not. A "macrostate" (not implemented yet) groups
 * microstates by net charge / total proton count; the eventual titration
 * curve comes from population-weighting microstates by pH using their
 * pKa contributions. This file only does site-finding and microstate
 * enumeration/structure-generation -- no pKa values, no populations, no
 * curve yet.
 *
 * --- Site detection: SMARTS pattern library ---
 *
 * Follows this project's existing RDKit.js SMARTS-matching convention
 * exactly (see js/smarts-filters.js): `RDKit.get_qmol(smarts)` compiled
 * once and cached, matched against a molblock-round-tripped RDKit mol
 * via `mol.get_substruct_matches(qmol)` (returns a JSON string; a
 * literal `"{}"` -- not `"[]"` -- means zero matches, confirmed from
 * js/qed.js's own handling of this exact API). Match atom indices are
 * 0-based and correspond 1:1 to `Array.from(molecule.atoms.values())`
 * order, the same convention every other RDKit-consuming file here
 * relies on.
 *
 * Patterns are listed MORE-SPECIFIC-FIRST and an atom already claimed by
 * an earlier pattern is skipped by later, more general ones -- this
 * matters a lot in practice: without it, every nitrogen of a guanidino
 * group (arginine's side chain) gets independently flagged as a generic
 * aliphatic amine on top of the correct single "guanidine" site, and a
 * sulfonamide nitrogen gets wrongly flagged as basic. Both were real
 * bugs caught by validating this pattern list against real RDKit (not
 * assumed correct from writing plausible-looking SMARTS) on molecules
 * with known ionizable groups before shipping:
 *   glycine (1 amine + 1 COOH), histidine (+ 1 imidazole basic N),
 *   arginine (1 amine + 1 COOH + 1 guanidine -- NOT 3 separate amines),
 *   lysine (2 amines + 1 COOH), aspirin (1 COOH, ester O correctly NOT
 *   matched as anything), caffeine (1 weakly-basic ring N, its two
 *   amide-like ring N's correctly excluded), methanesulfonamide
 *   (correctly 0 sites -- its N is neither a free amine nor does this
 *   version detect sulfonamide acidity), quaternary ammonium and a
 *   simple amide (correctly 0 sites each).
 *
 * KNOWN v1 SCOPE LIMITATIONS (disclosed, not silent):
 *   - Only detects groups drawn in their NEUTRAL/reference form (e.g. a
 *     -COOH, not a molecule someone already drew as -COO-) -- every
 *     acid SMARTS requires the explicit H, so an already-ionized input
 *     site won't be found. Already-charged atoms (N+) are explicitly
 *     excluded from the base patterns for the same reason.
 *   - No sulfonamide/amide-NH/hydroxamic-acid acidity, no phosphonate
 *     P-OH beyond simple phosphoric-acid-ester-style matching, no
 *     imidazole ring N-H acidity (pKa ~14, rarely relevant) -- common,
 *     well-understood groups are covered; exotic/weak ones are not yet.
 *   - Imidazole/guanidine/amidine are each treated as ONE ionizable
 *     site (their real, single delocalized cation), not one site per
 *     nitrogen -- correct for how these groups actually titrate.
 *
 * --- Microstate structure generation ---
 *
 * This app's RDKit.js build has no confirmed API for mutating an
 * already-parsed mol's formal charges/hydrogens in place (confirmed:
 * the only structural mutation used anywhere in this codebase is
 * `add_hs_in_place()`, a one-way "add every implicit H" operation, see
 * js/geomol-features.js). So each microstate's structure is built the
 * same way this app already builds everything RDKit touches: clone the
 * molecule (CC.Molecule.fromJSON/toJSON, a real, already-used shallow
 * copy), set each site atom's `.charge` field directly (the exact same
 * primitive js/tools.js's own charge tool uses), write it to a molblock
 * with the app's existing writer (js/molfile.js already supports M CHG
 * records), and hand that to RDKit.js for sanitization. RDKit recomputes
 * implicit hydrogen counts itself from the charge — this deliberately
 * never reimplements RDKit's own charge/valence model by hand.
 */

window.CC = window.CC || {};
CC.PKAMicrostates = window.CC.PKAMicrostates || {};

(function () {
  // More-specific-first: an atom claimed by an earlier pattern is
  // skipped by later, more general ones (see file header).
  const PATTERNS = [
    { name: 'carboxylic_acid', cls: 'acid', smarts: '[CX3](=O)[OX2H1]', atomIdx: 2 },
    { name: 'sulfonic_acid', cls: 'acid', smarts: '[SX4](=O)(=O)[OX2H1]', atomIdx: 3 },
    { name: 'sulfinic_acid', cls: 'acid', smarts: '[SX3](=O)[OX2H1]', atomIdx: 2 },
    { name: 'phosphoric_OH', cls: 'acid', smarts: '[PX4](=O)[OX2H1]', atomIdx: 2 },
    { name: 'phenol', cls: 'acid', smarts: '[OX2H1][c]', atomIdx: 0 },
    { name: 'thiol', cls: 'acid', smarts: '[#6][SX2H1]', atomIdx: 1 },
    { name: 'tetrazole_NH', cls: 'acid', smarts: '[nH]1nnnc1', atomIdx: 0 },
    { name: 'guanidine', cls: 'base', smarts: '[NX3][CX3](=[NX2])[NX3]', atomIdx: 2 },
    { name: 'amidine', cls: 'base', smarts: '[NX3][CX3]=[NX2;!$(N-a)]', atomIdx: 2 },
    { name: 'pyridine_n', cls: 'base', smarts: '[nX2;H0;!+]', atomIdx: 0 },
    // H2,H1,H0: primary/secondary/TERTIARY. Originally H2,H1 only, missing
    // an entire real class -- a tertiary N-aryl amine (N,N-dialkylaniline,
    // or a cyclic amine like morpholine/piperidine hung directly off an
    // aromatic ring, e.g. an aminotriazine) has zero N-H and was silently
    // undetected as a site at all: aromatic_amine rejected it on H-count,
    // and aliphatic_amine's own `!$(N-a)` exclusion rejects anything
    // bonded to an aromatic atom, so neither pattern covered it. Found
    // from a real reported case (a bis-morpholino-triazine + N-aryl-
    // piperidine scaffold) where both ring nitrogens went completely
    // undetected. H0 still correctly excludes amides/sulfonamides/
    // amidines/charged-N via the same exclusions already here.
    { name: 'aromatic_amine', cls: 'base', smarts: '[NX3;H2,H1,H0;!$(NC=O);!$(N[SX4](=O)(=O));!$(N[CX3]=[NX2]);$(N-a);!+]', atomIdx: 0 },
    { name: 'aliphatic_amine', cls: 'base', smarts: '[NX3;H2,H1,H0;!$(NC=O);!$(N=*);!$(N-a);!$(N[SX4](=O)(=O));!$(N[CX3]=[NX2]);!+]', atomIdx: 0 },
  ];

  let compiledPatterns = null; // lazy + cached, mirrors smarts-filters.js's own pattern

  function compilePatterns(RDKit) {
    if (compiledPatterns) return compiledPatterns;
    compiledPatterns = PATTERNS.map(function (p) {
      let qmol = null;
      try {
        qmol = RDKit.get_qmol(p.smarts);
        if (!qmol.is_valid()) { qmol.delete(); qmol = null; }
      } catch (err) {
        qmol = null;
      }
      return Object.assign({}, p, { qmol: qmol });
    }).filter(function (p) { return !!p.qmol; });
    return compiledPatterns;
  }

  /**
   * Detects ionizable sites in `molecule`. Returns an array of
   * { name, cls: 'acid'|'base', atomId, element }, one per detected
   * site (priority-deduplicated -- see file header), in no particular
   * order. Empty array for an empty molecule, RDKit not loaded yet, or
   * a molecule with no detected ionizable groups.
   */
  CC.PKAMicrostates.findIonizableSites = function (molecule) {
    const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
    if (!RDKit || !molecule || molecule.isEmpty()) return [];

    const patterns = compilePatterns(RDKit);
    const heavyAtoms = Array.from(molecule.atoms.values());
    const molblock = CC.moleculeToMolblock(molecule);
    const sites = [];
    const claimed = new Set();
    let mol = null;
    try {
      mol = RDKit.get_mol(molblock);
      if (!mol || !mol.is_valid()) return [];

      patterns.forEach(function (p) {
        let matches;
        try {
          matches = JSON.parse(mol.get_substruct_matches(p.qmol));
        } catch (err) {
          matches = null;
        }
        // get_substruct_matches returns the literal string "{}" (not
        // "[]") when there are zero matches -- confirmed from
        // js/qed.js's own handling of this exact RDKit.js quirk.
        if (!Array.isArray(matches)) return;
        matches.forEach(function (m) {
          const atomIdx = m.atoms[p.atomIdx];
          if (claimed.has(atomIdx)) return;
          const atom = heavyAtoms[atomIdx];
          if (!atom) return;
          claimed.add(atomIdx);
          sites.push({ name: p.name, cls: p.cls, atomId: atom.id, element: atom.element });
        });
      });
    } finally {
      if (mol) mol.delete();
    }
    return sites;
  };

  /**
   * Enumerates the full 2^N microstate space for `sites` (from
   * findIonizableSites). Each microstate:
   *   { protonation: [bool, ...] } -- one entry per site (same order as
   *     `sites`), true = protonated at this site in this microstate.
   *   netCharge -- this microstate's total formal charge from its
   *     ionizable sites (protonated acid / deprotonated base = neutral;
   *     deprotonated acid = -1 each; protonated base = +1 each). Doesn't
   *     include any fixed charges elsewhere in the molecule (e.g. a
   *     quaternary ammonium drawn as N+, which findIonizableSites never
   *     treats as a toggleable site in the first place) -- add those in
   *     separately if the caller needs the molecule's true total charge.
   *   numProtonsFromReference -- signed proton count relative to the
   *     single "all-neutral" reference microstate (every acid
   *     protonated, every base deprotonated) -- the natural axis for
   *     grouping microstates into macrostates later.
   * No molecular structures are built here -- see
   * buildMicrostateStructure() for that (kept separate: most uses of
   * this list, e.g. counting states per net charge, don't need one).
   */
  CC.PKAMicrostates.enumerateMicrostates = function (sites) {
    const n = sites.length;
    const total = Math.pow(2, n);
    const microstates = new Array(total);
    for (let mask = 0; mask < total; mask++) {
      const protonation = new Array(n);
      let netCharge = 0;
      let numProtonsFromReference = 0;
      for (let i = 0; i < n; i++) {
        const protonated = !!(mask & (1 << i));
        protonation[i] = protonated;
        if (sites[i].cls === 'acid') {
          netCharge += protonated ? 0 : -1;
          numProtonsFromReference += protonated ? 0 : -1; // reference: every acid protonated
        } else {
          netCharge += protonated ? 1 : 0;
          numProtonsFromReference += protonated ? 1 : 0; // reference: every base deprotonated
        }
      }
      microstates[mask] = { protonation: protonation, netCharge: netCharge, numProtonsFromReference: numProtonsFromReference };
    }
    return microstates;
  };

  /**
   * Builds the actual structure for one microstate (see file header for
   * why this goes through a molblock clone + RDKit sanitize rather than
   * mutating anything in place). Returns { molecule, smiles } — the
   * cloned, charge-adjusted CC.Molecule and its RDKit-canonicalized
   * SMILES — or null if RDKit rejects the result (not expected for a
   * chemically valid input + a well-formed microstate, but checked
   * rather than assumed).
   */
  CC.PKAMicrostates.buildMicrostateStructure = function (molecule, sites, microstate) {
    const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
    if (!RDKit) return null;

    const clone = CC.Molecule.fromJSON(molecule.toJSON());
    sites.forEach(function (site, i) {
      const atom = clone.atoms.get(site.atomId);
      if (!atom) return;
      const protonated = microstate.protonation[i];
      atom.charge = site.cls === 'acid' ? (protonated ? 0 : -1) : (protonated ? 1 : 0);
    });

    const molblock = CC.moleculeToMolblock(clone);
    let mol = null;
    try {
      mol = RDKit.get_mol(molblock);
      if (!mol || !mol.is_valid()) return null;
      return { molecule: clone, smiles: mol.get_smiles() };
    } catch (err) {
      return null;
    } finally {
      if (mol) mol.delete();
    }
  };
})();
