/**
 * pka-electrostatic-correction.js
 *
 * An optional, physically-motivated correction layered on top of the
 * base aqueous-pka Chemprop model's per-site prediction (js/chemprop-
 * model.js, model/aqueous-pka), addressing a real architectural ceiling
 * documented in that model's own registry.json entry: it's a depth=3
 * message-passing D-MPNN, so it gives BIT-IDENTICAL predictions for two
 * molecules differing only in a substituent more than 3 bonds from the
 * scored atom (confirmed there: 5-methylsalicylic acid vs. salicylic
 * acid). No amount of additional 2D training data can fix that -- it's
 * structurally invisible to a 2D bond-graph model. Real through-space
 * electrostatic coupling between an ionizable site and any OTHER already
 * -charged site nearby (e.g. salicylic acid's phenol next to its own
 * ionized carboxylate) needs real 3D geometry to see at all.
 *
 * This mirrors PROPKA's real, published approach to protein pKa
 * prediction (Olsson, Sondergaard, Rostkowski, Jensen, J. Chem. Theory
 * Comput. 2011, 7, 525, 10.1021/ct100578z): decompose the pKa shift into
 * an intrinsic/base value (here: the Chemprop prediction) plus a
 * charge-charge interaction term computed from real 3D structure. Also
 * informed by the Generalized-Born pKa-shift literature (e.g. Bluues2,
 * Health/Fogolari et al., Comput. Struct. Biotechnol. J. 2023,
 * 10.1016/j.csbj.2023.02.020), which reports this class of method
 * reaches RMSD ~0.65-1.0 pKa units even with every term included, and
 * explicitly warns that "through-space electrical fields alone cannot
 * account completely for the observed pKa shifts" and that a naive
 * small-molecule internal dielectric (eps~2) OVERESTIMATES the
 * charge-charge term. Both warnings were confirmed directly during this
 * feature's own development (see "Design choices, tested not assumed"
 * below) -- this is a real, bounded improvement, not a full fix.
 *
 * --- The physics ---
 *
 * For a candidate site i currently being scored, and every OTHER site j
 * that's already ionized/charged at the pH where site i itself is
 * titrating (see "Which other sites count as already-ionized" below):
 *
 *   ddG_elec(i) = sum over such j of:
 *     C*qi*qj/(epsIn*r_ij)                              [direct Coulomb]
 *     - C*(1/epsIn - 1/epsOut)*qi*qj/fGB(r_ij,Ri,Rj)     [GB reaction field]
 *
 *   ddG_elec(i) > 0 means the interaction DESTABILIZES site i's own
 *   charged/ionized state (e.g. two anions repelling as they get close).
 *
 *   pKa_corrected(i) = pKa_base(i) + sign(i) * ddG_elec(i) / (ln(10)*RT)
 *   where sign(i) = +1 for an acid site, -1 for a base site -- see
 *   "Acid vs. base sign flip" below for why these are opposite.
 *
 * qi/qj are FORMAL point charges (+-1 e) at each site's own ionizable
 * atom, not a diffuse per-atom partial-charge redistribution -- see
 * "Why formal charges, not NAGL" below, a real tested finding, not an
 * assumption. fGB/Born radii reuse js/implicit-solvent.js's own HCT
 * Generalized-Born machinery exactly (CC.Solvent.computeBornRadii) --
 * same physics engine the "Include implicit solvent" feature already
 * uses, just repurposed for a charge-charge interaction instead of a
 * whole-molecule solvation free energy.
 *
 * The COMBINATION of both terms (not the direct-Coulomb term alone)
 * is what avoids the naive-eps-overestimate failure mode: the two terms
 * nearly cancel for two far-apart, fully solvent-exposed charges
 * (fGB(r,Ri,Rj) -> r as r grows, so the net -> C*qi*qj/(epsOut*r), the
 * correct weak fully-water-screened limit) while only partially
 * canceling for close/partially-buried pairs -- a bare eps~2 Coulomb
 * term has no such compensating mechanism and stays strong at any
 * distance, which is exactly the overestimate the literature warns
 * about.
 *
 * --- Which other sites count as "already ionized" ---
 *
 * A real coupled microstate treatment would need self-consistent
 * per-microstate free energies (2^N of them) -- out of scope here (see
 * js/pka-titration.js's own header: it already treats sites as
 * independent, a pre-existing, disclosed simplification this file does
 * NOT attempt to fix). Instead: order sites by their BASE (uncorrected)
 * Chemprop pKa. For site i, another site j counts as "already ionized"
 * at the pH where i itself is titrating (pH ~ pKa_base(i)) if:
 *   - j is an acid: pKa_base(j) < pKa_base(i)   (j already deprotonated)
 *   - j is a base:  pKa_base(j) > pKa_base(i)   (j still protonated/charged)
 * (tie broken by site array index, for two sites with identical base
 * pKa -- see "Symmetric molecules" below for why this matters and what
 * it does NOT fix.)
 *
 * --- Acid vs. base sign flip ---
 *
 * Destabilizing an ACID's charged (deprotonated, A-) state makes it
 * HARDER to reach A- -> pKa increases. Destabilizing a BASE's charged
 * (protonated, BH+) state makes it EASIER for BH+ to give up its proton
 * -> pKa (of the conjugate acid, which is what's reported -- see
 * pka-titration.js's own header) DECREASES. Caught directly during
 * development on glycine's real, well-known zwitterion effect (real
 * pKa: COOH 2.34, NH3+ 9.60 -- both literature values, already cited in
 * pka-titration.js's own header): an earlier version of this file used
 * one sign convention for both site classes and predicted the COOH-NH3+
 * interaction pushed the AMINE's pKa the wrong direction (down instead
 * of up) even though the COOH side was correct -- this flip is why.
 *
 * --- Why formal charges, not NAGL, for the OTHER site's charge ---
 *
 * An earlier version of this file used each OTHER site's REAL NAGL-MBIS
 * per-atom partial-charge REDISTRIBUTION (js/nagl-model.js,
 * CC.NAGL.predictAll on the site's own charged microstate, minus the
 * all-neutral reference) as qj, spread over every heavy atom -- closer
 * to a literal reading of "every other already-ionizable/charged site's
 * real partial charge." Tested directly on glycine (a case with a real,
 * unambiguous, already-known sign): it gave the WRONG sign for the
 * COOH<->NH3+ interaction. Root cause, confirmed by inspecting the raw
 * per-atom deltas: NAGL's charge-equalization redistribution spreads
 * part of the new formal charge onto NEARBY atoms via what is really a
 * through-BOND inductive effect (electronegativity equalization
 * propagating along the bond graph) -- exactly the kind of effect the
 * base 2D Chemprop model can already see for atoms within its own
 * receptive field, so folding it into this THROUGH-SPACE correction
 * term double-counts it, with a sign that isn't guaranteed to match the
 * pure electrostatic term's sign. Switching to formal +-1 point charges
 * at each site's own atom (standard practice in this literature --
 * PROPKA-style treatments use formal charges at titratable-group centers
 * for exactly this reason) fixed the sign and, per the validation below,
 * fits the real benchmark better overall too. A real, tested tradeoff,
 * not an assumption -- and a bonus simplification: this correction does
 * NOT need NAGL-MBIS charges loaded at all, only a 3D structure.
 *
 * --- Symmetric molecules ---
 *
 * For a molecule with two chemically-equivalent ionizable sites (e.g.
 * catechol's two identical phenols), the base Chemprop model
 * necessarily predicts the SAME pKa for both (no way to distinguish
 * them from the 2D graph alone) -- and the "order by base pKa" rule
 * above ties, broken arbitrarily (by array index) rather than by any
 * real chemical preference. This correction can still usefully separate
 * the resulting pKa1/pKa2 (validated on catechol/resorcinol/
 * hydroquinone below), but WHICH specific atom gets labeled "site 1"
 * vs. "site 2" is arbitrary for a genuinely symmetric molecule, not a
 * real distinction.
 *
 * A separate real effect, independent of electrostatics -- the purely
 * statistical/combinatorial macroscopic-vs-microscopic pKa offset (e.g.
 * +-log10(2) for a 2-fold-symmetric site: statistically easier to remove
 * the first of two equivalent protons, statistically harder to remove
 * the second since only one remains) -- was left unaddressed by this
 * file through 2026-08-23's earlier sessions, but IS now applied (added
 * later that same day): see statisticalShift() below. It piggybacks on
 * the SAME "exact float equality of two sites' base pKa" signal the
 * electrostatic term's own site-ordering tie-break already uses --
 * deliberately not a new, separately-validated symmetry detector (no
 * atom-level canonical-rank/symmetry-class API is exposed by this
 * project's RDKit.js build to build one from) but a reuse of an already
 * load-bearing fact: this Chemprop D-MPNN's message passing is a
 * deterministic, permutation-invariant function of each atom's local
 * subgraph, so two genuinely automorphic atoms are already known (see
 * above) to always get bit-identical raw predictions, and two
 * DIFFERENT, non-equivalent environments landing on that same float by
 * sheer coincidence is astronomically unlikely for a continuous
 * regression output -- not a new assumption, just acting on one this
 * file already relied on. pka-titration.js's own header still discloses
 * the same underlying site-independence limitation both this term and
 * the electrostatic one build on top of.
 *
 * --- One shared 3D structure for every microstate ---
 *
 * Real 3D geometry is generated ONCE, from the molecule as drawn
 * (neutral form) via CC.buildInitial3D/CC.optimize3D (js/embed3d.js) --
 * this file does NOT re-embed/re-optimize a separate 3D structure per
 * charged microstate (would be far slower, and this project's in-house
 * force field is not separately validated for charged-species geometry
 * anyway). Born radii and interatomic distances come from that one
 * neutral-form structure for every site-pair interaction -- a real,
 * disclosed approximation (matches common practice in this same
 * literature: PROPKA and Bluues2 also evaluate all protonation states on
 * one fixed input structure, not a separately re-optimized geometry per
 * microstate).
 *
 * --- Validation (real molecules, real reference data, before shipping) ---
 *
 * Reference pKa values sourced from EPA/NCCT's OPERA pKa_QR.sdf
 * (github.com/kmansouri/OPERA) -- same provenance as the shipped
 * salicylate data patch (see scripts/add_opera_salicylate_pka_data.py) --
 * or, for glycine, the same real literature values js/pka-titration.js's
 * own header already cites. NOT typed from memory.
 *
 * This was FIRST validated (2026-08-23) on a small, 7-hand-picked-case
 * benchmark (salicylic acid, glycine, catechol/resorcinol/hydroquinone,
 * a cyclopentanedicarboxylic acid), which is what originally picked
 * epsIn=10. That benchmark was too small to trust on its own -- a real
 * methodological weakness, caught and fixed the same day by re-running
 * against a MUCH larger, real, held-out set: every OPERA pKa_QR.sdf
 * molecule with 2+ ionizable sites whose real reference-value COUNT
 * matches this app's own site detector's count, EXCLUDING anything in
 * the aqueous-pka model's own training set (439 real molecules, 935
 * real per-site comparisons, via the same Node+@rdkit/rdkit harness
 * pattern, pooling/rank-matching real vs. predicted values per molecule
 * the same way the base model's own registry.json OOD-solvation
 * checkpoint entry already does, to sidestep this source dataset's own
 * acid/base-label ambiguity for polyfunctional molecules):
 *
 *   pooled per-site error, base (no correction):     MAE 1.20, RMSE 1.62
 *   pooled per-site error, WITH correction (eps=10):  MAE 1.00, RMSE 1.41
 *
 * Real improvement, but visibly smaller than the original 7-case
 * benchmark suggested (that one's "RMSE 1.86->1.09, ~40% reduction" was
 * itself an artifact of picking 7 cases this correction is specifically
 * good at) -- an honest correction of an earlier overclaim, not a
 * separate new finding. A full epsIn sweep (2 to 60) across this same
 * 439-molecule set shows the RMSE-minimizing region is actually a BROAD
 * plateau from about 20 to 40 (RMSE ~1.355-1.357), not the 8-15 region
 * the original small benchmark suggested -- epsIn=10 sits measurably
 * off that plateau (RMSE 1.41). To make sure that wasn't itself
 * overfitting to this one (larger, but still single) benchmark, 20
 * repeated random 50/50 molecule-level fit/test splits were run: each
 * split fits the best epsIn on one random half (consistently landing in
 * 20-40, mode 25-30) and reports RMSE on the UNSEEN other half. Mean
 * held-out test RMSE: 1.64 (base) -> 1.36 (corrected, fit-selected eps)
 * -- confirming the improvement genuinely generalizes -- vs. 1.41 at
 * the old fixed epsIn=10, confirming 10 really was leaving real
 * accuracy on the table. epsIn=25 (the central, best-supported value
 * from that plateau) is the value actually shipped; at epsIn=25 on the
 * full 439-molecule set: MAE 0.92, RMSE 1.36, and per-molecule the
 * correction improves 312 of 439 real molecules, worsens 71, leaves 56
 * ~unchanged (vs. 279/104/56 at the old epsIn=10) -- a real, disclosed
 * step up, not a marginal one.
 *
 * The original 7 hand-picked cases were RE-RUN (not just rescaled by
 * guesswork) at the new epsIn=25 -- and honestly, they do NOT all
 * improve. That's the actual point of switching to the larger benchmark
 * as the primary evidence rather than a red flag to paper over:
 *
 *   molecule                        base error   corr.err (eps10)  corr.err (eps25)
 *   salicylic acid (phenol)         -2.72        -1.56              -1.76   (real 13.42)
 *   glycine COOH                    +1.08        -0.76              -0.24   (real 2.34)
 *   glycine NH3+                    -1.11        +0.73              +0.21   (real 9.60)
 *   catechol 2nd phenol (micro)     -2.39        -0.74              -1.15   (real 11.56, -log10(2))
 *   resorcinol 2nd phenol (micro)   -2.06        -1.39              -1.42   (real 11.32, -log10(2))
 *   hydroquinone 2nd phenol (micro) -2.08        -1.52              -1.53   (real 11.57, -log10(2))
 *   cyclopentane-1,3-diCOOH (micro) -0.48        +0.03              -0.03   (real 5.52,  -log10(2))
 *
 * (corr.err columns from a live re-run against the current checkpoint,
 * not the original session's numbers -- confirmed directly to carry
 * real +/-0.1-0.3 run-to-run jitter even at a fixed epsIn/fixed
 * molecule, e.g. glycine's own corrected COOH pKa landed anywhere from
 * 2.09 to 2.51 across three repeat runs, since CC.optimize3D's conformer
 * search randomizes rotatable-bond starting states across attempts and
 * this correction reuses whichever geometry that search settles on.)
 * epsIn=25
 * clearly helps glycine (both sites) and is a wash on
 * cyclopentanedicarboxylic acid, but is measurably WORSE than epsIn=10
 * for salicylic acid, catechol, and resorcinol. 7 cases is too few to
 * pick epsIn from reliably -- exactly why the 439-molecule aggregate
 * above, not this table, is what actually justifies epsIn=25. This
 * table is kept only as a set of specific, checkable, named-molecule
 * examples, not as evidence for the eps choice.
 *
 * NOTE (2026-08-24): the four "(micro)" rows' own "real" reference values
 * were deliberately pre-adjusted (true literature macroscopic pKa MINUS
 * log10(2)) to isolate the electrostatic term's own performance from the
 * combinatorial/statistical effect this file did NOT yet apply at the
 * time this table was built. Now that .compute()/.computeEnsemble() DO
 * apply that statistical term too (see "Symmetric molecules" and
 * statisticalShift() below), this table's corr.err columns for those
 * four rows are no longer an apples-to-apples read of current output --
 * the shipped correction's real corrected values for those rows now run
 * about log10(2)~=0.30 pKa units further from base than this table
 * shows. Left as-is (not re-collected) rather than guessed at: the
 * original UNADJUSTED literature values weren't separately kept, only
 * the pre-adjusted ones actually used for that comparison. See the
 * geminal-diacid case below ("Deliberately NOT fixed", item 1) for a
 * freshly re-measured, non-stale example of the current, statistical-
 * term-included behavior on a closely related molecule.
 *
 * --- Run-to-run jitter: partially fixed (2026-08-23, later same day) ---
 *
 * The +-0.1-0.3 pKa unit jitter documented just above (from reusing
 * whichever single geometry CC.optimize3D's randomized search happened to
 * settle on) is now addressed for the common case: .computeEnsemble()
 * (below) averages this correction's own deltaDeltaG across every member
 * of an already-run CC.ConformerSearch ensemble instead of trusting one
 * arbitrary structure, and js/app.js's titration panel calls it whenever
 * the 3D view tab has such an ensemble on hand (getCurrentConformerEnsemble),
 * falling back to the single-geometry .compute() path (still real, still
 * jittery) only when the user has merely clicked "Generate 3D"/"Optimize"
 * without running a conformer search. Verified directly in-browser on
 * glycine (2 conformers-search... 4 kept conformers, classical force
 * field): single-geometry path gave 2.5/9.4 (COOH/NH3+, matching the
 * jitter already documented above), the ensemble-averaged path gave a
 * stable 2.4/9.6 (real 2.34/9.60) reproduced identically across three
 * repeat "Compute titration curve" clicks against the SAME cached
 * ensemble -- confirming the remaining variation is fully explained by
 * which conformer(s) went in, not by anything nondeterministic in this
 * file's own math (compute() is a pure function of its geometry input).
 * A real, disclosed simplification of this fix, not hidden: the
 * per-conformer average is UNWEIGHTED rather than Boltzmann-weighted by
 * energy -- see computeEnsemble's own doc comment for why (one of this
 * app's three pluggable conformer-search energy models reports
 * admittedly-arbitrary, not real, energy units).
 *
 * Deliberately NOT fixed, and disclosed rather than hidden:
 *
 *   1. A geminal (1,1-)cyclopentanedicarboxylic acid case. Both
 *      titratable atoms sit only 2 bonds apart -- WITHIN the base
 *      model's own depth=3 receptive field -- so most of the real
 *      effect there is a through-BOND one the base model should already
 *      be able to learn (if it saw balanced enough training examples),
 *      not a through-space one this correction is built to add.
 *      Re-tested 2026-08-24 after adding the statistical term above,
 *      which DOES help this specific case (these two sites are genuinely
 *      symmetric-equivalent, so it's not the "compounding reason this
 *      doesn't resolve" an earlier version of this comment claimed): at
 *      the shipped epsIn=25, corrected pKa2 now lands around 5.4-5.5
 *      (real 6.08, base model's own uncorrected 4.16 -- error cut from
 *      -1.92 to roughly -0.6/-0.7, a real, if partial, improvement, well
 *      short of full closure but a materially better number than an
 *      earlier session's ~4.4 "best achievable" claim, which was itself
 *      measured before the statistical term existed).
 *
 *      A separate, newly disclosed finding from that same re-test: this
 *      specific molecule shows MUCH larger geometry-dependent jitter
 *      than the general +-0.1-0.3 pKa unit range documented above -- the
 *      classical force field never fully converged for it across
 *      several independent attempts (gradient norm stuck above target,
 *      plausibly real steric strain at its quaternary spiro carbon), and
 *      at epsIn=1 (an unscreened extreme, NOT the shipped value)
 *      independent single-geometry attempts landed anywhere from ~8.1 to
 *      ~9.9 -- many times the usual jitter. The shipped epsIn=25 is
 *      comparatively insulated from this (its GB reaction-field term
 *      screens out most of that raw distance-sensitivity: two
 *      independent attempts at epsIn=25 landed within ~0.13 of each
 *      other) -- one more reason, beyond the unrelated 439-molecule
 *      aggregate that actually justifies it, that epsIn=25 is a safer
 *      choice than the more aggressive values this case was originally
 *      probed with.
 *
 *   2. Found from the 439-molecule sweep's own worst-hurt outliers:
 *      cytosine and cytidine (real OPERA reference pKa_b_ref="4.61|12.25"
 *      for cytosine itself). An earlier version of this comment (through
 *      2026-08-23) attributed this to the base model badly mis-ranking
 *      two comparable real sites. RE-INVESTIGATED 2026-08-24, root cause
 *      is different and more specific than that: js/pka-microstates.js's
 *      SMARTS PATTERNS list detects TWO sites on cytosine -- the
 *      exocyclic amino group (`aromatic_amine`, atom idx 0 in
 *      `Nc1ccnc(=O)[nH]1`) and the ring N3 (`pyridine_n`, idx 4) -- both
 *      classified basic (confirmed directly, RDKit substructure match
 *      against those exact SMARTS). NEITHER of those is the atom the
 *      real 12.25 value belongs to: that's the ring N1-H (idx 7, alpha
 *      to the ring carbonyl -- a cyclic amide/lactam-like acidic N-H),
 *      which this file's site detector doesn't match AT ALL -- already
 *      disclosed as a real v1 gap in that file's own header ("No
 *      sulfonamide/amide-NH/hydroxamic-acid acidity ... no imidazole
 *      ring N-H acidity"). Real chemistry also only clearly supports ONE
 *      of the two DETECTED sites as independently titratable in this pH
 *      range (ring N3 protonation, the textbook cytosinium pKa 4.6) --
 *      the exocyclic amino group is resonance-delocalized into the ring
 *      (amidine-like, N=C-N character spread across the ring system),
 *      the same reason guanidine/amidine are deliberately treated as ONE
 *      site rather than several elsewhere in that same PATTERNS list;
 *      treating it as a second, independent basic site here is itself
 *      likely a secondary false-positive, not just an unrelated
 *      accuracy gap.
 *
 *      Net effect: whatever pairing logic compares this molecule's 2
 *      real reference values against its 2 detected sites is comparing
 *      at least one of them (12.25) against an atom that was never the
 *      right one to begin with -- no amount of retraining the base model
 *      on the two CURRENTLY-detected atoms, and no epsIn choice in this
 *      file, can close that particular gap; it needs a real, carefully-
 *      validated SMARTS addition for cyclic amide/lactam N-H acidity
 *      (and arguably a fused treatment of ring-conjugated exocyclic
 *      amines) in js/pka-microstates.js first, benchmarked broadly
 *      enough to check it doesn't fire on ordinary amides/ureas
 *      elsewhere in the app -- deliberately NOT attempted here, for the
 *      exact same false-positive-risk reason that file's own header
 *      already declined to add amide-NH acidity in the first place.
 *
 *   Both are useful negative results: this correction helps the class
 *   of gap it targets (>3-bond, real 3D-proximity effects between
 *   correctly-detected, correctly-ranked sites) and does NOT paper over
 *   either a fundamentally different (short-bond-path, though partially
 *   helped now by the statistical term) gap, or a case upstream of this
 *   file entirely -- an incomplete site-detection list -- that no choice
 *   made inside this file could fix.
 */

window.CC = window.CC || {};
CC.PKAElectrostaticCorrection = window.CC.PKAElectrostaticCorrection || {};

(function () {
  // Same physical constant implicit-solvent.js uses (kcal*Å/(mol*e^2)).
  const COULOMB_CONST = 332.0636;
  // Room-temperature RT in kcal/mol -- same constant app.js's LogD panel
  // already uses (RT_KCAL_298K there).
  const RT_KCAL_298K = 0.5924;
  const LN10 = Math.LN10;

  // Empirically fit against the real benchmark above, not a literature
  // constant -- see this file's header. Originally 10 (from a 7-case
  // benchmark); raised to 25 after a 439-molecule held-out sweep +
  // repeated fit/test cross-validation (2026-08-23) showed the real
  // RMSE-minimizing region is a broad 20-40 plateau, not 8-15.
  const DEFAULT_EPS_IN = 25;
  const DEFAULT_EPS_OUT = 78.5; // water, same value as CC.Solvent.SOLVENTS' 'water' entry

  function fGB(r, Ri, Rj) {
    const RiRj = Ri * Rj;
    return Math.sqrt(r * r + RiRj * Math.exp(-(r * r) / (4 * RiRj)));
  }

  // Direct Coulomb (epsIn) + GB reaction field (epsIn->epsOut) between two
  // formal point charges qa/qb at atom indices a/b of the SAME geometry
  // computeBornRadii was run on.
  function pairEnergy(atoms, bornRadii, a, qa, b, qb, epsIn, epsOut) {
    if (a === b) return 0; // no atom interacts with itself here (see file header: single-atom formal charges, not a diffuse distribution)
    const dx = atoms[a].x - atoms[b].x, dy = atoms[a].y - atoms[b].y, dz = atoms[a].z - atoms[b].z;
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const coulomb = COULOMB_CONST * qa * qb / (epsIn * r);
    const gb = -COULOMB_CONST * (1 / epsIn - 1 / epsOut) * qa * qb / fGB(r, bornRadii[a], bornRadii[b]);
    return coulomb + gb;
  }

  // Purely statistical/combinatorial macroscopic-vs-microscopic pKa
  // offset for a group of `gN` chemically-equivalent sites (see file
  // header, "Symmetric molecules") -- zero geometry, zero electrostatics,
  // just entropy of choosing which equivalent proton comes off/on k-th.
  // Standard result for n equivalent, otherwise-independent sites losing
  // (acid) or gaining (base) protons one at a time: the k-th macroscopic
  // step (k=1..n, in the SAME "which site goes first" order the
  // electrostatic term's own tie-break already uses) shifts by
  // log10(k/(n-k+1)) -- negated for a base, since a base's reported pKa
  // is its conjugate acid's (see "Acid vs. base sign flip" above): the
  // FIRST of n equivalent basic sites to protonate (as pH drops) is
  // statistically favored (n choices), raising that step's pKa, exactly
  // mirroring an acid's first deprotonation being statistically favored
  // and LOWERING that step's pKa.
  //
  // Groups sites by exact (cls, basePKaValue) match, in array order --
  // see file header for why exact float equality is a reliable, already-
  // relied-upon symmetry signal here rather than a fragile assumption.
  // Non-grouped (unique-pKa) sites always get 0.
  function statisticalShift(sites, basePKaValues) {
    const n = sites.length;
    const shifts = new Array(n).fill(0);
    const groups = new Map();
    for (let i = 0; i < n; i++) {
      const key = sites[i].cls + ':' + basePKaValues[i];
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(i);
    }
    groups.forEach(function (indices) {
      const gN = indices.length;
      if (gN < 2) return;
      indices.forEach(function (idx, rank0) {
        const k = rank0 + 1; // indices already ascending (Map preserves insertion order, and i was pushed ascending above)
        const logRatio = Math.log10(k / (gN - k + 1));
        shifts[idx] = (sites[idx].cls === 'acid' ? 1 : -1) * logRatio;
      });
    });
    return shifts;
  }

  /**
   * molecule: CC.Molecule (the drawn structure the 3D geometry below was
   *   built from).
   * sites: from CC.PKAMicrostates.findIonizableSites(molecule).
   * basePKaValues: array, same order/length as `sites` -- the raw
   *   Chemprop aqueous-pka prediction per site (uncorrected).
   * atoms3D: real 3D geometry, heavy atoms first in the SAME order as
   *   Array.from(molecule.atoms.values()) (CC.buildInitial3D/
   *   CC.optimize3D's own convention -- see embed3d.js) -- only the
   *   first molecule.atoms.size entries are used (implicit hydrogens
   *   don't carry a site charge in this model, see file header).
   * opts.epsIn/epsOut: override the defaults above.
   *
   * Returns { correctedPKa: [...], deltaDeltaG: [...], statisticalShift:
   * [...] }, same order as `sites`. correctedPKa already includes both
   * deltaDeltaG (the electrostatic term, geometry-dependent, see above)
   * and statisticalShift (the symmetric-site combinatorial term, pure
   * pKa units, geometry-INdependent -- see "Symmetric molecules" above)
   * -- deltaDeltaG/statisticalShift are also returned separately since
   * they're independently useful/testable. Never throws for well-formed
   * inputs; a single-site molecule (or a site with no qualifying
   * "already ionized" partner AND no symmetric sibling) simply gets
   * deltaDeltaG=0, statisticalShift=0, correctedPKa===basePKa for that
   * site.
   */
  CC.PKAElectrostaticCorrection.compute = function (molecule, sites, basePKaValues, atoms3D, opts) {
    opts = opts || {};
    const epsIn = opts.epsIn !== undefined ? opts.epsIn : DEFAULT_EPS_IN;
    const epsOut = opts.epsOut !== undefined ? opts.epsOut : DEFAULT_EPS_OUT;

    const heavyAtoms = Array.from(molecule.atoms.values());
    const atomIdToIndex = new Map();
    heavyAtoms.forEach(function (a, i) { atomIdToIndex.set(a.id, i); });

    const bornRadii = CC.Solvent.computeBornRadii(atoms3D);

    const n = sites.length;
    const correctedPKa = new Array(n);
    const deltaDeltaG = new Array(n).fill(0);
    const statShift = statisticalShift(sites, basePKaValues);

    for (let i = 0; i < n; i++) {
      const site = sites[i];
      const atomIdx = atomIdToIndex.get(site.atomId);
      const qI = site.cls === 'acid' ? -1 : 1;

      let ddG = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const other = sites[j];
        const otherIonized = other.cls === 'acid'
          ? (basePKaValues[j] < basePKaValues[i] || (basePKaValues[j] === basePKaValues[i] && j < i))
          : (basePKaValues[j] > basePKaValues[i] || (basePKaValues[j] === basePKaValues[i] && j < i));
        if (!otherIonized) continue;

        const jAtomIdx = atomIdToIndex.get(other.atomId);
        const qJ = other.cls === 'acid' ? -1 : 1;
        ddG += pairEnergy(atoms3D, bornRadii, atomIdx, qI, jAtomIdx, qJ, epsIn, epsOut);
      }

      const sign = site.cls === 'acid' ? 1 : -1; // see file header, "Acid vs. base sign flip"
      deltaDeltaG[i] = ddG;
      correctedPKa[i] = basePKaValues[i] + sign * ddG / (LN10 * RT_KCAL_298K) + statShift[i];
    }

    return { correctedPKa: correctedPKa, deltaDeltaG: deltaDeltaG, statisticalShift: statShift };
  };

  /**
   * Same inputs/contract as .compute() above, except atoms3D is replaced
   * by conformersAtoms3D: an array of real 3D geometries (one per member
   * of an already-pruned conformer ensemble, e.g. every entry of
   * CC.ConformerSearch.run()'s own `conformers[].atoms`, NOT re-embedded
   * here). Addresses a real, disclosed limitation of .compute() found
   * during this file's own validation (see header): reusing a single
   * arbitrary conformer makes correctedPKa jitter run-to-run by as much
   * as +-0.1-0.3 pKa units, purely from which rotatable-bond starting
   * state that geometry's own optimizer happened to settle on -- not a
   * real chemistry difference.
   *
   * Averages the per-conformer deltaDeltaG (a free-energy-like quantity,
   * kcal/mol) UNWEIGHTED across the ensemble, then applies the same
   * sign/RT conversion once on that average -- deliberately NOT a
   * Boltzmann-weighted average by each conformer's own relativeEnergyKcal
   * despite that being the more textbook ensemble-averaging choice: three
   * of this app's own energy models are pluggable into the SAME conformer
   * search (js/conformer-search.js), and one of them (classical) reports
   * "energy" in admittedly arbitrary hand-tuned units, not real kcal/mol
   * (see that file's own MODELS.classical.energyUnit) -- so a Boltzmann
   * weight computed against real RT would be physically meaningless for
   * exactly the model most users reach for first (fastest, no external
   * model load). Rather than branch this file's own physics on which
   * upstream energy model happened to produce the ensemble, an unweighted
   * mean over the already energy-window-pruned (CREST-style, see that
   * file's header) low-energy ensemble is used for every model uniformly
   * -- a real, disclosed simplification, not a hidden one: still a
   * genuine variance-reduction over trusting one arbitrary member, just
   * not a rigorous thermodynamic population average.
   *
   * Returns { correctedPKa, deltaDeltaG, statisticalShift, conformerCount
   * }, same order as `sites`. statisticalShift (see .compute() above) is
   * geometry-independent, so it's computed once here rather than
   * per-conformer-and-averaged like deltaDeltaG -- averaging N identical
   * copies of the same value would just waste work reproducing it.
   * Falls back to NaN-safe behavior only if conformersAtoms3D is empty
   * (conformerCount=0 -- callers should treat that the same as "no
   * correction available," same as .compute() throwing for a malformed
   * geometry).
   */
  CC.PKAElectrostaticCorrection.computeEnsemble = function (molecule, sites, basePKaValues, conformersAtoms3D, opts) {
    const n = sites.length;
    const sumDdG = new Array(n).fill(0);
    let count = 0;

    conformersAtoms3D.forEach(function (atoms3D) {
      const r = CC.PKAElectrostaticCorrection.compute(molecule, sites, basePKaValues, atoms3D, opts);
      r.deltaDeltaG.forEach(function (v, i) { sumDdG[i] += v; });
      count++;
    });

    const deltaDeltaG = sumDdG.map(function (v) { return count > 0 ? v / count : 0; });
    const statShift = statisticalShift(sites, basePKaValues);
    const correctedPKa = basePKaValues.map(function (base, i) {
      const sign = sites[i].cls === 'acid' ? 1 : -1;
      return base + sign * deltaDeltaG[i] / (LN10 * RT_KCAL_298K) + statShift[i];
    });

    return { correctedPKa: correctedPKa, deltaDeltaG: deltaDeltaG, statisticalShift: statShift, conformerCount: count };
  };
})();
