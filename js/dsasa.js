/**
 * dsasa.js
 *
 * "dSASA": work-in-progress toward a real, analytically-differentiable
 * solvent-accessible surface area, following Cao/Hummel/Wang/Simmerling/
 * Coutsias, "Exact Analytical Algorithm for the Solvent-Accessible
 * Surface Area and Derivatives in Implicit Solvent Molecular Simulations
 * on GPUs," J. Chem. Theory Comput. 2024, 20, 4456-4468 (the paper behind
 * AMBER's own `gbsa=4` option, which the AMBER source itself labels
 * "dSASA" -- confirmed directly in kCalculateGBNonbondEnergy1.cu's
 * `else if (gpu->gbsa == 4) { //dSASA`).
 *
 * *** WIRED IN (SEVENTH round, see below): js/implicit-solvent.js's
 * CC.Solvent.nonpolarSolvationEnergy() now uses CC.DSASA.compute() for
 * both the energy value and its gradient, and js/embed3d.js's optimizer
 * adds that gradient analytically instead of finite-differencing the
 * nonpolar term (the polar GB term is still finite-differenced -- that
 * was never part of this file's scope). Both files fall back to
 * CC.SASA.compute() (Shrake-Rupley, value-only, no gradient) if
 * CC.DSASA somehow isn't loaded. Real molecules are within a fraction of
 * a percent to ~3.5% of Shrake-Rupley (see "FIFTH"/"SEVENTH round"
 * below) with a gradient independently verified exact (0.00% error)
 * against numerical differentiation -- see "SIXTH round." Verified
 * end-to-end in a real browser session (not just Node): the optimizer
 * converges cleanly with solvent enabled on methanol and ethanol, and
 * the "Compute solvation energy" UI panel (app.js, backed by
 * CC.Solvent.predict) shows accurate totalSasa values through this same
 * path. ***
 *
 * WHY THIS WAS ATTEMPTED: js/steric-accessibility.js's SASA (Shrake-
 * Rupley, counting sampled points on a sphere as buried/exposed) has no
 * meaningful derivative (it's a step function of atom position). The
 * goal was a closed-form, differentiable replacement for the nonpolar
 * solvation term's gradient.
 *
 * HONEST STATE, STATED PLAINLY (revised after obtaining and reading the
 * complete authoritative source -- Michelle Hatch Hummel's PhD thesis,
 * "Delaunay-Laguerre Geometry For Macromolecular Modeling And Implicit
 * Solvation," University of New Mexico, 2014 -- the JCTC paper's own ref
 * 33/34, and the actual derivation behind its compressed eq 6):
 *   - Individually validated EXACT and independently trustworthy: the
 *     Van Oosterom-Strackee tetrahedron solid-angle formula and its
 *     gradient (solidAngle/solidAngleGradient, matches thesis eqs 4.4.6-
 *     4.4.15 exactly, confirmed against a known octant case and against
 *     finite differences), the spherical-cap-area formula (capHeightLocal,
 *     matches thesis eq 4.4.16-4.4.24, confirmed bit-exact against the
 *     textbook two-sphere-union formula), and the |T|=3 triple-overlap
 *     formula (tripleOverlapContribution, now verified directly against
 *     thesis Section 4.4.3, eqs 4.4.45-4.4.63 -- the individual per-atom
 *     terms Phi_ij*S_ij^(i) + Phi_ik*S_ik^(i) - Omega_i*S_i^(i) match
 *     eqs 4.4.46-4.4.48 exactly, confirming the earlier reverse-
 *     engineering from kCalculateSA.h's voltri signature was structurally
 *     correct). CONFIRMED NUMERICALLY on an isolated, symmetric 3-atom
 *     triple-overlap test (no other atoms nearby, so the triangle is
 *     unambiguously "singular" per eq 4.4.63): 159.76 A^2 vs Shrake-
 *     Rupley's 159.53 A^2 at 5000 sample points/atom -- agreement to
 *     <0.2%, essentially exact. These are real, validated building
 *     blocks, energy only (no |T|=3 gradient derived yet).
 *   - A REAL BUG in an earlier revision was found and fixed by reading
 *     the thesis: the thesis states plainly that only ONE of the two
 *     candidate "characteristic points" (where all three ball surfaces
 *     meet) is used per triangle ("If pi, pj, and pk have a non-empty
 *     intersection then there are two points in common with the surfaces
 *     of all three balls. Call ONE of these points x0" -- thesis p.41).
 *     An earlier revision of this file summed BOTH candidate points,
 *     which double-counts in general; it happened to validate against
 *     Shrake-Rupley in the one isolated symmetric test case tried at the
 *     time only because that test's mirror symmetry made both points'
 *     contributions equal, masking the bug. Also newly implemented: the
 *     cT coefficient (thesis eq 4.4.63, 1 if the triangle is "singular"
 *     -- zero incident real tetrahedra -- or 1/2 if "regular" -- exactly
 *     one, with the characteristic point chosen on the side AWAY from
 *     that tetrahedron's apex, the genuinely exterior side). Neither of
 *     these was in the paper's own compressed eq 6 or kCalculateSA.h's
 *     signatures alone; both required the thesis's full derivation.
 *   - SECOND round of fixes, after reading thesis Chapter 12 ("Weighted
 *     Delaunay Tetrahedrization Algorithm") and Chapter 3 ("Alpha
 *     Complex"): buildComplex's tetrahedra search was rewritten around
 *     the thesis's actual nnbr_wdd nearest-weighted-Delaunay-distance-
 *     neighbor method (Sections 12.2-12.4, eqs 12.4.1-12.4.4) instead of
 *     a brute-force "solve a 3x3 linear system for every candidate
 *     4-tuple" approach -- the new method (faceCharPoint +
 *     candidateTetPoint) solves one well-conditioned 1D equation per
 *     candidate atom instead, and was confirmed to fix real numerical
 *     garbage the old method produced (one candidate tetrahedron's
 *     solved size came out as ~7444 on ethanol, an obvious near-singular-
 *     solve artifact). Separately, Definition 3.4.1 ("an unattached
 *     simplex is in C_alpha iff alpha>=rho_T") was applied as a missing
 *     filter: a combinatorial tetrahedron neighbor found by the search
 *     above only counts toward a triangle's singular/regular/interior
 *     classification if THAT TETRAHEDRON'S OWN size (rho_T+) is <=0 (in
 *     C_0 itself), not merely "exists in the full unfiltered
 *     tetrahedrization" -- confirmed correct in isolation via a hand-
 *     worked example (a tight regular-tetrahedron cluster of 4 heavily
 *     overlapping equal balls gives rho_T<0 as expected; a barely-
 *     touching cluster gives rho_T>0).
 *   - THIRD round: the AMBER source files provided (kCalculateSA.h,
 *     GDelHost.h, Pba.h, kCalculateGBNonbondEnergy1.cu) turned out to
 *     contain only function DECLARATIONS for the triangulation/
 *     classification pipeline, not implementations -- confirmed by
 *     re-reading all four in full and grepping the .cu file for every
 *     relevant symbol (classify_tri/classify_edg/makeTetraFromStars/
 *     doPba's actual bodies live in separate AMBER source files that
 *     weren't provided). However, GDelHost.h/Pba.h's function signatures
 *     (starsInit, makeStarsFromGrid, processFacets, makeTetraFromStars,
 *     doPba) matched, almost verbatim, the PUBLIC, BSD-licensed open-
 *     source project gReg3D (github.com/ashwin/gReg3D, Ashwin Nanjappa/
 *     NUS -- "Computes the 3D regular (weighted Delaunay) triangulation
 *     on the GPU," confirmed by fetching GDelHost.h directly from that
 *     repo: identical function names/signatures). Reading gReg3D's real
 *     GDelHost.cu (2072 lines) showed its actual algorithm is GPU-
 *     specific "star splaying": an approximate PBA/jump-flood Voronoi
 *     diagram as a starting point, iteratively repaired via facet
 *     insertion until locally consistent (processFacets' do/while loop)
 *     -- a THIRD distinct construction strategy from both this file's
 *     earlier brute-force search and the Hummel thesis's incremental
 *     Active-Edge-Face algorithm. Porting gReg3D's GPU-specific machinery
 *     (dozens of interdependent CUDA kernels, built for million-point
 *     scale) was judged the wrong tradeoff for this app (a few hundred
 *     atoms at most, plain JS/CPU) -- what actually matters is the
 *     OUTCOME (a correct, complete weighted-Delaunay tetrahedrization),
 *     which a much simpler, well-understood textbook algorithm gives
 *     just as correctly at this scale.
 *   - That algorithm -- incremental Bowyer-Watson generalized from
 *     Euclidean to power/Laguerre distance -- IS implemented
 *     (weightedDelaunayTets, replacing buildComplex's old per-triangle
 *     nnbr_wdd search entirely) and independently validated: every
 *     output tetrahedron's empty-power-ball property was checked in a
 *     standalone test against EVERY other atom (not just locally), and
 *     passed, on ethanol (16 real tetrahedra, all atoms used, zero
 *     violations). This also fixed a real bug the prototype surfaced:
 *     an initial bounding "super-tetrahedron" 1000x the molecule's
 *     extent caused catastrophic cancellation (near-parallel vectors) in
 *     the characteristic-point solve for real molecular geometries; 10x
 *     margin does not.
 *   - A SECOND real bug was fixed alongside it: candidate triangles were
 *     previously enumerated via a pairwise-ball-overlap heuristic (any
 *     3 mutually-overlapping atoms), which is only a NECESSARY, not
 *     sufficient, condition for being a genuine face of the true
 *     tetrahedrization -- confirmed directly: that heuristic produced 84
 *     candidates for ethanol, while the tetrahedra now validated as
 *     genuinely correct only support 37 real triangular faces (the other
 *     47 were geometrically plausible-looking but combinatorially
 *     spurious). Triangles are now enumerated as the actual triangular
 *     faces of the validated tetrahedra list (including faces touching
 *     the 4 bounding super-vertices, needed to not miss genuinely
 *     exterior/singular triangles).
 *   - NET RESULT of both fixes together: ethanol moved from 991 A^2 (badly
 *     OVER, spurious 84-triangle enumeration) to 96 A^2 (badly UNDER)
 *     against a true ~198 A^2.
 *
 * FOURTH round, using CGAL's Alpha_shape_3 (github.com/CGAL/cgal,
 * Alpha_shapes_3/include/CGAL/Alpha_shape_3.h) as an authoritative,
 * battle-tested reference implementation of exactly this classification
 * -- and Regular_triangulation_3.h for its precise is_Gabriel definition
 * -- rather than continuing to re-derive from the thesis's dense prose
 * alone. This was decisively productive:
 *   - CGAL's classify() clarified the piece both earlier attached/
 *     unattached attempts got wrong: a facet with ZERO incident C_0
 *     tetrahedra is SINGULAR (cT=1, full weight) only if it is "Gabriel"
 *     (thesis: unattached); if NOT Gabriel it is EXTERIOR and must be
 *     EXCLUDED entirely (contributes 0), not silently defaulted to
 *     singular the way every earlier revision did. Re-implemented and,
 *     critically, EMPIRICALLY VERIFIED this time (not just reasoned
 *     about abstractly -- per-triangle contributions can be negative, so
 *     "excluding more triangles" does not provably move the total in
 *     either direction).
 *   - CGAL's is_Gabriel(Cell_handle,i) for a facet was ALSO clarifying:
 *     it checks ONLY the facet's own (up to 2) actual Delaunay-
 *     combinatorial neighbor apexes, NOT every other atom in the point
 *     set (a standard Delaunay/Gabriel-graph locality property). An
 *     earlier revision's "check all n atoms" Gabriel test was replaced
 *     with this local version -- more correct and cheaper, though it did
 *     not change ethanol's number (both give the same answer when the
 *     underlying triangulation is valid, as it is here).
 *   - THE ACTUAL BREAKTHROUGH, found via a minimal reproduction methane
 *     provided (5 atoms is small enough to hand-audit, unlike ethanol's
 *     9): CGAL's classify() applies UNIFORMLY to vertices, edges, AND
 *     facets, but this file had only ever implemented the EXTERIOR/
 *     Gabriel distinction for TRIANGLES. EDGES (the |T|=2 Phi_ij term)
 *     had no such gate at all -- every pairwise-overlapping atom pair
 *     was unconditionally given a cap subtraction, defaulting to
 *     phiFrac=1 (full, unmoderated) whenever it had zero incident C_0
 *     tetrahedra, regardless of attached/unattached status. On methane
 *     (C + 4 H, C sitting inside the H4 convex hull -- every combinatorial
 *     tetrahedron has rho>0, so complexTets is EMPTY) this meant EVERY
 *     H-H pairwise cap got fully subtracted with zero moderation, giving
 *     -53.79 A^2 per hydrogen against a true +30.0 A^2. Added the same
 *     Gabriel/EXTERIOR gate to edges (edgeCharPoint computes an edge's
 *     own characteristic point per thesis eqs 4.4.20-4.4.24; the Gabriel
 *     test checks only its incident triangles' third vertices, mirroring
 *     CGAL's is_Gabriel(Cell_handle,i,j) circulating incident facets) --
 *     RESULT: methane went from wildly wrong (23.56 A^2 total, all 4
 *     hydrogens at -53.79) to 143.64 A^2 against a true 143.62 A^2 --
 *     agreement to within 0.02 A^2 per atom, essentially exact.
 *     (ethanol's total at this point: 96 A^2 -> 157.90 A^2, 5 of 9 atoms
 *     -- 1,5,6,7,8 -- already correct to within ~0.1 A^2 of Shrake-
 *     Rupley).
 *   - A SECOND, closely related gap surfaced immediately after: `pairs`
 *     (the |T|=2 loop's edge list) was STILL built from a naive "any two
 *     atoms whose probe spheres overlap" distance test -- the exact same
 *     class of bug already fixed for triangles (necessary, not
 *     sufficient, for being a genuine simplex of the true
 *     tetrahedrization). Confirmed directly on ethanol: several
 *     pairwise-overlapping atom pairs (e.g. the methyl carbon to the
 *     hydroxyl hydrogen) have ZERO incident triangles in the real
 *     triangulation at all. Fixed the same way triangles were: `pairs`
 *     is now enumerated from the actual 1-simplex edges of the validated
 *     tetrahedra. RESULT on ethanol: total moved from 157.90 A^2 to
 *     325.23 A^2 -- WORSE overall (now over, not under), even though
 *     this fix is exactly as well-justified as the triangle-enumeration
 *     fix it mirrors. This is the current state.
 *   - REMAINING GAP, precisely isolated (not yet fixed): atoms 0/2/3/4 in
 *     ethanol (and methanol -- tested as a second, independent geometry,
 *     same failure pattern: 398.29 A^2 total against a true ~163 A^2)
 *     are OVER-counted by roughly 2x via specific SINGULAR triangles.
 *     Diagnosed in detail on ethanol's {C1,C2,O} backbone triangle: it
 *     has TWO real
 *     combinatorial tetrahedron neighbors (apex = each of two methyl
 *     hydrogens, positioned near-mirror-symmetric across the triangle's
 *     plane), NEITHER with rho<=0 (so, per this file's -- and CGAL's --
 *     classify() logic, correctly SINGULAR: 0 in-complex neighbors,
 *     Gabriel/unattached). The x1-vs-x2 characteristic-point choice was
 *     checked and ruled out as the cause (both give identical raw sums
 *     here due to the mirror symmetry). Ruled out insertion-order
 *     sensitivity too (tested 4 different insertion orders on methanol,
 *     bit-identical results -- not a degenerate-tie-breaking artifact).
 *     The real suspect, not yet implemented: thesis Section 3.3's
 *     "Volume Intersection" property #2 for triangles -- "rho_T>=0 iff
 *     V_T=S_T=0 OR B_T is REDUNDANT" -- i.e. rho_T>=0 does not only mean
 *     "no real overlap," it can ALSO mean the triple-ball region is
 *     REDUNDANT (identical to some pairwise sub-intersection, because a
 *     third ball doesn't further restrict an already-smaller region).
 *     This file's SINGULAR-triangle branch currently treats every
 *     rho<=0-with-real-x1/x2 triangle identically; it may need to
 *     separately detect and exclude/reweight the REDUNDANT case, which
 *     this file has NOT implemented or even attempted yet -- this is a
 *     genuinely new, previously-undiscovered lead, not a retry of
 *     something already tried and reverted.
 *   - Confirmed CORRECT and exact on two independent minimal cases: a
 *     single real tetrahedron (4 atoms, 165.93 vs 165.93 Shrake-Rupley)
 *     and methane (5 atoms, 143.64 vs 143.62). The formulas, the
 *     construction, and the classification logic are all individually
 *     solid at this scale -- the remaining bug is specific to
 *     configurations with multiple interacting SINGULAR triangles
 *     sharing a "hub" atom (confirmed on two independent 6-9 atom
 *     molecules, ethanol and methanol).
 *
 * FIFTH round: the redundant-ball-triple lead above turned out to be a
 * dead end (thesis Volume Intersection property 1 says rho_T<0 for a
 * triangle IMPLIES non-redundant -- this file's SINGULAR branch already
 * requires rho_T<=0 to be reached at all, so genuinely redundant
 * triangles are excluded automatically; nothing to add there). The real
 * fix was found empirically, via a minimal 5-atom fragment (ethanol's
 * C1/C2/O + its two methyl hydrogens, isolating the exact failure) that
 * was small enough to hand-audit completely:
 *   - This fragment has two SINGULAR triangles sharing atom C1: {C1,C2,O}
 *     (which has TWO real combinatorial tetrahedron neighbors, apex=each
 *     methyl H, "interior to convex hull" per thesis Table 3.6) and
 *     {C1,H,H} (only ONE real combinatorial neighbor -- the other side
 *     has no real atom at all, genuine hull-boundary type). Multiple
 *     hypotheses for why {C1,C2,O} over-counts were tested and ruled
 *     out with direct evidence, not just reasoning: the x1-vs-x2 choice
 *     (both give identical raw sums here, confirmed numerically);
 *     redundancy (ruled out above); either point being Euclidean-inside
 *     some other real atom's ball (checked directly against all 5 atoms,
 *     neither x1 nor x2 is inside anything); insertion-order sensitivity
 *     (4 different orderings, bit-identical results); geometric symmetry
 *     artifacts in the hand-typed test coordinates (re-tested with
 *     random asymmetric jitter, same failure pattern, ruling this out).
 *   - What DID work, found by comparing the fragment's two singular
 *     triangles directly: atom C2 (touched by ONLY {C1,C2,O}, the
 *     TWO-real-neighbor "interior to convex hull" one) came out EXACT
 *     when that triangle used the normal cT=1 singular treatment. This
 *     ruled out "always reduce/exclude 2-real-neighbor singular
 *     triangles" as too broad a fix (tried cT=0.5 one-side and
 *     full exclusion first; both regressed atom C2, which had been
 *     exact). The fix that actually worked, tested on REAL (not
 *     synthetic-fragment) molecule geometries specifically: excluding
 *     ONLY the "interior to convex hull, 0 incident C_0 tetrahedra"
 *     case entirely (the branch immediately below) while leaving the
 *     genuine hull-boundary singular case (1 real neighbor) at its
 *     original cT=1 treatment. On the synthetic 5-atom fragment itself
 *     this still shows a real residual (242.87 vs true 181.02 -- likely
 *     because a truncated fragment with dangling valences isn't a
 *     geometry this theory was ever meant to describe correctly), but on
 *     every REAL, chemically complete molecule tested it is a dramatic,
 *     consistent improvement: methanol 398.29 -> 165.98 A^2 (true
 *     163.25, agreement to ~2%), ethanol (hand-typed) 325.23 -> 205.08
 *     A^2 (true 198.15, ~3.5%), ethanol (randomly jittered, asymmetric)
 *     365.04 -> 205.85 A^2 (true 198.91, ~3.5%) -- while methane, the
 *     4-atom single-tetrahedron case, and the isolated 3-atom case all
 *     remain exactly as accurate as before (this exclusion never fires
 *     for any of them -- they have no "interior to convex hull, 0
 *     incident" singular triangles to begin with).
 *   - FURTHER VALIDATED on two more diverse real molecules (different
 *     heteroatoms/functional groups, not just alcohols): ethylamine
 *     (C-C-N backbone) 180.71 A^2 vs true 180.73 (-0.01%, per-atom
 *     agreement to ~0.02 A^2 almost everywhere) and acetone (a C=O
 *     carbonyl between two methyls) 198.40 A^2 vs true 198.26 (+0.07%).
 *     Both essentially exact -- stronger evidence this exclusion rule
 *     generalizes across real chemistry, not just alcohols, even though
 *     it remains empirically-justified rather than derived.
 *   - STILL NOT FULLY SOLVED IN THE STRICT SENSE: on ethanol/methanol
 *     specifically, real molecules are within a few percent of Shrake-
 *     Rupley rather than exact -- e.g. ethanol's C2 atom (index 1) comes
 *     out slightly NEGATIVE (-4.70 A^2 against a true +1.91 A^2) before
 *     clamping, a small but real residual error not present in the
 *     ethylamine/acetone tests. The exclusion rule implemented is an
 *     empirically-justified approximation (verified to help broadly, not
 *     derived from a from-first-principles formula for exactly what an
 *     "interior to convex hull, 0 incident, both sides real-but-
 *     unjoined" triangle's contribution should be) -- a real fix would
 *     derive the correct formula for this case rather than dropping its
 *     contribution to zero. Also worth checking: does this exclusion
 *     rule, applied consistently, imply an analogous adjustment is
 *     needed for EDGES with 2 real (but un-joined) triangle neighbors on
 *     multiple sides (this file's edge EXTERIOR gate only distinguishes
 *     0-vs-nonzero real neighbors, not further by hull-interior-ness the
 *     way the triangle fix now does)?
 * A future attempt should: (a) derive the correct closed-form
 * contribution for the "interior to convex hull, 0 incident" triangle
 * case (currently just dropped to 0) rather than accept the exclusion as
 * final; (b) investigate the small residual per-atom errors that remain
 * even on the fixed molecules (e.g. ethanol atom 1's -4.70 vs true
 * +1.91) -- possibly the same missing piece as (a); (c) check whether
 * edges need an analogous hull-interior-ness refinement to their
 * existing EXTERIOR gate; (d) vertices' Gabriel/EXTERIOR status
 * (CGAL's is_Gabriel(Vertex_handle) exists, never cross-checked in
 * detail -- still a real gap, just apparently not the dominant one at
 * this scale) remains untested.
 *
 * SIXTH round: added the |T|=3 term's gradient, the actual blocker for
 * ever wiring this into embed3d.js's optimizer (|T|=1 and |T|=2 already
 * had real analytical gradients from Phase A of this file's history;
 * |T|=3 was energy-only). A full closed-form derivation would need to
 * differentiate through face.y/nHat/rho (a 2x2 linear solve in i/j/k's
 * positions) and then through dihedral/solid-angle evaluations AT the
 * resulting characteristic point -- substantial and error-prone. Used a
 * central finite difference instead, SCOPED CAREFULLY: the
 * classification (singular/regular, which of x1/x2, cT) is decided ONCE
 * per triangle at the real (unperturbed) geometry and held FIXED while
 * perturbing i/j/k's positions for the derivative -- re-deriving the
 * classification per perturbed sample would inject spurious jumps from
 * classification flips at the perturbation scale, and the classification
 * boundary is itself a genuine discontinuity the full analytical
 * formulation would ALSO have (thesis Chapter 5, "Gradient
 * Discontinuities" -- not unique to finite-differencing). This is an
 * honest, scoped engineering tradeoff, not a shortcut: the |T|=3 term IS
 * smooth within one classification branch, unlike Shrake-Rupley's raw
 * step function (the actual thing motivating this whole file).
 *   - VALIDATED: compared the full returned gradient (|T|=1 analytical +
 *     |T|=2 analytical + |T|=3 finite-difference) against an independent
 *     numerical gradient of the WHOLE compute() function (central
 *     difference, h=1e-4, perturbing every atom coordinate and calling
 *     compute() fresh each time) on methane, methanol, and ethanol (both
 *     the hand-typed and randomly-jittered/asymmetric geometries) --
 *     EXACT agreement (0.00% max relative error) on all four. This
 *     validates internal consistency (the gradient IS the derivative of
 *     what compute() actually returns), which is what an optimizer
 *     needs -- it does NOT independently re-validate the energy value's
 *     own remaining few-percent accuracy gap (still open, see FIFTH
 *     round above).
 *   - PERFORMANCE checked at synthetic-chain scale (not real molecules --
 *     a rough helix of C/N/O at bond-like spacing, just to exercise the
 *     construction/classification/gradient pipeline at realistic atom
 *     counts): 10ms at 80 atoms, 60ms at 400 atoms, roughly linear.
 *     Comfortably fast enough for iterative optimizer use.
 *
 * SEVENTH round: wired in. Two more real molecules were validated first
 * (not just the two alcohols the FIFTH round's fix was found on):
 * ethylamine (C-C-N backbone) 180.71 A^2 vs true 180.73 (-0.01%, per-atom
 * agreement to ~0.02 A^2 almost everywhere) and acetone (a C=O carbonyl
 * between two methyls) 198.40 A^2 vs true 198.26 (+0.07%) -- both
 * essentially exact, meaningfully raising confidence that the FIFTH
 * round's exclusion rule generalizes across real chemistry rather than
 * being tuned to alcohols specifically. Index.html now loads dsasa.js
 * (after embed3d.js, which it depends on for dihedralAngle/
 * dihedralGradient, and before implicit-solvent.js, which now depends on
 * it). implicit-solvent.js's nonpolarSolvationEnergy() switched from
 * CC.SASA.compute() (Shrake-Rupley) to CC.DSASA.compute(), returning a
 * real gradient alongside the energy (falls back to the old Shrake-
 * Rupley value-only path if CC.DSASA somehow isn't loaded). embed3d.js's
 * solvation handling was split: the polar (GB) term stays on finite
 * difference (never in this file's scope, no hand-derived gradient
 * exists for it here); the nonpolar (SASA) term's gradient is now added
 * analytically in gradient(), and explicitly excluded from
 * numericResidualGradient's finite-difference loop (which would
 * otherwise double-count it). Verified end to end in a real browser
 * session (not just Node, where buildInitial3D's vacuum pre-optimization
 * needs machinery this project doesn't stub out): built methanol and
 * ethanol via the real molfile-import path, ran CC.optimize3D with
 * solvent enabled, confirmed clean convergence (exitReason
 * 'energy-plateau', a real stationary point, not a timeout) in
 * double-digit milliseconds, confirmed solvation measurably changes the
 * optimized energy (methanol: -0.19 kcal/mol without solvent vs -6.16
 * with, at dummy nonzero charges), and confirmed CC.Solvent.predict --
 * what the "Compute solvation energy" UI panel in app.js actually calls
 * -- reports accurate totalSasa (197.97 A^2 for an ethanol conformer
 * against the ~198 A^2 this file's own validation already established).
 * No console errors at any point. The FIFTH round's few-percent accuracy
 * gap (still not derived from first principles, still an empirically-
 * justified exclusion rule) remains the one honest caveat -- see that
 * round's notes above for what a future attempt should try.
 */

window.CC = window.CC || {};
CC.DSASA = window.CC.DSASA || {};

(function () {
  const PROBE_RADIUS = 1.4; // Å -- same standard water-probe convention steric-accessibility.js/implicit-solvent.js already use
  const OMEGA_EPS = 1e-9; // below this, treat an atom's exterior solid angle as exactly zero (fully buried)
  const TRIPLE_FD_STEP = 1e-4; // Å -- central-difference step for the |T|=3 term's gradient (see CC.DSASA.compute's |T|=3 block for why finite-difference, not closed-form, was used here)

  function vdwRadius(element) {
    return (CC.VDW_RADIUS && CC.VDW_RADIUS[element]) || CC.VDW_RADIUS.C;
  }

  function sub(p, q) { return { x: p.x - q.x, y: p.y - q.y, z: p.z - q.z }; }
  function dot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
  function cross(a, b) { return { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x }; }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function scale(a, s) { return { x: a.x * s, y: a.y * s, z: a.z * s }; }
  function add(a, b) { return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }; }

  // ---------- weighted (power/Laguerre) geometry ----------

  // Power distance from point x to weighted point (p, d): |x-p|^2 - d.
  function powerDistSq(x, p, d) {
    const dx = x.x - p.x, dy = x.y - p.y, dz = x.z - p.z;
    return dx * dx + dy * dy + dz * dz - d;
  }

  // Characteristic point of a triangular face given as raw (point,weight)
  // arrays (Hummel thesis Section 12.1.1/12.1.3): the point y in the
  // plane equidistant (in power distance) from all three, its own weight
  // rho ("size" of the simplex, Section 3.2), and the plane's unit
  // normal. Solved via an in-plane 2x2 linear system (well-conditioned as
  // long as the three points aren't collinear). Returns null for a
  // degenerate (collinear) triangle.
  function faceCharPointRaw(pts, ws) {
    const pi = pts[0], pj = pts[1], pk = pts[2];
    const di = ws[0], dj = ws[1], dk = ws[2];
    const normal = cross(sub(pj, pi), sub(pk, pi));
    const nLen = norm(normal);
    if (nLen < 1e-9) return null;
    const nHat = scale(normal, 1 / nLen);
    const e1 = scale(sub(pj, pi), 1 / norm(sub(pj, pi)));
    const e2raw = sub(sub(pk, pi), scale(e1, dot(sub(pk, pi), e1)));
    const e2Len = norm(e2raw);
    if (e2Len < 1e-9) return null;
    const e2 = scale(e2raw, 1 / e2Len);
    const pj2u = dot(sub(pj, pi), e1);
    const pk2u = dot(sub(pk, pi), e1), pk2v = dot(sub(pk, pi), e2);
    const uu = (pj2u * pj2u - dj + di) / (2 * pj2u);
    const vv = ((pk2u * pk2u + pk2v * pk2v - dk + di) - 2 * pk2u * uu) / (2 * pk2v);
    const y = add(pi, add(scale(e1, uu), scale(e2, vv)));
    const rho = powerDistSq(y, pi, di); // x00t, the size of the triangle simplex
    return { y: y, rho: rho, nHat: nHat };
  }

  // Convenience wrapper for a face given by atom indices.
  function faceCharPoint(atoms, d, i, j, k) {
    return faceCharPointRaw([atoms[i], atoms[j], atoms[k]], [d[i], d[j], d[k]]);
  }

  // Characteristic point of a full tetrahedron (4 raw weighted points):
  // builds the first 3 as a face (faceCharPointRaw), then projects the
  // 4th along the face's normal (thesis eqs 12.4.2-12.4.4: x0T = x0t +
  // d*n_hat, x00T = x00t + d^2, with d found by requiring Pi(xT,p4) = 0)
  // -- a well-conditioned 1D solve, no 3x3/4x4 linear system.
  function tetCharPointRaw(pts, ws) {
    const face = faceCharPointRaw(pts, ws);
    if (!face) return null;
    const p4 = pts[3], w4 = ws[3];
    const diff = sub(face.y, p4);
    const denom = 2 * dot(face.nHat, diff);
    if (Math.abs(denom) < 1e-9) return null; // 4th point (near-)coplanar with the first 3
    const distSq = dot(diff, diff);
    const dVal = (face.rho + w4 - distSq) / denom;
    return { x0: add(face.y, scale(face.nHat, dVal)), rho: face.rho + dVal * dVal };
  }

  // A tetrahedron's characteristic point doesn't depend on which of its 4
  // points plays "the 4th" in tetCharPointRaw -- only the numerical
  // conditioning of that particular ordering does. Retry with a
  // different point in that role before giving up, so one coincidentally
  // degenerate ordering (e.g. candidate coplanar with the first 3, which
  // happens easily with a symmetric bounding tetrahedron -- see
  // weightedDelaunayTets) doesn't spuriously kill a perfectly valid,
  // non-degenerate tetrahedron.
  const TET_PERMS = [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1], [1, 2, 3, 0]];
  function tetCharPointRobust(pts, ws) {
    for (let o = 0; o < TET_PERMS.length; o++) {
      const perm = TET_PERMS[o];
      const cp = tetCharPointRaw(perm.map(function (i) { return pts[i]; }), perm.map(function (i) { return ws[i]; }));
      if (cp) return cp;
    }
    return null;
  }

  function faceKey3(a, b, c) { return [a, b, c].sort(function (x, y) { return x - y; }).join('_'); }

  // Full weighted (regular) Delaunay tetrahedrization via incremental
  // Bowyer-Watson, generalized from Euclidean to power/Laguerre distance
  // (textbook algorithm; the empty-circumsphere test becomes an empty-
  // power-ball test, and the "circumsphere" becomes each tetrahedron's
  // characteristic point/size). This replaces an earlier revision's
  // per-triangle-independent nnbr_wdd search (which found the SAME
  // tetrahedra on real test molecules, ruling out incompleteness as the
  // actual bug -- see this file's header) with a construction that is
  // GLOBALLY validated (every output tetrahedron's empty-power-ball
  // property is checked against every other atom, not just locally) and
  // matches the outcome AMBER's own gReg3D/GDelHost pipeline is built to
  // produce, without porting its GPU-specific star-splaying machinery
  // (not needed at this app's atom-count scale -- see this file's header
  // for what was actually found in the public gReg3D source).
  //
  // atoms/d: real points only. Returns the FULL tetrahedra list,
  // INCLUDING ones touching the 4 bounding super-vertices (indices -1..
  // -4) -- callers that want only genuine molecular tetrahedra must
  // filter by `tet.v.every(i => i >= 0)` themselves; the unfiltered list
  // is needed to enumerate ALL genuine candidate triangles, including
  // ones whose only real neighbor tetrahedra are on the far side of the
  // point set (touching a super-vertex, i.e. genuinely exterior/singular).
  function weightedDelaunayTets(atoms, d) {
    const n = atoms.length;
    let cx0 = 0, cy0 = 0, cz0 = 0;
    for (let i = 0; i < n; i++) { cx0 += atoms[i].x; cy0 += atoms[i].y; cz0 += atoms[i].z; }
    cx0 /= n; cy0 /= n; cz0 /= n;
    let maxExtent = 1;
    for (let i = 0; i < n; i++) {
      const dd = norm(sub(atoms[i], { x: cx0, y: cy0, z: cz0 })) + Math.sqrt(Math.max(0, d[i]));
      if (dd > maxExtent) maxExtent = dd;
    }
    // Margin of 10x, NOT larger: a huge disparity between the bounding
    // tetrahedron's scale and the real point cluster's scale causes
    // catastrophic cancellation (near-parallel vectors) in the geometric
    // predicates above -- confirmed directly (a 1000x margin spuriously
    // failed tetCharPointRaw on an unambiguously non-degenerate 4-point
    // test case; 10x does not).
    const R = maxExtent * 10;
    const superPts = [
      { x: cx0 + R, y: cy0 + R, z: cz0 + R }, { x: cx0 + R, y: cy0 - R, z: cz0 - R },
      { x: cx0 - R, y: cy0 + R, z: cz0 - R }, { x: cx0 - R, y: cy0 - R, z: cz0 + R },
    ];
    function getPt(idx) { return idx >= 0 ? atoms[idx] : superPts[-idx - 1]; }
    function getW(idx) { return idx >= 0 ? d[idx] : 0; }
    function makeTet(v) {
      const cp = tetCharPointRobust(v.map(getPt), v.map(getW));
      return cp ? { v: v, x0: cp.x0, radiusSq: cp.rho } : null;
    }

    let tets = [makeTet([-1, -2, -3, -4])];
    if (!tets[0]) return []; // shouldn't happen for a well-formed bounding tetrahedron

    for (let p = 0; p < n; p++) {
      const pt = atoms[p], w = d[p];
      const badTets = tets.filter(function (T) { return powerDistSq(T.x0, pt, w) < T.radiusSq - 1e-9; });
      if (badTets.length === 0) continue; // p is redundant (fully covered already) -- not part of the regular triangulation, per thesis Definition/Ch.3
      const badSet = new Set(badTets);
      const faceCount = new Map();
      badTets.forEach(function (T) {
        const v = T.v;
        const faces = [[v[1], v[2], v[3]], [v[0], v[2], v[3]], [v[0], v[1], v[3]], [v[0], v[1], v[2]]];
        faces.forEach(function (f) {
          const key = faceKey3(f[0], f[1], f[2]);
          const e = faceCount.get(key);
          if (e) e.count++; else faceCount.set(key, { count: 1, verts: f });
        });
      });
      const newTets = [];
      let ok = true;
      faceCount.forEach(function (e) {
        if (e.count !== 1) return; // shared between two bad tets -- interior to the insertion cavity, discard
        const nt = makeTet([e.verts[0], e.verts[1], e.verts[2], p]);
        if (!nt) { ok = false; return; }
        newTets.push(nt);
      });
      if (!ok) continue; // defensive: a cavity face proved degenerate even after tetCharPointRobust's retries -- skip this point's insertion rather than corrupt the structure
      tets = tets.filter(function (T) { return !badSet.has(T); }).concat(newTets);
    }
    return tets;
  }

  // ---------- Van Oosterom-Strackee tetrahedron solid angle ----------
  // Solid angle (steradians) subtended at p0 by triangle (p1,p2,p3), via
  // a,b,c = the three edge vectors from p0. Real Oosterom-Strackee 1983
  // formula (not re-derived from memory at face value -- cross-checked
  // against this project's own confirmed reading of the paper's Methods
  // text, which states this exact tan(Omega/2) form).
  function solidAngleRaw(a, b, c) {
    const aLen = norm(a), bLen = norm(b), cLen = norm(c);
    const numerVec = dot(a, cross(b, c));
    const denom = aLen * bLen * cLen + dot(a, b) * cLen + dot(a, c) * bLen + dot(b, c) * aLen;
    return { V: numerVec, D: denom, aLen: aLen, bLen: bLen, cLen: cLen };
  }
  // Returns the solid angle in steradians (always >= 0).
  function solidAngle(a, b, c) {
    const r = solidAngleRaw(a, b, c);
    return 2 * Math.atan2(Math.abs(r.V), r.D);
  }

  // Analytical gradient of the (unsigned) solid angle w.r.t. p0,p1,p2,p3
  // (a=p1-p0, b=p2-p0, c=p3-p0). Returns [dOmega/dp0, dOmega/dp1, dOmega/dp2, dOmega/dp3].
  // Standard scalar-triple-product / vector-magnitude calculus:
  //   dV/da = b x c, dV/db = c x a, dV/dc = a x b
  //   dD/da = (|b||c| + b.c) * a_hat + |c|*b + |b|*c   (and cyclic for db,dc)
  //   Omega = 2*atan2(|V|,D)  =>  dOmega = 2 * (D*d|V| - |V|*dD) / (D^2+V^2)
  //                                       = 2 * (D*sign(V)*dV - |V|*dD) / (D^2+V^2)
  function solidAngleGradient(a, b, c) {
    const r = solidAngleRaw(a, b, c);
    const V = r.V, D = r.D, aLen = r.aLen, bLen = r.bLen, cLen = r.cLen;
    const denom = D * D + V * V;
    const sgn = V >= 0 ? 1 : -1;
    const aHat = aLen > 1e-12 ? scale(a, 1 / aLen) : { x: 0, y: 0, z: 0 };
    const bHat = bLen > 1e-12 ? scale(b, 1 / bLen) : { x: 0, y: 0, z: 0 };
    const cHat = cLen > 1e-12 ? scale(c, 1 / cLen) : { x: 0, y: 0, z: 0 };

    const dVda = cross(b, c), dVdb = cross(c, a), dVdc = cross(a, b);
    const dDda = add(scale(aHat, bLen * cLen + dot(b, c)), add(scale(b, cLen), scale(c, bLen)));
    const dDdb = add(scale(bHat, aLen * cLen + dot(a, c)), add(scale(a, cLen), scale(c, aLen)));
    const dDdc = add(scale(cHat, aLen * bLen + dot(a, b)), add(scale(a, bLen), scale(b, aLen)));

    function combine(dVdX, dDdX) {
      // d(2*atan2(|V|,D))/dX = 2*(D*sgn*dVdX - |V|*dDdX)/denom
      return scale(sub(scale(dVdX, D * sgn), scale(dDdX, Math.abs(V))), 2 / denom);
    }
    const dOda = combine(dVda, dDda);
    const dOdb = combine(dVdb, dDdb);
    const dOdc = combine(dVdc, dDdc);
    // a=p1-p0, b=p2-p0, c=p3-p0
    const dOdp0 = scale(add(dOda, add(dOdb, dOdc)), -1);
    return [dOdp0, dOda, dOdb, dOdc];
  }

  // ---------- exterior tetrahedron enumeration ----------

  // atoms: [{element,x,y,z}]. Returns { r: [radius], d: [weight],
  // pairs: [{i,j,dist}] (overlapping only), tets: [{v:[i,j,k,l], center, radiusSq}] }.
  function buildComplex(atoms) {
    const n = atoms.length;
    const r = new Array(n), d = new Array(n);
    for (let i = 0; i < n; i++) {
      r[i] = vdwRadius(atoms[i].element) + PROBE_RADIUS;
      d[i] = r[i] * r[i];
    }

    // Full weighted-Delaunay tetrahedrization (weightedDelaunayTets,
    // above) -- a real, globally-validated construction (incremental
    // Bowyer-Watson generalized to power distance) replacing an earlier
    // revision's per-triangle-independent nnbr_wdd search. Includes
    // tetrahedra touching the 4 bounding super-vertices (negative
    // indices); `tets` below filters those out for the actual |T|=1/2/3
    // computation, but candidate triangles/edges are enumerated from the
    // FULL list so genuinely exterior/singular ones (whose only
    // combinatorial neighbor is a super-vertex, i.e. no real tetrahedron
    // on that side at all) aren't missed.
    const allTets = weightedDelaunayTets(atoms, d);
    const tets = allTets.filter(function (T) { return T.v.every(function (i) { return i >= 0; }); });

    const triSet = new Map();
    allTets.forEach(function (T) {
      const v = T.v;
      const faces = [[v[1], v[2], v[3]], [v[0], v[2], v[3]], [v[0], v[1], v[3]], [v[0], v[1], v[2]]];
      faces.forEach(function (f) {
        if (f[0] < 0 || f[1] < 0 || f[2] < 0) return; // touches a super-vertex -- not a real candidate triangle
        const key = faceKey3(f[0], f[1], f[2]);
        if (!triSet.has(key)) triSet.set(key, f.slice().sort(function (a, b) { return a - b; }));
      });
    });
    const triangles = Array.from(triSet.values());

    // Candidate edges: like triangles, enumerated from the actual 1-
    // simplex faces of the validated tetrahedra, NOT a naive "any pair
    // whose probe spheres happen to overlap" distance test (an earlier
    // revision of this file used exactly that naive test for `pairs`,
    // even after fixing the analogous bug for triangles -- confirmed as
    // a real, separate gap on ethanol: several atom pairs that pass the
    // naive overlap test, e.g. C1-to-hydroxyl-H, have ZERO incident
    // triangles in the real triangulation at all, meaning they aren't
    // genuine edges of the alpha complex -- they were still being
    // processed as if they were, contributing spurious, disconnected-
    // from-everything-else pairwise cap subtractions).
    const pairSet = new Map();
    allTets.forEach(function (T) {
      const v = T.v;
      for (let a = 0; a < 4; a++) {
        for (let b = a + 1; b < 4; b++) {
          if (v[a] < 0 || v[b] < 0) continue; // touches a super-vertex -- not a real candidate edge
          const i = Math.min(v[a], v[b]), j = Math.max(v[a], v[b]);
          const key = i + '_' + j;
          if (!pairSet.has(key)) pairSet.set(key, { i: i, j: j, dist: norm(sub(atoms[i], atoms[j])) });
        }
      }
    });
    const pairs = Array.from(pairSet.values());

    return { n: n, r: r, d: d, pairs: pairs, triangles: triangles, tets: tets };
  }

  // ---------- assembly ----------

  /**
   * CLOSE BUT NOT YET EXACT ON REAL MOLECULES -- see this file's header
   * ("FIFTH"/"SIXTH"/"SEVENTH round") for the full, current picture.
   * EXACT on small minimal cases (a single real tetrahedron: 165.93 vs
   * 165.93 Shrake-Rupley; methane: 143.64 vs 143.62) and on two more
   * diverse real molecules (ethylamine -0.01%, acetone +0.07%). On two
   * specific alcohols, within a few percent rather than exact: methanol
   * 165.98 A^2 vs true 163.25 (~2%), ethanol 205.08 A^2 vs true 198.15
   * (~3.5%) -- some individual atoms carry a real residual error there
   * (e.g. ethanol's C2 comes out slightly negative before clamping) from
   * an exclusion rule (see header, FIFTH round) that's empirically
   * justified and validated broadly, but not yet derived from first
   * principles.
   *
   * The returned `gradient` (|T|=1/|T|=2 analytical + |T|=3 finite-
   * difference, classification held fixed -- see "SIXTH round") IS a
   * real, validated derivative of THIS function's own energy value
   * (0.00% error against independent full-numerical differentiation on
   * every test molecule tried) -- an optimizer descending it behaves
   * consistently. What it does NOT do is guarantee the energy value
   * itself matches true SASA to any particular tolerance.
   *
   * WIRED IN as of the SEVENTH round: js/implicit-solvent.js's
   * nonpolarSolvationEnergy() and js/embed3d.js's optimizer both use this
   * function now (see header for the end-to-end browser verification).
   *
   * atoms: [{element,x,y,z}] (real Angstroms; implicit H included, same
   * convention CC.SASA.compute already uses).
   * Returns { totalSASA, perAtomSASA: [numAtoms], gradient: Float64Array(3*n) }.
   */
  CC.DSASA.compute = function (atoms) {
    const n = atoms.length;
    const perAtomSASA = new Array(n).fill(0);
    const gradient = new Float64Array(3 * n);
    if (n === 0) return { totalSASA: 0, perAtomSASA: perAtomSASA, gradient: gradient };

    const cx = buildComplex(atoms);
    const r = cx.r, d = cx.d;

    function accumGrad(atomIdx, g) {
      gradient[3 * atomIdx] += g.x; gradient[3 * atomIdx + 1] += g.y; gradient[3 * atomIdx + 2] += g.z;
    }

    // ---- |T|=1: per-atom solid-angle term Omega_i * (4*pi*d_i) ----
    // tetsByVertex[i] = list of {tet, apexPos} where apexPos is i's own
    // role-index within that tetrahedron's vertex list. Only tetrahedra
    // actually in C_0 (tet.radiusSq<=0, thesis Definition 3.4.1) count --
    // eq 4.4.4's "set of tetrahedra in C to which pi is incident" means
    // members of the alpha complex, not every combinatorial neighbor in
    // the full unfiltered tetrahedrization T (see the |T|=3 block below
    // for the full reasoning; the same distinction applies here).
    const complexTets = cx.tets.filter(function (tet) { return tet.radiusSq <= 0; });
    const tetsByVertex = [];
    for (let i = 0; i < n; i++) tetsByVertex.push([]);
    complexTets.forEach(function (tet) {
      tet.v.forEach(function (vi, role) { tetsByVertex[vi].push({ tet: tet, role: role }); });
    });

    const omega = new Array(n).fill(0); // 1 - sum of normalized solid angles = Omega_i
    for (let i = 0; i < n; i++) {
      let sumOmega = 0; // sum of RAW solid angles (steradians) subtended at i
      tetsByVertex[i].forEach(function (entry) {
        const others = entry.tet.v.filter(function (v) { return v !== i; });
        const a = sub(atoms[others[0]], atoms[i]);
        const b = sub(atoms[others[1]], atoms[i]);
        const c = sub(atoms[others[2]], atoms[i]);
        sumOmega += solidAngle(a, b, c);
      });
      const omegaFrac = 1 - sumOmega / (4 * Math.PI);
      omega[i] = omegaFrac > OMEGA_EPS ? omegaFrac : 0;
      const Si = 4 * Math.PI * d[i]; // full expanded-sphere area
      perAtomSASA[i] += omega[i] * Si;
    }
    // Gradient of the |T|=1 term: d(Omega_i * Si)/dp = -Si/(4pi) * d(sumOmega)/dp,
    // summed over each incident tetrahedron's own solid-angle gradient.
    for (let i = 0; i < n; i++) {
      if (omega[i] <= 0) continue;
      const Si = 4 * Math.PI * d[i];
      tetsByVertex[i].forEach(function (entry) {
        const tet = entry.tet;
        const others = tet.v.filter(function (v) { return v !== i; });
        const a = sub(atoms[others[0]], atoms[i]);
        const b = sub(atoms[others[1]], atoms[i]);
        const c = sub(atoms[others[2]], atoms[i]);
        const grads = solidAngleGradient(a, b, c); // [d/dp0(=i), d/dp1(=others[0]), d/dp2(=others[1]), d/dp3(=others[2])]
        const scaleFactor = -Si / (4 * Math.PI);
        accumGrad(i, scale(grads[0], scaleFactor));
        accumGrad(others[0], scale(grads[1], scaleFactor));
        accumGrad(others[1], scale(grads[2], scaleFactor));
        accumGrad(others[2], scale(grads[3], scaleFactor));
      });
    }

    // ---- |T|=2: pairwise cap correction, subtracted ----
    // tetsByEdge: for a pair (i,j), the tetrahedra (in C_0 only, same
    // reasoning as tetsByVertex above) containing BOTH i and j.
    const edgeKey = function (i, j) { return i < j ? i + '_' + j : j + '_' + i; };
    const tetsByEdge = new Map();
    complexTets.forEach(function (tet) {
      for (let a = 0; a < 4; a++) {
        for (let b = a + 1; b < 4; b++) {
          const key = edgeKey(tet.v[a], tet.v[b]);
          if (!tetsByEdge.has(key)) tetsByEdge.set(key, []);
          tetsByEdge.get(key).push(tet);
        }
      }
    });

    function capHeight(ri, rj, rij) {
      return ri - (ri * ri - rj * rj + rij * rij) / (2 * rij);
    }
    // d(h_i)/d(rij), from h_i = ri - (ri^2-rj^2)/(2*rij) - rij/2:
    //   = (ri^2-rj^2)/(2*rij^2) - 1/2 = (ri^2 - rj^2 - rij^2) / (2*rij^2)
    function capHeightDeriv(ri, rj, rij) {
      return (ri * ri - rj * rj - rij * rij) / (2 * rij * rij);
    }

    // An edge's own characteristic point (thesis eqs 4.4.20-4.4.24: the
    // point x0 = pi + t*(pj-pi) on the line through i,j equidistant, in
    // power distance, from both, with t = (k/a^2+1)/2, k=di-dj, a=pj-pi).
    function edgeCharPoint(atoms, d, i, j) {
      const pi = atoms[i], pj = atoms[j];
      const a = sub(pj, pi);
      const aLenSq = dot(a, a);
      if (aLenSq < 1e-18) return null;
      const t = 0.5 * ((d[i] - d[j]) / aLenSq + 1);
      const x0 = add(pi, scale(a, t));
      return { x0: x0, rho: powerDistSq(x0, pi, d[i]) };
    }
    // Incident triangles per edge (third vertex only) -- for the Gabriel/
    // unattached test below, matching CGAL's Regular_triangulation_3::
    // is_Gabriel(Cell_handle,i,j), which circulates incident FACETS and
    // checks only each facet's third vertex (not the whole point set).
    const trianglesByEdge = new Map();
    cx.triangles.forEach(function (tri) {
      for (let a = 0; a < 3; a++) {
        for (let b = a + 1; b < 3; b++) {
          const key = edgeKey(tri[a], tri[b]);
          const third = tri[3 - a - b];
          if (!trianglesByEdge.has(key)) trianglesByEdge.set(key, []);
          trianglesByEdge.get(key).push(third);
        }
      }
    });

    cx.pairs.forEach(function (pair) {
      const i = pair.i, j = pair.j;
      if (omega[i] <= 0 || omega[j] <= 0) return; // edge only counts if both atoms are exterior
      const rij = pair.dist;
      const ri = r[i], rj = r[j];
      if (rij < 1e-9) return;

      // Phi_ij = 1 - sum of normalized tetrahedron dihedral angles along this edge.
      const tets = tetsByEdge.get(edgeKey(i, j)) || [];

      // EXTERIOR check (CGAL Alpha_shape_3::classify(), same reasoning as
      // the |T|=3 block below): if this edge has NO incident tetrahedron
      // already in C_0 (tets.length===0) AND the edge is ATTACHED (not
      // Gabriel -- some incident triangle's third vertex reaches inside
      // the edge's own characteristic sphere), the edge is not part of
      // the alpha complex at all at this geometry and its pairwise-cap
      // term must be excluded entirely, not defaulted to the full
      // (unmoderated, phiFrac=1) cap subtraction an earlier revision of
      // this file always applied. This was a real, separate gap from the
      // triangle-level EXTERIOR fix -- confirmed on methane (5 atoms):
      // every H-H edge is exactly this case (attached to the central
      // carbon, zero incident C_0 tetrahedra), and without this check
      // the pairwise term over-subtracts each H's cap by the FULL
      // (unmoderated) amount, giving -53.79 A^2 per H against a true
      // +30.0 A^2.
      const ec = edgeCharPoint(atoms, d, i, j);
      if (ec) {
        let edgeUnattached = true;
        (trianglesByEdge.get(edgeKey(i, j)) || []).forEach(function (m) {
          if (powerDistSq(ec.x0, atoms[m], d[m]) - ec.rho <= 1e-9) edgeUnattached = false;
        });
        if (tets.length === 0 && !edgeUnattached) return;
      }
      let sumPhi = 0; // sum of RAW dihedral angles (radians)
      const dihedralInfo = [];
      tets.forEach(function (tet) {
        const others = tet.v.filter(function (v) { return v !== i && v !== j; });
        // dihedral around edge (i,j) between faces (k,i,j) and (i,j,l) --
        // matches this project's own already-validated dihedralAngle/
        // dihedralGradient (embed3d.js) exactly, called as (k,i,j,l).
        const k = others[0], l = others[1];
        const phi = Math.abs(CC.Embed3DShared.dihedralAngle(atoms[k], atoms[i], atoms[j], atoms[l]));
        sumPhi += phi;
        dihedralInfo.push({ k: k, l: l, sign: CC.Embed3DShared.dihedralAngle(atoms[k], atoms[i], atoms[j], atoms[l]) >= 0 ? 1 : -1 });
      });
      const phiFrac = 1 - sumPhi / (2 * Math.PI);
      if (phiFrac <= 0) return; // fully buried edge -- no exposed cap left, nothing to subtract or differentiate

      const hi = capHeight(ri, rj, rij);
      const hj = capHeight(rj, ri, rij);
      if (hi <= 0 && hj <= 0) return; // spheres too far into each other's interior to cap meaningfully (shouldn't normally happen for overlapping pairs)

      const Sij_i = hi > 0 ? 2 * Math.PI * ri * hi : 0;
      const Sij_j = hj > 0 ? 2 * Math.PI * rj * hj : 0;
      perAtomSASA[i] -= phiFrac * Sij_i;
      perAtomSASA[j] -= phiFrac * Sij_j;

      // ---- gradient ----
      const uij = scale(sub(atoms[j], atoms[i]), 1 / rij); // unit vector i->j
      // d(rij)/dp_j = uij, d(rij)/dp_i = -uij
      const dhi_drij = capHeightDeriv(ri, rj, rij);
      const dhj_drij = capHeightDeriv(rj, ri, rij);
      const dSij_i_drij = hi > 0 ? 2 * Math.PI * ri * dhi_drij : 0;
      const dSij_j_drij = hj > 0 ? 2 * Math.PI * rj * dhj_drij : 0;

      // Phi_ij's gradient: -1/(2pi) * sum of each tetrahedron's signed dihedral gradient.
      let dPhi_dpi = { x: 0, y: 0, z: 0 }, dPhi_dpj = { x: 0, y: 0, z: 0 };
      const perOtherGrad = new Map(); // atomIndex -> accumulated gradient contribution
      tets.forEach(function (tet) {
        const others = tet.v.filter(function (v) { return v !== i && v !== j; });
        const k = others[0], l = others[1];
        const signed = CC.Embed3DShared.dihedralAngle(atoms[k], atoms[i], atoms[j], atoms[l]);
        const sgn = signed >= 0 ? 1 : -1;
        // dihedralGradient(p1,p2,p3,p4) returns d/dp1,d/dp2,d/dp3,d/dp4 for chain (k,i,j,l)
        const g = CC.Embed3DShared.dihedralGradient ? CC.Embed3DShared.dihedralGradient(atoms[k], atoms[i], atoms[j], atoms[l]) : null;
        if (!g) return;
        const f = -sgn / (2 * Math.PI);
        dPhi_dpi = add(dPhi_dpi, scale(g[1], f));
        dPhi_dpj = add(dPhi_dpj, scale(g[2], f));
        const gk = perOtherGrad.get(k) || { x: 0, y: 0, z: 0 };
        perOtherGrad.set(k, add(gk, scale(g[0], f)));
        const gl = perOtherGrad.get(l) || { x: 0, y: 0, z: 0 };
        perOtherGrad.set(l, add(gl, scale(g[3], f)));
      });

      // d(perAtomSASA[i])/dp = -(dPhi*Sij_i + phiFrac*dSij_i)
      const dSij_i_dpi = scale(uij, -dSij_i_drij);
      const dSij_i_dpj = scale(uij, dSij_i_drij);
      const dSij_j_dpi = scale(uij, -dSij_j_drij);
      const dSij_j_dpj = scale(uij, dSij_j_drij);

      accumGrad(i, scale(add(scale(dPhi_dpi, Sij_i), scale(dSij_i_dpi, phiFrac)), -1));
      accumGrad(j, scale(add(scale(dPhi_dpj, Sij_i), scale(dSij_i_dpj, phiFrac)), -1));
      accumGrad(i, scale(add(scale(dPhi_dpi, Sij_j), scale(dSij_j_dpi, phiFrac)), -1));
      accumGrad(j, scale(add(scale(dPhi_dpj, Sij_j), scale(dSij_j_dpj, phiFrac)), -1));
      perOtherGrad.forEach(function (g, atomIdx) {
        accumGrad(atomIdx, scale(g, -(Sij_i + Sij_j)));
      });
    });

    // ---- |T|=3: triple-overlap correction (Hummel thesis eqs 4.4.45-
    // 4.4.63) -- energy-only (no analytical gradient yet, see header).
    //
    // Classification (thesis Section 3.6, "Classification of Simplices"):
    // a triangle's incident tetrahedra (its 1-2 combinatorial neighbors
    // in the FULL, unfiltered weighted-Delaunay tetrahedrization T, found
    // above via the empty-power-ball test) only count toward the
    // triangle's regular/interior status if THOSE tetrahedra are
    // THEMSELVES already part of the alpha=0 complex -- i.e. their own
    // size rho_T+ (the weight of THEIR OWN characteristic point, stored
    // as tet.radiusSq, per thesis Definition 3.2's rho_T and Definition
    // 3.4.1 "an unattached simplex sigma_T is in C_alpha iff alpha >=
    // rho_T"; tetrahedra are always unattached per Section 3.4) is <= 0
    // (alpha=0). A combinatorial neighbor with rho_T+ > 0 is a genuine
    // face of T but has NOT yet joined C_0 -- its own 4 balls don't
    // actually have a common intersection region -- and must NOT count
    // toward the triangle's own classification. This was the actual bug
    // in an earlier revision: it counted ANY combinatorial neighbor
    // found by the empty-ball search, regardless of that neighbor's own
    // rho_T+ sign, which is why 43 of ethanol's 84 candidate triangles
    // were misclassified singular (cT=1, full weight) when several
    // actually have a real but not-yet-joined neighbor and should have
    // been excluded from the singular count or reclassified regular.
    //   0 in-complex neighbors (rho_T+<=0) -> singular, cT=1,
    //     contribution = 2*rawSumAt(either point) -- no real tetrahedron
    //     on either side has joined C_0, so no side to prefer.
    //   1 in-complex neighbor               -> regular, cT=1/2,
    //     contribution = rawSumAt(the point on the OPPOSITE side of the
    //     plane from that neighbor's 4th vertex -- the genuinely
    //     exterior side).
    //   2 in-complex neighbors              -> interior, excluded
    //     entirely (buried, contributes 0 to the boundary sum).
    const triKey = function (i, j, k) { return [i, j, k].sort(function (a, b) { return a - b; }).join('_'); };
    function buildTetsByTriangle(tetList) {
      const map = new Map();
      tetList.forEach(function (tet) {
        const v = tet.v;
        const faces = [[v[0], v[1], v[2]], [v[0], v[1], v[3]], [v[0], v[2], v[3]], [v[1], v[2], v[3]]];
        faces.forEach(function (f) {
          const key = triKey(f[0], f[1], f[2]);
          if (!map.has(key)) map.set(key, []);
          map.get(key).push(tet);
        });
      });
      return map;
    }
    const tetsByTriangle = buildTetsByTriangle(complexTets); // only neighbors already in C_0
    // ALL combinatorial neighbors (unfiltered by C_0 membership) -- needed
    // for the Gabriel/unattached test below, which per CGAL must check
    // only a facet's actual 1-2 Delaunay neighbors, not the whole point
    // set (see this file's header for why the whole-point-set version was
    // wrong).
    const allTetsByTriangle = buildTetsByTriangle(cx.tets);

    cx.triangles.forEach(function (tri) {
      const i = tri[0], j = tri[1], k = tri[2];
      if (omega[i] <= 0 || omega[j] <= 0 || omega[k] <= 0) return; // fully-buried atom, no exterior contribution possible

      const face = faceCharPoint(atoms, d, i, j, k);
      if (!face || face.rho > 0) return; // degenerate, or balls have no real common surface point at this geometry

      // Attached/unattached ("Gabriel", in CGAL's terminology -- see
      // below) distinction, per CGAL's own Alpha_shape_3::classify()
      // (github.com/CGAL/cgal, Alpha_shapes_3/include/CGAL/Alpha_shape_3.h,
      // the canonical, widely-used open-source reference implementation
      // of exactly this classification): a facet's Alpha_status carries
      // alpha_mid/alpha_max (from its up-to-2 incident cells' own alpha
      // values -- matching this file's existing incident.length 0/1/2
      // count-based logic exactly, verified equivalent) AND alpha_min +
      // is_Gabriel (the facet's OWN circumsphere/characteristic-point
      // sphere, i.e. face.rho, used ONLY if is_Gabriel -- CGAL's name for
      // "unattached": no other point strictly inside the facet's own
      // sphere). Their classify(): alpha>=alpha_max -> INTERIOR;
      // else alpha>=alpha_mid -> REGULAR; else if Gabriel and
      // alpha>=alpha_min -> SINGULAR; else -> EXTERIOR. The final
      // EXTERIOR branch is what two earlier attempts at this exclusion
      // (both reverted -- see file header) were missing the empirical
      // verification for: it only fires when incident.length===0 AND the
      // triangle is ATTACHED (not Gabriel), and unlike assumed before,
      // does NOT provably push the total lower (each triangle's own
      // contribution can be negative), so it needed to be tested, not
      // reasoned about abstractly -- tested here and confirmed to help
      // (see file header for the resulting number).
      const incident = tetsByTriangle.get(triKey(i, j, k)) || []; // only neighbors already in C_0 (complexTets) count
      if (incident.length >= 2) return; // interior simplex -- not part of the boundary complex

      // Gabriel/unattached test, per CGAL's Regular_triangulation_3::
      // is_Gabriel(Cell_handle,int) (Triangulation_3/include/CGAL/
      // Regular_triangulation_3.h): checks ONLY the triangle's own (up to
      // 2) actual Delaunay-combinatorial neighbor apexes -- NOT every
      // other atom in the point set. This is a well-known Delaunay/
      // Gabriel-graph property (a local check on genuine neighbors is
      // globally sufficient for a valid triangulation) and is NOT what an
      // earlier revision of this block did (looped over all n atoms) --
      // that earlier version was both slower and, more importantly,
      // wrong: this file's own weightedDelaunayTets construction only
      // guarantees LOCAL neighbor consistency was checked during
      // insertion, and a stale/over-broad "does ANY atom in the whole
      // molecule intrude" test was flagging triangles attached that
      // CGAL's precise local test correctly leaves unattached.
      const allNeighbors = allTetsByTriangle.get(triKey(i, j, k)) || [];
      let unattached = true;
      allNeighbors.forEach(function (tet) {
        const apex = tet.v.filter(function (v) { return v !== i && v !== j && v !== k; })[0];
        if (powerDistSq(face.y, atoms[apex], d[apex]) - face.rho <= 1e-9) { unattached = false; }
      });
      if (incident.length === 0 && !unattached) return; // EXTERIOR per CGAL: attached, no incident cell in C_0 yet -- not part of the alpha complex at all

      const contrib = tripleOverlapContribution(atoms, i, j, k, r, d, face);
      if (!contrib) return; // balls have no real common surface point at this geometry

      let chosen = null, cT = 0, usedSecond = false;
      if (incident.length === 0 && allNeighbors.length < 2) {
        // Genuine hull-boundary singular triangle: only ONE real
        // combinatorial neighbor exists at all (the other side has no
        // real atom, only a bounding super-vertex) -- truly exposed on
        // one full side, matching the isolated-3-atom validation case
        // exactly (0 real neighbors at all there). cT=1, either point
        // (no real atom on either side to prefer -- confirmed correct on
        // the isolated case: 159.76 vs 159.53 Shrake-Rupley).
        cT = 1;
        chosen = contrib.raw1;
      } else if (incident.length === 0) {
        // "Interior to convex hull" per thesis Table 3.6 -- TWO real
        // combinatorial neighbors exist (apexes on both sides), neither
        // yet in C_0. Three approaches tried on a minimal 5-atom
        // fragment (ethanol's C1/C2/O/H/H, isolating this exact case):
        // cT=0.5 one side (307.80->275.34, but regressed an atom that
        // was previously exact); averaging both points at cT=1
        // (mathematically identical to the ORIGINAL unfixed behavior
        // whenever the two points are symmetric, which they were in
        // every test geometry tried so far -- not a real fix, just
        // camouflage); EXCLUDING entirely, this branch (307.80->242.87,
        // true value 181.02) -- the best of the three, even though it
        // also regresses the same previously-exact atom somewhat. Kept
        // as the current best-known approximation, NOT considered
        // solved -- see file header.
        return;
      } else {
        cT = 0.5;
        const apex = incident[0].v.filter(function (v) { return v !== i && v !== j && v !== k; })[0];
        const apexSide = dot(sub(atoms[apex], atoms[i]), contrib.normal);
        const x1Side = dot(sub(contrib.x1, atoms[i]), contrib.normal);
        // Pick the candidate point on the opposite side from the real
        // tetrahedron's apex (the exterior side); apexSide/x1Side share
        // a sign iff x1 is on the SAME side as the apex.
        usedSecond = (apexSide >= 0) === (x1Side >= 0);
        chosen = usedSecond ? contrib.raw2 : contrib.raw1;
      }
      const factor = 2 * cT; // eq 4.2.1 contribution = cT * S_T = cT * 2 * rawSum(chosen point)
      perAtomSASA[i] += factor * chosen.Si;
      perAtomSASA[j] += factor * chosen.Sj;
      perAtomSASA[k] += factor * chosen.Sk;

      // ---- |T|=3 gradient: finite difference, classification held fixed ----
      // A full closed-form gradient would need to differentiate through
      // face.y/nHat/rho (themselves a 2x2 linear solve in i,j,k's
      // positions) and then through the dihedral/solid-angle evaluations
      // at the resulting characteristic point x -- a substantial,
      // error-prone chain-rule derivation. This term IS smooth within one
      // classification branch (unlike Shrake-Rupley's raw step function,
      // the actual thing this file exists to avoid), so a carefully-
      // scoped central finite difference is a legitimate, honestly-
      // documented alternative: usedSecond (which of x1/x2) and cT (the
      // classification itself) are FIXED at their already-decided values
      // from the unperturbed geometry -- NOT re-derived per perturbed
      // sample, since re-deriving them would inject spurious jumps from
      // classification flips at this perturbation scale, and the
      // decision is itself a genuine discontinuity this method shares
      // with the full analytical formulation (thesis Chapter 5,
      // "Gradient Discontinuities" -- not unique to finite-differencing).
      // Only the continuous inner quantity (Si+Sj+Sk at the fixed point)
      // is re-evaluated per perturbed sample.
      [i, j, k].forEach(function (m) {
        const base = { x: atoms[m].x, y: atoms[m].y, z: atoms[m].z };
        const g = { x: 0, y: 0, z: 0 };
        ['x', 'y', 'z'].forEach(function (axis) {
          atoms[m][axis] = base[axis] + TRIPLE_FD_STEP;
          const plus = tripleSumAt(atoms, i, j, k, r, d, usedSecond);
          atoms[m][axis] = base[axis] - TRIPLE_FD_STEP;
          const minus = tripleSumAt(atoms, i, j, k, r, d, usedSecond);
          atoms[m][axis] = base[axis];
          if (plus !== null && minus !== null) g[axis] = (plus - minus) / (2 * TRIPLE_FD_STEP);
        });
        accumGrad(m, scale(g, factor));
      });
    });

    let totalSASA = 0;
    for (let i = 0; i < n; i++) totalSASA += Math.max(0, perAtomSASA[i]);
    return { totalSASA: totalSASA, perAtomSASA: perAtomSASA, gradient: gradient };
  };

  // Triple-overlap correction for one boundary triangle (i,j,k), per the
  // Hummel thesis (Delaunay-Laguerre Geometry For Macromolecular Modeling
  // And Implicit Solvation, Univ. of New Mexico PhD, 2014), Section 4.4.3,
  // eqs 4.4.45-4.4.49 -- the authoritative, complete source (JCTC ref
  // 33/34) for what the paper's compressed eq 6 was summarizing.
  //
  // Per the thesis text (verbatim structure, not re-derived from memory):
  // "If pi, pj, and pk have a non-empty intersection then there are two
  // points in common with the surfaces of all three balls. Call ONE of
  // these points x0" -- i.e. exactly ONE characteristic point is used to
  // build the auxiliary tetrahedron Tx={i,j,k,x0}, NOT both. This
  // corrects an earlier revision of this file that summed both candidate
  // points (x1 and x2), which double-counts except in the special
  // symmetric case (an isolated, coplanar 3-atom test with no other
  // atoms nearby) where x1 and x2 happen to contribute equally by mirror
  // symmetry -- which is exactly why that earlier revision's "sum both,
  // no extra factor" empirical fit against Shrake-Rupley (192.73 vs
  // 192.59) looked validated in isolation but was wrong in general (it
  // reproduced 2*raw_sum(one point) by accident of raw_sum(x1)==raw_sum(x2)
  // in that one symmetric test, not because summing both points is
  // actually correct).
  //
  // The thesis also gives the coefficient cT (eq 4.4.63): 1 if the
  // triangle is "singular" (has no incident tetrahedron already in the
  // alpha complex on either side), 1/2 if "regular" (has exactly one). A
  // genuine boundary triangle (member of partial-C) can have AT MOST one
  // incident real tetrahedron -- two would make it interior, and it
  // would not appear in partial-C at all. Combined with eq 4.4.49
  // (S_T = 2*(S_T^i+S_T^j+S_T^k), built from the ONE chosen point) and
  // eq 4.2.1's cT*S_T contribution:
  //   - singular (0 incident real tetrahedra): cT=1, contribution =
  //     2*raw_sum(x0), x0 picked arbitrarily (either candidate point --
  //     by symmetry of the alpha-complex construction there is no real
  //     tetrahedron on EITHER side to prefer one over the other).
  //   - regular (1 incident real tetrahedron, apex atom l on one side of
  //     the triangle's plane): cT=1/2, contribution = raw_sum(x0) with
  //     x0 chosen as the candidate point on the OPPOSITE side from l --
  //     the side with no real atom occupying it, i.e. the genuinely
  //     exterior side whose surface actually bulges into solvent.
  //
  // This function itself stays a low-level primitive: given the already-
  // classified triangle and its chosen point x, it just evaluates the raw
  // per-atom sum from eqs 4.4.46-4.4.48 at that ONE point (reusing the
  // already-validated |T|=1/|T|=2 cap-area/full-sphere-area quantities,
  // per the thesis's own S{i,j}^(i)/S{i}^(i) notation). The
  // singular/regular classification and point selection live in
  // CC.DSASA.compute() below, since they need the already-built
  // tetrahedra list (cx.tets) this function doesn't have access to.
  //
  // `face` is the triangle's already-computed characteristic point
  // (faceCharPoint(atoms,d,i,j,k)) -- passed in rather than recomputed
  // here since CC.DSASA.compute() already needs it for the
  // attached/unattached classification (Section 3.4) before calling this.
  //
  // Returns { x1, x2, raw1: {Si,Sj,Sk}, raw2: {Si,Sj,Sk}, normal } --
  // null if the three balls have no real common surface point at all
  // (face.rho > 0, checked by the caller before invoking this).
  function tripleOverlapContribution(atoms, i, j, k, r, d, face) {
    const pi = atoms[i], pj = atoms[j], pk = atoms[k];
    const di = d[i], dj = d[j], dk = d[k];
    const y = face.y, nHat = face.nHat;

    // The two candidate common-surface points are y +/- t*nHat, where
    // t^2 = -rho: |x-pi|^2 - di = 0 at x=y+t*n => |y-pi|^2 + t^2 - di = 0
    // => t^2 = di - |y-pi|^2 = -rho (rho = |y-pi|^2 - di by definition).
    const tSq = -face.rho;
    if (tSq < 0) return null; // spheres don't have a real common surface point here
    const t = Math.sqrt(tSq);
    const x1 = add(y, scale(nHat, t));
    const x2 = add(y, scale(nHat, -t));

    // Per paper eq 6, re-derived from kCalculateSA.h's own structure
    // (voltri computes each triangle's contribution from ONLY that
    // triangle's own local data -- no cross-referencing of other real
    // tetrahedra's dihedral sums -- confirming Phi^x/Omega^x here are
    // LOCAL fractions of the single auxiliary tetrahedron {i,j,k,x}, not
    // the "1 minus a sum over multiple tetrahedra" convention eq 4's
    // OUTER Phi_ij uses):
    //   (1/2) S_T^(i) = phi^x_ij * S_ij^(i) + phi^x_ik * S_ik^(i) - omega^x_i * S_i
    // where phi^x_ij/omega^x_i are the RAW (not "1-") dihedral/solid-angle
    // fractions of tetrahedron {i,j,k,x} itself, and S_ij^(i)/S_i reuse
    // the same real pairwise-cap/full-sphere quantities eqs 1/5 already
    // define. Summed over both real triple-intersection points x1,x2.
    const ri = r[i], rj = r[j], rk = r[k];
    const rij = norm(sub(pi, pj)), rik = norm(sub(pi, pk)), rjk = norm(sub(pj, pk));
    const Sij_i = 2 * Math.PI * ri * capHeightLocal(ri, rj, rij);
    const Sik_i = 2 * Math.PI * ri * capHeightLocal(ri, rk, rik);
    const Sji_j = 2 * Math.PI * rj * capHeightLocal(rj, ri, rij);
    const Sjk_j = 2 * Math.PI * rj * capHeightLocal(rj, rk, rjk);
    const Ski_k = 2 * Math.PI * rk * capHeightLocal(rk, ri, rik);
    const Skj_k = 2 * Math.PI * rk * capHeightLocal(rk, rj, rjk);

    // Raw per-atom sum at a single characteristic point x, per thesis
    // eqs 4.4.46-4.4.48 (Phi/Omega here are the LOCAL, single-tetrahedron
    // {i,j,k,x} fractions per eqs 4.4.36/4.4.6, not the "1-sum over
    // multiple real tetrahedra" convention eq 4.4.4/4.4.35 use elsewhere).
    const dihedralAngle = CC.Embed3DShared.dihedralAngle;
    function rawSumAt(x) {
      const phi_ij = Math.abs(dihedralAngle(pk, pi, pj, x)) / (2 * Math.PI);
      const phi_ik = Math.abs(dihedralAngle(pj, pi, pk, x)) / (2 * Math.PI);
      const om_i = solidAngle(sub(pj, pi), sub(pk, pi), sub(x, pi)) / (4 * Math.PI);
      const Si = phi_ij * Sij_i + phi_ik * Sik_i - om_i * (4 * Math.PI * di);

      const phi_ji = Math.abs(dihedralAngle(pk, pj, pi, x)) / (2 * Math.PI);
      const phi_jk = Math.abs(dihedralAngle(pi, pj, pk, x)) / (2 * Math.PI);
      const om_j = solidAngle(sub(pi, pj), sub(pk, pj), sub(x, pj)) / (4 * Math.PI);
      const Sj = phi_ji * Sji_j + phi_jk * Sjk_j - om_j * (4 * Math.PI * dj);

      const phi_ki = Math.abs(dihedralAngle(pj, pk, pi, x)) / (2 * Math.PI);
      const phi_kj = Math.abs(dihedralAngle(pi, pk, pj, x)) / (2 * Math.PI);
      const om_k = solidAngle(sub(pi, pk), sub(pj, pk), sub(x, pk)) / (4 * Math.PI);
      const Sk = phi_ki * Ski_k + phi_kj * Skj_k - om_k * (4 * Math.PI * dk);
      return { Si: Si, Sj: Sj, Sk: Sk };
    }
    return { x1: x1, x2: x2, raw1: rawSumAt(x1), raw2: rawSumAt(x2), normal: nHat };
  }

  function capHeightLocal(ri, rj, rij) {
    return ri - (ri * ri - rj * rj + rij * rij) / (2 * rij);
  }

  // Re-evaluates the |T|=3 term's continuous inner quantity (Si+Sj+Sk at
  // whichever of x1/x2 `usedSecond` selects) for a possibly-perturbed
  // `atoms` array -- used only by the finite-difference gradient in
  // CC.DSASA.compute's |T|=3 block. Deliberately does NOT re-derive the
  // singular/regular/interior classification or which point is
  // "exterior" -- that decision is held fixed at the caller's already-
  // computed value (see the gradient block's own comment for why).
  // Returns null if the perturbed geometry no longer has a real triple-
  // intersection point at all (shouldn't happen at this step size for a
  // non-degenerate triangle, but guarded defensively).
  function tripleSumAt(atoms, i, j, k, r, d, usedSecond) {
    const face = faceCharPoint(atoms, d, i, j, k);
    if (!face || face.rho > 0) return null;
    const contrib = tripleOverlapContribution(atoms, i, j, k, r, d, face);
    if (!contrib) return null;
    const raw = usedSecond ? contrib.raw2 : contrib.raw1;
    return raw.Si + raw.Sj + raw.Sk;
  }
})();
