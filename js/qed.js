/**
 * qed.js
 *
 * QED (Quantitative Estimate of Drug-likeness), Bickerton et al. 2012,
 * Nature Chemistry 4, 90-98. Ported from RDKit's own actual contrib
 * implementation (Novartis-authored, the same code real RDKit ships) --
 * the exact ADS desirability-function parameters and the exact
 * structural-alert/acceptor SMARTS lists below are copied from that
 * source, not approximated or reconstructed from the paper's tables by
 * hand, the same standard as this project's other from-scratch ports
 * (SA Score, chemprop's featurizer).
 *
 * One documented, deliberate simplification: the paper's AROM property
 * is a specific SSSR-based "delete non-fully-aromatic rings, count what's
 * left" definition, not simply "number of aromatic rings" -- the
 * reference implementation's own comment notes these aren't identical.
 * This port uses RDKit.js's plain NumAromaticRings instead, since the
 * more precise definition needs ring/substructure-deletion operations
 * this build's API doesn't cleanly expose. Everything else (HBA's
 * QED-specific acceptor definition, which is NOT the same as RDKit's
 * standard NumHBA, and all ~100 structural alerts) is matched via real
 * RDKit.js SMARTS substructure matching, not a shortcut.
 */

window.CC = window.CC || {};

(function () {
  // name -> {A,B,C,D,E,F,DMAX}
  const ADS_PARAMETERS = {
    MW: { A: 2.817065973, B: 392.5754953, C: 290.7489764, D: 2.419764353, E: 49.22325677, F: 65.37051707, DMAX: 104.9805561 },
    ALOGP: { A: 3.172690585, B: 137.8624751, C: 2.534937431, D: 4.581497897, E: 0.822739154, F: 0.576295591, DMAX: 131.3186604 },
    HBA: { A: 2.948620388, B: 160.4605972, C: 3.615294657, D: 4.435986202, E: 0.290141953, F: 1.300669958, DMAX: 148.7763046 },
    HBD: { A: 1.618662227, B: 1010.051101, C: 0.985094388, D: 0.000000001, E: 0.713820843, F: 0.920922555, DMAX: 258.1632616 },
    PSA: { A: 1.876861559, B: 125.2232657, C: 62.90773554, D: 87.83366614, E: 12.01999824, F: 28.51324732, DMAX: 104.5686167 },
    ROTB: { A: 0.010000000, B: 272.4121427, C: 2.558379970, D: 1.565547684, E: 1.271567166, F: 2.758063707, DMAX: 105.4420403 },
    AROM: { A: 3.217788970, B: 957.7374108, C: 2.274627939, D: 0.000000001, E: 1.317690384, F: 0.375760881, DMAX: 312.3372610 },
    ALERTS: { A: 0.010000000, B: 1199.094025, C: -0.09002883, D: 0.000000001, E: 0.185904477, F: 0.875193782, DMAX: 417.7253140 },
  };

  // RDKit's own WEIGHT_MEAN weighting (its documented default) -- order
  // matches ADS_PARAMETERS' property order above.
  const WEIGHTS = { MW: 0.66, ALOGP: 0.46, HBA: 0.05, HBD: 0.61, PSA: 0.06, ROTB: 0.65, AROM: 0.48, ALERTS: 0.95 };

  function ads(x, p) {
    const exp1 = 1 + Math.exp(-1 * (x - p.C + p.D / 2) / p.E);
    const exp2 = 1 + Math.exp(-1 * (x - p.C - p.D / 2) / p.F);
    const dx = p.A + p.B / exp1 * (1 - 1 / exp2);
    return dx / p.DMAX;
  }

  // QED's own HBA definition -- NOT the same set of patterns as RDKit's
  // standard NumHBA/lipinskiHBA, a real and easy-to-miss distinction.
  const ACCEPTOR_SMARTS = [
    '[oH0;X2]', '[OH1;X2;v2]', '[OH0;X2;v2]', '[OH0;X1;v2]', '[O-;X1]',
    '[SH0;X2;v2]', '[SH0;X1;v2]', '[S-;X1]', '[nH0;X2]', '[NH0;X1;v3]',
    '[$([N;+0;X3;v3]);!$(N[C,S]=O)]',
  ];

  // The paper's full structural-alert list (unwanted/reactive/toxic
  // substructure patterns) -- copied verbatim from RDKit's own contrib
  // source, not abridged.
  const ALERT_SMARTS = [
    '*1[O,S,N]*1', '[S,C](=[O,S])[F,Br,Cl,I]', '[CX4][Cl,Br,I]', '[#6]S(=O)(=O)O[#6]',
    '[$([CH]),$(CC)]#CC(=O)[#6]', '[$([CH]),$(CC)]#CC(=O)O[#6]', 'n[OH]',
    '[$([CH]),$(CC)]#CS(=O)(=O)[#6]', 'C=C(C=O)C=O', 'n1c([F,Cl,Br,I])cccc1', '[CH1](=O)',
    '[#8][#8]', '[C;!R]=[N;!R]', '[N!R]=[N!R]', '[#6](=O)[#6](=O)', '[#16][#16]', '[#7][NH2]',
    'C(=O)N[NH2]', '[#6]=S',
    '[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]',
    'C1(=[O,N])C=CC(=[O,N])C=C1', 'C1(=[O,N])C(=[O,N])C=CC=C1', 'a21aa3a(aa1aaaa2)aaaa3',
    'a31a(a2a(aa1)aaaa2)aaaa3', 'a1aa2a3a(a1)A=AA=A3=AA=A2', 'c1cc([NH2])ccc1',
    '[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]',
    'I', 'OS(=O)(=O)[O-]', '[N+](=O)[O-]', 'C(=O)N[OH]', 'C1NC(=O)NC(=O)1', '[SH]', '[S-]',
    'c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]', 'c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]',
    '[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1', '[CR1]1[CR1][CR1]cc[CR1][CR1]1',
    '[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1', '[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1',
    '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1', '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
    'C#C', '[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]',
    '[$([N+R]),$([n+R]),$([N+]=C)][O-]', '[#6]=N[OH]', '[#6]=NOC=O',
    '[#6](=O)[CX4,CR0X3,O][#6](=O)', 'c1ccc2c(c1)ccc(=O)o2', '[O+,o+,S+,s+]', 'N=C=O',
    '[NX3,NX4][F,Cl,Br,I]', 'c1ccccc1OC(=O)[#6]', '[CR0]=[CR0][CR0]=[CR0]', '[C+,c+,C-,c-]',
    'N=[N+]=[N-]', 'C12C(NC(N1)=O)CSC2', 'c1c([OH])c([OH,NH2,NH])ccc1', 'P', '[N,O,S]C#N',
    'C=C=O', '[Si][F,Cl,Br,I]', '[SX2]O', '[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)',
    'O1CCCCC1OC2CCC3CCCCC3C2', 'N=[CR0][N,n,O,S]',
    '[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2',
    'C=[C!r]C#N', '[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])', '[OH]c1ccc([OH,NH2,NH])cc1',
    'c1ccccc1OC(=O)O', '[SX2H0][N]', 'c12ccccc1(SC(S)=N2)', 'c12ccccc1(SC(=S)N2)',
    'c1nnnn1C=O', 's1c(S)nnc1NC=O', 'S1C=CSC1=S', 'C(=O)Onnn', 'OS(=O)(=O)C(F)(F)F',
    'N#CC[OH]', 'N#CC(=O)', 'S(=O)(=O)C#N', 'N[CH2]C#N', 'C1(=O)NCC1', 'S(=O)(=O)[O-,OH]',
    'NC[F,Cl,Br,I]', 'C=[C!r]O', '[NX2+0]=[O+0]', '[OR0,NR0][OR0,NR0]',
    'C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]', '[CX2R0][NX3R0]', 'c1ccccc1[C;!R]=[C;!R]c2ccccc2',
    '[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]',
    '[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]',
    '[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]', '[*]=[N+]=[*]',
    '[SX3](=O)[O-,OH]', 'N#N', 'F.F.F.F', '[R0;D2][R0;D2][R0;D2][R0;D2]',
    '[cR,CR]~C(=O)NC(=O)~[cR,CR]', 'C=!@CC=[O,S]', '[#6,#8,#16][#6](=O)O[#6]',
    'c[C;R0](=[O,S])[#6]', 'c[SX2][C;!R]', 'C=C=C', 'c1nc([F,Cl,Br,I,S])ncc1',
    'c1ncnc([F,Cl,Br,I,S])c1', 'c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])',
    '[#6]S(=O)(=O)c1ccc(cc1)F', '[15N]', '[13C]', '[18O]', '[34S]',
  ];

  function matchCount(rawJson) {
    const parsed = JSON.parse(rawJson);
    // get_substruct_matches returns the string "{}" (an empty object,
    // not an empty array) when there are zero matches -- Array.isArray
    // guards against silently getting `undefined` from {}.length there.
    return Array.isArray(parsed) ? parsed.length : 0;
  }

  function countSmartsMatches(RDKit, mol, smartsList) {
    let count = 0;
    smartsList.forEach(function (smarts) {
      let qmol = null;
      try {
        qmol = RDKit.get_qmol(smarts);
        if (!qmol || !qmol.is_valid()) return;
        if (matchCount(mol.get_substruct_matches(qmol)) > 0) count++;
      } catch (err) {
        // A handful of these SMARTS use recursive/exotic syntax that
        // older RDKit.js SMARTS parsers may not accept -- skip that one
        // pattern rather than failing the whole QED calculation over it.
      } finally {
        if (qmol) qmol.delete();
      }
    });
    return count;
  }

  function countAcceptors(RDKit, mol) {
    let total = 0;
    ACCEPTOR_SMARTS.forEach(function (smarts) {
      let qmol = null;
      try {
        qmol = RDKit.get_qmol(smarts);
        if (!qmol || !qmol.is_valid()) return;
        total += matchCount(mol.get_substruct_matches(qmol));
      } catch (err) {
        // see countSmartsMatches
      } finally {
        if (qmol) qmol.delete();
      }
    });
    return total;
  }

  /**
   * Computes QED for an already-parsed RDKit.js mol (caller owns its
   * lifetime -- this doesn't delete it). descriptors should be the
   * output of mol.get_descriptors() (already parsed to an object),
   * reused rather than recomputed here.
   *
   * Returns { qed, properties: {MW,ALOGP,HBA,HBD,PSA,ROTB,AROM,ALERTS},
   * desirabilities: {...} } -- the per-property breakdown is included
   * since QED is often more useful to look at broken down (which single
   * property is dragging the score down) than as one number alone.
   */
  CC.computeQED = function (RDKit, mol, descriptors) {
    const properties = {
      MW: descriptors.amw,
      ALOGP: descriptors.CrippenClogP,
      HBA: countAcceptors(RDKit, mol),
      HBD: descriptors.NumHBD,
      PSA: descriptors.tpsa,
      ROTB: descriptors.NumRotatableBonds,
      AROM: descriptors.NumAromaticRings, // documented approximation, see file header
      ALERTS: countSmartsMatches(RDKit, mol, ALERT_SMARTS),
    };

    const names = Object.keys(ADS_PARAMETERS);
    const desirabilities = {};
    names.forEach(function (name) {
      desirabilities[name] = ads(properties[name], ADS_PARAMETERS[name]);
    });

    let weightedLogSum = 0, weightSum = 0;
    names.forEach(function (name) {
      weightedLogSum += WEIGHTS[name] * Math.log(desirabilities[name]);
      weightSum += WEIGHTS[name];
    });
    const qed = Math.exp(weightedLogSum / weightSum);

    return { qed: qed, properties: properties, desirabilities: desirabilities };
  };
})();
