/**
 * chemistry.js
 *
 * Thin wrapper around RDKit.js for: sanitization/valence check, canonical
 * SMILES, and a curated set of 2D molecular descriptors (no 3D needed —
 * these come straight off the molecular graph). Nothing here mutates the
 * molecule — it's a read-only analysis step run after each edit.
 *
 * RDKit.js JSMol objects hold WASM memory and must be freed manually with
 * .delete() — every path through this function does that before returning.
 */

window.CC = window.CC || {};

// Curated subset of RDKit's ~40-field descriptor JSON (see
// mol.get_descriptors()) — the common "drug-likeness" style properties,
// with display labels/units/precision. Field names below are RDKit's own,
// confirmed against several real get_descriptors() outputs.
CC.DESCRIPTOR_FIELDS = [
  { key: 'amw', label: 'Molecular weight', unit: 'g/mol', decimals: 2 },
  { key: 'CrippenClogP', label: 'LogP (Crippen)', unit: '', decimals: 2 },
  { key: 'tpsa', label: 'Topological PSA', unit: '\u00c5\u00b2', decimals: 1 },
  { key: 'NumHBD', label: 'H-bond donors', unit: '', decimals: 0 },
  { key: 'NumHBA', label: 'H-bond acceptors', unit: '', decimals: 0 },
  { key: 'NumRotatableBonds', label: 'Rotatable bonds', unit: '', decimals: 0 },
  { key: 'NumRings', label: 'Rings', unit: '', decimals: 0 },
  { key: 'NumAromaticRings', label: 'Aromatic rings', unit: '', decimals: 0 },
  { key: 'FractionCSP3', label: 'Fraction Csp3', unit: '', decimals: 2 },
  { key: 'NumAtomStereoCenters', label: 'Stereocenters (defined)', unit: '', decimals: 0 },
  { key: 'NumUnspecifiedAtomStereoCenters', label: 'Stereocenters (unspecified)', unit: '', decimals: 0 },
  { key: 'saScore', label: 'Synthetic accessibility (1=easy, 10=hard)', unit: '', decimals: 2 },
  { key: 'qed', label: 'QED (drug-likeness, 0-1)', unit: '', decimals: 3 },
];

CC.analyzeMolblock = function (molblock) {
  const RDKit = window.chemCanvasLibs && window.chemCanvasLibs.RDKit;
  if (!RDKit) {
    return { available: false };
  }

  let mol = null;
  try {
    mol = RDKit.get_mol(molblock);
  } catch (err) {
    return {
      available: true,
      valid: false,
      message: 'RDKit could not parse this structure',
    };
  }

  if (!mol || !mol.is_valid()) {
    if (mol) mol.delete();
    return {
      available: true,
      valid: false,
      message: 'Structure failed sanitization — check atom valence',
    };
  }

  let smiles = '';
  let descriptors = null;
  try {
    smiles = mol.get_smiles();
  } catch (err) {
    smiles = '';
  }
  try {
    descriptors = JSON.parse(mol.get_descriptors());
  } catch (err) {
    descriptors = null;
  }
  if (descriptors && CC.computeQED) {
    try {
      descriptors.qed = CC.computeQED(RDKit, mol, descriptors).qed;
    } catch (err) {
      // QED runs ~100 SMARTS substructure matches -- if one of RDKit.js's
      // SMARTS parses ever throws unexpectedly, don't let that take down
      // the rest of the (already-working) descriptor table with it.
      descriptors.qed = null;
    }
  }
  if (descriptors) {
    try {
      const molData = JSON.parse(mol.get_json()).molecules[0];
      const ext = (molData.extensions || []).find(function (e) { return e.name === 'rdkitRepresentation'; });
      // RDKit's JSON export omits the aromaticAtoms key entirely (not an
      // empty array) when a molecule has zero aromatic atoms -- a real
      // bug here previously: `ext.aromaticAtoms ? ... : null` treated
      // "key genuinely absent because there are none" the same as "this
      // failed," giving null (shown as "no data") for a molecule that
      // was correctly, successfully computed as 0% aromatic. Those are
      // different things -- only missing `ext` itself (RDKit truly
      // couldn't answer) should give null now.
      descriptors.aromaticAtomCount = ext ? (ext.aromaticAtoms ? ext.aromaticAtoms.length : 0) : null;
    } catch (err) {
      descriptors.aromaticAtomCount = null;
    }
  }

  mol.delete();

  return {
    available: true,
    valid: true,
    smiles: smiles,
    descriptors: descriptors,
  };
};
