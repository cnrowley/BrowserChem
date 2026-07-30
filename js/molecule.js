/**
 * molecule.js
 *
 * The core molecular graph: atoms and bonds, with no rendering or DOM
 * knowledge. Chemistry validation (valence, aromaticity, etc.) is NOT
 * here yet — this is deliberately just a graph container. That logic
 * arrives with the RDKit.js integration stage.
 */

window.CC = window.CC || {};

CC.Molecule = class Molecule {
  constructor() {
    this.atoms = new Map();
    this.bonds = new Map();
    this._nextAtomId = 1;
    this._nextBondId = 1;
  }

  addAtom(element, x, y, opts) {
    opts = opts || {};
    const id = 'a' + this._nextAtomId++;
    const atom = {
      id: id,
      element: element,
      x: x,
      y: y,
      charge: opts.charge || 0,
    };
    this.atoms.set(id, atom);
    return atom;
  }

  removeAtom(atomId) {
    this.atoms.delete(atomId);
    for (const bond of Array.from(this.bonds.values())) {
      if (bond.a1 === atomId || bond.a2 === atomId) {
        this.bonds.delete(bond.id);
      }
    }
  }

  addBond(a1, a2, order, opts) {
    if (a1 === a2) return null;
    const existing = this.getBondBetween(a1, a2);
    if (existing) return existing;
    opts = opts || {};
    const id = 'b' + this._nextBondId++;
    // stereo: 'none' | 'wedge' | 'hash'. Only meaningful on single bonds —
    // wedge/hash is how 2D drawings show a tetrahedral stereocenter, and
    // by MDL molfile convention the narrow end sits at a1 (the
    // stereocenter) and the wide end at a2 (the substituent going up/down
    // out of the plane).
    const bond = { id: id, a1: a1, a2: a2, order: order || 1, stereo: opts.stereo || 'none' };
    this.bonds.set(id, bond);
    return bond;
  }

  setBondStereo(bondId, stereo) {
    const bond = this.bonds.get(bondId);
    if (!bond) return;
    bond.stereo = stereo;
    if (stereo === 'wedge' || stereo === 'hash') bond.order = 1; // chemistry constraint
  }

  removeBond(bondId) {
    this.bonds.delete(bondId);
  }

  getBondBetween(a1, a2) {
    for (const bond of this.bonds.values()) {
      if ((bond.a1 === a1 && bond.a2 === a2) || (bond.a1 === a2 && bond.a2 === a1)) {
        return bond;
      }
    }
    return null;
  }

  getBondsForAtom(atomId) {
    const result = [];
    for (const bond of this.bonds.values()) {
      if (bond.a1 === atomId || bond.a2 === atomId) result.push(bond);
    }
    return result;
  }

  getDegree(atomId) {
    return this.getBondsForAtom(atomId).length;
  }

  isEmpty() {
    return this.atoms.size === 0;
  }

  toJSON() {
    return {
      atoms: Array.from(this.atoms.values()).map(function (a) { return Object.assign({}, a); }),
      bonds: Array.from(this.bonds.values()).map(function (b) { return Object.assign({}, b); }),
      nextAtomId: this._nextAtomId,
      nextBondId: this._nextBondId,
    };
  }

  static fromJSON(data) {
    const mol = new CC.Molecule();
    for (const a of data.atoms) mol.atoms.set(a.id, Object.assign({}, a));
    for (const b of data.bonds) mol.bonds.set(b.id, Object.assign({}, b));
    mol._nextAtomId = data.nextAtomId;
    mol._nextBondId = data.nextBondId;
    return mol;
  }
};
