/**
 * tools.js
 *
 * The interaction layer: translates pointer events on the SVG canvas into
 * molecule edits, according to whichever tool is currently active. Nothing
 * in here talks to RDKit or does chemistry validation — it just builds and
 * edits the graph. Every completed action calls commit(), which snapshots
 * the molecule into the undo/redo history.
 */

window.CC = window.CC || {};

CC.Controller = class Controller {
  constructor(svg, molecule, history, callbacks) {
    this.svg = svg;
    this.molecule = molecule;
    this.history = history;
    this.selection = new Set();
    this.currentTool = 'bond';
    this.currentElement = 'C';
    this.currentBondStyle = 'plain'; // 'plain' | 'wedge' | 'hash'
    this.currentRingSize = 6;
    this.currentRingAromatic = false;
    this.drag = null;
    this.hoveredAtomId = null;
    this.hoveredBondId = null; // set on pointermove, only when no atom is being hovered -- see _onPointerMove

    callbacks = callbacks || {};
    this.onMoleculeChanged = callbacks.onMoleculeChanged || function () {};
    this.onSelectionChanged = callbacks.onSelectionChanged || function () {};
    // Both keyboard-only actions -- neither touches the molecule/RDKit
    // directly from here, per this file's own "no chemistry validation,
    // no RDKit access" boundary (see header comment). Tool-switching just
    // needs to also update the toolbar's own UI state, and cleanup needs
    // RDKit's get_new_coords(), both of which live in app.js.
    this.onToolShortcut = callbacks.onToolShortcut || function () {};
    this.onCleanupShortcut = callbacks.onCleanupShortcut || function () {};
    // Fired on every pointermove (and cleared on pointerleave) with either
    // null or {x, y, r} in molecule-space coords -- lets app.js position
    // the hover-highlight circle without this file touching the DOM
    // directly, matching this file's "no rendering" boundary.
    this.onHoverChanged = callbacks.onHoverChanged || function () {};

    this.svg.addEventListener('pointerdown', this._onPointerDown.bind(this));
    this.svg.addEventListener('pointermove', this._onPointerMove.bind(this));
    this.svg.addEventListener('pointerup', this._onPointerUp.bind(this));
    this.svg.addEventListener('pointerleave', this._onPointerLeave.bind(this));
    document.addEventListener('keydown', this._onKeyDown.bind(this));
  }

  setTool(tool) {
    this.currentTool = tool;
    this.selection.clear();
    this.onSelectionChanged();
  }

  setElement(element) {
    this.currentElement = element;
  }

  setBondStyle(style) {
    this.currentBondStyle = style;
  }

  setRingSize(n) {
    this.currentRingSize = n;
  }

  setRingAromatic(aromatic) {
    this.currentRingAromatic = aromatic;
  }

  setMolecule(molecule) {
    this.molecule = molecule;
    this.selection.clear();
    this.onSelectionChanged();
  }

  deleteSelection() {
    if (this.selection.size === 0) return false;
    for (const atomId of this.selection) {
      this.molecule.removeAtom(atomId);
    }
    this.selection.clear();
    this._commit();
    return true;
  }

  // ---------- internals ----------

  _point(e) {
    return CC.svgPoint(this.svg, e.clientX, e.clientY);
  }

  _hitAtom(e) {
    const target = e.target.closest('[data-atom-id]');
    return target ? target.dataset.atomId : null;
  }

  _hitBond(e) {
    const target = e.target.closest('[data-bond-id]');
    return target ? target.dataset.bondId : null;
  }

  _commit() {
    this.history.commit(this.molecule.toJSON());
    this.onMoleculeChanged();
  }

  _onPointerDown(e) {
    this.svg.setPointerCapture(e.pointerId);
    const atomId = this._hitAtom(e);
    const bondId = !atomId ? this._hitBond(e) : null;
    const p = this._point(e);

    switch (this.currentTool) {
      case 'select':
        return this._selectPointerDown(atomId, p);
      case 'atom':
        return this._atomPointerDown(atomId, p);
      case 'bond':
        if (!atomId && bondId) return this._bondClickPointerDown(bondId);
        return this._bondOrChainPointerDown(atomId, p);
      case 'chain':
        return this._bondOrChainPointerDown(atomId, p);
      case 'ring':
        return this._ringPointerDown(atomId, p);
      case 'charge':
        return this._chargePointerDown(atomId, e.shiftKey);
      case 'erase':
        return this._erasePointerDown(atomId, bondId);
    }
  }

  /**
   * Clicking directly on an existing bond with the Bond tool active — no
   * drag needed. With the "Plain" style selected this cycles bond order
   * (single -> double -> triple -> single), matching the drag-to-cycle
   * behavior. With "Wedge"/"Hash" selected, it sets that bond's stereo
   * (clicking the same style again clears it back to plain).
   */
  _bondClickPointerDown(bondId) {
    const bond = this.molecule.bonds.get(bondId);
    if (!bond) return;

    if (this.currentBondStyle === 'plain') {
      bond.order = bond.order >= 3 ? 1 : bond.order + 1;
      bond.stereo = 'none'; // cycling order on a stereo bond clears it — order>1 can't carry wedge/hash
    } else if (bond.stereo === this.currentBondStyle) {
      this.molecule.setBondStereo(bondId, 'none');
    } else {
      this.molecule.setBondStereo(bondId, this.currentBondStyle);
    }
    this._commit();
  }

  _onPointerMove(e) {
    const p = this._point(e);
    this.hoveredAtomId = CC.nearestAtomWithinRadius(this.molecule, p.x, p.y, CC.HOVER_RADIUS);
    // Same precedence _hoverTarget() below already uses for the visual
    // highlight: a bond is only "hovered" for keyboard-shortcut purposes
    // when no atom is -- the two are mutually exclusive, matching how an
    // atom's own hover-highlight circle always wins over a bond under it.
    this.hoveredBondId = this.hoveredAtomId ? null : this._hitBond(e);

    if (this.drag) {
      if (this.drag.type === 'move-atom') {
        const atom = this.molecule.atoms.get(this.drag.atomId);
        if (atom) {
          atom.x = p.x;
          atom.y = p.y;
          this.onMoleculeChanged();
        }
      } else if (this.drag.type === 'bond') {
        const startAtom = this.molecule.atoms.get(this.drag.startAtomId);
        if (startAtom) {
          const hoverAtomId = this._hitAtom(e);
          const end = (hoverAtomId && hoverAtomId !== this.drag.startAtomId)
            ? this.molecule.atoms.get(hoverAtomId)
            : CC.snapToAngle(startAtom, p);
          this.drag.previewEnd = end;
          this.drag.hoverAtomId = (hoverAtomId && hoverAtomId !== this.drag.startAtomId) ? hoverAtomId : null;
          this.onMoleculeChanged({ preview: this.drag });
        }
      } else if (this.drag.type === 'chain') {
        this._updateChainPreview(p);
        this.onMoleculeChanged({ preview: this.drag });
      }
    }

    // Published after any drag-triggered render above, so a fresh render
    // doesn't wipe the highlight circle out from under it.
    this.onHoverChanged(this._hoverTarget(e));
  }

  // What the hover-highlight circle should show right now: the nearest
  // atom within snap range (same radius the chain tool uses to close
  // rings, so what's highlighted is exactly what dragging onto it would
  // connect to), else a bond under the pointer highlighted at its
  // midpoint, else nothing.
  _hoverTarget(e) {
    if (this.hoveredAtomId) {
      const atom = this.molecule.atoms.get(this.hoveredAtomId);
      if (atom) return { x: atom.x, y: atom.y, r: CC.HOVER_RADIUS };
    }
    const bondId = this._hitBond(e);
    if (bondId) {
      const bond = this.molecule.bonds.get(bondId);
      const a1 = bond && this.molecule.atoms.get(bond.a1);
      const a2 = bond && this.molecule.atoms.get(bond.a2);
      if (a1 && a2) return { x: (a1.x + a2.x) / 2, y: (a1.y + a2.y) / 2, r: 8 };
    }
    return null;
  }

  _onPointerUp() {
    if (!this.drag) return;

    if (this.drag.type === 'move-atom') {
      this._commit();
    } else if (this.drag.type === 'bond') {
      this._finalizeBond();
      this._commit();
    } else if (this.drag.type === 'chain') {
      this._finalizeChain();
      this._commit();
    }

    this.drag = null;
    this.onMoleculeChanged();
  }

  _onPointerLeave() {
    this.hoveredAtomId = null;
    this.hoveredBondId = null;
    this.onHoverChanged(null);
  }

  // Keyboard interface for the whole canvas, ChemDraw-style single-key
  // shortcuts layered on top of the existing tool/cleanup shortcuts below
  // rather than replacing them -- each new binding only fires while
  // hovering the specific thing it applies to (an atom for element keys,
  // a bond for order/stereo keys), same gating principle the original
  // C/N/O/S/F element shortcuts already used, so 1-7/L keep their
  // existing meaning everywhere else on the canvas:
  //   - hover a BOND, press 1/2/3 -> set that bond's order directly
  //     (single/double/triple); Z is an alias for 3 (ChemDraw's own
  //     alkyne/triple shortcut)
  //   - hover a BOND, press 4/W -> set wedge stereo; 5/H -> set hash
  //     stereo (via molecule.setBondStereo, same setter the bond-style
  //     click-to-cycle interaction uses)
  //   - hover an ATOM, type a letter -> change that atom's element
  //     directly: C/N/O/S/F/P/I/H for their own symbol, L for Cl and B
  //     for Br (ChemDraw's own mapping -- neither halogen gets its own
  //     first letter since C/B are already Carbon/Boron's keys)
  //   - 1-7 (no hover needed) -> switch tool, matching the tool rail's
  //     own left-to-right order (Select, Atom, Bond, Chain, Ring,
  //     Charge, Erase)
  //   - L (no atom hovered) -> clean up: redraw the whole structure with
  //     standard bond lengths/angles (via RDKit) and recenter it
  // All gated the same way: never when a modifier key is held (so this
  // doesn't fight Ctrl+Z etc.), and never while a text input elsewhere on
  // the page has focus (the SMILES box, etc.) just because the mouse
  // also happens to be resting over the canvas.
  _onKeyDown(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    const target = e.target;
    const tag = target && target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || (target && target.isContentEditable)) return;

    const key = e.key.toUpperCase();

    if (this.hoveredBondId) {
      const bond = this.molecule.bonds.get(this.hoveredBondId);
      if (bond) {
        const BOND_ORDER_KEYS = { 1: 1, 2: 2, 3: 3, Z: 3 };
        if (key in BOND_ORDER_KEYS) {
          e.preventDefault();
          bond.order = BOND_ORDER_KEYS[key];
          bond.stereo = 'none'; // order>1 can't carry wedge/hash -- same rule the click-to-cycle interaction already applies
          this._commit();
          this.onMoleculeChanged();
          return;
        }
        const BOND_STEREO_KEYS = { 4: 'wedge', W: 'wedge', 5: 'hash', H: 'hash' };
        if (key in BOND_STEREO_KEYS) {
          e.preventDefault();
          this.molecule.setBondStereo(this.hoveredBondId, BOND_STEREO_KEYS[key]);
          this._commit();
          this.onMoleculeChanged();
          return;
        }
      }
    }

    const TOOL_NUMBER_KEYS = { '1': 'select', '2': 'atom', '3': 'bond', '4': 'chain', '5': 'ring', '6': 'charge', '7': 'erase' };
    if (e.key in TOOL_NUMBER_KEYS) {
      e.preventDefault();
      this.onToolShortcut(TOOL_NUMBER_KEYS[e.key]);
      return;
    }

    if (key === 'L' && !this.hoveredAtomId) {
      e.preventDefault();
      this.onCleanupShortcut();
      return;
    }

    if (!this.hoveredAtomId) return;

    const ELEMENT_KEYS = { C: 'C', N: 'N', O: 'O', S: 'S', F: 'F', P: 'P', I: 'I', H: 'H', L: 'Cl', B: 'Br' };
    if (!(key in ELEMENT_KEYS)) return;

    const atom = this.molecule.atoms.get(this.hoveredAtomId);
    if (!atom) return;

    e.preventDefault();
    atom.element = ELEMENT_KEYS[key];
    this._commit();
    this.onMoleculeChanged();
  }

  // ---------- select tool ----------

  _selectPointerDown(atomId, p) {
    if (atomId) {
      this.selection = new Set([atomId]);
      this.onSelectionChanged();
      this.drag = { type: 'move-atom', atomId: atomId };
    } else {
      this.selection.clear();
      this.onSelectionChanged();
    }
    this.onMoleculeChanged();
    void p;
  }

  // ---------- atom tool ----------

  _atomPointerDown(atomId, p) {
    if (atomId) {
      const atom = this.molecule.atoms.get(atomId);
      atom.element = this.currentElement;
    } else {
      this.molecule.addAtom(this.currentElement, p.x, p.y);
    }
    this._commit();
  }

  // ---------- bond / chain tools ----------

  _bondOrChainPointerDown(atomId, p) {
    let startAtomId = atomId;
    if (!startAtomId) {
      const atom = this.molecule.addAtom('C', p.x, p.y);
      startAtomId = atom.id;
    }
    this.drag = { type: this.currentTool, startAtomId: startAtomId };
    this.onMoleculeChanged();
  }

  _finalizeBond() {
    const startAtomId = this.drag.startAtomId;
    let endAtomId = this.drag.hoverAtomId;

    if (!endAtomId) {
      const startAtom = this.molecule.atoms.get(startAtomId);
      const pos = this.drag.previewEnd;
      if (startAtom && pos && CC.distance(startAtom, pos) > 8) {
        const newAtom = this.molecule.addAtom('C', pos.x, pos.y);
        endAtomId = newAtom.id;
      }
    }

    if (endAtomId && endAtomId !== startAtomId) {
      const existing = this.molecule.getBondBetween(startAtomId, endAtomId);
      if (existing) {
        if (this.currentBondStyle === 'plain') {
          existing.order = existing.order >= 3 ? 1 : existing.order + 1;
          existing.stereo = 'none';
        } else {
          // Re-dragging the same bond with a wedge/hash style selected
          // sets that stereo directly, narrow end at startAtomId — matches
          // how a fresh wedge/hash bond below is drawn (a1 = stereocenter).
          existing.a1 = startAtomId;
          existing.a2 = endAtomId;
          this.molecule.setBondStereo(existing.id, this.currentBondStyle);
        }
      } else if (this.currentBondStyle === 'plain') {
        this.molecule.addBond(startAtomId, endAtomId, 1);
      } else {
        // Wedge/hash bonds are always single — narrow end (stereocenter)
        // is startAtomId, the atom the drag began from.
        this.molecule.addBond(startAtomId, endAtomId, 1, { stereo: this.currentBondStyle });
      }
    }
  }

  _updateChainPreview(p) {
    const startAtom = this.molecule.atoms.get(this.drag.startAtomId);
    if (!startAtom) return;
    const dist = CC.distance(startAtom, p);
    const segments = Math.max(1, Math.round(dist / CC.BOND_LENGTH));
    const baseAngle = Math.atan2(p.y - startAtom.y, p.x - startAtom.x);
    const zigzag = (30 * Math.PI) / 180;

    const points = [{ x: startAtom.x, y: startAtom.y }];
    for (let i = 0; i < segments; i++) {
      const last = points[points.length - 1];
      const angle = baseAngle + (i % 2 === 0 ? -zigzag / 2 : zigzag / 2);
      points.push({
        x: last.x + Math.cos(angle) * CC.BOND_LENGTH,
        y: last.y + Math.sin(angle) * CC.BOND_LENGTH,
      });
    }
    this.drag.previewPoints = points;

    // If the chain's free end has landed near an existing atom (other
    // than the one it started from), releasing should close a ring onto
    // that atom instead of dropping a fresh overlapping one on top of it.
    const last = points[points.length - 1];
    this.drag.snapAtomId = CC.nearestAtomWithinRadius(
      this.molecule, last.x, last.y, CC.HOVER_RADIUS, this.drag.startAtomId
    );
  }

  _finalizeChain() {
    const points = this.drag.previewPoints;
    if (!points || points.length < 2) return;
    let prevId = this.drag.startAtomId;
    // Every interior point always becomes a new atom; only the final
    // point may instead snap onto an existing one to close a ring.
    for (let i = 1; i < points.length - 1; i++) {
      const atom = this.molecule.addAtom('C', points[i].x, points[i].y);
      this.molecule.addBond(prevId, atom.id, 1);
      prevId = atom.id;
    }

    const last = points[points.length - 1];
    let lastId = this.drag.snapAtomId;
    if (!lastId) {
      lastId = this.molecule.addAtom('C', last.x, last.y).id;
    }
    if (!this.molecule.getBondBetween(prevId, lastId)) {
      this.molecule.addBond(prevId, lastId, 1);
    }
  }

  // ---------- ring tool ----------

  _ringPointerDown(atomId, p) {
    const n = this.currentRingSize;
    const r = CC.BOND_LENGTH / (2 * Math.sin(Math.PI / n));
    let center;
    let attachAtomId = null;
    let dir = 0; // angle from the attach atom OUT to the ring's center; unused when attaching to empty canvas

    if (atomId) {
      const molecule = this.molecule;
      const atom = molecule.atoms.get(atomId);
      const pointerDir = Math.atan2(p.y - atom.y, p.x - atom.x) || -Math.PI / 2;
      // Orient the ring so its two bonds at the attachment atom split
      // evenly around whatever open angular space is left there, rather
      // than always placing the ring at a fixed "north" rotation
      // regardless of what's already drawn. For the common case (one
      // existing bond, six-membered ring), the vertex-to-center line of
      // a regular polygon always bisects that vertex's own interior
      // angle, so this puts both new ring bonds at exactly 120° from the
      // existing one -- standard chemistry-drawing placement, not a
      // fixed rotation that happens to look right sometimes. Pointer
      // direction only breaks ties between equally-open gaps (e.g. two
      // existing bonds exactly opposite each other); see
      // CC.bestOpenDirection.
      const existingAngles = molecule.getBondsForAtom(atomId).map(function (b) {
        const otherId = b.a1 === atomId ? b.a2 : b.a1;
        const other = molecule.atoms.get(otherId);
        return Math.atan2(other.y - atom.y, other.x - atom.x);
      });
      dir = CC.bestOpenDirection(existingAngles, pointerDir);
      center = { x: atom.x + Math.cos(dir) * r, y: atom.y + Math.sin(dir) * r };
      attachAtomId = atomId;
    } else {
      center = p;
    }

    const ids = [];
    for (let i = 0; i < n; i++) {
      // With an attach atom, vertex 0 is placed at angle (dir + π) from
      // center -- exactly the reverse of how center was placed relative
      // to the atom, so vertex 0 lands back on the atom's own (unmoved)
      // position by construction, not by teleporting it there after the
      // fact. Without one (empty-canvas click), there's nothing to stay
      // consistent with, so this keeps the original fixed "north" start.
      const angle = attachAtomId
        ? dir + Math.PI + (Math.PI * 2 * i) / n
        : (Math.PI * 2 * i) / n - Math.PI / 2;
      const x = center.x + Math.cos(angle) * r;
      const y = center.y + Math.sin(angle) * r;
      if (i === 0 && attachAtomId) {
        const atom = this.molecule.atoms.get(attachAtomId);
        atom.x = x;
        atom.y = y;
        ids.push(attachAtomId);
      } else {
        const atom = this.molecule.addAtom('C', x, y);
        ids.push(atom.id);
      }
    }
    // Aromatic template (benzene): alternating single/double bonds --
    // the same Kekulized representation this app uses everywhere else
    // (there's no separate "aromatic" bond order in the data model;
    // RDKit perceives aromaticity from this pattern the same way it
    // would from a hand-drawn alternating ring).
    for (let i = 0; i < n; i++) {
      const order = this.currentRingAromatic && i % 2 === 1 ? 2 : 1;
      this.molecule.addBond(ids[i], ids[(i + 1) % n], order);
    }
    this._commit();
  }

  // ---------- charge tool ----------

  _chargePointerDown(atomId, decrement) {
    if (!atomId) return;
    const atom = this.molecule.atoms.get(atomId);
    let charge = atom.charge + (decrement ? -1 : 1);
    charge = Math.max(-3, Math.min(3, charge));
    atom.charge = charge;
    this._commit();
  }

  // ---------- erase tool ----------

  _erasePointerDown(atomId, bondId) {
    if (atomId) {
      this.molecule.removeAtom(atomId);
      this._commit();
    } else if (bondId) {
      this.molecule.removeBond(bondId);
      this._commit();
    }
  }
};
