/**
 * history.js
 *
 * Snapshot-based undo/redo. Every committed edit stores a full JSON
 * snapshot of the molecule rather than an inverse operation — simpler to
 * get right than a command-pattern undo stack, and molecules in a 2D
 * editor are small enough that this is cheap.
 */

window.CC = window.CC || {};

CC.History = class History {
  constructor(onChange) {
    this.stack = [];
    this.index = -1;
    this.onChange = onChange || function () {};
  }

  commit(state) {
    this.stack = this.stack.slice(0, this.index + 1);
    this.stack.push(state);
    this.index = this.stack.length - 1;
    this.onChange();
  }

  canUndo() {
    return this.index > 0;
  }

  canRedo() {
    return this.index < this.stack.length - 1;
  }

  undo() {
    if (!this.canUndo()) return null;
    this.index -= 1;
    this.onChange();
    return this.stack[this.index];
  }

  redo() {
    if (!this.canRedo()) return null;
    this.index += 1;
    this.onChange();
    return this.stack[this.index];
  }

  current() {
    return this.stack[this.index];
  }
};
