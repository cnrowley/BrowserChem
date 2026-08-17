/**
 * logger.js
 *
 * App-wide status/event log -- a lightweight timestamped ring buffer
 * (info/success/warning/error), completely independent of any UI. Any
 * file can call CC.Logger.info/success/warning/error(...) without
 * knowing whether a console is even open; js/app.js's setupLogConsole()
 * is the only place that actually renders these into the DOM (see
 * index.html's #log-console) via CC.Logger.subscribe(...), and it's also
 * where "Download log" turns CC.Logger.toText() into a file.
 *
 * Loaded early (see index.html's <script> order) so lib-loader.js and
 * every other file can log from the moment they run, not just after
 * app.js's own DOMContentLoaded setup -- CC.Logger.getEntries() is used
 * to backfill anything logged before setupLogConsole() had a chance to
 * subscribe.
 */

window.CC = window.CC || {};

(function () {
  const MAX_ENTRIES = 2000; // ring buffer -- a long session shouldn't grow this file-download-sized forever

  const entries = [];
  const listeners = [];

  function pad(n) { return n < 10 ? '0' + n : String(n); }

  function formatTime(date) {
    return pad(date.getHours()) + ':' + pad(date.getMinutes()) + ':' + pad(date.getSeconds());
  }

  function log(level, message) {
    const entry = { time: new Date(), level: level, message: String(message) };
    entries.push(entry);
    if (entries.length > MAX_ENTRIES) entries.shift();
    listeners.forEach(function (fn) { fn(entry); });
    return entry;
  }

  function formatEntry(entry) {
    return '[' + entry.time.toISOString() + '] [' + entry.level.toUpperCase() + '] ' + entry.message;
  }

  CC.Logger = {
    info: function (message) { return log('info', message); },
    success: function (message) { return log('success', message); },
    warning: function (message) { return log('warning', message); },
    error: function (message) { return log('error', message); },

    getEntries: function () { return entries.slice(); },

    clear: function () {
      entries.length = 0;
      listeners.forEach(function (fn) { fn(null); }); // null = "the log was cleared", not a new entry
    },

    /** fn(entry) is called for every future log call; fn(null) on clear(). Does NOT replay past entries -- call getEntries() first for those. */
    subscribe: function (fn) { listeners.push(fn); },

    formatEntry: formatEntry,
    formatTime: formatTime,

    /** Plain-text rendering of every entry currently held, one per line -- what "Download log" saves. */
    toText: function () {
      return entries.map(formatEntry).join('\n') + (entries.length ? '\n' : '');
    },
  };
})();
