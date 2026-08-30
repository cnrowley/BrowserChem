#!/bin/bash
# run_sharded.sh <script.js> <in.csv> <out.csv> [n_shards=16]
#
# Parallelizes any of this harness's own row-at-a-time CLI scripts
# (score_batch.js, add_extra_features.js -- both take
# <in.csv> <out.csv> [startRow] [endRow]) across N independent Node
# processes. Each row scores completely independently (no shared state),
# so this is embarrassingly parallel -- on a 32-core machine, 24 shards
# took ~15 minutes wall-clock for 2,908 rows of score_batch.js's own
# (expensive, real-GB/SA-solvent) work that would have been ~4.6 hours
# single-threaded (real measurement, not an estimate -- see the
# pka-microstate-freeenergy registry entry's own training notes).
#
# Shard intermediates live in their own tmp dir (not this directory) so
# two invocations (e.g. one still running) never collide on filenames.
#
# Requires a static server already running at CC_BASE_URL (default
# http://localhost:8000/) serving the repo root -- see harness.js.
set -e
cd "$(dirname "$0")"
SCRIPT="$1"
IN_CSV="$2"
OUT_CSV="$3"
N_SHARDS="${4:-16}"

export CC_BASE_URL="${CC_BASE_URL:-http://localhost:8000/}"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

TOTAL=$(($(wc -l < "$IN_CSV") - 1))
CHUNK=$(( (TOTAL + N_SHARDS - 1) / N_SHARDS ))
echo "script=$SCRIPT total rows=$TOTAL, shards=$N_SHARDS, chunk=$CHUNK, workdir=$WORKDIR" >&2

pids=()
for ((i=0; i<N_SHARDS; i++)); do
  s=$((i * CHUNK))
  e=$(((i + 1) * CHUNK))
  if [ "$s" -ge "$TOTAL" ]; then break; fi
  node "$SCRIPT" "$IN_CSV" "$WORKDIR/shard_$i.csv" "$s" "$e" > "$WORKDIR/shard_$i.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} shard processes: ${pids[*]}" >&2
wait
echo "all shards done" >&2

# Merge: one header, then all shard bodies in order.
head -n1 "$WORKDIR/shard_0.csv" > "$OUT_CSV"
for ((i=0; i<N_SHARDS; i++)); do
  f="$WORKDIR/shard_$i.csv"
  [ -f "$f" ] && tail -n +2 "$f" >> "$OUT_CSV"
done
echo "merged into $OUT_CSV: $(wc -l < "$OUT_CSV") lines" >&2
grep -h "^done\." "$WORKDIR"/shard_*.log >&2 || true
