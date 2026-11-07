#!/usr/bin/env bash
# Wait for a Kaggle job from gpu/run_job.sh to finish, then copy its
# results/ into the repo's results/ and its log to results/kaggle_<name>.log.
#
#   gpu/fetch_job.sh <job-name> [--no-wait]
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="$1"
USER_NAME=$(kaggle config view | awk '/username/ {print $3}')
REF="$USER_NAME/cs205-$NAME"
if [ "${2:-}" != "--no-wait" ]; then
  until s=$(kaggle kernels status "$REF" 2>&1); echo "$s" | grep -qiE 'complete|error|cancel'; do
    sleep 30
  done
  echo "$s"
fi
OUT="${TMPDIR:-/tmp}/cs205_kaggle_out/$NAME"
rm -rf "$OUT" && mkdir -p "$OUT"
kaggle kernels output "$REF" -p "$OUT" >/dev/null
[ -d "$OUT/results" ] && cp -R "$OUT/results/." results/
[ -d "$OUT/images" ] && cp -Rn "$OUT/images/." images/   # never overwrite local images
# the log is JSON lines of {stream_name, data}; flatten to plain text
python3 -c "
import json, sys
for e in json.load(open(sys.argv[1])): sys.stdout.write(e.get('data', ''))
" "$OUT/cs205-$NAME.log" > "results/kaggle_$NAME.log" 2>/dev/null || true
echo "fetched -> results/ (log: results/kaggle_$NAME.log)"
ls "$OUT/results" 2>/dev/null || true
