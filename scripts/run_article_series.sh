#!/bin/bash
# Run all series of the article (series.py) for the four seeds, at most JOBS runs at the same time.
# Usage:  bash scripts/run_article_series.sh [series ...]      (default: all series)
#         JOBS=6 bash scripts/run_article_series.sh SR0 SG0
# Output in runs/ (c23_z<seed>_<series>.npz + manifest; log per run in runs/logs/). Restartable: a run whose log ends
# with END is skipped. A 4-hour run takes about 2 hours of computing time on one core of an Apple M1.
cd "$(dirname "$0")/.." || exit 1
JOBS=${JOBS:-3}
SERIES=${*:-$(python3 -c "import series; print(' '.join(series.SERIES))")}
mkdir -p runs/logs
python3 run_model.py 271 SR0 TEIND=1 NAAMSUF=_build > /dev/null 2>&1; rm -f runs/c23_z271_build*   # build the C library once
for s in $SERIES; do
  for z in 271 272 273 274; do
    log="runs/logs/${s}_z$z.log"
    grep -q END "$log" 2>/dev/null && continue
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
    echo "$(date '+%F %T') start $s seed $z"
    python3 run_model.py "$z" "$s" > "$log" 2>&1 &
    sleep 1
  done
done
wait
echo "$(date '+%F %T') all runs finished"
