#!/bin/sh
# End the E5 campaign once Lips finishes, so HoneyBee never starts.
#
# Why stop there. The E5 gate is "ml dominates variance at matched speedup in
# >=2 of the 3 validation sequences", which FlowerPan + Lips already decide.
# HoneyBee is the sequence the original tau grid was calibrated on, so it is the
# least independent of the three -- the smallest addition for the ~13h it costs.
#
# Why a watcher instead of editing the driver: the driver is already running,
# and sh reads a script lazily by file offset. Editing it in place would corrupt
# the execution of a campaign that has hours of encoding behind it.
set -u

LOG=/workspace/results/benchmark/e5_ablation.log
PIDFILE=/workspace/results/benchmark/e5_ablation/.pid

while ! grep -q 'Lips  end=' "$LOG" 2>/dev/null; do
    sleep 20
done

echo "STOP: Lips finished; ending before HoneyBee $(date -u +%FT%TZ)" >> "$LOG"

# The driver shell first, so it cannot advance to the next sequence...
if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null
fi
# ...then anything it left mid-flight. The `!/awk/` guard keeps this from
# matching the awk process that carries the pattern in its own command line --
# a self-match here would leave the real encoder running.
for p in $(ps -eo pid,args | awk '/libaom_perf\/aomenc|ablation_attrib\.py/ && !/awk/ {print $1}'); do
    kill "$p" 2>/dev/null
done

echo "E5_ALL_DONE (stopped after Lips, by design) $(date -u +%FT%TZ)" >> "$LOG"
