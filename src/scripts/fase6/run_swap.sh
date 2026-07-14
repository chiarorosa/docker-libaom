#!/bin/bash
# Fase 6 extension -- native CNN vs H9a swap at cpu-used=1/2/3.
#
# Runs native_cpuN + h9a_bal/aggr_cpuN (9 configs) x 8 seqs x 4 cq x 15 frames,
# then the report (referenced to the Fase 6 cpu-used=0 anchor). Resumable.
# Needs the AV1_DISABLE_NATIVE_CNN toggle in build/libaom_perf (rebuild first).
#
# Launch DETACHED (robust; see run_fase6.sh notes on docker exec -d):
#   docker exec av1_bench bash -lc 'nohup setsid bash \
#     /workspace/src/scripts/fase6/run_swap.sh >/dev/null 2>&1 &'
# Follow:
#   docker exec av1_bench tail -f /workspace/results/benchmark/fase6_swap/run.log
set -euo pipefail

PY=/workspace/build/venv-ml/bin/python
DIR=/workspace/src/scripts/fase6
OUT=/workspace/results/benchmark/fase6_swap
mkdir -p "$OUT"

{
  echo "===== FASE 6 SWAP START $(date -u +%FT%TZ) ====="
  "$PY" "$DIR/encode_swap.py" --out-dir "$OUT"
  echo "===== REPORT $(date -u +%FT%TZ) ====="
  "$PY" "$DIR/report_swap.py" --out-dir "$OUT"
  echo "===== FASE 6 SWAP DONE $(date -u +%FT%TZ) ====="
} 2>&1 | tee -a "$OUT/run.log"
