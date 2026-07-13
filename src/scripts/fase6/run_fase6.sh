#!/bin/bash
# Fase 6 -- CTC final-results benchmark (All Intra, Class A1, 4K, 10-bit).
#
# Runs the full 6-config x 8-seq x 4-cq x 15-frame encode grid, then the
# report. Resumable: re-running skips encodes already in raw_results.csv, so a
# killed container just needs this script re-launched.
#
# Launch DETACHED inside the persistent container:
#   docker exec -d av1_bench bash /workspace/src/scripts/fase6/run_fase6.sh
# Follow progress:
#   docker exec av1_bench tail -f /workspace/results/benchmark/fase6/run.log
set -euo pipefail

PY=/workspace/build/venv-ml/bin/python
DIR=/workspace/src/scripts/fase6
OUT=/workspace/results/benchmark/fase6
mkdir -p "$OUT"

{
  echo "===== FASE 6 START $(date -u +%FT%TZ) ====="
  "$PY" "$DIR/encode_ctc.py" --out-dir "$OUT"
  echo "===== REPORT $(date -u +%FT%TZ) ====="
  "$PY" "$DIR/report_ctc.py" --out-dir "$OUT"
  echo "===== FASE 6 DONE $(date -u +%FT%TZ) ====="
} 2>&1 | tee -a "$OUT/run.log"
