#!/bin/sh
# E5 -- attribution ablation on the VALIDATION split (HoneyBee/FlowerPan/Lips),
# 10 frames, cpu-used=0, matched policy (NONE-commit only for every arm).
#
# What this repairs. The attribution ablation already ran on the test split at 10
# frames (Fase 5), but the ml and variance speedup ranges came out DISJOINT in
# 3/3 sequences, so no matched-speedup comparison existed. The cause is known:
# the tau grid was frozen on the HoneyBee calibration, where variance@0.95 gave
# 1.34x; on other content the same tau already jumps to ~1.9-2.1x. Extending the
# variance grid is legitimate HERE and only here -- validation is the split whose
# stated role is choosing operating thresholds. Doing it on test would be tuning
# a configuration while looking at test data.
#
# The ml grid is deliberately UNCHANGED (the frozen 0.95..0.50). Only the
# variance arm gains conservative points, which is the arm whose range was too
# aggressive to overlap.
#
# Order: FlowerPan and Lips first. HoneyBee is where the original grid was
# calibrated, so it is the least independent of the three -- if the run is cut
# short, the two independent sequences are the ones already in hand.
set -u

PY=/workspace/build/venv-ml/bin/python
DRIVER=/workspace/src/scripts/benchmark/ablation_attrib.py
SAMPLES=/workspace/src/samples
OUT=/workspace/results/benchmark/e5_ablation

mkdir -p "$OUT"
echo $$ > "$OUT/.pid"

run_seq() {
    name=$1
    file=$2
    echo "########## $name  start=$(date -u +%FT%TZ) ##########"
    $PY "$DRIVER" \
        --seq "$SAMPLES/$file" \
        --frames 10 --cqs 20 32 43 55 \
        --methods ml variance random \
        --tau-none-for \
            ml=0.95,0.90,0.80,0.70,0.60,0.50 \
            variance=0.999,0.995,0.99,0.97,0.95,0.90,0.80 \
            random=0.95,0.90,0.80,0.70 \
        --out-dir "$OUT/$name" \
        || echo "!!! $name FAILED (rc=$?) -- continuing with the next sequence"
    echo "########## $name  end=$(date -u +%FT%TZ) ##########"
}

run_seq FlowerPan FlowerPan_3840x2160_50fps_420_8bit_YUV_RAW.yuv
run_seq Lips      Lips_3840x2160_120fps_420_8bit_YUV_RAW.yuv
run_seq HoneyBee  HoneyBee_3840x2160_120fps_420_8bit_YUV_RAW.yuv

echo "E5_ALL_DONE $(date -u +%FT%TZ)"
