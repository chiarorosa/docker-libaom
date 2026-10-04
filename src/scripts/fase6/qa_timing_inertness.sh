set -e
SEQ=/workspace/src/samples/aomctc_test_set/BoxingPractice_3840x2160_5994fps_10bit_420.y4m
COMMON="--cpu-used=1 --passes=1 --end-usage=q --cq-level=32 --kf-min-dist=0 --kf-max-dist=0 --deltaq-mode=0 --enable-tpl-model=0 --enable-keyframe-filtering=0 --tile-columns=1 --tile-rows=0 --threads=2 --row-mt=0 --bit-depth=10 --limit=3 --obu"
export AV1_DISABLE_NATIVE_CNN=1 AV1_STUDENT_TAU_NONE=0.95 AV1_STUDENT_TAU_SPLIT=0.90 AV1_STUDENT_TAU_REST=0.20 AV1_STUDENT_H9D_ENABLE=1
mkdir -p /tmp/qa
/workspace/build/libaom_perf_h9d/aomenc $COMMON -o /tmp/qa/ref.obu $SEQ 2>/dev/null
/workspace/build/libaom_perf_timing/aomenc $COMMON -o /tmp/qa/off.obu $SEQ 2>/dev/null
AV1_PRUNER_TIMING=1 /workspace/build/libaom_perf_timing/aomenc $COMMON -o /tmp/qa/on.obu $SEQ 2>/tmp/qa/on.err
md5sum /tmp/qa/*.obu
grep -A8 PRUNER_TIMING /tmp/qa/on.err
