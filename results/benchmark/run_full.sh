#!/bin/bash
cd /workspace
for S in Jockey_3840x2160_120fps_420_8bit_YUV_RAW RaceNight_3840x2160_50fps_420_8bit_YUV_RAW RiverBank_3840x2160_50fps_420_8bit_YUV_RAW; do
  name=${S%%_*}; Y=src/samples/$S.yuv; O=results/benchmark/h9_test/$name
  echo "########## SEQ $name ##########"
  build/venv-ml/bin/python src/scripts/benchmark/h7h8_bench.py --seq $Y --frames 10 --cqs 20 32 43 55 --preset safe --out-dir $O/curve_safe
  build/venv-ml/bin/python src/scripts/benchmark/h7h8_bench.py --seq $Y --frames 10 --cqs 20 32 43 55 --preset aggressive --out-dir $O/curve_aggr
  build/venv-ml/bin/python src/scripts/benchmark/ablation_attrib.py --seq $Y --frames 10 --cqs 20 32 43 55 --methods ml variance random --tau-none 0.95 0.90 0.80 0.70 0.60 0.50 --out-dir $O/ablation
  echo "########## SEQ $name DONE ##########"
done
echo FULL_DONE
