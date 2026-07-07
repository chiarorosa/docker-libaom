# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This is the **editable working copy of libaom (AV1 reference codec, tag v3.10.0)** inside an
AV1 encoder-optimization research lab. The current line of research is **partition heuristics**
(see the `test-partition-heuristic` branch and `av1/encoder/partition_*` files).

The git repository root is **`C:\dev\av1-docker`**, not this directory. This `src/aom` tree is one
of several checked-in components. The layout that matters:

- `src/aom` (here) — working copy you edit and modify. **This is where optimizations go.**
- `src/aom_baseline` — NEVER READ THIS FOLDER. NEVER EDIT THIS FOLDER. It's a pristine, untouched v3.10.0. The "blind control" for benchmarks.
- `docs/GUIA_builds.md` — the authoritative experiment workflow guide (in Portuguese). Read it before
  changing the build/test flow.
- `logs/`, `results/` — build/test logs and encoder output, shared with the container.

## Critical workflow: edit on Windows, build/test in Docker

Code is edited on Windows (this path) but **all builds and tests run inside a Linux Docker container**,
where this tree is mounted at `/workspace/src/aom`. You cannot build natively on Windows here.

Start the container from the repo root:

```bash
cd C:\dev\av1-docker
docker compose run --rm research_env /bin/bash
```

Inside the container, paths map as: `src/ → /workspace/src`, `logs/ → /workspace/logs`,
`results/ → /workspace/results`, and `build/` is a named Docker volume at `/workspace/build`
(ccache lives there and persists between sessions).

## Build variants

There are three distinct build configurations, each in its own `build/` subdir. Never mix them.

**Daily dev loop (use this ~90% of the time)** — pure C, debug symbols, tests enabled:

```bash
# Configure once:
cmake -S /workspace/src/aom -B /workspace/build/libaom_dev_generic \
  -G Ninja -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DAOM_TARGET_CPU:STRING=generic \
  -DENABLE_CCACHE:BOOL=1 -DENABLE_EXAMPLES:BOOL=ON -DENABLE_TESTS:BOOL=ON \
  -DENABLE_DOCS:BOOL=OFF -DCONFIG_INTERNAL_STATS:BOOL=1

# Rebuild after every code change (seconds, thanks to ccache + Ninja):
cmake --build /workspace/build/libaom_dev_generic -j"$(nproc)" 2>&1 \
  | tee /workspace/logs/build-libaom_dev_generic.log
```

`AOM_TARGET_CPU=generic` is essential: it forces pure-C code paths so the hand-written SIMD/assembly
does **not** shadow your C modifications. Validate logic here first.

## Testing

Run the AOMedia unit tests (`test_libaom`) with a filter scoped to the areas under study:

```bash
/workspace/build/libaom_dev_generic/test_libaom \
  --gtest_filter='BlockdTest*:C/*DrPredTest*:IntrabcTest*:MvCostTest*:C/SADavgTest*:EncodeAPI.AllIntra*:KeyValAPI.*partition*:KeyValAPI.*intra*' \
  2>&1 | tee /workspace/logs/test_run_$(date +%Y%m%d_%H%M%S).log
```

**Bitstream sanity check** — confirm a change still produces a valid AV1 stream before heavy runs:

```bash
dd if=/dev/zero bs=115200 count=10 of=/tmp/quick_test.yuv          # 10-frame 320x240 YUV
/workspace/build/libaom_dev_generic/aomenc -w 320 -h 240 --fps=30/1 \
  --limit=10 --passes=1 --psnr -o /workspace/results/test_dev.ivf /tmp/quick_test.yuv
```

## Codebase architecture (libaom)

Standard libaom layout. The pieces most relevant to this research:

- `av1/encoder/` — the encoder. Partition-decision surface (the experiment target):
  `partition_search.c/.h` (recursive RD partition search), `partition_strategy.c/.h`
  (early-termination / ML-guided pruning), `var_based_part.c` (variance-based fast partitioning),
  `external_partition.*` and `av1_ml_partition_models.h` / `partition_*_weights.h` (learned models).
- `av1/common/` — bitstream-format code shared by encoder and decoder (transforms, prediction,
  loop filter, CDEF). `av1_rtcd_defs.pl` declares the run-time-CPU-detect function tables; the build
  generates `av1_rtcd.h` from it, selecting C vs SIMD per function — the reason `generic` builds matter.
- `av1/decoder/` — the decoder.
- `av1/av1_cx_iface.c` — encoder public API / option plumbing (`aom_codec_*`, control IDs).
- `aom_dsp/`, `aom_scale/`, `aom_mem/`, `aom_ports/`, `aom_util/` — shared DSP and platform primitives,
  each with `arm/`, `x86/` SIMD subtrees mirroring the C reference.
- `apps/`, `examples/` — `aomenc`/`aomdec` CLIs and API usage samples.
- `test/` — GoogleTest-based unit and encode/decode conformance tests.

Configuration is CMake-driven: `ENABLE_*` flags control the build; `CONFIG_*` flags
(defaults in `build/cmake/aom_config_defaults.cmake`) control codec features. Both are set at
`cmake` configure time, not build time.

## Conventions

- C code is formatted with clang-format (`.clang-format`, Google-based style); CMake with
  `.cmake-format.py`. Match surrounding style — this is a mirror of an upstream project.
- Keep `src/aom_baseline` pristine. All modifications go in `src/aom` on a dedicated branch
  (`git checkout -b <name>`) so experiments stay isolated from the control.
