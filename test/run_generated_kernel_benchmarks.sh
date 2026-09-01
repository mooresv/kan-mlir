#!/usr/bin/env bash
set -euo pipefail

ROOT=/src/kan-mlir
BUILD=$ROOT/build
ARTIFACT=$ROOT/knot_poly_fits.pt

BATCHES=(256 1024 4096 16384 65536)

WARMUP=100
ITERS=1000

OUT=/tmp/kan_generated_bench
CSV=$OUT/results.csv

mkdir -p "$OUT"

echo \
"batch,kernel_us,samples_per_s,outputs_per_s,max_abs_error,rms_error" \
> "$CSV"

for B in "${BATCHES[@]}"; do
    echo
    echo "============================================================"
    echo "Batch $B"
    echo "============================================================"

    SRC="$OUT/kanlib_bench_${B}.mlir"
    GPU="$OUT/kanlib_bench_${B}_gpu.mlir"
    BUF="$OUT/kanlib_bench_${B}_bufferized.mlir"
    NVVM="$OUT/kanlib_bench_${B}_nvvm_bare.mlir"
    FATBIN="$OUT/kanlib_bench_${B}.fatbin"
    DATA="$OUT/data_${B}"
    LOG="$OUT/run_${B}.log"

    # ----------------------------------------------------------
    # Generate static-shape KAN dialect benchmark.
    # ----------------------------------------------------------

    cd "$ROOT"

    python test/make_kanlib_gpu_benchmark.py \
      --artifact "$ARTIFACT" \
      --batch-size "$B" \
      --output "$SRC"

    # ----------------------------------------------------------
    # KAN -> GPU.
    # ----------------------------------------------------------

    cd "$BUILD"

    ./tools/kan-opt/kan-opt \
      "$SRC" \
      --lower-kan-piecewise-poly-to-gpu \
      -o "$GPU"

    ./tools/kan-opt/kan-opt \
      "$GPU" \
      --verify-each \
      -o /dev/null

    # ----------------------------------------------------------
    # Bufferize.
    # ----------------------------------------------------------

    ./tools/kan-opt/kan-opt \
      "$GPU" \
      --one-shot-bufferize="bufferize-function-boundaries" \
      -o "$BUF"

    # ----------------------------------------------------------
    # GPU -> sm_90 fatbin with bare-pointer kernel ABI.
    # ----------------------------------------------------------

    ./tools/kan-opt/kan-opt \
      "$BUF" \
      --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90 opt-level=3 kernel-bare-ptr-calling-convention=true" \
      -o "$NVVM"

    # ----------------------------------------------------------
    # Extract exact compiler-generated NVIDIA fatbin.
    # ----------------------------------------------------------

    cd "$ROOT"

    python test/extract_fatbin.py \
      "$NVVM" \
      "$FATBIN"

    # ----------------------------------------------------------
    # Generate deterministic input/reference data.
    # ----------------------------------------------------------

    rm -rf "$DATA"

    python test/export_kanlib_benchmark_data.py \
      --artifact "$ARTIFACT" \
      --batch-size "$B" \
      --output-dir "$DATA"

    # ----------------------------------------------------------
    # Benchmark.
    # ----------------------------------------------------------

    /tmp/bench_generated_kernel \
      "$FATBIN" \
      "$DATA" \
      "$B" \
      "$WARMUP" \
      "$ITERS" \
      | tee "$LOG"

    RESULT=$(grep '^RESULT,' "$LOG")

    if [[ -z "$RESULT" ]]; then
        echo "ERROR: no RESULT line for batch $B"
        exit 1
    fi

    # Strip RESULT, prefix.
    echo "${RESULT#RESULT,}" >> "$CSV"
done

echo
echo "============================================================"
echo "Raw results"
echo "============================================================"
cat "$CSV"

echo
echo "Saved to:"
echo "$CSV"
