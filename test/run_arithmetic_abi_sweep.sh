
#!/usr/bin/env bash
set -euo pipefail

cd /src/kan-mlir

OUT=/tmp/kan_arithmetic_abi_sweep
BASE=/tmp/kan_stable_abi_sweep
KANOPT=build/tools/kan-opt/kan-opt
BENCH="$BASE/bench_stable_abi"

mkdir -p "$OUT"

CSV="$OUT/results.csv"

echo "batch,kernel_us,samples_per_s,outputs_per_s,max_abs_error,rms_error" \
  > "$CSV"

for B in 64 256 1024 4096 8192; do
    echo
    echo "========================================"
    echo "Arithmetic interval selection sweep: batch=$B"
    echo "========================================"

    SRC="$OUT/kan_${B}.mlir"
    GPU="$OUT/kan_${B}_gpu.mlir"
    BUF="$OUT/kan_${B}_buf.mlir"
    NVVM="$OUT/kan_${B}_nvvm.mlir"
    FATBIN="$OUT/kan_${B}.fatbin"
    LOG="$OUT/bench_${B}.log"

    python test/make_kanlib_gpu_benchmark.py \
      --artifact knot_poly_fits_8x32_g10.pt \
      --batch-size "$B" \
      --output "$SRC"

    "$KANOPT" "$SRC" \
      --lower-kan-piecewise-poly-to-gpu \
      -o "$GPU"

    "$KANOPT" "$GPU" \
      --one-shot-bufferize="bufferize-function-boundaries" \
      -o "$BUF"

    "$KANOPT" "$BUF" \
      --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90 opt-level=3 kernel-bare-ptr-calling-convention=true" \
      -o "$NVVM"

    python test/extract_fatbin.py \
      "$NVVM" "$FATBIN"

    DATA="$BASE/data_${B}"

    if [[ ! -d "$DATA" ]]; then
        echo "Missing benchmark data: $DATA" >&2
        exit 1
    fi

    KERNEL="kan_piecewise_arithmetic_b${B}_i8_o32_p16_c4"

    KAN_TRANSPOSE_COEFFS=1 \
    KAN_KERNEL_NAME="$KERNEL" \
      "$BENCH" \
        "$FATBIN" \
        "$DATA" \
        "$B" 100 1000 8 32 16 4 \
      | tee "$LOG"

    RESULT=$(grep '^RESULT,' "$LOG" | tail -1)

    if [[ -z "$RESULT" ]]; then
        echo "Missing RESULT line for batch $B" >&2
        exit 1
    fi

    echo "${RESULT#RESULT,}" >> "$CSV"
done

echo
echo "Results saved to $CSV"
cat "$CSV"
