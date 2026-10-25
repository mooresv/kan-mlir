#!/usr/bin/env bash
set -euo pipefail

# Sweep polynomial vs. de Boor for:
#   DIN=32, DOUT=128, grid=5, cubic splines
#
# Assumes:
#   - run from /src/kan-mlir
#   - kanenv is activated
#   - knot_poly_32x128_g5.pt exists
#   - /tmp/bench_compiled_32x128_g5 exists
#   - /tmp/bench_deboor_32x128_g5 exists
#
# We use fewer iterations for this exploratory sweep.
WARMUP=50
ITERS=200

ARTIFACT="knot_poly_32x128_g5.pt"
FIT="knots_p5_d3"

POLY_BENCH="/tmp/bench_compiled_32x128_g5"
DEBOOR_BENCH="/tmp/bench_deboor_32x128_g5"

BATCHES=(256 512 1024 2048 4096 8192 16384 32768 65536)

RESULTS="/tmp/sweep_32x128_g5_results.csv"

# Start fresh.
echo "implementation,batch,time_us,samples_per_s,outputs_per_s,max_abs_error,rms_error" \
    > "${RESULTS}"

for B in "${BATCHES[@]}"; do
    echo
    echo "============================================================"
    echo "BATCH = ${B}"
    echo "============================================================"

    DATA_DIR="/tmp/kan_bench_32x128_g5_${B}"
    BASE="/tmp/kanlib_32x128_g5_${B}"

    # --------------------------------------------------------
    # Export input/reference data.
    # --------------------------------------------------------
    echo
    echo "--- Exporting benchmark data ---"

    python test/export_kanlib_benchmark_data.py \
        --artifact "${ARTIFACT}" \
        --fit "${FIT}" \
        --batch-size "${B}" \
        --output-dir "${DATA_DIR}"

    # --------------------------------------------------------
    # Generate static-shape KAN MLIR.
    # --------------------------------------------------------
    echo
    echo "--- Generating MLIR ---"

    python test/make_kanlib_gpu_benchmark.py \
        --artifact "${ARTIFACT}" \
        --batch-size "${B}" \
        --output "${BASE}.mlir"

    # --------------------------------------------------------
    # Lower KAN -> GPU.
    # --------------------------------------------------------
    echo
    echo "--- Lowering KAN -> GPU ---"

    build/tools/kan-opt/kan-opt \
        "${BASE}.mlir" \
        --lower-kan-piecewise-poly-to-gpu \
        -o "${BASE}_gpu.mlir"

    # --------------------------------------------------------
    # Bufferize.
    # --------------------------------------------------------
    echo
    echo "--- Bufferizing ---"

    build/tools/kan-opt/kan-opt \
        "${BASE}_gpu.mlir" \
        --one-shot-bufferize="bufferize-function-boundaries" \
        -o "${BASE}_bufferized.mlir"

    # --------------------------------------------------------
    # Lower GPU -> NVVM.
    # --------------------------------------------------------
    echo
    echo "--- Lowering GPU -> NVVM ---"

    build/tools/kan-opt/kan-opt \
        "${BASE}_bufferized.mlir" \
        --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90 opt-level=3 kernel-bare-ptr-calling-convention=true" \
        -o "${BASE}_nvvm.mlir"

    # --------------------------------------------------------
    # Check ABI.
    #
    # We expect the same 17-argument ABI established for
    # 32x128/g5. Print it so any unexpected CSE/capture change
    # is visible in the log.
    # --------------------------------------------------------
    echo
    echo "--- gpu.launch_func ABI ---"

    grep 'gpu.launch_func' "${BASE}_nvvm.mlir"

    # --------------------------------------------------------
    # Extract fatbin.
    # --------------------------------------------------------
    echo
    echo "--- Extracting fatbin ---"

    python test/extract_fatbin.py \
        "${BASE}_nvvm.mlir" \
        "${BASE}.fatbin"

    # --------------------------------------------------------
    # Polynomial benchmark.
    # --------------------------------------------------------
    echo
    echo "--- Polynomial ---"

    POLY_OUTPUT=$(
        "${POLY_BENCH}" \
            "${BASE}.fatbin" \
            "${DATA_DIR}" \
            "${B}" \
            "${WARMUP}" \
            "${ITERS}"
    )

    echo "${POLY_OUTPUT}"

    POLY_RESULT=$(echo "${POLY_OUTPUT}" | grep '^RESULT,')

    # Existing RESULT format:
    # RESULT,batch,time_us,host_submit_us,samples/s,outputs/s,maxerr,rmserr
    IFS=',' read -r _ PB PT PH PS PO PE PR <<< "${POLY_RESULT}"

    echo "polynomial,${PB},${PT},${PS},${PO},${PE},${PR}" \
        >> "${RESULTS}"

    # --------------------------------------------------------
    # de Boor benchmark.
    # --------------------------------------------------------
    echo
    echo "--- de Boor ---"

    DEBOOR_OUTPUT=$(
        "${DEBOOR_BENCH}" \
            "${DATA_DIR}" \
            "${B}" \
            "${DATA_DIR}/expected.bin" \
            "${WARMUP}" \
            "${ITERS}"
    )

    echo "${DEBOOR_OUTPUT}"

    DEBOOR_RESULT=$(echo "${DEBOOR_OUTPUT}" | grep '^RESULT,')

    # Existing RESULT format:
    # RESULT,batch,time_us,samples/s,outputs/s,maxerr,rmserr
    IFS=',' read -r _ DB DT DS DO DE DR <<< "${DEBOOR_RESULT}"

    echo "deboor,${DB},${DT},${DS},${DO},${DE},${DR}" \
        >> "${RESULTS}"

    echo
    echo "--- Running results ---"
    cat "${RESULTS}"
done

echo
echo "============================================================"
echo "SWEEP COMPLETE"
echo "============================================================"
cat "${RESULTS}"
