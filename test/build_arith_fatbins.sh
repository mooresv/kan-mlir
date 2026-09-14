#!/usr/bin/env bash
set -euo pipefail

KANMLIR=/src/kan-mlir
KANOPT=${KANMLIR}/build/tools/kan-opt/kan-opt
GEN=${KANMLIR}/test/make_kanlib_gpu_benchmark.py
EXTRACT=${KANMLIR}/test/extract_fatbin.py
ARTIFACT=${KANMLIR}/knot_poly_fits.pt

for B in 256 1024 4096 16384 65536; do
    echo "========================================"
    echo "Building arithmetic-poly fatbin for B=${B}"
    echo "========================================"

    python ${GEN} \
      --artifact ${ARTIFACT} \
      --batch-size ${B} \
      --output /tmp/kanlib_bench_${B}.mlir

    ${KANOPT} \
      /tmp/kanlib_bench_${B}.mlir \
      --lower-kan-piecewise-poly-to-gpu \
      -o /tmp/kanlib_bench_${B}_gpu_arith.mlir

    ${KANOPT} \
      /tmp/kanlib_bench_${B}_gpu_arith.mlir \
      --one-shot-bufferize="bufferize-function-boundaries" \
      -o /tmp/kanlib_bench_${B}_bufferized_arith.mlir

    ${KANOPT} \
      /tmp/kanlib_bench_${B}_bufferized_arith.mlir \
      --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90 opt-level=3 kernel-bare-ptr-calling-convention=true" \
      -o /tmp/kanlib_bench_${B}_nvvm_arith.mlir

    python ${EXTRACT} \
      /tmp/kanlib_bench_${B}_nvvm_arith.mlir \
      /tmp/kanlib_bench_${B}_arith.fatbin

    echo
done

echo "All arithmetic-poly fatbins built."
