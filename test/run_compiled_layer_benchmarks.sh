#!/usr/bin/env bash
set -euo pipefail

BASE=/tmp/kan_generated_bench
OUT=/tmp/kan_layer_bench
CSV=$OUT/results.csv

BATCHES=(256 1024 4096 16384 65536)

WARMUP=100
ITERS=1000

mkdir -p "$OUT"

echo \
"batch,gpu_layer_us,host_submit_us,samples_per_s,outputs_per_s,max_abs_error,rms_error" \
> "$CSV"

for B in "${BATCHES[@]}"; do

    echo
    echo "============================================================"
    echo "Compiled layer batch $B"
    echo "============================================================"

    FATBIN="$BASE/kanlib_bench_${B}.fatbin"
    DATA="$BASE/data_${B}"
    LOG="$OUT/run_${B}.log"

    if [[ ! -f "$FATBIN" ]]; then
        echo "Missing fatbin: $FATBIN"
        exit 1
    fi

    if [[ ! -d "$DATA" ]]; then
        echo "Missing data directory: $DATA"
        exit 1
    fi

    /tmp/bench_compiled_layer \
      "$FATBIN" \
      "$DATA" \
      "$B" \
      "$WARMUP" \
      "$ITERS" \
      | tee "$LOG"

    RESULT=$(grep '^RESULT,' "$LOG")

    if [[ -z "$RESULT" ]]; then
        echo "ERROR: no RESULT line for B=$B"
        exit 1
    fi

    echo "${RESULT#RESULT,}" >> "$CSV"

done

echo
echo "============================================================"
echo "Compiled layer results"
echo "============================================================"

column -s, -t "$CSV" 2>/dev/null || cat "$CSV"

echo
echo "Saved to:"
echo "$CSV"
