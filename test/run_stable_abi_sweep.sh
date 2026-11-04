#!/usr/bin/env bash
set -euo pipefail

ROOT=/src/kan-mlir
ARTIFACT="$ROOT/knot_poly_fits_8x32_g10.pt"
OUT=/tmp/kan_stable_abi_sweep
WARMUP=100
ITERS=1000
REPEATS=3

if (( $# > 0 )); then
    BATCHES=("$@")
else
    BATCHES=(64 256 1024 4096 8192)
fi

mkdir -p "$OUT"
CSV="$OUT/results.csv"

echo "batch,repeat,kernel_us,samples_per_s,outputs_per_s,max_abs_error,rms_error" > "$CSV"

cd "$ROOT"

nvcc -O3 -std=c++17 \
    test/bench_generated_kernel_stable_abi.cpp \
    -o "$OUT/bench_stable_abi" -lcuda

for B in "${BATCHES[@]}"; do
    echo "===== Batch $B ====="

    SRC="$OUT/kan_${B}.mlir"
    GPU="$OUT/kan_${B}_gpu.mlir"
    BUF="$OUT/kan_${B}_bufferized.mlir"
    NVVM="$OUT/kan_${B}_nvvm.mlir"
    FATBIN="$OUT/kan_${B}.fatbin"
    DATA="$OUT/data_${B}"

    python test/make_kanlib_gpu_benchmark.py \
        --artifact "$ARTIFACT" \
        --batch-size "$B" \
        --output "$SRC"

    build/tools/kan-opt/kan-opt "$SRC" \
        --lower-kan-piecewise-poly-to-gpu -o "$GPU"

    build/tools/kan-opt/kan-opt "$GPU" \
        --verify-each -o /dev/null

    KERNEL=$(python - "$GPU" <<'PY'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text()
names = re.findall(r'\bgpu\.func\s+@([\w.$-]+)\s*\(', text)
if len(names) != 1:
    raise SystemExit(f"Expected one GPU kernel, found: {names}")
print(names[0])
PY
)

    echo "Kernel: $KERNEL"

    build/tools/kan-opt/kan-opt "$GPU" \
        --one-shot-bufferize="bufferize-function-boundaries" \
        -o "$BUF"

    build/tools/kan-opt/kan-opt "$BUF" \
        --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90 opt-level=3 kernel-bare-ptr-calling-convention=true" \
        -o "$NVVM"

    python test/extract_fatbin.py \
        --binary-name kan_piecewise_gpu \
        "$NVVM" "$FATBIN"

    python test/export_kanlib_benchmark_data.py \
        --artifact "$ARTIFACT" \
        --batch-size "$B" \
        --output-dir "$DATA"

    for R in $(seq 1 "$REPEATS"); do
        LOG="$OUT/run_${B}_rep${R}.log"

        KAN_KERNEL_NAME="$KERNEL" \
        "$OUT/bench_stable_abi" \
            "$FATBIN" "$DATA" "$B" \
            "$WARMUP" "$ITERS" 8 32 16 4 \
            | tee "$LOG"

        RESULT=$(grep '^RESULT,' "$LOG" | tail -1)

        if [[ -z "$RESULT" ]]; then
            echo "Missing RESULT for batch $B repeat $R" >&2
            exit 1
        fi

        IFS=, read -r tag batch us sps ops maxerr rmserr <<< "$RESULT"

        python - "$maxerr" "$rmserr" <<'PY'
import math, sys
mx, rms = map(float, sys.argv[1:])
if not (math.isfinite(mx) and math.isfinite(rms)
        and mx <= 1e-5 and rms <= 1e-5):
    raise SystemExit(f"Correctness failed: max={mx}, rms={rms}")
PY

        echo "$batch,$R,$us,$sps,$ops,$maxerr,$rmserr" >> "$CSV"
    done
done

python - "$CSV" "$OUT/summary.csv" <<'PY'
import csv, statistics, sys
from collections import defaultdict

with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)
for row in rows:
    groups[int(row["batch"])].append(row)

with open(sys.argv[2], "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "batch", "median_kernel_us", "min_kernel_us",
        "max_kernel_us", "median_samples_per_s",
        "median_outputs_per_s", "max_abs_error", "max_rms_error"
    ])
    for batch in sorted(groups):
        rs = groups[batch]
        values = lambda key: [float(r[key]) for r in rs]
        writer.writerow([
            batch,
            statistics.median(values("kernel_us")),
            min(values("kernel_us")),
            max(values("kernel_us")),
            statistics.median(values("samples_per_s")),
            statistics.median(values("outputs_per_s")),
            max(values("max_abs_error")),
            max(values("rms_error"))
        ])

print("\nSweep summary:")
print(open(sys.argv[2]).read())
PY
