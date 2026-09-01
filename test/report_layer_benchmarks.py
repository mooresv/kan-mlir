#!/usr/bin/env python3

import csv


KANLIB_US = {
    256: 208.833,
    1024: 213.007,
    4096: 218.245,
    16384: 208.705,
    65536: 209.745,
}


KERNEL_CSV = "/tmp/kan_generated_bench/results.csv"
LAYER_CSV = "/tmp/kan_layer_bench/results.csv"


def load_csv(path):
    result = {}

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            result[int(row["batch"])] = row

    return result


kernel = load_csv(KERNEL_CSV)
layer = load_csv(LAYER_CSV)


print()
print(
    f"{'Batch':>8} "
    f"{'KANLib us':>11} "
    f"{'Kernel us':>11} "
    f"{'Layer us':>11} "
    f"{'Host us':>10} "
    f"{'Layer speedup':>14} "
    f"{'Layer/kernel':>13}"
)

print("-" * 95)

for B in sorted(KANLIB_US):

    kanlib_us = KANLIB_US[B]

    kernel_us = float(
        kernel[B]["kernel_us"])

    layer_us = float(
        layer[B]["gpu_layer_us"])

    host_us = float(
        layer[B]["host_submit_us"])

    speedup = kanlib_us / layer_us

    overhead_ratio = layer_us / kernel_us

    print(
        f"{B:8d} "
        f"{kanlib_us:11.3f} "
        f"{kernel_us:11.3f} "
        f"{layer_us:11.3f} "
        f"{host_us:10.3f} "
        f"{speedup:13.2f}x "
        f"{overhead_ratio:12.3f}x"
    )
