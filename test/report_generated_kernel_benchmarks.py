#!/usr/bin/env python3

import argparse
import csv


KANLIB_US = {
    256: 208.833,
    1024: 213.007,
    4096: 218.245,
    16384: 208.705,
    65536: 209.745,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv",
        nargs="?",
        default="/tmp/kan_generated_bench/results.csv",
    )
    args = ap.parse_args()

    rows = []

    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            batch = int(row["batch"])
            kernel_us = float(row["kernel_us"])
            samples_s = float(row["samples_per_s"])
            outputs_s = float(row["outputs_per_s"])
            max_err = float(row["max_abs_error"])

            kanlib_us = KANLIB_US.get(batch)

            speedup = (
                kanlib_us / kernel_us
                if kanlib_us is not None
                else float("nan")
            )

            rows.append(
                (
                    batch,
                    kanlib_us,
                    kernel_us,
                    speedup,
                    samples_s,
                    outputs_s,
                    max_err,
                )
            )

    print()
    print(
        f"{'Batch':>8} "
        f"{'KANLib us':>12} "
        f"{'Kernel us':>12} "
        f"{'Speedup':>10} "
        f"{'Samples/s':>14} "
        f"{'Outputs/s':>14} "
        f"{'Max error':>12}"
    )

    print("-" * 100)

    for (
        batch,
        kanlib_us,
        kernel_us,
        speedup,
        samples_s,
        outputs_s,
        max_err,
    ) in rows:
        print(
            f"{batch:8d} "
            f"{kanlib_us:12.3f} "
            f"{kernel_us:12.3f} "
            f"{speedup:9.2f}x "
            f"{samples_s:14.3e} "
            f"{outputs_s:14.3e} "
            f"{max_err:12.3e}"
        )


if __name__ == "__main__":
    main()
