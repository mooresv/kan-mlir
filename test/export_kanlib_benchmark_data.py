#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch


def evaluate_piecewise(x, boundaries, coeffs):
    """
    x:          [B, Din]
    boundaries: [Din, P+1]
    coeffs:     [Dout, Din, P, degree+1]
                ascending powers a0,a1,...,ad

    Returns:
        [B, Dout]
    """
    B, Din = x.shape
    Dout, coeff_Din, pieces, ncoeff = coeffs.shape

    assert Din == coeff_Din

    degree = ncoeff - 1
    y = np.zeros((B, Dout), dtype=np.float32)

    for b in range(B):
        for o in range(Dout):
            acc = np.float32(0.0)

            for i in range(Din):
                xv = np.float32(x[b, i])

                piece = 0
                for p in range(1, pieces):
                    if xv >= boundaries[i, p]:
                        piece = p

                poly = np.float32(
                    coeffs[o, i, piece, degree]
                )

                for k in range(degree - 1, -1, -1):
                    poly = np.float32(
                        np.float32(poly * xv)
                        + coeffs[o, i, piece, k]
                    )

                acc = np.float32(acc + poly)

            y[b, o] = acc

    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifact",
        default="knot_poly_fits.pt",
    )
    ap.add_argument(
        "--fit",
        default="knots_p5_d3",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    ap.add_argument(
        "--output-dir",
        default="/tmp/kan_bench_data",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=12345,
    )
    args = ap.parse_args()

    saved = torch.load(
        args.artifact,
        map_location="cpu",
        weights_only=False,
    )

    fit = saved["fits"][args.fit]

    boundaries = (
        fit["boundaries"]
        .detach()
        .cpu()
        .float()
        .numpy()
        .astype(np.float32)
    )

    coeffs = (
        fit["coefficients"]
        .detach()
        .cpu()
        .float()
        .numpy()
        .astype(np.float32)
    )

    Dout, Din, pieces, ncoeff = coeffs.shape

    rng = np.random.default_rng(args.seed)

    x = rng.uniform(
        -1.0,
        1.0,
        size=(args.batch_size, Din),
    ).astype(np.float32)

    expected = evaluate_piecewise(
        x,
        boundaries,
        coeffs,
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    x.tofile(outdir / "input.bin")
    boundaries.tofile(outdir / "bounds.bin")
    coeffs.tofile(outdir / "coeffs.bin")
    expected.tofile(outdir / "expected.bin")

    print(f"batch       : {args.batch_size}")
    print(f"din         : {Din}")
    print(f"dout        : {Dout}")
    print(f"pieces      : {pieces}")
    print(f"degree      : {ncoeff - 1}")
    print(f"input       : {x.shape}")
    print(f"boundaries  : {boundaries.shape}")
    print(f"coefficients: {coeffs.shape}")
    print(f"expected    : {expected.shape}")
    print(f"output dir  : {outdir}")
    print(f"expected[0] : {expected[0]}")


if __name__ == "__main__":
    main()
