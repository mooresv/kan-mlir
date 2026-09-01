#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact",
        default="knot_poly_fits.pt",
        help="Path to knot-aligned polynomial-fit artifact",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Static batch size",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output MLIR file",
    )

    args = parser.parse_args()

    saved = torch.load(
        args.artifact,
        map_location="cpu",
        weights_only=False,
    )

    fit = saved["fits"]["knots_p5_d3"]
    config = saved["config"]

    din = config["in_features"]
    dout = config["out_features"]

    boundaries = fit["boundaries"].float()
    coefficients = fit["coefficients"].float()

    pieces = coefficients.shape[2]
    coeff_count = coefficients.shape[3]

    batch = args.batch_size

    assert tuple(boundaries.shape) == (din, pieces + 1)
    assert coefficients.shape[0] == dout
    assert coefficients.shape[1] == din

    mlir = f"""module {{
  func.func @kanlib_piecewise_benchmark(
      %input: tensor<{batch}x{din}xf32>,
      %bounds: tensor<{din}x{pieces + 1}xf32>,
      %coeffs: tensor<{dout}x{din}x{pieces}x{coeff_count}xf32>
  ) -> tensor<{batch}x{dout}xf32> {{

    %result = kan.piecewise_poly_linear
        %input, %bounds, %coeffs
        : tensor<{batch}x{din}xf32>,
          tensor<{din}x{pieces + 1}xf32>,
          tensor<{dout}x{din}x{pieces}x{coeff_count}xf32>
          -> tensor<{batch}x{dout}xf32>

    return %result : tensor<{batch}x{dout}xf32>
  }}
}}
"""

    output = Path(args.output)
    output.write_text(mlir)

    print(f"Wrote:        {output}")
    print(f"batch:        {batch}")
    print(f"din:          {din}")
    print(f"dout:         {dout}")
    print(f"pieces:       {pieces}")
    print(f"degree:       {coeff_count - 1}")
    print(f"boundaries:   {tuple(boundaries.shape)}")
    print(f"coefficients: {tuple(coefficients.shape)}")


if __name__ == "__main__":
    main()
