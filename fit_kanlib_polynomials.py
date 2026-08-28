#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch

from numpy.polynomial import Chebyshev, Polynomial
from kanlib.nn.bspline import Linear


def get_feature_domains(layer: Linear):
    """
    Return the non-extended spline domain [xmin, xmax] for each input feature.

    KANLib stores an extended knot grid. For spline order p, the original
    grid begins at grid[:, p] and ends at grid[:, -(p+1)].
    """
    grid = layer.basis.grid.detach().cpu()
    p = layer.basis.spline_order

    xmin = grid[:, p]
    xmax = grid[:, -(p + 1)]

    return xmin, xmax


def get_feature_knot_boundaries(layer: Linear, input_index: int):
    """
    Return the original, non-extended spline-grid boundaries for one feature.

    For grid_size = G, this returns G+1 boundaries and therefore G intervals.
    """
    grid = layer.basis.grid.detach().cpu()
    p = layer.basis.spline_order
    g = layer.basis.grid_size

    # Extended grid layout:
    #
    # [p left-extension knots]
    # [G+1 original grid points]
    # [p right-extension knots]
    #
    boundaries = grid[input_index, p : p + g + 1]

    return boundaries.numpy().astype(np.float64)


@torch.no_grad()
def evaluate_edge_function(
    layer: Linear,
    input_index: int,
    output_index: int,
    x_values: torch.Tensor,
):
    """
    Evaluate one learned edge function

        phi_{o,i}(x) = sum_k c[o,i,k] B_{i,k}(x)
    """
    device = next(layer.parameters()).device
    dtype = next(layer.parameters()).dtype

    x_values = x_values.to(device=device, dtype=dtype)

    n = x_values.numel()

    x = torch.zeros(
        n,
        layer.in_features,
        device=device,
        dtype=dtype,
    )
    x[:, input_index] = x_values

    basis = layer.basis(x)  # [N, Din, Nbasis]

    basis_i = basis[:, input_index, :]
    coeff = layer.weighted_coefficients[
        output_index, input_index, :
    ]

    return basis_i @ coeff


def fit_single_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
):
    """
    Fit using Chebyshev coordinates for numerical stability, then convert
    to ordinary power-basis coefficients

        a0 + a1*x + ... + ad*x^d

    suitable for Horner/FMA lowering.
    """
    cheb = Chebyshev.fit(x, y, degree)
    poly = cheb.convert(kind=Polynomial)

    coeff = np.asarray(poly.coef, dtype=np.float64)

    if len(coeff) < degree + 1:
        coeff = np.pad(
            coeff,
            (0, degree + 1 - len(coeff)),
        )

    coeff = coeff[: degree + 1]

    return coeff


def make_uniform_boundaries(lo: float, hi: float, pieces: int):
    return np.linspace(
        lo,
        hi,
        pieces + 1,
        dtype=np.float64,
    )


def fit_piecewise_edge(
    layer: Linear,
    input_index: int,
    output_index: int,
    degree: int,
    boundaries: np.ndarray,
    fit_points_per_piece: int,
    test_points_per_piece: int,
):
    """
    Fit one polynomial per interval.

    Returns:
      coeffs:
        [num_pieces, degree+1]

      error metrics over an independent validation grid.
    """
    num_pieces = len(boundaries) - 1

    coeffs = np.empty(
        (num_pieces, degree + 1),
        dtype=np.float64,
    )

    all_exact = []
    all_approx = []

    for piece in range(num_pieces):
        lo = boundaries[piece]
        hi = boundaries[piece + 1]

        # ------------------------------------------------------------
        # Fitting samples
        # ------------------------------------------------------------
        x_fit = np.linspace(
            lo,
            hi,
            fit_points_per_piece,
            dtype=np.float64,
        )

        y_fit = evaluate_edge_function(
            layer,
            input_index,
            output_index,
            torch.from_numpy(x_fit),
        ).cpu().double().numpy()

        coeff = fit_single_polynomial(
            x_fit,
            y_fit,
            degree,
        )

        coeffs[piece, :] = coeff

        # ------------------------------------------------------------
        # Independent validation points.
        #
        # Keep them slightly inside each interval so internal grid
        # boundaries are not double-counted.
        # ------------------------------------------------------------
        if test_points_per_piece > 1:
            step = (hi - lo) / test_points_per_piece
            x_test = np.linspace(
                lo + 0.5 * step,
                hi - 0.5 * step,
                test_points_per_piece,
                dtype=np.float64,
            )
        else:
            x_test = np.array(
                [0.5 * (lo + hi)],
                dtype=np.float64,
            )

        y_exact = evaluate_edge_function(
            layer,
            input_index,
            output_index,
            torch.from_numpy(x_test),
        ).cpu().double().numpy()

        y_approx = np.polynomial.polynomial.polyval(
            x_test,
            coeff,
        )

        all_exact.append(y_exact)
        all_approx.append(y_approx)

    all_exact = np.concatenate(all_exact)
    all_approx = np.concatenate(all_approx)

    error = all_approx - all_exact

    max_abs = np.max(np.abs(error))
    rms = np.sqrt(np.mean(error**2))

    scale = np.max(np.abs(all_exact))
    relative_max = max_abs / scale if scale > 0 else 0.0

    return coeffs, max_abs, rms, relative_max


def determine_piece_indices(
    x_feature: torch.Tensor,
    boundaries: torch.Tensor,
):
    """
    Determine interval index for each x.

    boundaries has shape [pieces+1].

    Returns indices in [0, pieces-1].
    """
    # bucketize against internal boundaries only
    internal = boundaries[1:-1]

    idx = torch.bucketize(
        x_feature.contiguous(),
        internal.contiguous(),
    )

    return idx


def evaluate_piecewise_polynomial_layer(
    x: torch.Tensor,
    coeff: torch.Tensor,
    boundaries: torch.Tensor,
):
    """
    Reference piecewise-polynomial layer.

    x:
      [B, Din]

    coeff:
      [Dout, Din, pieces, degree+1]

    boundaries:
      [Din, pieces+1]

    Semantics:

      y[b,o] = sum_i p[o,i,piece(x[b,i])](x[b,i])
    """
    batch, din = x.shape

    dout, coeff_din, pieces, ncoeff = coeff.shape

    assert din == coeff_din
    assert boundaries.shape == (din, pieces + 1)

    degree = ncoeff - 1

    output = torch.zeros(
        batch,
        dout,
        device=x.device,
        dtype=x.dtype,
    )

    for i in range(din):
        xi = x[:, i]

        piece_idx = determine_piece_indices(
            xi,
            boundaries[i],
        )

        # coeff_i:
        # [Dout, pieces, degree+1]
        coeff_i = coeff[:, i, :, :]

        # Select one piece for every batch element and output.
        #
        # Rearranged to:
        # [B, Dout, degree+1]
        selected = coeff_i[:, piece_idx, :]
        selected = selected.permute(1, 0, 2)

        # Horner evaluation.
        p = selected[:, :, degree]

        xi2 = xi.unsqueeze(1)

        for k in range(degree - 1, -1, -1):
            p = p * xi2 + selected[:, :, k]

        output += p

    return output


def build_boundaries(
    layer: Linear,
    piece_mode: str,
    pieces: int,
):
    """
    Return boundaries for each input dimension.

    Uniform:
        [Din, pieces+1]

    Knots:
        [Din, grid_size+1]
    """
    xmin, xmax = get_feature_domains(layer)

    all_boundaries = []

    for i in range(layer.in_features):
        if piece_mode == "uniform":
            b = make_uniform_boundaries(
                xmin[i].item(),
                xmax[i].item(),
                pieces,
            )

        elif piece_mode == "knots":
            b = get_feature_knot_boundaries(
                layer,
                i,
            )

        else:
            raise ValueError(
                f"Unknown piece mode: {piece_mode}"
            )

        all_boundaries.append(b)

    lengths = {
        len(b) for b in all_boundaries
    }

    if len(lengths) != 1:
        raise RuntimeError(
            "This prototype assumes the same number of pieces "
            "for all input features."
        )

    return np.stack(all_boundaries, axis=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in-features", type=int, default=2)
    parser.add_argument("--out-features", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--spline-order", type=int, default=3)

    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[3, 5],
    )

    parser.add_argument(
        "--piece-mode",
        choices=["uniform", "knots"],
        default="uniform",
    )

    parser.add_argument(
        "--pieces",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help=(
            "Piece counts to test in uniform mode. "
            "Ignored in knots mode."
        ),
    )

    parser.add_argument(
        "--fit-points-per-piece",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--test-points-per-piece",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--layer-test-batch",
        type=int,
        default=65536,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="kanlib_piecewise_poly_fits.pt",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    layer = Linear(
        in_features=args.in_features,
        out_features=args.out_features,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        use_output_bias=False,
        use_residual_branch=False,
        use_spline_weight=False,
    ).to(device)

    if args.checkpoint is not None:
        state = torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=True,
        )
        layer.load_state_dict(state)

    layer.eval()

    print("Layer:")
    print(layer)

    print("\nParameters:")
    for name, p in layer.named_parameters():
        print(
            f"  {name:20s} {tuple(p.shape)}"
        )

    print("\nBuffers:")
    for name, b in layer.named_buffers():
        print(
            f"  {name:20s} {tuple(b.shape)}"
        )

    xmin, xmax = get_feature_domains(layer)

    print("\nSpline domains:")
    for i in range(args.in_features):
        print(
            f"  input {i}: "
            f"[{xmin[i].item():.8f}, "
            f"{xmax[i].item():.8f}]"
        )

    results = {
        "config": vars(args),
        "grid": layer.basis.grid.detach().cpu(),
        "source_coefficients":
            layer.weighted_coefficients.detach().cpu(),
        "domains": torch.stack(
            [xmin, xmax],
            dim=1,
        ),
        "fits": {},
    }

    if args.piece_mode == "knots":
        piece_counts = [args.grid_size]
    else:
        piece_counts = args.pieces

    for pieces_requested in piece_counts:

        boundaries_np = build_boundaries(
            layer,
            args.piece_mode,
            pieces_requested,
        )

        actual_pieces = boundaries_np.shape[1] - 1

        print()
        print("#" * 78)
        print(
            f"Piece mode: {args.piece_mode}   "
            f"pieces: {actual_pieces}"
        )
        print("#" * 78)

        print("\nBoundaries:")
        for i in range(args.in_features):
            print(
                f"  input {i}: "
                + " ".join(
                    f"{v:.6f}"
                    for v in boundaries_np[i]
                )
            )

        for degree in args.degrees:

            print()
            print("=" * 78)
            print(
                f"Degree {degree}, "
                f"pieces {actual_pieces}, "
                f"mode {args.piece_mode}"
            )
            print("=" * 78)

            poly_coeff = torch.empty(
                args.out_features,
                args.in_features,
                actual_pieces,
                degree + 1,
                dtype=torch.float64,
            )

            edge_max_errors = []
            edge_rms_errors = []
            edge_relative_errors = []

            for o in range(args.out_features):
                for i in range(args.in_features):

                    coeffs, max_err, rms_err, rel_err = \
                        fit_piecewise_edge(
                            layer=layer,
                            input_index=i,
                            output_index=o,
                            degree=degree,
                            boundaries=boundaries_np[i],
                            fit_points_per_piece=
                                args.fit_points_per_piece,
                            test_points_per_piece=
                                args.test_points_per_piece,
                        )

                    poly_coeff[o, i, :, :] = \
                        torch.from_numpy(coeffs)

                    edge_max_errors.append(max_err)
                    edge_rms_errors.append(rms_err)
                    edge_relative_errors.append(rel_err)

                    print(
                        f"edge (o={o:2d}, i={i:2d})  "
                        f"max={max_err:11.4e}  "
                        f"rms={rms_err:11.4e}  "
                        f"rel-max={rel_err:11.4e}"
                    )

            # --------------------------------------------------------
            # Full-layer validation
            # --------------------------------------------------------

            x_layer = torch.empty(
                args.layer_test_batch,
                args.in_features,
                device=device,
            )

            for i in range(args.in_features):
                lo = xmin[i].item()
                hi = xmax[i].item()
                x_layer[:, i].uniform_(lo, hi)

            with torch.no_grad():

                y_spline = layer(x_layer)

                coeff_gpu = poly_coeff.to(
                    device=device,
                    dtype=x_layer.dtype,
                )

                boundaries_gpu = torch.from_numpy(
                    boundaries_np
                ).to(
                    device=device,
                    dtype=x_layer.dtype,
                )

                y_poly = evaluate_piecewise_polynomial_layer(
                    x=x_layer,
                    coeff=coeff_gpu,
                    boundaries=boundaries_gpu,
                )

                layer_error = y_poly - y_spline

                layer_max = (
                    layer_error.abs().max().item()
                )

                layer_rms = torch.sqrt(
                    torch.mean(layer_error**2)
                ).item()

                output_scale = (
                    y_spline.abs().max().item()
                )

                layer_rel = (
                    layer_max / output_scale
                    if output_scale > 0
                    else 0.0
                )

            print()
            print("Aggregate edge errors:")
            print(
                f"  worst max abs : "
                f"{max(edge_max_errors):.6e}"
            )
            print(
                f"  mean max abs  : "
                f"{np.mean(edge_max_errors):.6e}"
            )
            print(
                f"  mean RMS      : "
                f"{np.mean(edge_rms_errors):.6e}"
            )
            print(
                f"  worst rel max : "
                f"{max(edge_relative_errors):.6e}"
            )

            print()
            print("Full-layer error:")
            print(
                f"  max abs       : "
                f"{layer_max:.6e}"
            )
            print(
                f"  RMS           : "
                f"{layer_rms:.6e}"
            )
            print(
                f"  relative max  : "
                f"{layer_rel:.6e}"
            )

            key = (
                f"{args.piece_mode}"
                f"_p{actual_pieces}"
                f"_d{degree}"
            )

            results["fits"][key] = {
                "piece_mode": args.piece_mode,
                "pieces": actual_pieces,
                "degree": degree,
                "boundaries":
                    torch.from_numpy(boundaries_np),
                "coefficients": poly_coeff,
                "edge_max_errors":
                    torch.tensor(edge_max_errors),
                "edge_rms_errors":
                    torch.tensor(edge_rms_errors),
                "edge_relative_errors":
                    torch.tensor(edge_relative_errors),
                "layer_max_error": layer_max,
                "layer_rms_error": layer_rms,
                "layer_relative_max_error": layer_rel,
            }

    output = Path(args.output)
    torch.save(results, output)

    print()
    print("=" * 78)
    print(f"Saved polynomial fits to {output}")
    print("=" * 78)


if __name__ == "__main__":
    main()
