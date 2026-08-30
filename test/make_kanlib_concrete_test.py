import torch
import torch.nn.functional as F
from pathlib import Path

from kanlib.nn.bspline import Linear


ARTIFACT = Path("knot_poly_fits.pt")
OUTPUT = Path("test/kanlib_concrete_piecewise_poly.mlir")


def dense(arr):
    """
    Emit a nested MLIR dense literal.
    """
    if arr.ndim == 0:
        v = float(arr)
        if v.is_integer():
            return f"{v:.1f}"
        return f"{v:.9g}"
    return "[" + ", ".join(dense(x) for x in arr) + "]"


def evaluate_piecewise(x, boundaries, coeff):
    """
    x:
        [B, Din]

    boundaries:
        [Din, P+1]

    coeff:
        [Dout, Din, P, degree+1]

    Returns:
        [B, Dout]
    """
    B, Din = x.shape
    Dout, Din2, pieces, degree_plus_one = coeff.shape

    assert Din == Din2

    y = torch.zeros(B, Dout, dtype=x.dtype, device=x.device)

    for i in range(Din):
        xi = x[:, i]

        # Match the MLIR lowering:
        #
        # piece = 0
        # if x >= bounds[1]: piece = 1
        # ...
        #
        # bucketize using internal boundaries gives exactly that behavior.
        piece = torch.bucketize(
            xi,
            boundaries[i, 1:-1],
            right=True,
        )

        for o in range(Dout):
            c = coeff[o, i, piece, :]

            # Horner evaluation.
            p = c[:, -1]
            for d in range(degree_plus_one - 2, -1, -1):
                p = p * xi + c[:, d]

            y[:, o] += p

    return y


def main():
    saved = torch.load(
        ARTIFACT,
        map_location="cpu",
        weights_only=False,
    )

    fit = saved["fits"]["knots_p5_d3"]

    config = saved["config"]

    in_features = config["in_features"]
    out_features = config["out_features"]
    grid_size = config["grid_size"]
    spline_order = config["spline_order"]

    grid = saved["grid"].float()
    source_coeff = saved["source_coefficients"].float()

    boundaries = fit["boundaries"].float()
    poly_coeff = fit["coefficients"].float()

    print("grid shape:", tuple(grid.shape))
    print("source coefficients shape:", tuple(source_coeff.shape))
    print("boundaries shape:", tuple(boundaries.shape))
    print("polynomial coefficients shape:", tuple(poly_coeff.shape))

    # ------------------------------------------------------------
    # Reconstruct the original KANLib spline layer.
    # ------------------------------------------------------------

    layer = Linear(
        in_features=in_features,
        out_features=out_features,
        grid_size=grid_size,
        spline_order=spline_order,
        use_output_bias=False,
        use_residual_branch=False,
        use_spline_weight=False,
    ).cpu()

    with torch.no_grad():
        layer.coefficients.copy_(source_coeff)

        # KANLib stores the grid in the basis object.
        layer.basis.grid.copy_(grid)

    layer.eval()

    # ------------------------------------------------------------
    # Deterministic test inputs.
    #
    # We deliberately exercise all five knot intervals.
    # ------------------------------------------------------------

    x = torch.tensor(
        [
            [-0.90, -0.75],
            [-0.55, -0.35],
            [-0.10,  0.10],
            [ 0.35,  0.50],
            [ 0.75,  0.90],
            [-0.80,  0.00],
            [ 0.00,  0.80],
            [ 0.59, -0.59],
        ],
        dtype=torch.float32,
    )

    # ------------------------------------------------------------
    # Source KANLib evaluation.
    # ------------------------------------------------------------

    with torch.no_grad():
        y_kan = layer(x)

    # ------------------------------------------------------------
    # Piecewise-polynomial evaluation.
    # ------------------------------------------------------------

    y_poly = evaluate_piecewise(
        x,
        boundaries,
        poly_coeff,
    )

    diff = y_poly - y_kan

    print("\nInput:")
    print(x)

    print("\nKANLib output:")
    print(y_kan)

    print("\nPiecewise polynomial output:")
    print(y_poly)

    print("\nDifference:")
    print(diff)

    print(
        "\nMax abs error:",
        diff.abs().max().item(),
    )

    print(
        "RMS error:",
        torch.sqrt(torch.mean(diff * diff)).item(),
    )

    # ------------------------------------------------------------
    # Emit self-contained MLIR.
    # ------------------------------------------------------------

    B = x.shape[0]
    Din = in_features
    Dout = out_features

    pieces = poly_coeff.shape[2]
    coeff_count = poly_coeff.shape[3]

    mlir = f"""module {{
  func.func private @print_f32(f32)

  func.func @kanlib_piecewise_concrete()
      -> tensor<{B}x{Dout}xf32> {{

    %x = arith.constant dense<{dense(x)}>
      : tensor<{B}x{Din}xf32>

    %bounds = arith.constant dense<{dense(boundaries)}>
      : tensor<{Din}x{pieces + 1}xf32>

    %coeffs = arith.constant dense<{dense(poly_coeff)}>
      : tensor<{Dout}x{Din}x{pieces}x{coeff_count}xf32>

    %y = kan.piecewise_poly_linear
        %x, %bounds, %coeffs
        : tensor<{B}x{Din}xf32>,
          tensor<{Din}x{pieces + 1}xf32>,
          tensor<{Dout}x{Din}x{pieces}x{coeff_count}xf32>
          -> tensor<{B}x{Dout}xf32>

    return %y : tensor<{B}x{Dout}xf32>
  }}

  func.func @main() {{
    %result = func.call @kanlib_piecewise_concrete()
      : () -> tensor<{B}x{Dout}xf32>

"""

    for b in range(B):
        for o in range(Dout):
            mlir += f"""    %c_b{b}_o{o}_b = arith.constant {b} : index
    %c_b{b}_o{o}_o = arith.constant {o} : index
    %v_b{b}_o{o} = tensor.extract %result[
        %c_b{b}_o{o}_b,
        %c_b{b}_o{o}_o
    ] : tensor<{B}x{Dout}xf32>
    func.call @print_f32(%v_b{b}_o{o}) : (f32) -> ()

"""

    mlir += """    return
  }
}
"""

    OUTPUT.write_text(mlir)

    print(f"\nWrote {OUTPUT}")

    print("\nExpected flattened piecewise-polynomial output:")
    for v in y_poly.flatten():
        print(f"{v.item():.8f}")


if __name__ == "__main__":
    main()
