import numpy as np

x = np.array([
    [-0.8,  0.1],
    [ 0.7, -0.4],
], dtype=np.float32)

bounds = np.array([
    [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
    [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
], dtype=np.float32)

# coeff[o, i, piece, degree]
# Power basis: a0 + a1*x + a2*x^2 + a3*x^3
coeff = np.zeros((2, 2, 5, 4), dtype=np.float32)

for r in range(5):
    # phi_00 = r + x
    coeff[0, 0, r, :] = [r, 1.0, 0.0, 0.0]

    # phi_01 = 10 + r + 2*x
    coeff[0, 1, r, :] = [10.0 + r, 2.0, 0.0, 0.0]

    # phi_10 = 20 + r - x
    coeff[1, 0, r, :] = [20.0 + r, -1.0, 0.0, 0.0]

    # phi_11 = 30 + r + 0.5*x
    coeff[1, 1, r, :] = [30.0 + r, 0.5, 0.0, 0.0]


def piece(xval, b):
    r = 0
    for j in range(1, len(b) - 1):
        if xval >= b[j]:
            r = j
    return r


y = np.zeros((2, 2), dtype=np.float32)

for b in range(2):
    for o in range(2):
        for i in range(2):
            xv = x[b, i]
            r = piece(xv, bounds[i])
            a0, a1, a2, a3 = coeff[o, i, r]
            p = ((a3 * xv + a2) * xv + a1) * xv + a0
            y[b, o] += p

print("Input:")
print(x)
print()
print("Piece indices:")
for b in range(2):
    print([piece(x[b, i], bounds[i]) for i in range(2)])

print()
print("Expected output:")
print(y)

def dense(arr):
    if arr.ndim == 0:
        v = float(arr)
        if v.is_integer():
            return f"{v:.1f}"
        return f"{v:.8g}"
    return "[" + ", ".join(dense(x) for x in arr) + "]"
mlir = f"""module {{

  func.func private @print_f32(f32)

  func.func @piecewise_poly_concrete()
      -> tensor<2x2xf32> {{

    %x = arith.constant dense<[
      [-0.8, 0.1],
      [0.7, -0.4]
    ]> : tensor<2x2xf32>

    %bounds = arith.constant dense<[
      [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
      [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
    ]> : tensor<2x6xf32>

    %coeffs = arith.constant dense<{dense(coeff)}>
        : tensor<2x2x5x4xf32>

    %y = kan.piecewise_poly_linear
        %x, %bounds, %coeffs
        : tensor<2x2xf32>,
          tensor<2x6xf32>,
          tensor<2x2x5x4xf32>
          -> tensor<2x2xf32>

    return %y : tensor<2x2xf32>
  }}

  func.func @main() {{
    %result = func.call @piecewise_poly_concrete()
        : () -> tensor<2x2xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %v00 = tensor.extract %result[%c0, %c0]
        : tensor<2x2xf32>
    %v01 = tensor.extract %result[%c0, %c1]
        : tensor<2x2xf32>
    %v10 = tensor.extract %result[%c1, %c0]
        : tensor<2x2xf32>
    %v11 = tensor.extract %result[%c1, %c1]
        : tensor<2x2xf32>

    func.call @print_f32(%v00) : (f32) -> ()
    func.call @print_f32(%v01) : (f32) -> ()
    func.call @print_f32(%v10) : (f32) -> ()
    func.call @print_f32(%v11) : (f32) -> ()

    return
  }}
}}
"""

with open("test/concrete_piecewise_poly.mlir", "w") as f:
    f.write(mlir)

print()
print("Wrote test/concrete_piecewise_poly.mlir")
