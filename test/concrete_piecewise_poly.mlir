module {

  func.func private @print_f32(f32)

  func.func @piecewise_poly_concrete()
      -> tensor<2x2xf32> {

    %x = arith.constant dense<[
      [-0.8, 0.1],
      [0.7, -0.4]
    ]> : tensor<2x2xf32>

    %bounds = arith.constant dense<[
      [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
      [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
    ]> : tensor<2x6xf32>

    %coeffs = arith.constant dense<[[[[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0], [3.0, 1.0, 0.0, 0.0], [4.0, 1.0, 0.0, 0.0]], [[10.0, 2.0, 0.0, 0.0], [11.0, 2.0, 0.0, 0.0], [12.0, 2.0, 0.0, 0.0], [13.0, 2.0, 0.0, 0.0], [14.0, 2.0, 0.0, 0.0]]], [[[20.0, -1.0, 0.0, 0.0], [21.0, -1.0, 0.0, 0.0], [22.0, -1.0, 0.0, 0.0], [23.0, -1.0, 0.0, 0.0], [24.0, -1.0, 0.0, 0.0]], [[30.0, 0.5, 0.0, 0.0], [31.0, 0.5, 0.0, 0.0], [32.0, 0.5, 0.0, 0.0], [33.0, 0.5, 0.0, 0.0], [34.0, 0.5, 0.0, 0.0]]]]>
        : tensor<2x2x5x4xf32>

    %y = kan.piecewise_poly_linear
        %x, %bounds, %coeffs
        : tensor<2x2xf32>,
          tensor<2x6xf32>,
          tensor<2x2x5x4xf32>
          -> tensor<2x2xf32>

    return %y : tensor<2x2xf32>
  }

  func.func @main() {
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
  }
}
