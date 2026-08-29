module {
  func.func @piecewise_poly(
      %x : tensor<64x2xf32>,
      %bounds : tensor<2x6xf32>,
      %coeffs : tensor<8x2x5x4xf32>)
      -> tensor<64x8xf32> {

    %y = kan.piecewise_poly_linear
        %x, %bounds, %coeffs
        : tensor<64x2xf32>,
          tensor<2x6xf32>,
          tensor<8x2x5x4xf32>
          -> tensor<64x8xf32>

    return %y : tensor<64x8xf32>
  }
}
