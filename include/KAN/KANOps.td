#ifndef KAN_OPS
#define KAN_OPS

include "KAN/KANDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class KAN_Op<string mnemonic, list<Trait> traits = []>
    : Op<KAN_Dialect, mnemonic, traits>;

def KAN_PiecewisePolyLinearOp
    : KAN_Op<"piecewise_poly_linear", [Pure]> {

  let summary =
      "Evaluate a KAN layer represented by piecewise polynomials";

  let description = [{
    Evaluates a collection of learned piecewise-polynomial edge functions.

    The input has shape:

      [batch, input_features]

    The boundary tensor has shape:

      [input_features, num_pieces + 1]

    The coefficient tensor has shape:

      [output_features, input_features,
       num_pieces, degree + 1]

    Coefficients are stored in ascending power order:

      a_0, a_1, ..., a_degree

    For input x[b,i], the operation first determines the piece r containing
    x[b,i], evaluates

      p_{o,i,r}(x) =
          a_0 + a_1*x + ... + a_degree*x^degree,

    and computes

      y[b,o] = sum_i p_{o,i,r}(x[b,i]).

    The implementation strategy is deliberately unspecified at this level.
  }];

  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$boundaries,
      AnyRankedTensor:$coefficients
  );

  let results = (outs
      AnyRankedTensor:$output
  );

  let assemblyFormat = [{
    $input `,` $boundaries `,` $coefficients
    attr-dict `:`
    type($input) `,`
    type($boundaries) `,`
    type($coefficients)
    `->` type($output)
  }];
}

#endif // KAN_OPS
