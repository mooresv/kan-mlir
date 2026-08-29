#ifndef KAN_DIALECT
#define KAN_DIALECT

include "mlir/IR/DialectBase.td"

def KAN_Dialect : Dialect {
  let name = "kan";
  let cppNamespace = "::mlir::kan";

  let summary = "Dialect for Kolmogorov--Arnold Network compilation";

  let description = [{
    The KAN dialect represents learned functions and compiler-selected
    function representations for Kolmogorov--Arnold Networks.

    The initial prototype contains a piecewise-polynomial linear operator
    representing a KAN layer whose learned edge functions have been
    converted to piecewise polynomial form.
  }];

}

#endif // KAN_DIALECT
