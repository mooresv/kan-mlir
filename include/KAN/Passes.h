#ifndef KAN_PASSES_H
#define KAN_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace kan {

std::unique_ptr<Pass> createLowerPiecewisePolyPass();

void registerKANPasses();

} // namespace kan
} // namespace mlir

#endif
