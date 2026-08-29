#include "KAN/KANDialect.h"
#include "KAN/KANOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::kan;

// Generated dialect definitions.
#include "KAN/KANDialect.cpp.inc"

// Generated operation definitions.
#define GET_OP_CLASSES
#include "KAN/KANOps.cpp.inc"

void KANDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "KAN/KANOps.cpp.inc"
      >();
}
