#include "KAN/KANDialect.h"
#include "KAN/KANOps.h"

using namespace mlir;
using namespace mlir::kan;

#include "KAN/KANDialect.cpp.inc"

void KANDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "KAN/KANOps.cpp.inc"
      >();
}
