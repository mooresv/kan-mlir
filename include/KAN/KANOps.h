#ifndef KAN_KANOPS_H
#define KAN_KANOPS_H

#include "KAN/KANDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "KAN/KANOps.h.inc"

#endif // KAN_KANOPS_H
