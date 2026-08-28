#include "KAN/KANDialect.h"
#include "KAN/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::kan::registerKANPasses();

  mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);

  registry.insert<
      mlir::kan::KANDialect,
      mlir::arith::ArithDialect,
      mlir::func::FuncDialect,
      mlir::scf::SCFDialect,
      mlir::tensor::TensorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(
          argc,
          argv,
          "KAN optimizer driver\n",
          registry));
}
