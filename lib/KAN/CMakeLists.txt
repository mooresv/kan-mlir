add_mlir_library(MLIRKANDialect
  KANDialect.cpp
  LowerPiecewisePoly.cpp

  DEPENDS
  KANDialectIncGen
  KANOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransformUtils
)
