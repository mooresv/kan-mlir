add_mlir_library(MLIRKANDialect
  KANDialect.cpp
  LowerPiecewisePoly.cpp
  LowerPiecewisePolyGPU.cpp

  DEPENDS
  KANDialectIncGen
  KANOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRMathDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransformUtils
)
