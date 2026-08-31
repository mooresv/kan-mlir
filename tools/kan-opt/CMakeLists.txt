add_executable(kan-opt
  kan-opt.cpp
)

target_link_libraries(kan-opt PRIVATE
  MLIRKANDialect

  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMlirOptMain
  MLIRPass
  MLIRRegisterAllExtensions

  MLIRGPUToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport

  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
)
