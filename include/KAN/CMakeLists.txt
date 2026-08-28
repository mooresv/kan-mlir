set(LLVM_TARGET_DEFINITIONS KANDialect.td)

mlir_tablegen(KANDialect.h.inc -gen-dialect-decls)
mlir_tablegen(KANDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(KANDialectIncGen)


set(LLVM_TARGET_DEFINITIONS KANOps.td)

mlir_tablegen(KANOps.h.inc -gen-op-decls)
mlir_tablegen(KANOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(KANOpsIncGen)

add_dependencies(KANOpsIncGen KANDialectIncGen)
