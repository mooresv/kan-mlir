module {
  func.func @gpu_smoke(%arg0 : memref<1024xf32>) {
    %c1 = arith.constant 1.0 : f32
    %c1024 = arith.constant 1024 : index
    %c256 = arith.constant 256 : index
    %c1_idx = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    gpu.launch
        blocks(%bx, %by, %bz) in (%gx = %c4, %gy = %c1_idx, %gz = %c1_idx)
        threads(%tx, %ty, %tz) in (%bxsz = %c256,
                                   %bysz = %c1_idx,
                                   %bzsz = %c1_idx) {

      %base = arith.muli %bx, %c256 : index
      %idx = arith.addi %base, %tx : index

      %inside = arith.cmpi ult, %idx, %c1024 : index

      scf.if %inside {
        %v = memref.load %arg0[%idx] : memref<1024xf32>
        %r = arith.addf %v, %c1 : f32
        memref.store %r, %arg0[%idx] : memref<1024xf32>
      }

      gpu.terminator
    }

    return
  }
}
