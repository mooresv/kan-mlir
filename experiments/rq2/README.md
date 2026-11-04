# RQ2: Performance portability of equivalent KAN representations

> **Research question:** Is there a universally preferable computational representation, or does the best representation depend on model shape, workload, and hardware?

This directory documents experiments comparing GPU implementations of learned KAN edge functions. RQ1 investigates whether alternative representations preserve model behavior; RQ2 measures the execution costs of those representations and their implementation choices. RQ3 will investigate compiler-directed selection subject to an error bound.

## Current scope and status

The current measurements are for **one KAN configuration on one NVIDIA H100 GPU**. They demonstrate meaningful differences between representation and lowering strategies, but **do not yet establish performance portability across model shapes or hardware**. All results below are preliminary single-sweep measurements; repeat runs and report variability before publication.

The optimized reciprocal implementation is currently **benchmark-specialized**: its compiler lowering uses `invH = 5.0f`, assuming uniform knot spacing `h = 0.2`. This assumption has not yet been validated for arbitrary trained models, and the implementation must not be treated as a general-purpose lowering without a guard or fallback.

## Experimental configuration

| Parameter | Value |
| --- | --- |
| GPU | NVIDIA H100 80 GB HBM3 (`sm_90`) |
| Input dimensions (`DIN`) | 8 |
| Output dimensions (`DOUT`) | 32 |
| Polynomial pieces | 16 |
| Polynomial degree | 3 (4 coefficients per piece) |
| Boundary tensor | `[8, 17]` |
| Original coefficient layout | `[DOUT, DIN, PIECES, NCOEFF]` |
| Transposed coefficient layout | `[DIN, PIECES, NCOEFF, DOUT]` |
| Batch sizes | 64, 256, 1024, 4096, 8192 |
| CUDA block size | 256 threads |
| Warmup / timed iterations | 100 / 1000 |
| Timing | GPU kernel time; excludes coefficient transposition and setup |

The benchmark artifact used in the experiments is `knot_poly_fits_8x32_g10.pt`, with fit key `knots_p16_d3`. Kernel execution uses one thread per `(batch, output)` element and accumulates over the eight input dimensions.

## Implementations compared

1. **Original polynomial:** piecewise cubic polynomial evaluation (Horner form), original coefficient layout, linear boundary scan.
2. **Transposed polynomial:** same polynomial evaluation and linear scan, with the output dimension contiguous in the coefficient layout.
3. **Arithmetic-division polynomial:** transposed coefficients; estimates the piece using `floor((x - b0) / h)`, clamps it, and applies boundary corrections.
4. **Reciprocal polynomial:** transposed coefficients; replaces division with multiplication by the compile-time constant `5.0f` for the assumed spacing `h = 0.2`, retaining boundary corrections.
5. **Original and transposed de Boor:** spline recurrence reference implementations, with different coefficient layouts.

Coefficient-layout transformations are implementation changes within a representation; polynomial/Horner versus de Boor is a representation/evaluation choice. The distinction matters when interpreting the research question.

## Preliminary H100 kernel latency

All times are **microseconds per kernel launch** (lower is better). Measurements were collected in separate benchmark sweeps.

| Batch | Original polynomial | Transposed polynomial | Arithmetic division | Original de Boor | Transposed de Boor | Reciprocal polynomial |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 7.451 | 5.038 | 5.515 | 5.294 | 5.000 | **3.707** |
| 256 | 7.608 | 5.273 | 5.734 | 5.683 | 5.246 | **3.809** |
| 1024 | 7.794 | 5.418 | 5.549 | 5.769 | 5.170 | **4.016** |
| 4096 | 21.872 | 6.449 | 5.829 | 9.357 | 5.683 | **4.703** |
| 8192 | 41.629 | 9.156 | 8.853 | 15.428 | 8.612 | **6.860** |

The reciprocal results above are rounded from the latest sweep:

| Batch | Kernel time (µs) | Max absolute error | RMS error |
| ---: | ---: | ---: | ---: |
| 64 | 3.706559896 | 0 | 0 |
| 256 | 3.809087992 | 0 | 0 |
| 1024 | 4.015999794 | 0 | 0 |
| 4096 | 4.702976227 | 0 | 0 |
| 8192 | 6.860064030 | 0 | 0 |

At batch 8192, reciprocal polynomial evaluation is approximately **6.07× faster** than the original polynomial kernel and **1.26× faster** than transposed de Boor. At batch 4096, it is approximately **1.24× faster** than the arithmetic-division variant. These are ratios of reported single-sweep times, not statistical confidence intervals.

### Numerical validation

The benchmark compares outputs against `expected.bin`. The reciprocal kernel reported zero maximum absolute error and zero RMS error for all five tested batch sizes. This is **not** a proof of equivalence for arbitrary inputs. In particular, test exact knot boundaries, adjacent floating-point values, out-of-range inputs, NaNs, and nonuniform knot configurations before claiming general correctness. The float-to-integer conversion path also deserves explicit exceptional-input testing.

### PTX observations

Inspection of the batch-4096 reciprocal kernel with `cuobjdump --dump-ptx` showed no `div.` or `rcp.` instructions. Eight `cvt.rmi.f32.f32` instructions were observed, consistent with floor operations across the eight input dimensions. PTX inspection alone does not establish final SASS instruction costs; use Nsight Compute or SASS inspection for deeper attribution.

## Reproducing the current reciprocal experiment

Run these commands **from the repository root**, after configuring/building the compiler and benchmark harness for the H100 environment:

```bash
cmake --build build --target kan-opt -j 8
bash test/run_reciprocal_abi_sweep.sh
cat /tmp/kan_reciprocal_abi_sweep/results.csv
```

The current scripts remain under `test/` in the `stable-gpu-abi` branch. The reciprocal sweep writes generated MLIR, a fatbin, and a CSV under `/tmp/kan_reciprocal_abi_sweep/`. Its benchmark harness uses `KAN_TRANSPOSE_COEFFS=1` and selects a `kan_piecewise_reciprocal_b...` kernel through `KAN_KERNEL_NAME`.

For PTX inspection of the batch-4096 artifact:

```bash
cuobjdump --dump-ptx /tmp/kan_reciprocal_abi_sweep/kan_4096.fatbin \
  > /tmp/kan_reciprocal_4096.ptx
grep -nE 'div\.|rcp\.|mul\.f32|fma\.|cvt\.rmi' \
  /tmp/kan_reciprocal_4096.ptx
```

**Historical sweep caveat:** `test/run_stable_abi_sweep.sh`, `test/run_transposed_abi_sweep.sh`, and `test/run_arithmetic_abi_sweep.sh` are retained as experimental scripts, but the compiler currently emits the reciprocal implementation. Those scripts **do not independently recreate their historical kernel variants from the current checkout**. Reproduction requires the corresponding historical lowering revision or a future compiler option to select the lowering strategy. Do not interpret merely rerunning all four scripts on the current checkout as a valid comparative experiment.

## Next experiments

- **Reproducibility:** capture exact Git commit, CUDA/toolchain versions, GPU clock/power settings, and multiple independent timing repetitions; report dispersion.
- **Correctness:** verify uniform spacing in exported artifacts, test knot-boundary neighborhoods and exceptional inputs, and add a safe fallback for nonuniform knots.
- **Selectable lowerings:** expose linear scan, arithmetic division, and guarded reciprocal as distinct compiler strategies, rather than editing the lowering between runs.
- **Shape sweep:** vary batch size, `DIN`, `DOUT`, knot/piece count, and polynomial degree; record kernel-only and end-to-end latency.
- **Hardware sweep:** repeat equivalent experiments on other GPU architectures and CPU targets where supported.
- **Profiling:** investigate memory transactions, occupancy, instruction mix, register pressure, and the two remaining dependent boundary-correction loads.
- **RQ3 handoff:** use the resulting performance and accuracy measurements to inform a compiler selection policy with explicit error constraints.

## Repository organization

This README is the starting point for RQ2. The experimental sweep scripts currently live in `test/`, and generated outputs live under `/tmp/`. A follow-up organizational change can move research-specific drivers and standalone de Boor benchmarks into `experiments/rq2/` while leaving shared correctness tests and utility scripts in `test/`; adjust script paths and verify execution as part of that move.
