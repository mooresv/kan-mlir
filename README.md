# KAN-MLIR

Prototype MLIR compiler infrastructure for transforming and lowering
learned functions in Kolmogorov-Arnold Networks (KANs).

This repository currently focuses on a first proof of concept:
representing a KAN layer whose learned B-spline edge functions have been
transformed into piecewise polynomials, and lowering that representation
to standard MLIR dialects.

## Research Goal

A KAN layer can be written as

$$
\phi_{o,i}(x) = \sum_k c_{o,i,k} B_{i,k}(x),
$$

with layer output

$$
y_o = \sum_i \phi_{o,i}(x_i).
$$

For a degree-d B-spline, each learned edge function is a degree-d
polynomial within each knot interval. This permits an exact change of
function representation from B-spline basis evaluation to direct
piecewise-polynomial evaluation.

The compiler prototype is intended to investigate two related
transformations:

1.  **Accuracy-preserving representation transformation:** convert the
    B-spline basis expansion to a knot-aligned piecewise polynomial.
2.  **Controlled approximation:** reduce the number of pieces and/or
    change the polynomial representation to trade numerical error for
    execution cost.

Longer term, the compiler will use numerical constraints and target
hardware characteristics to select among alternative learned-function
representations and implementations.

## Current Preliminary Result

Experiments with a small KANLib layer use:

-   input features: 2
-   output features: 8
-   B-spline grid size: 5
-   spline order/degree: 3
-   pure spline path (residual branch and separate spline weights
    disabled)

For this configuration, KANLib stores coefficients with shape:

``` text
[8, 2, 8]
```

and a grid with shape:

``` text
[2, 12]
```

The manually reconstructed basis-expansion computation agrees exactly
with the KANLib layer in the initial test.

Polynomial fitting produced the following full-layer maximum absolute
errors:

  -------------------------------------------------------------------------
  Representation               Pieces             Degree     Max abs. error
  ---------------- ------------------ ------------------ ------------------
  Global                            1                  3       1.342379e-01
  polynomial                                             

  Global                            1                  5       1.026966e-01
  polynomial                                             

  Uniform                           2                  3       6.734483e-02
  piecewise                                              
  polynomial                                             

  Uniform                           2                  5       1.093743e-02
  piecewise                                              
  polynomial                                             

  Uniform                           4                  3       8.056574e-03
  piecewise                                              
  polynomial                                             

  Uniform                           4                  5       8.786395e-04
  piecewise                                              
  polynomial                                             

  Uniform                           8                  3       9.142682e-04
  piecewise                                              
  polynomial                                             

  Uniform                           8                  5       1.177043e-04
  piecewise                                              
  polynomial                                             

  **Knot-aligned                **5**              **3**   **9.238720e-07**
  piecewise                                              
  polynomial**                                           
  -------------------------------------------------------------------------

The knot-aligned cubic representation is essentially exact at FP32
precision. This is expected because a cubic B-spline combination is
itself a cubic polynomial on each knot interval.

The important compiler distinction is therefore:

``` text
B-spline basis expansion
        |
        | exact representation change
        v
knot-aligned piecewise cubic
        |
        | optional approximation
        v
lower-cost approximate representation
```

## Repository Layout

``` text
kan-mlir/
├── CMakeLists.txt
├── include/
│   └── KAN/
│       ├── CMakeLists.txt
│       ├── KANDialect.td
│       ├── KANOps.td
│       ├── KANDialect.h
│       ├── KANOps.h
│       └── Passes.h
├── lib/
│   └── KAN/
│       ├── CMakeLists.txt
│       ├── KANDialect.cpp
│       └── LowerPiecewisePoly.cpp
├── tools/
│   └── kan-opt/
│       ├── CMakeLists.txt
│       └── kan-opt.cpp
└── test/
    ├── piecewise_poly.mlir
    ├── make_concrete_test.py
    ├── concrete_piecewise_poly.mlir
    ├── make_kanlib_concrete_test.py
    └── kanlib_concrete_piecewise_poly.mlir
```

## Initial KAN Dialect Operation

The first operation is:

``` mlir
kan.piecewise_poly_linear
```

For the current test case, its types are:

``` mlir
%x      : tensor<64x2xf32>
%bounds : tensor<2x6xf32>
%coeffs : tensor<8x2x5x4xf32>
%result : tensor<64x8xf32>
```

A complete example is:

``` mlir
module {
  func.func @piecewise_poly(
      %x : tensor<64x2xf32>,
      %bounds : tensor<2x6xf32>,
      %coeffs : tensor<8x2x5x4xf32>)
      -> tensor<64x8xf32> {

    %y = kan.piecewise_poly_linear
        %x, %bounds, %coeffs
        : tensor<64x2xf32>,
          tensor<2x6xf32>,
          tensor<8x2x5x4xf32>
          -> tensor<64x8xf32>

    return %y : tensor<64x8xf32>
  }
}
```

### Semantics

For batch element `b`, output feature `o`, and input feature `i`, the
operation selects a piece `r` according to `x[b,i]` and evaluates

$$
p_{o,i,r}(x) =
a_0 + a_1x + \cdots + a_dx^d.
$$

It then computes

$$
y_{b,o} = \sum_i p_{o,i,r}(x_{b,i}).
$$

The coefficient tensor is organized as

``` text
[output_feature, input_feature, piece, coefficient]
```

with coefficients stored in ascending power order:

``` text
[a0, a1, ..., ad]
```

The number of pieces and polynomial degree are inferred from tensor
shapes rather than stored as redundant attributes.

## First Lowering

The initial pass is:

``` text
--lower-kan-piecewise-poly
```

It lowers

``` text
kan.piecewise_poly_linear
```

to:

-   `scf`
-   `tensor`
-   `arith`

The current lowering is deliberately straightforward and serves as a
correctness and performance baseline.

Conceptually, it generates:

``` text
for b = 0 .. batch:
  for o = 0 .. output_features:
    acc = 0
    for i = 0 .. input_features:
      x = input[b,i]

      r = select_piece(x, boundaries[i])

      p = coefficients[o,i,r,degree]
      for d = degree-1 .. 0:
        p = p*x + coefficients[o,i,r,d]

      acc += p

    output[b,o] = acc
```

### Piece Selection

For the current five-piece case, interval selection is unrolled:

``` text
piece = 0
if x >= bounds[i,1]: piece = 1
if x >= bounds[i,2]: piece = 2
if x >= bounds[i,3]: piece = 3
if x >= bounds[i,4]: piece = 4
```

The MLIR lowering implements this with `arith.cmpf` and `arith.select`.

This intentionally gives us a simple baseline against which to compare
hardware-aware alternatives such as:

-   arithmetic selection for uniform intervals,
-   binary-search selection,
-   exponent-derived interval selection,
-   specialized selection for known knot structures.

Polynomial evaluation currently uses Horner's method with `arith.mulf`
and `arith.addf`. FMA generation and GPU-specific optimization are
deferred to later lowering/optimization stages.

## MLIR Environment

The current MLIR installation comes from the existing torch-mlir unified
build:

``` text
MLIR_DIR=/src/torch-mlir/build-unified/lib/cmake/mlir
LLVM_DIR=/src/torch-mlir/build-unified/lib/cmake/llvm
```

CMake has successfully located:

``` text
/src/torch-mlir/build-unified/lib/cmake/mlir/MLIRConfig.cmake
/src/torch-mlir/build-unified/lib/cmake/llvm/LLVMConfig.cmake
```

The Python/KANLib environment is:

``` text
/opt/kanenv
```

Activate it with:

``` bash
source /opt/kanenv/bin/activate
```

The environment uses Python 3.13 and was created because KANLib requires
Python 3.13 or later.

**Important:** `/opt/kanenv` is inside the development Docker container.
It survives stopping and restarting that container, but will not appear
in a newly created container.

## Recovering the Development Container

On the DGX host, list containers:

``` bash
docker ps -a --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}'
```

Restart the development container if necessary:

``` bash
docker start <container-name-or-id>
docker exec -it <container-name-or-id> bash
```

Then reactivate the environment:

``` bash
source /opt/kanenv/bin/activate
```

## Configuring KAN-MLIR

Always configure from the top-level `kan-mlir` directory, **not** from
`tools/kan-opt`.

``` bash
cd /src/kan-mlir
rm -rf build
mkdir build
cd build

cmake -G Ninja .. \
  -DMLIR_DIR=/src/torch-mlir/build-unified/lib/cmake/mlir \
  -DLLVM_DIR=/src/torch-mlir/build-unified/lib/cmake/llvm
```

Configuration has successfully completed with this setup.

## Building `kan-opt`

The next target is:

``` bash
cd /src/kan-mlir/build
ninja kan-opt
```

### Current Build Status

`kan-opt` now builds successfully against the torch-mlir unified MLIR
build. The KAN dialect parses and prints correctly, and the
`--lower-kan-piecewise-poly` pass is registered and runs successfully.

The torch-mlir unified build needed the `MLIRMlirOptMain` target to be
built once:

``` bash
ninja -C /src/torch-mlir/build-unified MLIRMlirOptMain
```

After that:

``` bash
cd /src/kan-mlir/build
ninja kan-opt
```

## Testing

Once `kan-opt` builds, first check that the pass is registered:

``` bash
./tools/kan-opt/kan-opt --help | grep kan
```

Then test parsing without lowering:

``` bash
./tools/kan-opt/kan-opt \
  ../test/piecewise_poly.mlir
```

Then run the first lowering:

``` bash
./tools/kan-opt/kan-opt \
  ../test/piecewise_poly.mlir \
  --lower-kan-piecewise-poly
```

Successful lowering should eliminate:

``` text
kan.piecewise_poly_linear
```

and produce IR containing operations such as:

``` text
tensor.empty
scf.for
tensor.extract
arith.cmpf
arith.select
arith.mulf
arith.addf
tensor.insert
```

## Synthetic End-to-End Semantic Validation

The current lowering has been validated by compiling and executing a
small concrete test case independently checked in Python. The test uses
batch size 2, 2 input features, 2 output features, 5 pieces, degree 3,
and boundaries `[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]`.

The input

``` text
[[-0.8,  0.1],
 [ 0.7, -0.4]]
```

selects pieces `[0, 2]` and `[4, 1]`. The independent Python reference
is:

``` text
[[11.4, 52.85],
 [14.9, 54.1]]
```

Generate and lower the test:

``` bash
cd /src/kan-mlir
source /opt/kanenv/bin/activate
python test/make_concrete_test.py

cd build
./tools/kan-opt/kan-opt \
  ../test/concrete_piecewise_poly.mlir \
  --lower-kan-piecewise-poly \
  -o concrete_lowered.mlir
```

Bufferize and lower structured control flow:

``` bash
./tools/kan-opt/kan-opt \
  concrete_lowered.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  -o concrete_bufferized.mlir

./tools/kan-opt/kan-opt \
  concrete_bufferized.mlir \
  --convert-scf-to-cf \
  -o concrete_cf.mlir
```

For the current MLIR build, explicit LLVM conversion passes are used:

``` bash
./tools/kan-opt/kan-opt \
  concrete_cf.mlir \
  --convert-arith-to-llvm \
  --convert-index-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o concrete_llvm.mlir
```

The generic `--convert-to-llvm` pass is not currently used because this
custom driver does not yet register the conversion-interface extension
promised by the `ub` dialect.

Translate to LLVM IR and compile:

``` bash
mlir-translate --mlir-to-llvmir concrete_llvm.mlir -o concrete.ll
llc -filetype=obj concrete.ll -o concrete.o
```

The test declares an external `print_f32` helper:

``` c
#include <stdio.h>

void print_f32(float x)
{
    printf("%.8f\n", x);
}
```

Compile, link, and execute with:

``` bash
gcc -c print_f32.c -o print_f32.o
gcc -no-pie concrete.o print_f32.o -o concrete_test
./concrete_test
```

The compiled MLIR computation produces:

``` text
11.39999962
52.84999847
14.89999962
54.09999847
```

These values agree with the Python reference to FP32 rounding. The
validated path is therefore:

``` text
kan.piecewise_poly_linear
        |
        v
SCF + tensor + arith
        |
        v
bufferized memref IR
        |
        v
control-flow + LLVM dialect
        |
        v
LLVM IR
        |
        v
native x86 executable
```

This establishes that the initial KAN representation lowering is not
only syntactically valid MLIR but preserves the intended numerical
semantics on a concrete executable test.

## KANLib-Derived End-to-End Validation

The compiler path has also been validated using an actual KANLib-derived
knot-aligned cubic representation rather than synthetic coefficients.
The saved fitting artifact contains the original KANLib B-spline grid
and source coefficients together with the fitted knot-aligned
representation:

``` text
source grid             [2, 12]
source coefficients     [8, 2, 8]
piece boundaries        [2, 6]
polynomial coefficients [8, 2, 5, 4]
```

The five knot intervals are defined by:

``` text
[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
```

`test/make_kanlib_concrete_test.py` reconstructs the original KANLib
spline layer from the saved grid and coefficients, evaluates it on a
deterministic 8-by-2 input batch, evaluates the saved knot-aligned cubic
representation on the same inputs, and emits a self-contained MLIR test.

For this deterministic batch, the original KANLib B-spline evaluation
and the Python knot-aligned cubic evaluation agree closely:

``` text
maximum absolute error: 2.8312206268310547e-07
RMS error:              8.069594059634255e-08
```

The generated MLIR uses the real fitted coefficient tensor in
`kan.piecewise_poly_linear` and is lowered through tensor/SCF,
bufferization, control flow, the LLVM dialect, LLVM IR, and finally a
native x86 executable.

The native executable produces all 64 output values identically to the
Python piecewise-polynomial reference at the printed eight-decimal
precision:

``` text
original KANLib B-spline
        |
        | representation transformation
        | max abs. difference on test batch: 2.83e-07
        v
Python knot-aligned cubic
        |
        | KAN dialect lowering + LLVM compilation
        | identical at printed 8-decimal precision
        v
native compiled result
```

The complete compilation sequence is:

``` bash
cd /src/kan-mlir/build

./tools/kan-opt/kan-opt \
  ../test/kanlib_concrete_piecewise_poly.mlir \
  --lower-kan-piecewise-poly \
  -o kanlib_lowered.mlir

./tools/kan-opt/kan-opt \
  kanlib_lowered.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  -o kanlib_bufferized.mlir

./tools/kan-opt/kan-opt \
  kanlib_bufferized.mlir \
  --convert-scf-to-cf \
  -o kanlib_cf.mlir

./tools/kan-opt/kan-opt \
  kanlib_cf.mlir \
  --convert-arith-to-llvm \
  --convert-index-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o kanlib_llvm.mlir

mlir-translate --mlir-to-llvmir kanlib_llvm.mlir -o kanlib.ll
llc -filetype=obj kanlib.ll -o kanlib.o
gcc -no-pie kanlib.o print_f32.o -o kanlib_test
./kanlib_test
```

This demonstrates end to end that a learned B-spline KAN layer can be
transformed to a knot-aligned polynomial function representation,
expressed in the KAN dialect, lowered through standard MLIR and LLVM
infrastructure, and executed while preserving the expected FP32
numerical behavior.

## Immediate Milestones

Completed:

1.  Build `MLIRMlirOptMain` and `kan-opt`.
2.  Parse and print `kan.piecewise_poly_linear`.
3.  Lower `kan.piecewise_poly_linear` to `scf`, `tensor`, and `arith`.
4.  Bufferize the lowered tensor computation.
5.  Lower through control flow and LLVM dialects to LLVM IR.
6.  Compile the LLVM IR to a native executable.
7.  Validate the lowering against a synthetic independently computed
    reference.
8.  Export real knot-aligned cubic coefficients derived from KANLib into
    MLIR.
9.  Reconstruct the source KANLib layer and compare it with the
    polynomial form.
10. Compile and execute the KANLib-derived MLIR test end to end.
11. Verify that the native result matches the Python polynomial
    reference.

Next:

1.  Establish a baseline GPU implementation and timing.
2.  Lower the piecewise-polynomial computation toward GPU execution.
3.  Implement hardware-aware interval-selection and
    polynomial-evaluation alternatives.
4.  Compare KANLib, baseline MLIR, and optimized MLIR implementations.
5.  Measure numerical error, execution time, memory traffic, instruction
    mix, and generated GPU code.

## Preliminary Evaluation Plan

The first performance table is expected to compare at least:

  ---------------------------------------------------------------------------------
  Representation /       Pieces       Degree        Error     GPU time      Speedup
  implementation                                                       
  ---------------- ------------ ------------ ------------ ------------ ------------
  KANLib B-spline            --            3    reference          TBD         1.0x

  Knot-aligned                5            3    \~9.24e-7          TBD          TBD
  polynomial                                                           

  MLIR baseline               5            3      matches          TBD          TBD
  lowering                                         Python              
                                               polynomial              
                                             reference at              
                                                  printed              
                                                8-decimal              
                                                precision              

  MLIR optimized              5            3          TBD          TBD          TBD
  lowering                                                             
  ---------------------------------------------------------------------------------

Useful additional measurements include:

-   kernel execution time,
-   end-to-end layer execution time,
-   intermediate storage/materialization,
-   global-memory traffic,
-   instruction mix,
-   achieved memory bandwidth,
-   achieved compute throughput,
-   interval-selection overhead.

## Longer-Term Compiler Direction

The initial operation and lowering are only a starting point. The
intended compiler structure is roughly:

``` text
trained KAN / learned computation
              |
              v
      learned-function IR
              |
              v
    range and error analysis
              |
              v
 representation generation
       and selection
              |
              v
 hardware-aware lowering
              |
              v
   GPU / accelerator code
```

The main research question is whether a compiler can treat learned
scalar functions as transformable IR objects and select function
representations and implementations according to numerical, reliability,
and hardware constraints. This distinguishes the project from work that
proposes a single alternative KAN basis or a single hand-optimized GPU
implementation.
