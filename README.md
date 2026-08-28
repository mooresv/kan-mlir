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

\[ y_o = `\sum`{=tex}*i `\phi`{=tex}*{o,i}(x_i). \]

For a degree-(d) B-spline, each learned edge function is a degree-(d)
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
    └── piecewise_poly.mlir
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

\[ p\_{o,i,r}(x) = a_0 + a_1x + `\cdots `{=tex}+ a_dx\^d. \]

It then computes

\[ y\_{b,o} = `\sum`{=tex}*i p*{o,i,r}(x\_{b,i}). \]

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

CMake configuration succeeds, but the most recent build stopped with:

``` text
ninja: error:
'/src/torch-mlir/build-unified/lib/libMLIRMlirOptMain.a',
needed by 'tools/kan-opt/kan-opt',
missing and no known rule to make it
```

The exported MLIR CMake configuration references `MLIRMlirOptMain`, but
that library has apparently not yet been built in the torch-mlir unified
build.

The next step is therefore:

``` bash
ninja -C /src/torch-mlir/build-unified MLIRMlirOptMain
```

Verify that the library exists:

``` bash
ls -l /src/torch-mlir/build-unified/lib/libMLIRMlirOptMain*
```

Then retry:

``` bash
cd /src/kan-mlir/build
ninja kan-opt
```

If additional MLIR libraries are reported as missing, build the
corresponding targets in `/src/torch-mlir/build-unified` and retry.

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

## Immediate Milestones

The short-term path is:

1.  Build `MLIRMlirOptMain`.
2.  Build `kan-opt`.
3.  Parse the test `kan.piecewise_poly_linear` operation.
4.  Run `--lower-kan-piecewise-poly`.
5.  Fix any API differences between the prototype code and the exact
    MLIR version in the torch-mlir build.
6.  Export the already-generated knot-aligned cubic coefficients from
    Python into an MLIR test case.
7.  Execute the lowered computation and compare its output against
    KANLib.
8.  Establish the baseline GPU implementation and timing.
9.  Implement one or more hardware-aware lowering alternatives.
10. Compare numerical error, execution time, and generated GPU code.

## Preliminary Evaluation Plan

The first performance table is expected to compare at least:

  ---------------------------------------------------------------------------------
  Representation /       Pieces       Degree        Error     GPU time      Speedup
  implementation                                                       
  ---------------- ------------ ------------ ------------ ------------ ------------
  KANLib B-spline            --            3    reference          TBD         1.0x

  Knot-aligned                5            3    \~9.24e-7          TBD          TBD
  polynomial                                                           

  MLIR baseline               5            3          TBD          TBD          TBD
  lowering                                                             

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

The central research question is not simply whether a particular
polynomial KAN implementation can be made fast. Rather, it is whether a
compiler can treat learned scalar functions as transformable IR objects
and select function representations and implementations according to
numerical, interpretability, and hardware constraints.

This distinguishes the project from work that proposes a single
alternative KAN basis or a single hand-optimized GPU implementation.
