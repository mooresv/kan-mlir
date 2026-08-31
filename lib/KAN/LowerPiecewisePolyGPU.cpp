#include "KAN/KANOps.h"
#include "KAN/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cstdint>
#include <memory>

using namespace mlir;

namespace {

static Value cidx(PatternRewriter &rewriter,
                  Location loc,
                  int64_t value) {
  return rewriter.create<arith::ConstantIndexOp>(loc, value);
}

static Value cf32(PatternRewriter &rewriter,
                  Location loc,
                  float value) {
  return rewriter.create<arith::ConstantFloatOp>(
      loc, rewriter.getF32Type(), APFloat(value));
}

static bool hasTerminator(Block *block) {
  if (block->empty())
    return false;

  return block->back().hasTrait<OpTrait::IsTerminator>();
}

static void setBeforeTerminatorOrEnd(
    PatternRewriter &rewriter,
    Block *block) {

  if (hasTerminator(block))
    rewriter.setInsertionPoint(&block->back());
  else
    rewriter.setInsertionPointToEnd(block);
}

static void setSCFYield(
    PatternRewriter &rewriter,
    Location loc,
    Block *block,
    ValueRange values = {}) {

  if (hasTerminator(block))
    rewriter.eraseOp(&block->back());

  rewriter.setInsertionPointToEnd(block);

  rewriter.create<scf::YieldOp>(
      loc,
      values);
}

static void setGPUTerminator(
    PatternRewriter &rewriter,
    Location loc,
    Block *block) {

  if (hasTerminator(block))
    rewriter.eraseOp(&block->back());

  rewriter.setInsertionPointToEnd(block);

  rewriter.create<gpu::TerminatorOp>(loc);
}

// Copy rank-2 tensor -> rank-2 host-shared memref.
static void copyTensor2DToMemref(
    PatternRewriter &rewriter,
    Location loc,
    Value tensor,
    Value memref,
    int64_t d0,
    int64_t d1) {

  Value c0 = cidx(rewriter, loc, 0);
  Value c1 = cidx(rewriter, loc, 1);
  Value cD0 = cidx(rewriter, loc, d0);
  Value cD1 = cidx(rewriter, loc, d1);

  auto l0 =
      rewriter.create<scf::ForOp>(
          loc, c0, cD0, c1);

  Block *body0 = l0.getBody();

  setBeforeTerminatorOrEnd(
      rewriter, body0);

  Value i0 = l0.getInductionVar();

  auto l1 =
      rewriter.create<scf::ForOp>(
          loc, c0, cD1, c1);

  Block *body1 = l1.getBody();

  setBeforeTerminatorOrEnd(
      rewriter, body1);

  Value i1 = l1.getInductionVar();

  Value v =
      rewriter.create<tensor::ExtractOp>(
          loc,
          tensor,
          ValueRange{i0, i1});

  rewriter.create<memref::StoreOp>(
      loc,
      v,
      memref,
      ValueRange{i0, i1});

  setSCFYield(
      rewriter, loc, body1);

  setSCFYield(
      rewriter, loc, body0);

  rewriter.setInsertionPointAfter(l0);
}

static void copyTensor4DToMemref(
    PatternRewriter &rewriter,
    Location loc,
    Value tensor,
    Value memref,
    int64_t d0,
    int64_t d1,
    int64_t d2,
    int64_t d3) {

  Value c0 = cidx(rewriter, loc, 0);
  Value c1 = cidx(rewriter, loc, 1);

  Value cD0 = cidx(rewriter, loc, d0);
  Value cD1 = cidx(rewriter, loc, d1);
  Value cD2 = cidx(rewriter, loc, d2);
  Value cD3 = cidx(rewriter, loc, d3);

  auto l0 =
      rewriter.create<scf::ForOp>(
          loc, c0, cD0, c1);
  Block *body0 = l0.getBody();
  setBeforeTerminatorOrEnd(rewriter, body0);
  Value i0 = l0.getInductionVar();

  auto l1 =
      rewriter.create<scf::ForOp>(
          loc, c0, cD1, c1);
  Block *body1 = l1.getBody();
  setBeforeTerminatorOrEnd(rewriter, body1);
  Value i1 = l1.getInductionVar();

  auto l2 =
      rewriter.create<scf::ForOp>(
          loc, c0, cD2, c1);
  Block *body2 = l2.getBody();
  setBeforeTerminatorOrEnd(rewriter, body2);
  Value i2 = l2.getInductionVar();

  auto l3 =
      rewriter.create<scf::ForOp>(
          loc, c0, cD3, c1);
  Block *body3 = l3.getBody();
  setBeforeTerminatorOrEnd(rewriter, body3);
  Value i3 = l3.getInductionVar();

  Value v =
      rewriter.create<tensor::ExtractOp>(
          loc,
          tensor,
          ValueRange{i0, i1, i2, i3});

  rewriter.create<memref::StoreOp>(
      loc,
      v,
      memref,
      ValueRange{i0, i1, i2, i3});

  setSCFYield(rewriter, loc, body3);
  setSCFYield(rewriter, loc, body2);
  setSCFYield(rewriter, loc, body1);
  setSCFYield(rewriter, loc, body0);

  rewriter.setInsertionPointAfter(l0);
}

// Copy rank-2 memref -> tensor.

static Value copyMemref2DToTensor(
    PatternRewriter &rewriter,
    Location loc,
    Value memref,
    RankedTensorType tensorTy,
    int64_t d0,
    int64_t d1) {

  Value c0 = cidx(rewriter, loc, 0);
  Value c1 = cidx(rewriter, loc, 1);
  Value cD0 = cidx(rewriter, loc, d0);
  Value cD1 = cidx(rewriter, loc, d1);

  Value empty =
      rewriter.create<tensor::EmptyOp>(
          loc,
          ArrayRef<int64_t>{d0, d1},
          tensorTy.getElementType());

  auto l0 =
      rewriter.create<scf::ForOp>(
          loc,
          c0,
          cD0,
          c1,
          ValueRange{empty});

  Block *body0 = l0.getBody();

  setBeforeTerminatorOrEnd(
      rewriter, body0);

  Value i0 = l0.getInductionVar();
  Value outerTensor =
      l0.getRegionIterArgs()[0];

  auto l1 =
      rewriter.create<scf::ForOp>(
          loc,
          c0,
          cD1,
          c1,
          ValueRange{outerTensor});

  Block *body1 = l1.getBody();

  setBeforeTerminatorOrEnd(
      rewriter, body1);

  Value i1 = l1.getInductionVar();
  Value innerTensor =
      l1.getRegionIterArgs()[0];

  Value v =
      rewriter.create<memref::LoadOp>(
          loc,
          memref,
          ValueRange{i0, i1});

  Value updated =
      rewriter.create<tensor::InsertOp>(
          loc,
          v,
          innerTensor,
          ValueRange{i0, i1});

  setSCFYield(
      rewriter,
      loc,
      body1,
      ValueRange{updated});

  setSCFYield(
      rewriter,
      loc,
      body0,
      ValueRange{l1.getResult(0)});

  rewriter.setInsertionPointAfter(l0);

  return l0.getResult(0);
}

class LowerPiecewisePolyLinearGPUPattern
    : public OpRewritePattern<kan::PiecewisePolyLinearOp> {
public:
  using OpRewritePattern<
      kan::PiecewisePolyLinearOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(kan::PiecewisePolyLinearOp op,
                  PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    Value input = op.getInput();
    Value boundaries = op.getBoundaries();
    Value coefficients = op.getCoefficients();

    auto inputTy =
        dyn_cast<RankedTensorType>(input.getType());
    auto boundariesTy =
        dyn_cast<RankedTensorType>(boundaries.getType());
    auto coefficientsTy =
        dyn_cast<RankedTensorType>(coefficients.getType());
    auto outputTy =
        dyn_cast<RankedTensorType>(op.getOutput().getType());

    if (!inputTy || !boundariesTy ||
        !coefficientsTy || !outputTy)
      return rewriter.notifyMatchFailure(
          op, "all operands/results must be ranked tensors");

    if (inputTy.getRank() != 2 ||
        boundariesTy.getRank() != 2 ||
        coefficientsTy.getRank() != 4 ||
        outputTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "unexpected operand/result rank");

    if (!inputTy.hasStaticShape() ||
        !boundariesTy.hasStaticShape() ||
        !coefficientsTy.hasStaticShape() ||
        !outputTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "GPU prototype currently requires static shapes");

    Type elementTy = inputTy.getElementType();

    if (!elementTy.isF32())
      return rewriter.notifyMatchFailure(
          op, "GPU prototype currently supports f32 only");

    if (boundariesTy.getElementType() != elementTy ||
        coefficientsTy.getElementType() != elementTy ||
        outputTy.getElementType() != elementTy)
      return rewriter.notifyMatchFailure(
          op, "all tensors must have the same element type");

    const int64_t batch = inputTy.getShape()[0];
    const int64_t din = inputTy.getShape()[1];

    const int64_t dout =
        coefficientsTy.getShape()[0];
    const int64_t coeffDin =
        coefficientsTy.getShape()[1];
    const int64_t pieces =
        coefficientsTy.getShape()[2];
    const int64_t numCoefficients =
        coefficientsTy.getShape()[3];

    const int64_t degree =
        numCoefficients - 1;

    if (coeffDin != din)
      return rewriter.notifyMatchFailure(
          op, "coefficient input dimension mismatch");

    if (boundariesTy.getShape()[0] != din ||
        boundariesTy.getShape()[1] != pieces + 1)
      return rewriter.notifyMatchFailure(
          op, "boundary dimensions inconsistent");

    if (outputTy.getShape()[0] != batch ||
        outputTy.getShape()[1] != dout)
      return rewriter.notifyMatchFailure(
          op, "output dimensions inconsistent");

    if (degree < 0 || pieces < 1)
      return rewriter.notifyMatchFailure(
          op, "invalid polynomial dimensions");

    //
    // First GPU version:
    //
    // Use host_shared allocations so host staging and device execution
    // see the same underlying buffers. This is intentionally a
    // correctness implementation, not the final performance path.
    //
    auto inputMemTy =
        MemRefType::get({batch, din}, elementTy);

    auto boundsMemTy =
        MemRefType::get({din, pieces + 1}, elementTy);

    auto coeffMemTy =
        MemRefType::get(
            {dout, din, pieces, numCoefficients},
            elementTy);

    auto outputMemTy =
        MemRefType::get({batch, dout}, elementTy);

    UnitAttr hostShared =
        rewriter.getUnitAttr();

    auto inputAlloc =
        rewriter.create<gpu::AllocOp>(
            loc,
            inputMemTy,
            Type(),
            ValueRange{},
            ValueRange{},
            ValueRange{},
            hostShared);

    auto boundsAlloc =
        rewriter.create<gpu::AllocOp>(
            loc,
            boundsMemTy,
            Type(),
            ValueRange{},
            ValueRange{},
            ValueRange{},
            hostShared);

    auto coeffAlloc =
        rewriter.create<gpu::AllocOp>(
            loc,
            coeffMemTy,
            Type(),
            ValueRange{},
            ValueRange{},
            ValueRange{},
            hostShared);

    auto outputAlloc =
        rewriter.create<gpu::AllocOp>(
            loc,
            outputMemTy,
            Type(),
            ValueRange{},
            ValueRange{},
            ValueRange{},
            hostShared);

    Value inputMem = inputAlloc.getMemref();
    Value boundsMem = boundsAlloc.getMemref();
    Value coeffMem = coeffAlloc.getMemref();
    Value outputMem = outputAlloc.getMemref();

    //
    // Host staging.
    //
    copyTensor2DToMemref(
        rewriter, loc,
        input, inputMem,
        batch, din);

    copyTensor2DToMemref(
        rewriter, loc,
        boundaries, boundsMem,
        din, pieces + 1);

    copyTensor4DToMemref(
        rewriter, loc,
        coefficients, coeffMem,
        dout, din, pieces, numCoefficients);

    //
    // Launch geometry.
    //
    constexpr int64_t threadsPerBlock = 256;

    const int64_t totalOutputs =
        batch * dout;

    const int64_t blocks =
        (totalOutputs + threadsPerBlock - 1) /
        threadsPerBlock;

    Value gridX =
        cidx(rewriter, loc, blocks);

    Value one =
        cidx(rewriter, loc, 1);

    Value blockX =
        cidx(rewriter, loc, threadsPerBlock);

    auto launch =
        rewriter.create<gpu::LaunchOp>(
            loc,
            gridX, one, one,
            blockX, one, one);

    Block &kernelBody =
        launch.getBody().front();

    Value blockIdX =
        launch.getBlockIds().x;

    Value threadIdX =
        launch.getThreadIds().x;

    //
    // Insert kernel body.
    //
    setBeforeTerminatorOrEnd(
        rewriter, &kernelBody);

    Value blockBase =
        rewriter.create<arith::MulIOp>(
            loc, blockIdX, blockX);

    Value tid =
        rewriter.create<arith::AddIOp>(
            loc, blockBase, threadIdX);

    Value total =
        cidx(rewriter, loc, totalOutputs);

    Value inRange =
        rewriter.create<arith::CmpIOp>(
            loc,
            arith::CmpIPredicate::ult,
            tid,
            total);

    auto ifOp =
        rewriter.create<scf::IfOp>(
            loc,
            TypeRange{},
            inRange,
            false);

    Block *thenBody =
        &ifOp.getThenRegion().front();

    setBeforeTerminatorOrEnd(
        rewriter, thenBody);

    Value cDout =
        cidx(rewriter, loc, dout);

    Value b =
        rewriter.create<arith::DivUIOp>(
            loc, tid, cDout);

    Value o =
        rewriter.create<arith::RemUIOp>(
            loc, tid, cDout);

    Value c0 =
        cidx(rewriter, loc, 0);

    Value c1 =
        cidx(rewriter, loc, 1);

    Value cDin =
        cidx(rewriter, loc, din);

    Value zero =
        cf32(rewriter, loc, 0.0f);

    //
    // acc = 0
    // for i = 0 .. din
    //
    auto inputLoop =
        rewriter.create<scf::ForOp>(
            loc,
            c0,
            cDin,
            c1,
            ValueRange{zero});

    Block *inputBody =
        inputLoop.getBody();

    setBeforeTerminatorOrEnd(
        rewriter,
        inputBody);

    Value i =
        inputLoop.getInductionVar();

    Value acc =
        inputLoop.getRegionIterArgs()[0];

    Value x =
        rewriter.create<memref::LoadOp>(
            loc,
            inputMem,
            ValueRange{b, i});

    //
    // Piece selection:
    //
    // piece = 0
    // for p=1..pieces-1:
    //   if x >= bounds[i,p]:
    //      piece = p
    //
    Value piece =
        cidx(rewriter, loc, 0);

    for (int64_t p = 1; p < pieces; ++p) {
      Value pIdx =
          cidx(rewriter, loc, p);

      Value boundary =
          rewriter.create<memref::LoadOp>(
              loc,
              boundsMem,
              ValueRange{i, pIdx});

      Value ge =
          rewriter.create<arith::CmpFOp>(
              loc,
              arith::CmpFPredicate::OGE,
              x,
              boundary);

      piece =
          rewriter.create<arith::SelectOp>(
              loc,
              ge,
              pIdx,
              piece);
    }

    //
    // Horner evaluation.
    //
    // Coefficients are stored:
    // [output, input, piece, coefficient]
    //
    Value degreeIdx =
        cidx(rewriter, loc, degree);

    Value poly =
        rewriter.create<memref::LoadOp>(
            loc,
            coeffMem,
            ValueRange{o, i, piece, degreeIdx});

    for (int64_t k = degree - 1; k >= 0; --k) {
      Value kIdx =
          cidx(rewriter, loc, k);

      Value a =
          rewriter.create<memref::LoadOp>(
              loc,
              coeffMem,
              ValueRange{o, i, piece, kIdx});

      Value mul =
          rewriter.create<arith::MulFOp>(
              loc, poly, x);

      poly =
          rewriter.create<arith::AddFOp>(
              loc, mul, a);
    }

    Value nextAcc =
        rewriter.create<arith::AddFOp>(
            loc, acc, poly);

    setSCFYield(
        rewriter,
        loc,
        inputBody,
        ValueRange{nextAcc});

    //
    // Store one y[b,o].
    //
    rewriter.setInsertionPointAfter(inputLoop);

    rewriter.create<memref::StoreOp>(
        loc,
        inputLoop.getResult(0),
        outputMem,
        ValueRange{b, o});

    setSCFYield(
        rewriter,
        loc,
        thenBody);

    //
    // End gpu.launch.
    //
    rewriter.setInsertionPointAfter(ifOp);

    setGPUTerminator(
        rewriter,
        loc,
        &kernelBody);

    //
    // Back on host.
    //
    rewriter.setInsertionPointAfter(launch);

    Value result =
        copyMemref2DToTensor(
            rewriter,
            loc,
            outputMem,
            outputTy,
            batch,
            dout);

    //
    // Free host-shared allocations.
    //
// FIXME: gpu-to-llvm in the pinned MLIR build lowers gpu.alloc to
// mgpuMemAlloc but leaves gpu.dealloc consuming memref values through
// unrealized LLVM-descriptor-to-memref casts. Those casts prevent
// mlir-translate --mlir-to-llvmir from succeeding.
//
// For the current correctness prototype, omit gpu.dealloc. The managed
// allocations are reclaimed when the short-lived test process exits.
// Restore proper deallocation once the allocation/deallocation lowering
// is handled consistently.
//
/*    rewriter.create<gpu::DeallocOp>(
        loc,
        Type(),
        ValueRange{},
        inputMem);

    rewriter.create<gpu::DeallocOp>(
        loc,
        Type(),
        ValueRange{},
        boundsMem);

    rewriter.create<gpu::DeallocOp>(
        loc,
        Type(),
        ValueRange{},
        coeffMem);

    rewriter.create<gpu::DeallocOp>(
        loc,
        Type(),
        ValueRange{},
        outputMem);
*/

    rewriter.replaceOp(op, result);

    return success();
  }
};

struct LowerPiecewisePolyGPUPass
    : public PassWrapper<
          LowerPiecewisePolyGPUPass,
          OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerPiecewisePolyGPUPass)

  StringRef getArgument() const final {
    return "lower-kan-piecewise-poly-to-gpu";
  }

  StringRef getDescription() const final {
    return "Lower kan.piecewise_poly_linear to a GPU kernel";
  }

  void getDependentDialects(
      DialectRegistry &registry) const override {

    registry.insert<
        arith::ArithDialect,
        gpu::GPUDialect,
        memref::MemRefDialect,
        scf::SCFDialect,
        tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    patterns.add<LowerPiecewisePolyLinearGPUPattern>(
        &getContext());

    if (failed(applyPatternsGreedily(
            getOperation(),
            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace kan {

std::unique_ptr<Pass> createLowerPiecewisePolyGPUPass() {
  return std::make_unique<LowerPiecewisePolyGPUPass>();
}

void registerKANGPUPasses() {
  PassRegistration<LowerPiecewisePolyGPUPass>();
}


} // namespace kan
} // namespace mlir
