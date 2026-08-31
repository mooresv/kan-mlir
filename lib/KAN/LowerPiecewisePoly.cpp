#include "KAN/KANOps.h"
#include "KAN/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>

using namespace mlir;

namespace {

static Value createIndexConstant(PatternRewriter &rewriter,
                                 Location loc,
                                 int64_t value) {
  return rewriter.create<arith::ConstantIndexOp>(loc, value);
}

class LowerPiecewisePolyLinearPattern
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

    if (inputTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "input must have rank 2");

    if (boundariesTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "boundaries must have rank 2");

    if (coefficientsTy.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "coefficients must have rank 4");

    if (outputTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "output must have rank 2");

    if (!inputTy.hasStaticShape() ||
        !boundariesTy.hasStaticShape() ||
        !coefficientsTy.hasStaticShape() ||
        !outputTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op,
          "prototype lowering currently requires static shapes");
    }

    Type elementTy = inputTy.getElementType();

    if (!elementTy.isF32())
      return rewriter.notifyMatchFailure(
          op, "prototype currently supports f32 only");

    if (boundariesTy.getElementType() != elementTy ||
        coefficientsTy.getElementType() != elementTy ||
        outputTy.getElementType() != elementTy) {
      return rewriter.notifyMatchFailure(
          op, "all tensors must have the same element type");
    }

    const int64_t batch =
        inputTy.getShape()[0];
    const int64_t din =
        inputTy.getShape()[1];

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

    if (degree < 0 || pieces < 1)
      return rewriter.notifyMatchFailure(
          op, "invalid polynomial dimensions");

    if (coeffDin != din)
      return rewriter.notifyMatchFailure(
          op,
          "coefficient input dimension does not match input");

    if (boundariesTy.getShape()[0] != din)
      return rewriter.notifyMatchFailure(
          op,
          "boundary input dimension does not match input");

    if (boundariesTy.getShape()[1] != pieces + 1)
      return rewriter.notifyMatchFailure(
          op,
          "boundary dimension must equal pieces + 1");

    if (outputTy.getShape()[0] != batch ||
        outputTy.getShape()[1] != dout)
      return rewriter.notifyMatchFailure(
          op,
          "output shape is inconsistent with input/coefficients");

    //
    // Constants used by the loops.
    //
    Value c0 =
        createIndexConstant(rewriter, loc, 0);
    Value c1 =
        createIndexConstant(rewriter, loc, 1);

    Value cBatch =
        createIndexConstant(rewriter, loc, batch);
    Value cDout =
        createIndexConstant(rewriter, loc, dout);
    Value cDin =
        createIndexConstant(rewriter, loc, din);

    Value zero =
        rewriter.create<arith::ConstantFloatOp>(
            loc,
            rewriter.getF32Type(),
            APFloat(0.0f));

    //
    // We do not need to initialize the tensor because every [b,o]
    // location is written exactly once before it is observed.
    //
    Value initialOutput =
        rewriter.create<tensor::EmptyOp>(
            loc,
            ArrayRef<int64_t>{batch, dout},
            elementTy);

    //
    // for b = 0 .. batch
    //
    auto batchLoop =
        rewriter.create<scf::ForOp>(
            loc,
            c0,
            cBatch,
            c1,
            ValueRange{initialOutput});

    Block *batchBody =
        batchLoop.getBody();

    Value b =
        batchLoop.getInductionVar();

    Value batchTensor =
        batchLoop.getRegionIterArgs()[0];

    rewriter.setInsertionPointToEnd(batchBody);

    //
    // for o = 0 .. dout
    //
    auto outputLoop =
        rewriter.create<scf::ForOp>(
            loc,
            c0,
            cDout,
            c1,
            ValueRange{batchTensor});

    Block *outputBody =
        outputLoop.getBody();

    Value o =
        outputLoop.getInductionVar();

    Value outputTensor =
        outputLoop.getRegionIterArgs()[0];

    rewriter.setInsertionPointToEnd(outputBody);

    //
    // acc = 0
    //
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

    Value i =
        inputLoop.getInductionVar();

    Value acc =
        inputLoop.getRegionIterArgs()[0];

    rewriter.setInsertionPointToEnd(inputBody);

    //
    // x = input[b,i]
    //
    Value x =
        rewriter.create<tensor::ExtractOp>(
            loc,
            input,
            ValueRange{b, i});

    //
    // Determine the polynomial piece.
    //
    // piece = 0
    //
    // for p = 1 ... pieces-1:
    //
    //   if x >= boundaries[i,p]
    //       piece = p
    //
    // Because "pieces" is compile-time constant in this prototype,
    // this becomes a short chain of cmp/select operations.
    //
    Value piece =
        createIndexConstant(rewriter, loc, 0);

    for (int64_t p = 1; p < pieces; ++p) {

      Value pIndex =
          createIndexConstant(rewriter, loc, p);

      Value boundary =
          rewriter.create<tensor::ExtractOp>(
              loc,
              boundaries,
              ValueRange{i, pIndex});

      Value cmp =
          rewriter.create<arith::CmpFOp>(
              loc,
              arith::CmpFPredicate::OGE,
              x,
              boundary);

      piece =
          rewriter.create<arith::SelectOp>(
              loc,
              cmp,
              pIndex,
              piece);
    }

    //
    // Horner evaluation:
    //
    // p = a_degree
    //
    // for d = degree-1 ... 0
    //     p = p*x + a_d
    //
    Value degreeIndex =
        createIndexConstant(
            rewriter, loc, degree);

    Value polynomial =
        rewriter.create<tensor::ExtractOp>(
            loc,
            coefficients,
            ValueRange{
                o, i, piece, degreeIndex});

    for (int64_t d = degree - 1;
         d >= 0;
         --d) {

      Value dIndex =
          createIndexConstant(
              rewriter, loc, d);

      Value coefficient =
          rewriter.create<tensor::ExtractOp>(
              loc,
              coefficients,
              ValueRange{
                  o, i, piece, dIndex});

      Value product =
          rewriter.create<arith::MulFOp>(
              loc,
              polynomial,
              x);

      polynomial =
          rewriter.create<arith::AddFOp>(
              loc,
              product,
              coefficient);
    }

    //
    // acc += polynomial
    //
    Value newAcc =
        rewriter.create<arith::AddFOp>(
            loc,
            acc,
            polynomial);

    //
    // Explicitly terminate the input loop with the updated
    // accumulator.
    //
    rewriter.create<scf::YieldOp>(
        loc,
        ValueRange{newAcc});

    //
    // After the i loop, insert the accumulated result into
    // output[b,o].
    //
    rewriter.setInsertionPointAfter(inputLoop);

    Value finalAcc =
        inputLoop.getResult(0);

    Value updatedTensor =
        rewriter.create<tensor::InsertOp>(
            loc,
            finalAcc,
            outputTensor,
            ValueRange{b, o});

    //
    // Explicitly terminate the output loop.
    //
    rewriter.setInsertionPointToEnd(outputBody);

    rewriter.create<scf::YieldOp>(
        loc,
        ValueRange{updatedTensor});

    //
    // After the o loop, propagate the completed tensor through
    // the batch loop.
    //
    rewriter.setInsertionPointAfter(outputLoop);

    Value completedOutputRow =
        outputLoop.getResult(0);

    rewriter.setInsertionPointToEnd(batchBody);

    rewriter.create<scf::YieldOp>(
        loc,
        ValueRange{completedOutputRow});

    //
    // Replace the KAN operation with the final tensor.
    //
    rewriter.replaceOp(
        op,
        batchLoop.getResult(0));

    return success();
  }
};

class LowerKANPiecewisePolyPass
    : public PassWrapper<
          LowerKANPiecewisePolyPass,
          OperationPass<ModuleOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerKANPiecewisePolyPass)

  StringRef getArgument() const final {
    return "lower-kan-piecewise-poly";
  }

  StringRef getDescription() const final {
    return
        "Lower kan.piecewise_poly_linear to "
        "SCF, tensor, and arithmetic operations";
  }

  void getDependentDialects(
      DialectRegistry &registry) const override {

    registry.insert<
        arith::ArithDialect,
        scf::SCFDialect,
        tensor::TensorDialect>();
  }

  void runOnOperation() override {

    MLIRContext *context =
        &getContext();

    RewritePatternSet patterns(context);

    patterns.add<
        LowerPiecewisePolyLinearPattern>(
            context);

    if (failed(
            applyPatternsAndFoldGreedily(
                getOperation(),
                std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//
// Register the pass automatically when the library is loaded.
//
namespace mlir {
namespace kan {

std::unique_ptr<Pass> createLowerPiecewisePolyPass() {
  return std::make_unique<LowerKANPiecewisePolyPass>();
}

void registerKANPasses() {
  PassRegistration<LowerKANPiecewisePolyPass>();
}

} // namespace kan
} // namespace mlir
