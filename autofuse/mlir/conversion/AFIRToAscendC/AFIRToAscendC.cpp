/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRToAscendC.cpp - AFIR to ascendc lowering --------------*- C++ -*-===//
//
// Autofuse MLIR migration demo — lowers AFIR ops to the PyAsc `ascendc`
// dialect. Ported from Ascend-MLIR-gsr lib/Conversion/AFIRToASCIR (scope:
// binary elementwise add/sub/mul/div), written as a self-contained
// PassWrapper (no Passes.td tablegen infra) for the demo.
//
//===----------------------------------------------------------------------===//

#include "AFIRToAscendC/AFIRToAscendC.h"

#include "AFIR/AFIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Type converter: AFIR tensors -> ascendc LocalTensor
//===----------------------------------------------------------------------===//
class AFIRToAscendCTypeConverter : public TypeConverter {
 public:
  AFIRToAscendCTypeConverter() {
    addConversion([](Type type) -> std::optional<Type> {
      if (auto tensorType = dyn_cast<TensorType>(type)) {
        return ascendc::LocalTensorType::get(tensorType.getShape(), tensorType.getElementType());
      }
      return type;
    });
  }
};

//===----------------------------------------------------------------------===//
// Read the scheduled buffer position off an AFIR op's `outputs` attribute.
//
// autofuse's schedule already decided memory placement, and the bridge carries
// it on AFIR as AscTensorGroups(position). The lowering *honors* that decision
// rather than re-deriving it (codegen is a pure emitter). Unmapped/GM/default
// positions fall back to VECCALC (a compute-local buffer) for this demo; GM
// inputs would become GlobalTensor in a full implementation.
//===----------------------------------------------------------------------===//
static ascendc::TPosition readScheduledPosition(Operation *op) {
  auto outs = op->getAttrOfType<ArrayAttr>("outputs");
  if (!outs || outs.empty()) {
    return ascendc::TPosition::VECCALC;
  }
  auto group = dyn_cast<afir::AscTensorGroupsAttr>(outs[0]);
  if (!group) {
    return ascendc::TPosition::VECCALC;
  }
  switch (group.getPositionConfig().getPosition()) {
    case afir::Position::VECTOR_IN:
      return ascendc::TPosition::VECIN;
    case afir::Position::VECTOR_OUT:
      return ascendc::TPosition::VECOUT;
    case afir::Position::VECTOR_CALC:
      return ascendc::TPosition::VECCALC;
    default:
      return ascendc::TPosition::VECCALC;
  }
}

//===----------------------------------------------------------------------===//
// Binary elementwise: afir.add -> ascendc.add_l3, etc.
//===----------------------------------------------------------------------===//
template <typename AFIRBinaryOp, typename ASCBinaryOp>
struct ConvertBinaryElementwise : public ConversionPattern {
  ConvertBinaryElementwise(const TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, AFIRBinaryOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = op->getResult(0).getType();
    auto shapedType = dyn_cast<ShapedType>(resultType);
    if (!shapedType) {
      return rewriter.notifyMatchFailure(op, "expected shaped result type");
    }

    // Allocate the destination buffer at the position the schedule chose.
    auto bufferTy = ascendc::TBufType::get(op->getContext(), readScheduledPosition(op));
    Value tbuf = ascendc::TBufOp::create(rewriter, op->getLoc(), bufferTy);
    auto localTType = ascendc::LocalTensorType::get(shapedType.getShape(), shapedType.getElementType());
    Value dst = ascendc::TBufGetTensorOp::create(rewriter, op->getLoc(), localTType, tbuf, /*len=*/Value());

    ASCBinaryOp::create(rewriter, op->getLoc(), dst, operands[0], operands[1]);
    rewriter.replaceOp(op, dst);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// func.return operand type conversion
//===----------------------------------------------------------------------===//
struct ConvertFuncReturn : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct ConvertAFIRToAscendCPass : public PassWrapper<ConvertAFIRToAscendCPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAFIRToAscendCPass)

  StringRef getArgument() const final {
    return "convert-afir-to-ascendc";
  }
  StringRef getDescription() const final {
    return "Lower AFIR ops to the ascendc dialect (demo: binary elementwise)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ascendc::AscendCDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addIllegalDialect<afir::AFIRDialect>();
    target.addLegalDialect<ascendc::AscendCDialect>();

    AFIRToAscendCTypeConverter typeConverter;
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op.getOperandTypes()); });

    RewritePatternSet patterns(context);
    patterns.add<ConvertBinaryElementwise<afir::AddOp, ascendc::AddL3Op>>(typeConverter, context);
    patterns.add<ConvertBinaryElementwise<afir::SubOp, ascendc::SubL3Op>>(typeConverter, context);
    patterns.add<ConvertBinaryElementwise<afir::MulOp, ascendc::MulL3Op>>(typeConverter, context);
    patterns.add<ConvertBinaryElementwise<afir::DivOp, ascendc::DivL3Op>>(typeConverter, context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    patterns.add<ConvertFuncReturn>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mlir {
namespace afir {

std::unique_ptr<Pass> createConvertAFIRToAscendCPass() {
  return std::make_unique<ConvertAFIRToAscendCPass>();
}

void registerAFIRToAscendCPass() {
  PassRegistration<ConvertAFIRToAscendCPass>();
}

}  // namespace afir
}  // namespace mlir
