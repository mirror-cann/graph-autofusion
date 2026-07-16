/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Ops/Basic/math.cpp - AFIR math op implementations ---------*- C++ -*-===//
//
// Verifiers + return-type inference for AFIR math ops. Ported from
// Ascend-MLIR-gsr.
//
//===----------------------------------------------------------------------===//

#include "AFIR/AFIR.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

LogicalResult verifyUnaryElementwiseOp(Operation *op) {
  auto inputType = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());

  if (!inputType || !resultType) {
    return op->emitOpError("expected tensor operand and result");
  }

  if (inputType.hasRank() && resultType.hasRank()) {
    if (inputType.getRank() != resultType.getRank()) {
      return op->emitOpError("operand and result must have the same rank");
    }
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != ShapedType::kDynamic && resultShape[i] != ShapedType::kDynamic &&
          inputShape[i] != resultShape[i]) {
        return op->emitOpError("operand and result must have compatible shapes");
      }
    }
  }
  return success();
}

LogicalResult verifyLeadingDimensions(Operation *op, ArrayRef<int64_t> largeShape, ArrayRef<int64_t> resultShape,
                                      int rankOffset) {
  for (int i = 0; i < rankOffset; ++i) {
    if (largeShape[i] != ShapedType::kDynamic && resultShape[i] != ShapedType::kDynamic &&
        resultShape[i] != largeShape[i]) {
      return op->emitOpError("operands must have compatible shapes");
    }
  }
  return success();
}

LogicalResult verifySharedDimensions(Operation *op, ArrayRef<int64_t> largeShape, ArrayRef<int64_t> smallShape,
                                     ArrayRef<int64_t> resultShape, int rankOffset) {
  for (int i = rankOffset; i < static_cast<int>(largeShape.size()); ++i) {
    const int64_t large = largeShape[i];
    const int64_t small = smallShape[i - rankOffset];
    if (large != ShapedType::kDynamic && small != ShapedType::kDynamic && large != 1 && small != 1 && large != small &&
        large != resultShape[i]) {
      return op->emitOpError("operands must have compatible shapes");
    }
  }
  return success();
}

LogicalResult verifyBinaryElementwiseOp(Operation *op) {
  auto lhsType = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto rhsType = mlir::dyn_cast<ShapedType>(op->getOperand(1).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());

  if (!lhsType || !rhsType || !resultType) {
    return op->emitOpError("expected tensor operands and result");
  }

  if (!lhsType.hasRank() || !rhsType.hasRank()) {
    return success();
  }

  SmallVector<int64_t> largeShape;
  SmallVector<int64_t> smallShape;
  if (lhsType.getRank() >= rhsType.getRank()) {
    largeShape.assign(lhsType.getShape().begin(), lhsType.getShape().end());
    smallShape.assign(rhsType.getShape().begin(), rhsType.getShape().end());
  } else {
    smallShape.assign(lhsType.getShape().begin(), lhsType.getShape().end());
    largeShape.assign(rhsType.getShape().begin(), rhsType.getShape().end());
  }

  if (!resultType.hasRank() || resultType.getRank() != static_cast<int64_t>(largeShape.size())) {
    return op->emitOpError("result rank must match the broadcasted operand rank");
  }

  const int rankOffset = static_cast<int>(largeShape.size() - smallShape.size());
  if (failed(verifyLeadingDimensions(op, largeShape, resultType.getShape(), rankOffset))) return failure();
  return verifySharedDimensions(op, largeShape, smallShape, resultType.getShape(), rankOffset);
}

template <typename OpAdaptor>
LogicalResult inferBroadcastReturnTypes(MLIRContext * /*context*/, std::optional<Location> /*location*/,
                                        OpAdaptor adaptor, Type elementType,
                                        SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
  int64_t newShapeRank = 0;
  for (auto oper : adaptor.getOperands()) {
    if (auto opType = llvm::dyn_cast<RankedTensorType>(oper.getType())) {
      newShapeRank = std::max(newShapeRank, opType.getRank());
    } else {
      inferredReturnTypes.push_back(ShapedTypeComponents(elementType));
      return success();
    }
  }

  SmallVector<int64_t> newShape(newShapeRank, 1);
  SmallVector<int64_t> operOffset(adaptor.getOperands().size());
  for (size_t i = 0; i < adaptor.getOperands().size(); ++i) {
    operOffset[i] = newShapeRank - llvm::cast<RankedTensorType>(adaptor.getOperands()[i].getType()).getRank();
  }
  for (int64_t i = newShapeRank; i != 0; --i) {
    const int64_t index = i - 1;
    for (size_t j = 0; j < adaptor.getOperands().size(); ++j) {
      if (index < operOffset[j]) {
        continue;
      }
      auto shape = llvm::cast<RankedTensorType>(adaptor.getOperands()[j].getType()).getShape();
      if (shape[index - operOffset[j]] == 1) {
        continue;
      }
      if (newShape[index] == ShapedType::kDynamic || newShape[index] == 1) {
        newShape[index] = shape[index - operOffset[j]];
      }
      if (shape[index - operOffset[j]] == ShapedType::kDynamic) {
        continue;
      }
      if (newShape[index] != shape[index - operOffset[j]]) {
        return failure();
      }
    }
  }
  inferredReturnTypes.push_back(ShapedTypeComponents(newShape, elementType));
  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Unary op verifiers
//===----------------------------------------------------------------------===//
LogicalResult AbsOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult ExpOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult LnOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult SqrtOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult RsqrtOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult ReciprocalOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult ErfOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult TanhOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult ReluOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult NegOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult SigmoidOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult LogicalNotOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult IsnanOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult IsFiniteOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult LeakyReluOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}

//===----------------------------------------------------------------------===//
// Binary op verifiers
//===----------------------------------------------------------------------===//
LogicalResult AddOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult SubOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult MulOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult DivOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult MinimumOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult MaximumOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult TrueDivOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult PowOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult BitwiseAndOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult FloorDivOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult GeluOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult SignOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LogicalOrOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LogicalAndOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

//===----------------------------------------------------------------------===//
// Compare op verifiers
//===----------------------------------------------------------------------===//
LogicalResult GeOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult EqOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult NeOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult GtOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LeOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LtOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

//===----------------------------------------------------------------------===//
// Ternary op verifier (clip_by_value)
//===----------------------------------------------------------------------===//
LogicalResult ClipByValue::verify() {
  auto inputType = mlir::dyn_cast<ShapedType>(getOperand(0).getType());
  auto minType = mlir::dyn_cast<ShapedType>(getOperand(1).getType());
  auto maxType = mlir::dyn_cast<ShapedType>(getOperand(2).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(getResult().getType());

  if (!inputType || !minType || !maxType || !resultType) {
    return emitOpError("expected tensor operands and result");
  }

  auto checkCompatible = [&](ShapedType other, const char *what) -> LogicalResult {
    if (!inputType.hasRank() || !other.hasRank()) {
      return success();
    }
    if (inputType.getRank() != other.getRank()) {
      return emitOpError("input and ") << what << " must have the same rank";
    }
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> otherShape = other.getShape();
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != ShapedType::kDynamic && otherShape[i] != ShapedType::kDynamic &&
          inputShape[i] != otherShape[i] && inputShape[i] != 1 && otherShape[i] != 1) {
        return emitOpError("input and ") << what << " must have compatible shapes";
      }
    }
    return success();
  };

  if (failed(checkCompatible(minType, "min"))) return failure();
  if (failed(checkCompatible(maxType, "max"))) return failure();
  if (failed(checkCompatible(resultType, "result"))) return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Return-type inference (InferTensorTypeAdaptor -> inferReturnTypeComponents)
//===----------------------------------------------------------------------===//

#define AFIR_INFER_BROADCAST(OP)                                                                             \
  LogicalResult OP::inferReturnTypeComponents(MLIRContext *context, ::std::optional<Location> location,      \
                                              OP##Adaptor adaptor,                                           \
                                              SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) { \
    auto tensorType = llvm::dyn_cast<TensorType>(adaptor.getOperands()[0].getType());                        \
    if (!tensorType) {                                                                                       \
      return failure();                                                                                      \
    }                                                                                                        \
    return inferBroadcastReturnTypes<OP##Adaptor>(context, location, adaptor, tensorType.getElementType(),   \
                                                  inferredReturnShapes);                                     \
  }

AFIR_INFER_BROADCAST(AddOp)
AFIR_INFER_BROADCAST(SubOp)
AFIR_INFER_BROADCAST(MulOp)
AFIR_INFER_BROADCAST(DivOp)
AFIR_INFER_BROADCAST(MinimumOp)
AFIR_INFER_BROADCAST(MaximumOp)
AFIR_INFER_BROADCAST(TrueDivOp)
AFIR_INFER_BROADCAST(PowOp)
AFIR_INFER_BROADCAST(BitwiseAndOp)
AFIR_INFER_BROADCAST(FloorDivOp)
AFIR_INFER_BROADCAST(GeluOp)
AFIR_INFER_BROADCAST(SignOp)
AFIR_INFER_BROADCAST(ClipByValue)

#undef AFIR_INFER_BROADCAST

#define AFIR_INFER_BROADCAST_LOGICAL(OP)                                                                        \
  LogicalResult OP::inferReturnTypeComponents(MLIRContext *context, ::std::optional<Location> location,         \
                                              OP##Adaptor adaptor,                                              \
                                              SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {    \
    return inferBroadcastReturnTypes<OP##Adaptor>(                                                              \
        context, location, adaptor, IntegerType::get(context, 8, IntegerType::Unsigned), inferredReturnShapes); \
  }

AFIR_INFER_BROADCAST_LOGICAL(LogicalAndOp)
AFIR_INFER_BROADCAST_LOGICAL(LogicalOrOp)
AFIR_INFER_BROADCAST_LOGICAL(GeOp)
AFIR_INFER_BROADCAST_LOGICAL(EqOp)
AFIR_INFER_BROADCAST_LOGICAL(NeOp)
AFIR_INFER_BROADCAST_LOGICAL(GtOp)
AFIR_INFER_BROADCAST_LOGICAL(LeOp)
AFIR_INFER_BROADCAST_LOGICAL(LtOp)

#undef AFIR_INFER_BROADCAST_LOGICAL
