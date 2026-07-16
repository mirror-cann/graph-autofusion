/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Ops/Basic/data.cpp - AFIR data op implementations ---------*- C++ -*-===//

#include "AFIR/AFIR.h"
#include "AFIR/Ops/VerifierUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

LogicalResult verifyNullaryOp(Operation *op) {
  if (op->getNumOperands() != 0) {
    return op->emitOpError("expected 0 operands");
  }
  if (op->getNumResults() != 1) {
    return op->emitOpError("expected 1 result");
  }
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!resultType) {
    return op->emitOpError("expected tensor result");
  }
  return success();
}

// load/store verifier. Unlike unary *elementwise* math ops (verified in math.cpp),
// load/store reshape across the tiling boundary: a load maps a GM tensor at its
// logical rank onto a merged/tiled local view, and a store maps back. The
// scheduler (which runs before ScheduleToAfir) legitimately assigns operand and
// result DIFFERENT ranks/shapes here — e.g. GM <?x?x?xf16> load → tiled
// <?x?x?xf16> over merged+tiled axes. So we only require both sides be shaped
// tensors of the same element type; rank/shape may differ by design.
LogicalResult verifyUnaryOp(Operation *op) {
  auto types = getUnaryShapedTypes(op);
  if (failed(types)) return failure();
  ShapedType inputType = types->input;
  ShapedType resultType = types->result;
  if (inputType.getElementType() != resultType.getElementType()) {
    return op->emitOpError("operand and result must have the same element type");
  }
  return success();
}

}  // namespace

LogicalResult ScalarOp::verify() {
  return verifyNullaryOp(getOperation());
}
LogicalResult IndexExprOp::verify() {
  return verifyNullaryOp(getOperation());
}
LogicalResult LoadOp::verify() {
  return verifyUnaryOp(getOperation());
}
LogicalResult StoreOp::verify() {
  return verifyUnaryOp(getOperation());
}
