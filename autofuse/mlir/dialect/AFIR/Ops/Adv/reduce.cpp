/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Ops/Adv/reduce.cpp - AFIR reduce op implementation --------*- C++ -*-===//
//
// Ported from Ascend-MLIR-gsr: reduce verifiers + return-type inference
// (result drops the reduced axis).
//
//===----------------------------------------------------------------------===//

#include "AFIR/AFIR.h"
#include "AFIR/Ops/VerifierUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

LogicalResult verifyReduceOp(Operation *op) {
  auto types = getUnaryShapedTypes(op);
  if (failed(types)) return failure();
  ShapedType inputType = types->input;
  ShapedType resultType = types->result;
  if (inputType.hasRank() && resultType.hasRank()) {
    if (resultType.getRank() > inputType.getRank()) {
      return op->emitOpError("result rank must be <= input rank for reduce");
    }
  }
  return success();
}

template <typename OpAdaptor>
LogicalResult inferReduceReturnTypeComponents(MLIRContext * /*context*/, std::optional<Location> /*location*/,
                                              OpAdaptor adaptor,
                                              SmallVectorImpl<ShapedTypeComponents> &inferredReturnType) {
  auto opType = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!opType) {
    return failure();
  }
  const int64_t rank = opType.getRank();
  const int64_t axis = adaptor.getAxis();
  if (axis < 0 || axis >= rank) {
    return failure();
  }
  SmallVector<int64_t> newShape(opType.getShape().begin(), opType.getShape().end());
  newShape.erase(newShape.begin() + axis);
  inferredReturnType.push_back(ShapedTypeComponents(newShape, opType.getElementType()));
  return success();
}

}  // namespace

LogicalResult MaxOp::verify() {
  return verifyReduceOp(getOperation());
}
LogicalResult MinOp::verify() {
  return verifyReduceOp(getOperation());
}
LogicalResult SumOp::verify() {
  return verifyReduceOp(getOperation());
}
LogicalResult MeanOp::verify() {
  return verifyReduceOp(getOperation());
}
LogicalResult ProdOp::verify() {
  return verifyReduceOp(getOperation());
}
LogicalResult AnyOp::verify() {
  return verifyReduceOp(getOperation());
}
LogicalResult AllOp::verify() {
  return verifyReduceOp(getOperation());
}

#define AFIR_INFER_REDUCE(OP)                                                                                \
  LogicalResult OP::inferReturnTypeComponents(MLIRContext *context, ::std::optional<Location> location,      \
                                              OP##Adaptor adaptor,                                           \
                                              SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) { \
    return inferReduceReturnTypeComponents<OP##Adaptor>(context, location, adaptor, inferredReturnShapes);   \
  }

AFIR_INFER_REDUCE(MaxOp)
AFIR_INFER_REDUCE(MinOp)
AFIR_INFER_REDUCE(SumOp)
AFIR_INFER_REDUCE(MeanOp)
AFIR_INFER_REDUCE(ProdOp)
AFIR_INFER_REDUCE(AnyOp)
AFIR_INFER_REDUCE(AllOp)

#undef AFIR_INFER_REDUCE
