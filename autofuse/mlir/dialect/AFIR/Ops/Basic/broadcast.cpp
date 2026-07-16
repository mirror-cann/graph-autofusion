/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Ops/Basic/broadcast.cpp - AFIR broadcast op impl ----------*- C++ -*-===//

#include "AFIR/AFIR.h"
#include "AFIR/Ops/VerifierUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

LogicalResult verifyBroadcastOp(Operation *op) {
  auto types = getUnaryShapedTypes(op);
  if (failed(types)) return failure();
  ShapedType inputType = types->input;
  ShapedType resultType = types->result;

  if (inputType.hasRank() && resultType.hasRank()) {
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    const size_t inputRank = inputShape.size();
    const size_t resultRank = resultShape.size();
    if (resultRank < inputRank) {
      return op->emitOpError("result rank must be >= input rank for broadcast");
    }

    const size_t offset = resultRank - inputRank;
    for (size_t i = 0; i < inputRank; ++i) {
      const int64_t inputDim = inputShape[i];
      const int64_t resultDim = resultShape[i + offset];
      if (inputDim != ShapedType::kDynamic && resultDim != ShapedType::kDynamic && inputDim != 1 &&
          inputDim != resultDim) {
        return op->emitOpError("input shape must be broadcastable to result shape");
      }
    }
  }
  return success();
}

}  // namespace

LogicalResult BroadcastOp::verify() {
  return verifyBroadcastOp(getOperation());
}
