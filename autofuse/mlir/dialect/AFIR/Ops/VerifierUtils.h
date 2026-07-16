/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_OPS_VERIFIERUTILS_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_OPS_VERIFIERUTILS_H

#include "AFIR/AFIR.h"

namespace mlir::afir {

struct UnaryShapedTypes {
  ShapedType input;
  ShapedType result;
};

inline FailureOr<UnaryShapedTypes> getUnaryShapedTypes(Operation *op) {
  if (op->getNumOperands() != 1) {
    op->emitOpError("expected 1 operand");
    return failure();
  }
  if (op->getNumResults() != 1) {
    op->emitOpError("expected 1 result");
    return failure();
  }
  auto input = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto result = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!input || !result) {
    op->emitOpError("expected tensor operand and result");
    return failure();
  }
  return UnaryShapedTypes{input, result};
}

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_OPS_VERIFIERUTILS_H
