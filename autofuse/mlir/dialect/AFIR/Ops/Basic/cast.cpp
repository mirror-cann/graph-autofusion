/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Ops/Basic/cast.cpp - AFIR cast op implementation ----------*- C++ -*-===//
//
// Ported from Ascend-MLIR-gsr: verifies shape match + supported dtype cast pair.
//
//===----------------------------------------------------------------------===//

#include "AFIR/AFIR.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

enum class CastTypeId : unsigned {
  F32 = 0,
  F16 = 1,
  I64 = 2,
  I32 = 3,
  I16 = 4,
  BF16 = 5,
  I8 = 6,
  UI8 = 7,
  I4 = 8,
  UI32 = 9,
  UI16 = 10,
  UI64 = 11,
  Unknown = 999
};

CastTypeId getSignedIntegerTypeId(unsigned width) {
  switch (width) {
    case 64:
      return CastTypeId::I64;
    case 32:
      return CastTypeId::I32;
    case 16:
      return CastTypeId::I16;
    case 8:
      return CastTypeId::I8;
    case 4:
      return CastTypeId::I4;
    default:
      return CastTypeId::Unknown;
  }
}

CastTypeId getUnsignedIntegerTypeId(unsigned width) {
  switch (width) {
    case 64:
      return CastTypeId::UI64;
    case 32:
      return CastTypeId::UI32;
    case 16:
      return CastTypeId::UI16;
    case 8:
      return CastTypeId::UI8;
    default:
      return CastTypeId::Unknown;
  }
}

CastTypeId getTypeId(Type type) {
  if (auto floatTy = llvm::dyn_cast<FloatType>(type)) {
    if (floatTy.isF32()) return CastTypeId::F32;
    if (floatTy.isF16()) return CastTypeId::F16;
    if (floatTy.isBF16()) return CastTypeId::BF16;
    return CastTypeId::Unknown;
  }
  if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
    if (intTy.isUnsigned()) return getUnsignedIntegerTypeId(intTy.getWidth());
    if (intTy.isSignless() || intTy.isSigned()) return getSignedIntegerTypeId(intTy.getWidth());
  }
  return CastTypeId::Unknown;
}

const llvm::DenseSet<std::pair<CastTypeId, CastTypeId>> &getSupportedCastPairs() {
  using CT = CastTypeId;
  static const llvm::DenseSet<std::pair<CT, CT>> supportedCasts = {
      {CT::F32, CT::F32},  {CT::F32, CT::F16},  {CT::F32, CT::I64},  {CT::F32, CT::I32},  {CT::F32, CT::I16},
      {CT::F32, CT::BF16}, {CT::F16, CT::F32},  {CT::F16, CT::I32},  {CT::F16, CT::I16},  {CT::F16, CT::I8},
      {CT::F16, CT::UI8},  {CT::F16, CT::I4},   {CT::F16, CT::I64},  {CT::I4, CT::F16},   {CT::UI8, CT::F16},
      {CT::UI8, CT::F32},  {CT::UI8, CT::I32},  {CT::UI8, CT::I16},  {CT::UI8, CT::I8},   {CT::UI8, CT::I4},
      {CT::I8, CT::F16},   {CT::I8, CT::UI8},   {CT::I16, CT::F16},  {CT::I16, CT::F32},  {CT::I16, CT::UI16},
      {CT::I32, CT::F32},  {CT::I32, CT::I64},  {CT::I32, CT::I16},  {CT::I32, CT::F16},  {CT::I32, CT::UI32},
      {CT::I64, CT::I32},  {CT::BF16, CT::F32}, {CT::BF16, CT::I32}, {CT::UI32, CT::I32}, {CT::UI16, CT::I16},
      {CT::UI64, CT::I64}};
  return supportedCasts;
}

bool isSupportedCast(Type inputType, Type outputType) {
  const CastTypeId inputId = getTypeId(inputType);
  const CastTypeId outputId = getTypeId(outputType);
  if (inputId == CastTypeId::Unknown || outputId == CastTypeId::Unknown) {
    return false;
  }
  return getSupportedCastPairs().count({inputId, outputId}) > 0;
}

}  // namespace

LogicalResult CastOp::verify() {
  auto inputTensorType = mlir::dyn_cast<RankedTensorType>(getInput().getType());
  auto outputTensorType = mlir::dyn_cast<RankedTensorType>(getResult().getType());
  if (!inputTensorType || !outputTensorType) {
    return emitOpError("input and output must be ranked tensors");
  }
  if (inputTensorType.getShape() != outputTensorType.getShape()) {
    return emitOpError("input and output shapes must match, got input shape ")
           << inputTensorType.getShape() << " and output shape " << outputTensorType.getShape();
  }
  Type inputElemType = inputTensorType.getElementType();
  Type outputElemType = outputTensorType.getElementType();
  if (!isSupportedCast(inputElemType, outputElemType)) {
    return emitOpError("unsupported cast from ") << inputElemType << " to " << outputElemType;
  }
  return success();
}
