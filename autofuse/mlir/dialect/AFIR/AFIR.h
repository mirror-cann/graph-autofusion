/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIR.h - AFIR dialect declaration -------------------------*- C++ -*-===//
//
// Autofuse MLIR migration demo — AFIR dialect public header.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_AFIR_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_AFIR_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include dialect definition
#include "AFIR/Dialect.h.inc"

// Include enums (must be before attributes)
#include "AFIR/Enums.h.inc"

// Include attribute definitions
#define GET_ATTRDEF_CLASSES
#include "AFIR/Attrs.h.inc"

// Include operation definitions
#define GET_OP_CLASSES
#include "AFIR/Ops.h.inc"

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_AFIR_H
