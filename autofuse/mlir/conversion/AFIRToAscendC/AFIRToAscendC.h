/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRToAscendC.h - AFIR to ascendc lowering ----------------*- C++ -*-===//
//
// Autofuse MLIR migration demo — lowering pass declaration.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DEMO_AFIRTOASCENDC_H
#define AUTOFUSE_MLIR_DEMO_AFIRTOASCENDC_H

#include <memory>

namespace mlir {
class Pass;
namespace afir {

// Create the `convert-afir-to-ascendc` pass: lowers AFIR ops to PyAsc `ascendc`
// dialect ops (demo scope: binary elementwise add/sub/mul/div).
std::unique_ptr<Pass> createConvertAFIRToAscendCPass();

// Register the pass with the global PassRegistry (for af-opt CLI).
void registerAFIRToAscendCPass();

}  // namespace afir
}  // namespace mlir

#endif  // AUTOFUSE_MLIR_DEMO_AFIRTOASCENDC_H
