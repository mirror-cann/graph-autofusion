/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRToAscendCQueue.h - queue-faithful AFIR lowering -------*- C++ -*-===//
//
// Autofuse MLIR migration demo — queue-faithful lowering pass declaration.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DEMO_AFIRTOASCENDCQUEUE_H
#define AUTOFUSE_MLIR_DEMO_AFIRTOASCENDCQUEUE_H

#include <memory>

namespace mlir {
class Pass;
namespace afir {

// Create the `convert-afir-to-ascendc-queue` pass: lowers a buffer-annotated
// AFIR func (load/elementwise/store) to the full queue-driven ascendc form
// (pipe + queue + alloc/enque/deque/free + global_tensor + data_copy_l2 +
// *_l2), i.e. the shape a real Ascend C kernel compiles from.
std::unique_ptr<Pass> createConvertAFIRToAscendCQueuePass();
void registerAFIRToAscendCQueuePass();

}  // namespace afir
}  // namespace mlir

#endif  // AUTOFUSE_MLIR_DEMO_AFIRTOASCENDCQUEUE_H
