/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ScheduleToAfir.h - FusedScheduledResult → AFIR (in-process) --*- C++ -*-===//
//
// Autofuse MLIR migration demo — in-process direct bridge.
//
// Converts a real ascir::FusedScheduledResult (the codegen input, held in
// memory by Codegen::Generate) into an AFIR MLIR module, with zero
// serialization. This walks the graph structure directly in-process and
// produces MLIR ops via AfirBuilder.
//
// Hooked at codegen.cpp's Codegen::GenerateForInductor, gated by
// ENABLE_AUTOFUSE_MLIR. Default off (no legacy behavior change);
// dumping is driven by the AF_MLIR_AFIR_DUMP_DIR env var.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DEMO_SCHEDULETOAFIR_H
#define AUTOFUSE_MLIR_DEMO_SCHEDULETOAFIR_H

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "schedule_result.h"  // ascir::FusedScheduledResult

namespace af_demo {

// Build an AFIR module directly from a real FusedScheduledResult.
// Returns null on error (diagnostics emitted to the context).
mlir::OwningOpRef<mlir::ModuleOp> ScheduleResultToAfir(mlir::MLIRContext &context,
                                                       const ascir::FusedScheduledResult &result);

// Convenience entry for the codegen hook: if AF_MLIR_AFIR_DUMP_DIR is set,
// convert `result` to AFIR and dump `module.print()` to a file under that dir.
// No-op when the env var is unset. Never throws into the codegen path.
void MaybeDumpAfirFromSchedule(const std::string &stage, const ascir::FusedScheduledResult &result);

}  // namespace af_demo

#endif  // AUTOFUSE_MLIR_DEMO_SCHEDULETOAFIR_H
