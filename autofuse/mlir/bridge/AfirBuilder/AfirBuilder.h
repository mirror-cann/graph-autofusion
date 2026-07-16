/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AfirBuilder.h - Build an AFIR module from neutral schedule info -*- C++ -*-===//
//
// Autofuse MLIR migration demo — Step 2 bridge core.
//
// This is the *backend-neutral* half of the schedule-result bridge: it builds
// an AFIR MLIR module from plain structs (no autofuse backend types). The
// in-process adapter (ScheduleToAfir, compiled into the codegen lib) fills
// these structs directly from ascir::FusedScheduledResult and calls
// BuildAfirModule — zero serialization, all in memory.
//
// Keeping the MLIR-construction logic here (depending only on MLIR + the AFIR
// dialect) makes it buildable/testable in the standalone mlir/ build, and is
// the piece codegen-migration engineers actually need to learn.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DEMO_AFIRBUILDER_H
#define AUTOFUSE_MLIR_DEMO_AFIRBUILDER_H

#include <cstdint>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace af_demo {

// A tensor's static type + memory placement + logical axis ids.
struct TensorInfo {
  std::vector<int64_t> shape;  // static dims (ShapedType::kDynamic for dynamic)
  std::string dtype;           // "float16" | "float32" | "int32" | "int16" | "uint8" | ...
  int64_t tensor_id = -1;
  int64_t position = 0;  // afir::Position value (0=gm,1=vector_in,2=vector_out,3=vector_calc,...)
  std::vector<int64_t> axis_ids;
  std::vector<int64_t> vectorized_axis;  // axis ids stored in the buffer (API vector-length axes)
};

// One node of the schedule graph. Inputs reference producer values by name
// (either a kernel input tensor name or an earlier node's name).
struct NodeInfo {
  std::string name;
  std::string type;  // "Load"|"Store"|"Broadcast"|"Add"|"Sub"|"Mul"|"Div"|"Cast"|"Data"|"Output"
  std::vector<std::string> input_names;
  TensorInfo output;
  std::vector<int64_t> sched_axis;  // loop axis ids, outer -> inner (SchedInfo.axis)
  int64_t loop_axis = -1;           // innermost loop axis id (SchedInfo.loop_axis)
};

// A loop axis (carried into AFIR as attributes for later tiling).
struct AxisInfo {
  int64_t id = -1;
  std::string name;
  std::string type = "original";  // afir::AxisType spelling
  std::string size;               // symbolic size expression string
};

// One impl graph -> one func.func.
struct GraphInfo {
  std::string name;
  std::vector<AxisInfo> axes;
  std::vector<std::string> size_vars;
  std::vector<NodeInfo> nodes;  // topological order
  int64_t tiling_key = 0;
};

// Named kernel input/output tensors (become func args / results).
struct IoTensor {
  std::string name;
  TensorInfo info;
};

struct KernelInfo {
  std::string name;
  std::vector<IoTensor> inputs;
  std::vector<IoTensor> outputs;
  std::vector<GraphInfo> graphs;
};

// Build an AFIR module from neutral schedule info. Returns null on error
// (diagnostics are emitted to the context).
mlir::OwningOpRef<mlir::ModuleOp> BuildAfirModule(mlir::MLIRContext &context, const KernelInfo &kernel);

}  // namespace af_demo

#endif  // AUTOFUSE_MLIR_DEMO_AFIRBUILDER_H
