/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCIR_H__
#define __ASCIR_H__

#include <cstdint>
#include <memory>
#include <vector>

#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/symbolizer/symbolic.h"

namespace ascir {
  using Graph = af::AscGraph;
  using HintGraph = af::AscGraph;
  using ImplGraph = af::AscGraph;
  using NodeView = af::AscNodePtr;
  using NodeViewVisitorConst = af::AscNodeVisitor;
  using AxisId = af::AxisId;
  using Axis = af::Axis;
  using TensorAttr = af::AscTensor;
  using SizeExpr = af::Expression;
  //using TensorView = af::AscTensorAttr*;
  using TensorView = af::AscTensor;
  using TensorPtr = af::AscTensorAttr*;
  using SizeVar = af::SizeVar;

  // enum
  using ComputeType = af::ComputeType;
  using ComputeUnit = af::ComputeUnit;
  using ApiType = af::ApiType;
  using AllocType = af::AllocType;
  using MemHardware = af::MemHardware;
  using Position = af::Position;

  // id  int64_t
  using Identifier = int64_t;
  using TensorId = Identifier;
  using BufId = Identifier;
  using QueId = Identifier;
  using MergeScopeId = Identifier;
  using ReuseId = Identifier;

}

#endif