/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Bridge API that lets GE add nodes into AscGraph without instantiating af::Operator or
// af::ascir_op types in GE translation units.  All af-internal construction happens inside
// libaihac_ir.so, which is compiled without AUTOFUSE_USE_GE_METADEF.
//
// Two usage patterns in GE:
//
// Pattern A – caller already has a ge::OpDescPtr:
//   was: auto op = af::Operator(OpDescUtils::CreateOperatorFromOpDesc(op_desc));
//        asc_graph.AddNode(op);
//   now: af::AscGraphAddNodeFromOpDesc(asc_graph, op_desc);
//
// Pattern B – caller needs a specific ascir_op node (e.g. Split, Output):
//   was: std::shared_ptr<ascir_op::Split> op = std::make_shared<ascir_op::Split>(name);
//        op->InstanceOutputy(out_num);
//        asc_graph.AddNode(static_cast<af::Operator &>(*op));
//   now: af::AscGraphAddSplitNode(asc_graph, name, out_num);
//
// For other ascir_op types used in MakeRawNode<T>, the template is rewritten to call
// AscGraphAddAscirNodeByType() which dispatches by T::Type string inside libaihac_ir.so.

#ifndef METADEF_CXX_ASC_GRAPH_GE_BRIDGE_H_
#define METADEF_CXX_ASC_GRAPH_GE_BRIDGE_H_

#include <cstdint>
#include <string>
#include <vector>
#include "ascendc_ir.h"

namespace af {

// Pattern A: add a node from an existing OpDescPtr.
// The OpDesc must already have IR inputs/outputs declared (AppendIrInput / AppendIrOutput).
// AscNodeAttr is created automatically inside AscNode's constructor.
AscNodePtr AscGraphAddNodeFromOpDesc(AscGraph &asc_graph, OpDescPtr op_desc);

// Pattern B helpers for specific ascir_op types that GE constructs directly.

// Equivalent to: ascir_op::Split op(name); op.InstanceOutputy(out_num); asc_graph.AddNode(op);
AscNodePtr AscGraphAddSplitNode(AscGraph &asc_graph, const char *name, uint32_t out_num);

// Generic helper for ascir_op types used via MakeRawNode<T> in asc_overrides.h.
// Constructs the named ascir_op inside libaihac_ir.so and adds it to asc_graph.
// dynamic_input_num > 0  => calls op.DynamicInputRegister("x", dynamic_input_num)
// dynamic_output_num > 0 => calls op.DynamicOutputRegister("y", dynamic_output_num)
AscNodePtr AscGraphAddAscirNodeByType(AscGraph &asc_graph,
                                      const char *op_type,
                                      const char *name,
                                      size_t dynamic_input_num,
                                      size_t dynamic_output_num);

AxisId AscGraphCreateAxisBySerializedExpr(AscGraph &asc_graph,
                                          const char *name,
                                          const char *serialized_expr);

bool IsScalarInputBySerializedExprs(const std::vector<std::string> &serialized_exprs);

AxisId AscGraphCreateAxisBySerializedExpr(AscGraph &asc_graph,
                                          const char *name,
                                          const char *serialized_expr);

bool IsScalarInputBySerializedExprs(const std::vector<std::string> &serialized_exprs);

}  // namespace af

#endif  // METADEF_CXX_ASC_GRAPH_GE_BRIDGE_H_
