/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_specific_params_builder.h"

#include <algorithm>
#include <memory>
#include <set>

#include "base/att_const_values.h"
#include "common/checker.h"

namespace att {
namespace {
bool IsReducePerfNode(const std::string &node_type) {
  static const std::set<std::string> kReduceTypes = {kMax, kMin,       kMean,      kProd,      kSum,      kAny,
                                                     kAll, kReduceMax, kReduceMin, kReduceAny, kReduceAll};
  return kReduceTypes.count(node_type) != 0U;
}

af::Status FillReduceReuseSource(const af::AscNodePtr &ge_node, codegen::ReduceReuseInfo &reuse) {
  GE_ASSERT_NOTNULL(ge_node);
  auto node_in_anchor = ge_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(node_in_anchor);
  auto peer_out_anchor = node_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor);
  const auto &in_node = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(in_node);
  reuse.valid = true;
  reuse.is_reuse_source = in_node->GetOutAllNodes().size() == 1UL;
  return ge::SUCCESS;
}

bool FindAxisInTensor(const TensorPtr &tensor, const SubAxis *axis, size_t &axis_index) {
  for (size_t i = 0U; i < tensor->dim_info.size(); ++i) {
    if (tensor->dim_info[i] == axis) {
      axis_index = i;
      return true;
    }
  }
  return false;
}

struct ReduceAxisCount {
  std::set<const SubAxis *> visited;
  int64_t total_count{0};
  int64_t valid_count{0};
};

void CountReduceAxis(const TensorPtr &input, const TensorPtr &output, const SubAxis *axis, ReduceAxisCount &count) {
  if (axis == nullptr || !count.visited.insert(axis).second) {
    return;
  }
  size_t output_axis_index = 0U;
  if (FindAxisInTensor(output, axis, output_axis_index)) {
    ++count.total_count;
    const bool output_zero =
        output_axis_index < output->stride.size() && output->stride[output_axis_index] == af::sym::kSymbolZero;
    size_t input_axis_index = 0U;
    const bool input_zero = FindAxisInTensor(input, axis, input_axis_index) &&
                            input_axis_index < input->stride.size() &&
                            input->stride[input_axis_index] == af::sym::kSymbolZero;
    count.valid_count += (output_zero && !input_zero) ? 1 : 0;
    return;
  }
  for (const auto *from_axis : axis->parent_axis) {
    CountReduceAxis(input, output, from_axis, count);
  }
  for (const auto *from_axis : axis->orig_axis) {
    CountReduceAxis(input, output, from_axis, count);
  }
}

const SubAxis *FindLoopAxis(const NodeInfo &node_info, const ScheduleAttr &attrs) {
  const auto axis_iter =
      std::find_if(node_info.loop_axes.begin(), node_info.loop_axes.end(),
                   [&attrs](const SubAxis *axis) { return axis != nullptr && axis->id == attrs.loop_axis_id; });
  return axis_iter == node_info.loop_axes.end() ? nullptr : *axis_iter;
}

Expr GetReduceMergeTimes(const NodeInfo &node_info, const ScheduleAttr &attrs) {
  const auto *loop_axis = FindLoopAxis(node_info, attrs);
  if (loop_axis == nullptr || !loop_axis->repeat.IsValid()) {
    return CreateExpr(1);
  }
  return loop_axis->repeat;
}

bool IsNeedMultiReduceFromNodeInfo(const NodeInfo &node_info, const ScheduleAttr &attrs) {
  const auto *loop_axis = FindLoopAxis(node_info, attrs);
  if (loop_axis == nullptr || node_info.inputs.empty() || node_info.outputs.empty()) {
    return true;
  }
  ReduceAxisCount count;
  CountReduceAxis(node_info.inputs[0], node_info.outputs[0], loop_axis, count);
  return count.total_count == count.valid_count;
}

codegen::ReducePattern GetReducePatternFromNodeInfo(const NodeInfo &node_info) {
  if (node_info.outputs.empty() || node_info.outputs[0] == nullptr || node_info.outputs[0]->stride.empty()) {
    return codegen::ReducePattern::kUnknown;
  }
  return node_info.outputs[0]->stride.back() == af::sym::kSymbolZero ? codegen::ReducePattern::kAR
                                                                     : codegen::ReducePattern::kRA;
}

af::Status BuildReduceSpecificParamInput(const af::AscNodePtr &ge_node, const ScheduleAttr &attrs,
                                         const NodeInfo &node_info,
                                         codegen::ReduceSpecificParamBuildInput &input) {
  GE_ASSERT_TRUE(!node_info.inputs.empty() && node_info.inputs[0] != nullptr, "Reduce input tensor is empty, node[%s].",
                 node_info.name.c_str());
  GE_ASSERT_TRUE(!node_info.outputs.empty() && node_info.outputs[0] != nullptr,
                 "Reduce output tensor is empty, node[%s].", node_info.name.c_str());

  input.node_name = node_info.name;
  input.reduce_type = node_info.node_type;
  input.input_repeats = node_info.inputs[0]->repeat;
  input.input_strides = node_info.inputs[0]->stride;
  input.output_dims = node_info.outputs[0]->repeat;
  input.output_strides = node_info.outputs[0]->stride;
  input.dtype_size = node_info.inputs[0]->data_type_size;
  input.pattern = GetReducePatternFromNodeInfo(node_info);
  input.need_multi_reduce = IsNeedMultiReduceFromNodeInfo(node_info, attrs);
  input.merge_times = input.need_multi_reduce ? GetReduceMergeTimes(node_info, attrs) : CreateExpr(1);
  GE_ASSERT_SUCCESS(FillReduceReuseSource(ge_node, input.reuse));
  return ge::SUCCESS;
}
}  // namespace

af::Status FillReduceSpecificParams(const af::AscNodePtr &ge_node, const ScheduleAttr &attrs, NodeInfo &node_info) {
  if (!IsReducePerfNode(node_info.node_type)) {
    return ge::SUCCESS;
  }
  codegen::ReduceSpecificParamBuildInput input;
  GE_ASSERT_SUCCESS(BuildReduceSpecificParamInput(ge_node, attrs, node_info, input));
  GE_ASSERT_SUCCESS(codegen::BuildReduceSpecificParams(input, node_info.reduce_specific_params));
  return ge::SUCCESS;
}
}  // namespace att
