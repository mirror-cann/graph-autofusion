/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "vector_function_graph_parser.h"
#include <algorithm>
#include "base/att_const_values.h"
#include "common/checker.h"
#include "ascir_ops.h"
#include "base_types_printer.h"
#include "ascir/meta/ascir_ops_utils.h"
#include "common_utils.h"
namespace att {
namespace {
constexpr af::char_t kSubGraphName[] = "sub_graph_name";
constexpr af::char_t kInputPrefix[] = "_input_";
constexpr af::char_t kOutputPrefix[] = "_output_";
}  // namespace
using namespace af::ops;
using namespace af::ascir_op;
af::Status VectorFunctionGraphParser::ParseNodeInfos(const af::AscNodePtr &node, NodeInfo &node_info) {
  GE_ASSERT_SUCCESS(ParseInputTensors(node, node_info), "Parse input tensors failed, node name:%s, %s.",
                    node->GetNamePtr(), node->GetTypePtr());
  GE_ASSERT_SUCCESS(ParseOutputTensors(node, node_info), "Parse output tensors failed, node name:%s, %s.",
                    node->GetNamePtr(), node->GetTypePtr());
  return af::SUCCESS;
}

bool VectorFunctionGraphParser::GetVectorizedAxisInfo(const af::AscTensorAttr &tensor_attr, int64_t axis_id,
                                                      size_t vectorized_axis_index, Expr &repeat, Expr &stride) const {
  if (vectorized_axis_index < tensor_attr.vectorized_strides.size()) {
    for (const auto &axis : graph_.GetAllAxis()) {
      if (axis != nullptr && axis->id == axis_id) {
        repeat = axis->size;
        stride = tensor_attr.vectorized_strides[vectorized_axis_index];
        return true;
      }
    }
  }

  const auto axis_iter = std::find(tensor_attr.axis.begin(), tensor_attr.axis.end(), axis_id);
  if (axis_iter == tensor_attr.axis.end()) {
    return false;
  }
  const auto axis_index = static_cast<size_t>(std::distance(tensor_attr.axis.begin(), axis_iter));
  if (axis_index >= tensor_attr.repeats.size() || axis_index >= tensor_attr.strides.size()) {
    return false;
  }
  repeat = tensor_attr.repeats[axis_index];
  stride = tensor_attr.strides[axis_index];
  return true;
}

af::Status VectorFunctionGraphParser::GetVectorizedAxes(const TensorPtr &tensor,
                                                        const af::AscTensorAttr &tensor_attr) const {
  for (size_t i = 0UL; i < tensor_attr.vectorized_axis.size(); ++i) {
    const auto vectorized_axis_id = tensor_attr.vectorized_axis[i];
    Expr vectorized_axis_repeat;
    Expr vectorized_axis_stride;
    GE_ASSERT_TRUE(
        GetVectorizedAxisInfo(tensor_attr, vectorized_axis_id, i, vectorized_axis_repeat, vectorized_axis_stride),
        "Vectorized axis[%ld] not in graph axis or tensor axis, tensor info is %s, tensor attr axis=%s, "
        "repeats=%s, strides=%s, vectorized_axis=%s, vectorized_strides=%s.",
        vectorized_axis_id, tensor->ToString().c_str(), ascgen_utils::VectorToStr(tensor_attr.axis).c_str(),
        ascgen_utils::VectorToStr(tensor_attr.repeats).c_str(), ascgen_utils::VectorToStr(tensor_attr.strides).c_str(),
        ascgen_utils::VectorToStr(tensor_attr.vectorized_axis).c_str(),
        ascgen_utils::VectorToStr(tensor_attr.vectorized_strides).c_str());
    tensor->repeat.emplace_back(vectorized_axis_repeat);
    tensor->stride.emplace_back(vectorized_axis_stride);
  }
  return af::SUCCESS;
}

af::Status VectorFunctionGraphParser::ParseTensorInfo(const af::AscNodePtr &node, const af::AscTensorAttr &attr,
                                                      const TensorPtr &tensor, size_t index) {
  GE_ASSERT_SUCCESS(GetVectorizedAxes(tensor, attr), "Get vectorized axis tensor size[%s] failed, graph name[%s].",
                    tensor->name.c_str(), graph_.GetName().c_str());
  tensor->owner_node = node.get();
  tensor->data_type = BaseTypeUtils::DtypeToStr(attr.dtype);
  tensor->data_type_size = af::GetSizeByDataType(attr.dtype);
  // Vector function默认都放在UB
  tensor->loc = HardwareDef::UB;
  GELOGI("[VF_PERF_DFX] vf node [%s] sub node [%s] tensor[%zu] datatype [%d] name[%s], tensor info is %s",
         asc_node_->GetNamePtr(), node->GetNamePtr(), index, static_cast<int32_t>(attr.dtype), tensor->name.c_str(),
         tensor->ToString().c_str());
  return af::SUCCESS;
}

af::Status VectorFunctionGraphParser::ParseInputTensors(const af::AscNodePtr &node, NodeInfo &node_info) {
  for (size_t in_id = 0U; in_id < node->inputs.Size(); in_id++) {
    TensorPtr tensor = std::make_shared<Tensor>();
    GE_ASSERT_NOTNULL(tensor, "Create tensor failed, node name:%s, %s.", node->GetNamePtr(), node->GetTypePtr());
    tensor->name = node->GetName() + kInputPrefix + std::to_string(in_id);
    GE_ASSERT_SUCCESS(ParseTensorInfo(node, node->inputs[in_id].attr, tensor, in_id),
                      "Parse tensor info failed, node name:%s, %s.", node->GetNamePtr(), node->GetTypePtr());
    node_info.inputs.emplace_back(tensor);
  }
  return af::SUCCESS;
}

af::Status VectorFunctionGraphParser::ParseOutputTensors(const af::AscNodePtr &node, NodeInfo &node_info) {
  for (size_t out_id = 0U; out_id < node->outputs().size(); out_id++) {
    TensorPtr tensor = std::make_shared<Tensor>();
    GE_ASSERT_NOTNULL(tensor, "Create tensor failed, node name:%s, %s.", node->GetNamePtr(), node->GetTypePtr());
    tensor->name = node->GetName() + kOutputPrefix + std::to_string(out_id);
    GE_ASSERT_SUCCESS(ParseTensorInfo(node, node->outputs[out_id].attr, tensor, out_id),
                      "Parse tensor info failed, node name:%s, %s.", node->GetNamePtr(), node->GetTypePtr());
    node_info.outputs.emplace_back(tensor);
  }
  return af::SUCCESS;
}

af::Status VectorFunctionGraphParser::Parse() {
  GE_ASSERT_NOTNULL(asc_node_);
  // 仅解析VectorFunc节点
  if (asc_node_->GetTypePtr() != kVectorFunc) {
    return af::SUCCESS;
  }
  GELOGD("Node:%s, %s, sub_graph_name start to parse subgraph", asc_node_->GetNamePtr(), asc_node_->GetTypePtr());
  const std::string *graph_name = af::AttrUtils::GetStr(asc_node_->GetOpDescBarePtr(), kSubGraphName);
  GE_ASSERT_NOTNULL(graph_name, "Get sub graph name failed, vf node:%s", asc_node_->GetNamePtr());
  af::AscGraph sub_graph("vf_sub_graph");
  GE_ASSERT_SUCCESS(graph_.FindSubGraph(*graph_name, sub_graph), "Get sub_graph failed, vf node:%s, sub_graph_name:%s",
                    asc_node_->GetNamePtr(), graph_name);
  GELOGI("VF node:%s, sub_graph_name:%s", asc_node_->GetNamePtr(), graph_name->c_str());
  for (auto node : sub_graph.GetAllNodes()) {
    // subgraph上的Load api直接使用Tpipe上保存的UB tensor, 因此vf子图上Data节点的输出Tensor不必保存在tensor manager中.
    if (IsOps<Output>(node) || IsOps<Data>(node)) {
      continue;
    }
    NodeInfo node_info;
    node_info.node_type = node->GetType();
    node_info.name = node->GetName();
    GE_ASSERT_SUCCESS(ParseNodeInfos(node, node_info), "Parse node infos failed, node name:%s, %s.", node->GetNamePtr(),
                      node->GetTypePtr());
    nodes_infos_.emplace_back(node_info);
  }
  return af::SUCCESS;
}
}  // namespace att
