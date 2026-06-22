/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_GRAPH_CONSTRUCT_UTILS_H
#define ATT_GRAPH_CONSTRUCT_UTILS_H
#include <vector>
#include "ascir_ops.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/normal_graph/ge_tensor_impl.h"
namespace att {
class GraphConstructUtils {
 public:
  static void UpdateVectorizedStride(const std::vector<int64_t> &axis, const std::vector<af::Expression> &strides,
                                     const std::vector<int64_t> &vectorized_axis,
                                     std::vector<af::Expression> &vectorized_strides);
  static void UpdateGraphVectorizedStride(af::AscGraph &graph);
  static void UpdateGraphsVectorizedStride(std::vector<af::AscGraph> &impl_graphs);
  static af::Status UpdateTensorAxes(const std::vector<af::Axis> &axes, af::AscOpOutput &output, int32_t loop_id = -1);
  static af::Status UpdateOutputTensorAxes(const std::vector<af::Axis> &axes, std::vector<af::AscOpOutput> &&outputs,
                                           int32_t loop_id = -1);
  static af::Status CreateSimpleLoadStoreOp(af::AscGraph &graph);
  static af::AscNodePtr ConstructSingleOp(const std::string &op_type, int32_t in_cnt, int32_t out_cnt);
  static af::Status BuildConcatGroupAscendGraphS0S1ReduceMultiTiling(af::AscGraph &graph);
};

class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) {
    graph_ = std::make_shared<af::ComputeGraph>(name);
  }

  GraphBuilder(const std::string &name, const std::string &node_type) {
    graph_ = std::make_shared<af::ComputeGraph>(name);
    node_type_ = node_type;
  }

  af::NodePtr AddNode(const std::string &name, const std::string &type, const int in_cnt, const int out_cnt,
                      const std::vector<int64_t> shape = {1, 1, 1, 1}) {
    auto tensor_desc = std::make_shared<af::GeTensorDesc>();
    tensor_desc->SetShape(af::GeShape(std::move(shape)));
    tensor_desc->SetFormat(af::FORMAT_NCHW);
    tensor_desc->SetDataType(af::DT_FLOAT);

    auto op_desc = std::make_shared<af::OpDesc>(name, (node_type_.empty()) ? type : "AscGraph");
    for (std::int32_t i = 0; i < in_cnt; ++i) {
      op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (std::int32_t i = 0; i < out_cnt; ++i) {
      op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    op_desc->AddInferFunc([](af::Operator &op) { return af::GRAPH_SUCCESS; });
    return graph_->AddNode(op_desc);
  }

  af::Status AddDataEdge(const af::NodePtr &src_node, const std::int32_t src_idx, const af::NodePtr &dst_node,
                         const std::int32_t dst_idx) {
    return af::GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
  }

  af::ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  af::ComputeGraphPtr graph_;
  std::string node_type_;
};
}  // namespace att
#endif  // ATT_GRAPH_CONSTRUCT_UTILS_H
