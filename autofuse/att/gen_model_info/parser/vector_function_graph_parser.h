/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_VECTOR_FUNCTION_GRAPH_PARSER_H
#define AUTOFUSE_VECTOR_FUNCTION_GRAPH_PARSER_H

#include "ascendc_ir_core/ascendc_ir.h"
#include "gen_model_info/parser/tuning_space.h"

namespace att {
class VectorFunctionGraphParser {
 public:
  VectorFunctionGraphParser(const af::AscNodePtr &asc_node, const af::AscGraph &graph)
      : asc_node_(asc_node), graph_(graph) {}
  ~VectorFunctionGraphParser() = default;
  af::Status Parse();
  [[nodiscard]] const std::vector<NodeInfo> &GetNodesInfos() const {
    return nodes_infos_;
  }

 private:
  af::Status ParseNodeInfos(const af::AscNodePtr &node, NodeInfo &node_info);
  af::Status ParseInputTensors(const af::AscNodePtr &node, NodeInfo &node_info);
  af::Status ParseOutputTensors(const af::AscNodePtr &node, NodeInfo &node_info);
  af::Status GetVectorizedAxes(const TensorPtr &tensor, const af::AscTensorAttr &tensor_attr) const;
  bool GetVectorizedAxisInfo(const af::AscTensorAttr &tensor_attr, int64_t axis_id, size_t vectorized_axis_index,
                             Expr &repeat, Expr &stride) const;
  af::Status ParseTensorInfo(const af::AscNodePtr &node, const af::AscTensorAttr &attr, const TensorPtr &tensor,
                             size_t index);

  std::vector<NodeInfo> nodes_infos_;
  const af::AscNodePtr &asc_node_;
  const af::AscGraph &graph_;
};
}  // namespace att

#endif  // AUTOFUSE_VECTOR_FUNCTION_GRAPH_PARSER_H
