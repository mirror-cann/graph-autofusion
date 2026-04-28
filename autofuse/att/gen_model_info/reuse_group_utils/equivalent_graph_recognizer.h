/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_ATT_GEN_MODEL_INFO_REUSE_GROUP_UTILS_EQUIVALENT_GRAPH_RECOGNIZER_H_
#define AUTOFUSE_ATT_GEN_MODEL_INFO_REUSE_GROUP_UTILS_EQUIVALENT_GRAPH_RECOGNIZER_H_

#include "base/model_info.h"
namespace af { namespace att {
class EquivalentGraphRecognizer {
 public:
  EquivalentGraphRecognizer(const af::AscGraph &graph_to, const af::AscGraph &graph_from,
                            const ReuseScheduleGroupInfo &group_info_to, const ReuseScheduleGroupInfo &group_info_from);
  ~EquivalentGraphRecognizer() = default;
  bool IsEquivalent();
  const std::vector<std::string> &GetMappedInputAxesNames() const {
    return graph_to_ordered_input_names_;
  }
 private:
  bool IsAscNodeEquivalent(af::AscNode &node1, af::AscNode &node2);
  bool IsAscTensorEquivalent(const af::AscTensorAttr &tensor1, const af::AscTensorAttr &tensor2);
  bool IsAscNodeAttrEquivalent(const af::AscNodeAttr &node_attr_to, const af::AscNodeAttr &node_attr_from);
  bool CompareExpression(const af::Expression &expr1, const af::Expression &expr2);
  bool CanExprEquivalentAfterReplace(const af::Expression &replace_expr, const af::Expression &reuse_expr);
  bool IsTensorViewEquivalent(const af::AscTensorAttr &tensor1,
                              const af::AscTensorAttr &tensor2);
  bool CompareExprs(const std::vector<af::Expression> &exprs1, const std::vector<af::Expression> &exprs2);
  bool CompareAxis(const int64_t axis_id, const int64_t axis_id2) const;
  bool IsMemEquivalent(const af::AscTensorAttr &tensor1, const af::AscTensorAttr &tensor2) const;
  bool IsInputNodeSame(const af::AscNodePtr &asc_node1, const af::AscNodePtr &asc_node2);
  bool IsInputVar(const Expr &expr1, const Expr &expr2) const;
  std::string ReplaceSearchVarStr(const std::string &str) const;
  bool UpdateOrderedInputNames();
  bool IsInputAxesFromDuplicityMapped() const;
  // 检查graph_to_是否可以复用graph_from_的图
  af::AscGraph graph_to_;
  af::AscGraph graph_from_;
  const ReuseScheduleGroupInfo &group_info_to_;
  const ReuseScheduleGroupInfo &group_info_from_;
  std::set<std::string> search_axes_name_to_;
  std::set<std::string> search_axes_name_from_;
  std::set<std::string> input_axes_name_to_;
  std::set<std::string> input_axes_name_from_;
  std::map<int64_t, af::AxisPtr> axis_id_to_axis_map_to_;
  std::map<int64_t, af::AxisPtr> axis_id_to_axis_map_from_;
  // 按照graph_from_的输入轴映射到graph_to_输入轴名称
  std::map<std::string, std::string> mapped_input_axes_names_;
  // 按照顺序排布的graph_to_的输入轴，与graph_from_的一一映射(输出)
  std::vector<std::string> graph_to_ordered_input_names_;
};
}
}  // namespace af
#endif  // AUTOFUSE_ATT_GEN_MODEL_INFO_REUSE_GROUP_UTILS_EQUIVALENT_GRAPH_RECOGNIZER_H_
