/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CONCAT_SCHEDULE_CASE_GENERATOR_H_
#define ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CONCAT_SCHEDULE_CASE_GENERATOR_H_

#include "ascir_ops.h"
#include "ascir/meta/ascir.h"
#include "common/ascgen_log.h"
#include "optimize/task_generator/schedule_case_generator.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace optimize {
struct ExpressionStaticCheckEq {
  // concat子图表达式可能不一致, 依赖guard
  bool operator()(const af::Expression &lhs, const af::Expression &rhs) const {
    return af::SymbolicUtils::StaticCheckEq(lhs, rhs) == af::TriBool::kTrue;
  }
};
using ConcatDimAxisMap = std::unordered_map<af::Expression, af::AxisId, af::ExpressionHash, ExpressionStaticCheckEq>;
class ConcatFusionCaseGenerator : public FusionCaseGenerator {
 public:
  Status Generate(ascir::HintGraph &graph,
                  std::vector<ascir::ImplGraph> &graphs,
                  std::vector<std::string> &score_functions) override;
  Status EliminateConcat(ascir::HintGraph &graph, const af::AscNodePtr &concat_node);
  [[nodiscard]] bool HasLoadStoreConversion() const override {
    return true;
  }

 private:
  Status AddTemplatesForFirstDimConcat(const af::AscNodePtr &concat_node,
                                       ascir::HintGraph &graph,
                                       std::vector<ascir::ImplGraph> &graphs);
  Status AddTemplateForSplitConcat(const ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs);
  static Status AddTemplateForSmallTail(const ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs);
  bool NeedDynSmallTailTemplate(const af::AscNodePtr &concat_node) const;
  Status GenerateScoreFunctions(const std::vector<ascir::ImplGraph> &graphs,
                                size_t concat_dim,
                                std::vector<std::string> &score_functions) const;
  static std::vector<af::AscNodePtr> FindConcatNodes(const ascir::HintGraph &owner_graph, bool *has_unsupported_op = nullptr);
  Status ConvertConcatToStores(ascir::HintGraph &owner_graph, const af::AscNodePtr &concat_node);
  Status SplitConcats(ascir::HintGraph &owner_graph, const af::AscNodePtr &concat_node, bool &split);
  Status Prepare(const af::AscNodePtr &concat_node, size_t concat_dim);
  Status ReplaceWithStore(const af::AscNodePtr &concat_node, const af::InDataAnchorPtr &concat_in_anchor,
                          const af::Axis &replace_axis);
  Status ConvertSingleInput(ascir::HintGraph &owner_graph, const af::AscNodePtr &concat_node, size_t in_index,
                            size_t group_idx, ConcatDimAxisMap &repeat_to_axis_id);
  static Status PropagateAxisChanges(af::Node *start_node, const std::vector<ascir::AxisId> &new_axis_ids);
  Status ReplaceWithConcat(::ascir::ImplGraph &owner_graph,
                           const af::AscNodePtr &concat_node,
                           size_t start,
                           size_t end);
  static Status RemoveUnusedNodes(const af::AscNodePtr &concat_node, const std::vector<af::AscNodePtr> &nodes = {});
  static Status SplitDataForDifferentConcatDim(ascir::ImplGraph &owner_graph);
  static af::Status SetConcatOpAttr(af::ascir_op::Concat &concat_op,
                                    const af::AscNodePtr &concat_node,
                                    size_t concat_dim,
                                    size_t start,
                                    size_t end);
  static Status CollectBackwardNodes(const af::NodePtr &concat_node, std::vector<af::AscNodePtr> &nodes);
  static Status CollectReachableLoadNodes(const af::NodePtr &concat_node, std::set<af::AscNodePtr> &nodes);
  Status CloneNonConcatNodes(const af::Axis &new_axis, size_t index,
                             std::vector<af::InDataAnchorPtr> &in_anchors,
                             const std::vector<ascir::AxisId> &new_axis_ids,
                             std::unordered_map<std::string, af::NodePtr> &name_to_new_node);
  static af::Status ReplaceAxis(const af::AscNodePtr &node, size_t axis_index, const af::Axis &to_axis,
                                const std::vector<ascir::AxisId> &new_axis_ids);
  static af::Status UpdateRepeatAndStrides(const af::AscNodePtr &node, size_t axis_index,
                                           const af::Expression &axis_size, af::AscTensorAttr &tensor_attr);
  static Status InsertAxis(const ascir::ImplGraph &optimized_graph);
  static Status AddTemplateIfCanFitInOneKernel(const af::AscNodePtr &concat_node, ascir::HintGraph &graph,
                                               std::vector<ascir::ImplGraph> &graphs);
  static Status MarkNoMergeFirstAxis(const std::vector<ascir::ImplGraph> &graphs);
  bool KeepOriginGraph(const af::AscNodePtr &concat_node) const;
  static bool IsSmallBlock(const af::AscNodePtr &concat_node, size_t concat_dim);
  static Status ReconnectIfShareSameAncestor(const std::unordered_map<std::string, af::NodePtr> &name_to_node, const af::InDataAnchorPtr &in_anchor);
  static Status AddExtraShapeEnv(const af::AscNodePtr &concat_node, size_t concat_dim);
  Status PrepareForModifyingGraph(const af::AscNodePtr &concat_node);
  static Status RunCastOptimizationPass(std::vector<ascir::ImplGraph> &graphs);

  std::vector<af::AscNodePtr> post_concat_nodes_;
  std::set<af::AscNodePtr> reachable_load_nodes_;;
  std::map<std::string, std::vector<int32_t>> out_node_name_to_indices_;
  std::vector<af::Expression> concat_dim_offsets_;
  ascir::AxisId concat_axis_id_ = -1;
  size_t concat_dim_ = std::numeric_limits<size_t>::max();
  bool support_small_tail_ = false;
  bool split_concat_ = false;
  bool has_recompute_ = false;
};
}  // namespace optimize

#endif  // ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CONCAT_SCHEDULE_CASE_GENERATOR_H_
