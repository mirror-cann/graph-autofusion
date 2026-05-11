/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OPTIMIZE_FUSED_GRAPH_FUSED_GRAPH_UNFOLDER_H__
#define __OPTIMIZE_FUSED_GRAPH_FUSED_GRAPH_UNFOLDER_H__

#include "ascendc_ir.h"
#include "ascgen_log.h"
#include "node.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace optimize {
const char *const kAscGraphNodeType = "AscGraph";
const char *const kAscBackendType = "AscBackend";
using AscGraphPtr = af::AscGraph;

class FusedGraphUnfolder {
 public:
  static Status UnfoldFusedGraph(const af::ComputeGraphPtr &fused_graph,
                                 std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                 af::AscGraph &unfolded_asc_graph);

 private:
  static Status SelectCommonLoopAxis(std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                     std::vector<af::AxisPtr> &new_loop_axes);
  static Status MarkAllOutputAxisId(af::Node *concat_ascbc_node,
                                    std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                    const af::AxisId &axis_id,
                                    std::map<const af::AscGraph *, af::AxisId> &seen_graph_to_changed_axis_id,
                                    std::set<af::Node *> &seen_node);
  // 需要考虑向前以及向后的场景
  static Status MarkAllInputAxisId(af::Node *concat_input_node,
                                   std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                   const af::AxisId &axis_id,
                                   std::map<const af::AscGraph *, af::AxisId> &seen_graph_to_changed_axis_id,
                                   std::set<af::Node *> &seen_node);

  static Status ApplyMergedLoopAxis(const af::AscGraph &graph, const std::vector<af::AxisPtr> &new_loop_axes,
                                    const std::vector<af::AxisId> &loop_axis_ids, const size_t concat_dim);
  static Status UnfoldAscbcNode(af::Node *const &ascbc_node, const af::AscGraph &asc_graph,
                                const af::ComputeGraphPtr &target_computer_graph);
  static Status ReAssembleDataIrAttr(const af::ComputeGraphPtr &fused_graph,
                                     const std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph);
  static Status ReAssembleOutputIndex(const af::ComputeGraphPtr &fused_graph);

  static Status TransferInControlEdges(const std::set<af::NodePtr> &src_nodes, af::Node *const &asc_backend);
  static Status MergeInputNodes(const af::ComputeGraphPtr &graph, af::Node *const &asc_backend);
  static Status MergeOutputNodes(const af::ComputeGraphPtr &graph, af::Node *const &asc_backend);
  static Status DoSameLoadCse(const af::ComputeGraphPtr &fused_graph);
  static bool IsSameLoadNode(const af::AscNodePtr &lhs, const af::AscNodePtr &rhs);
  static Status RemoveRedundantLoads(const af::ComputeGraphPtr &graph);
  static Status RemoveUnusedNode(const af::ComputeGraphPtr &graph, const af::NodePtr &node, const bool force = false);
  static Status CollectPostConcatAscGraphs(af::Node *concat_ascbc_node,
                                           const std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                           const std::vector<af::AxisPtr> &new_loop_axes,
                                           const std::vector<af::AxisId> &loop_axis_ids,
                                           std::map<af::Node *, af::AscGraph> &post_concat_node_to_asc_graph);
  static Status DoAxisMappingForConstPostAscGraph(const af::AscGraph &graph,
                                                  const std::vector<af::AxisPtr> &new_loop_axes,
                                                  const std::vector<af::AxisId> &loop_axis_ids);
};
}  // namespace optimize

#endif  // __OPTIMIZE_FUSED_GRAPH_FUSED_GRAPH_UNFOLDER_H__
