/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_graph_unfolder.h"

#include <cstdint>
#include <map>
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "ascendc_ir/utils/asc_graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/gnode.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "graph_dump_utils.h"
#include "schedule_utils.h"
#include "ascir_utils.h"
#include "ascgraph_info_complete.h"

namespace optimize {
Status FusedGraphUnfolder::RemoveUnusedNode(const af::ComputeGraphPtr &graph, const af::NodePtr &node,
                                            const bool force) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(node);
  if (force && !node->GetAllOutDataAnchors().empty()) {
    auto out_anchor = node->GetOutDataAnchor(0);
    GE_CHECK_NOTNULL(out_anchor);
    out_anchor->UnlinkAll();
  }
  if (node->GetOutNodes().empty()) {
    GELOGI("%s node [%s] is unused, remove it.", node->GetTypePtr(), node->GetNamePtr());
    af::NodeUtils::UnlinkAll(*node);
    af::GraphUtils::RemoveNodeWithoutRelink(graph, node);
    return af::SUCCESS;
  }
  GELOGD("%s node [%s] has %zu outputs, keep it.", node->GetTypePtr(), node->GetNamePtr(), node->GetOutNodes().size());
  return af::SUCCESS;
}

Status FusedGraphUnfolder::TransferInControlEdges(const std::set<af::NodePtr> &src_nodes,
                                                  af::Node *const &asc_backend) {
  for (auto &src_node : src_nodes) {
    const auto &in_nodes = src_node->GetInAllNodes();
    const std::set<af::NodePtr> in_node_set(in_nodes.begin(), in_nodes.end());
    for (auto &ctrl_node : asc_backend->GetInControlNodes()) {
      GE_CHECK_NOTNULL(ctrl_node);
      if (in_node_set.count(ctrl_node) > 0UL) {
        continue;
      }
      GELOGI("[%s] Restore control edge to [%s]", ctrl_node->GetName().c_str(), src_node->GetName().c_str());
      // 添加连边
      af::GraphUtils::AddEdge(ctrl_node->GetOutControlAnchor(), src_node->GetInControlAnchor());
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::MergeInputNodes(const af::ComputeGraphPtr &graph, af::Node *const &asc_backend) {
  GE_CHECK_NOTNULL(asc_backend);
  std::set<af::NodePtr> src_nodes;
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() != af::DATA) {
      if (node->GetInAllNodes().empty()) {
        (void)src_nodes.emplace(node);
      }
      continue;
    }
    int64_t index = -1;
    (void)ScheduleUtils::GetNodeIrAttrIndex(node, index);
    GE_ASSERT_TRUE(index >= 0, "Ir attr index is invalid, node:[%s].", node->GetNamePtr());
    const auto parent_node_in_anchor = asc_backend->GetInDataAnchor(static_cast<int32_t>(index));
    GE_CHECK_NOTNULL(parent_node_in_anchor, "Parent output anchor is nullptr, data:[%s], index:[%d].",
                     node->GetNamePtr(), static_cast<int32_t>(index));
    const auto src_out_anchor = parent_node_in_anchor->GetPeerOutAnchor();
    if ((src_out_anchor == nullptr) || (src_out_anchor->GetOwnerNodeBarePtr() == nullptr)) {
      continue;
    }
    parent_node_in_anchor->UnlinkAll();
    // link src to outputs of DataNode
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        auto dst_node = peer_in_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(dst_node);
        const auto &in_nodes = dst_node->GetInDataNodes();
        if (std::all_of(in_nodes.begin(), in_nodes.end(),
                        [](const af::NodePtr &n) { return n->GetType() == af::DATA; })) {
          (void)src_nodes.emplace(dst_node);
        }
        GE_CHK_STATUS_RET(af::GraphUtils::ReplaceEdgeSrc(out_data_anchor, peer_in_anchor, src_out_anchor),
                          "[Replace][DataEdge] failed");
      }
    }
    // when unfold AscBackend, if data have control edges, which will be left in final graph
    // which cause topo sort failed.
    auto out_control_anchor = node->GetOutControlAnchor();
    GE_CHECK_NOTNULL(out_control_anchor);
    out_control_anchor->UnlinkAll();
    // remove isolated data node
    GE_CHK_STATUS_RET(af::GraphUtils::RemoveNodeWithoutRelink(graph, node), "Failed to remove data node [%s].",
                      node->GetNamePtr());
  }

  // transfer in control edges to all root nodes
  GE_CHK_STATUS_RET(TransferInControlEdges(src_nodes, asc_backend), "TransferInControlEdges failed.");

  asc_backend->GetInControlAnchor()->UnlinkAll();
  return af::SUCCESS;
}

bool FusedGraphUnfolder::IsSameLoadNode(const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
  // 如果不都是load节点，则返回
  if (!af::ops::IsOps<af::ascir_op::Load>(lhs) || !af::ops::IsOps<af::ascir_op::Load>(rhs)) {
    return false;
  }
  if (lhs->attr.sched.axis != rhs->attr.sched.axis) {
    return false;
  }

  af::Expression cur_load_offset;
  (void)ScheduleUtils::GetNodeIrAttrOffset(rhs, cur_load_offset);
  af::Expression old_load_offset;
  (void)ScheduleUtils::GetNodeIrAttrOffset(lhs, old_load_offset);
  if (af::SymbolicUtils::StaticCheckEq(cur_load_offset, old_load_offset) != af::TriBool::kTrue) {
    return false;
  }

  const auto &lhs_repeats = lhs->outputs[0].attr.repeats;
  const auto &lhs_strides = lhs->outputs[0].attr.strides;
  const auto &rhs_repeats = rhs->outputs[0].attr.repeats;
  const auto &rhs_strides = rhs->outputs[0].attr.strides;

  if ((lhs_repeats.size() != rhs_repeats.size()) || (lhs_strides.size() != rhs_strides.size())) {
    return false;
  }
  for (size_t i = 0UL; i < lhs_repeats.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(lhs_repeats[i], rhs_repeats[i]) != af::TriBool::kTrue) {
      return false;
    }
  }

  for (size_t i = 0UL; i < lhs_strides.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(lhs_strides[i], rhs_strides[i]) != af::TriBool::kTrue) {
      return false;
    }
  }

  GELOGI("Node(%s) and Node(%s) are same load node.", lhs->GetNamePtr(), rhs->GetNamePtr());
  return true;
}

Status FusedGraphUnfolder::DoSameLoadCse(const af::ComputeGraphPtr &fused_graph) {
  for (const auto &node : fused_graph->GetDirectNodePtr()) {
    GE_CHECK_NOTNULL(node);
    if ((node->GetType() != af::DATA) || (node->GetOutDataNodesSize() <= 1UL)) {
      continue;
    }
    auto output_anchor = node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(output_anchor);
    auto peer_in_anchors = output_anchor->GetPeerInDataAnchorsPtr();
    auto pre_load_anchor = peer_in_anchors[0UL];
    GE_ASSERT_NOTNULL(pre_load_anchor);
    auto pre_load_node = std::dynamic_pointer_cast<af::AscNode>(pre_load_anchor->GetOwnerNode());
    GE_ASSERT_NOTNULL(pre_load_node);
    for (size_t i = 1UL; i < peer_in_anchors.size(); ++i) {
      auto rear_load_anchor = peer_in_anchors[i];
      GE_ASSERT_NOTNULL(rear_load_anchor);
      auto rear_load_node = std::dynamic_pointer_cast<af::AscNode>(rear_load_anchor->GetOwnerNode());
      GE_ASSERT_NOTNULL(rear_load_node);
      const bool is_load_same = IsSameLoadNode(pre_load_node, rear_load_node);
      if (is_load_same) {
        // relink load's output
        auto rear_load_out_anchor = rear_load_node->GetOutDataAnchor(0);
        GE_ASSERT_NOTNULL(rear_load_out_anchor);
        auto pre_load_out_anchor = pre_load_node->GetOutDataAnchor(0);
        for (const auto &cur_load_next_in_anchor : rear_load_out_anchor->GetPeerInDataAnchors()) {
          GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(rear_load_out_anchor, cur_load_next_in_anchor));
          GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(pre_load_out_anchor, cur_load_next_in_anchor));
        }
        // Remove cur load
        af::NodeUtils::UnlinkAll(*rear_load_node);
        GE_CHK_STATUS_RET(af::GraphUtils::RemoveNodeWithoutRelink(fused_graph, rear_load_node),
                          "Failed to remove load node [%s].", rear_load_node->GetNamePtr());
      }
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::RemoveRedundantLoads(const af::ComputeGraphPtr &graph) {
  for (auto &load_node : graph->GetAllNodes()) {
    // step1: find Load node
    GE_ASSERT_NOTNULL(load_node);
    if (!af::ops::IsOps<af::ascir_op::Load>(load_node)) {
      continue;
    }
    GE_ASSERT_TRUE(load_node->GetInDataNodesSize() == 1UL);  // Load node has only one input.

    // step2: find Output node
    auto output_node = load_node->GetInDataNodes().at(0);
    GE_ASSERT_NOTNULL(output_node);
    if (!af::ops::IsOps<af::ascir_op::Output>(output_node)) {
      continue;
    }
    GE_ASSERT_TRUE(output_node->GetInDataNodesSize() == 1UL);  // Output node has only one input.

    // step3: find Store node
    auto store_node = output_node->GetInDataNodes().at(0);
    GE_ASSERT_NOTNULL(store_node);
    if (!af::ops::IsOps<af::ascir_op::Store>(store_node)) {
      GELOGW("The input of Output node[%s] is %s, not Store node, ", output_node->GetNamePtr(),
             store_node->GetNamePtr());
      continue;
    }
    GE_ASSERT_TRUE(store_node->GetInDataNodesSize() == 1UL);  // Store node has only one input.

    // step4: Pattern like Store+Output+Load can be optimized.
    GELOGD("Find Store+Output+Load pattern: [%s]+[%s]+[%s]", store_node->GetNamePtr(), output_node->GetNamePtr(),
           load_node->GetNamePtr());
    // find the input node of Store
    auto store_in_anchor = store_node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(store_in_anchor);
    auto pre_node_out_anchor = store_in_anchor->GetPeerOutAnchor();  // the input of Store node
    GE_ASSERT_NOTNULL(pre_node_out_anchor);
    auto pre_node = pre_node_out_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(pre_node);
    for (const auto &load_out_anchor : load_node->GetAllOutDataAnchors()) {
      GE_ASSERT_NOTNULL(load_out_anchor);
      // disconnect Load from its output nodes; And relink input of Store to the output of Load
      for (const auto &peer_in_anchor : load_out_anchor->GetPeerInDataAnchors()) {
        GE_ASSERT_NOTNULL(peer_in_anchor);
        auto load_output_node = peer_in_anchor->GetOwnerNodeBarePtr();
        GE_ASSERT_NOTNULL(load_output_node);
        GELOGD("Disconnect %s[%d] and %s[%d], Relink to %s[%d]", load_node->GetNamePtr(), load_out_anchor->GetIdx(),
               load_output_node->GetNamePtr(), peer_in_anchor->GetIdx(), pre_node->GetNamePtr(),
               pre_node_out_anchor->GetIdx());
        GE_ASSERT_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(load_out_anchor, peer_in_anchor, pre_node_out_anchor));
      }
    }

    // step5: Remove redundant Load, Output and Store
    RemoveUnusedNode(graph, load_node);
    RemoveUnusedNode(graph, output_node);
    RemoveUnusedNode(graph, store_node);
  }
  // step6: Remove NetOutput. Cannot merge two for loop because this depends on the results of previous loop.
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == af::NETOUTPUT) {
      RemoveUnusedNode(graph, node);
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::MergeOutputNodes(const af::ComputeGraphPtr &graph, af::Node *const &asc_backend) {
  GE_CHECK_NOTNULL(asc_backend);
  for (auto &output_node : graph->GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Output>(output_node)) {
      continue;
    }
    int64_t index = -1;
    (void)ScheduleUtils::GetNodeIrAttrIndex(output_node, index);
    GE_ASSERT_TRUE((index >= 0), "Get invalid attr index [%ld], node = %s[%s]", index, output_node->GetNamePtr(),
                   output_node->GetTypePtr());

    // Skip relinking when an ascbc node does not have a peer_in_data_anchor, which means that the data will be the
    // output in the ascgraph.
    const af::OutDataAnchorPtr &parent_out_anchor = asc_backend->GetOutDataAnchor(static_cast<int32_t>(index));
    GE_CHECK_NOTNULL(parent_out_anchor, "Parent output anchor is null, output_node:[%s], index:[%d].",
                     output_node->GetNamePtr(), static_cast<int32_t>(index));
    auto peer_in_anchor_in_parent = parent_out_anchor->GetPeerInDataAnchors();

    auto output_anchor = output_node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(output_anchor);
    for (af::InDataAnchorPtr &dst_in_anchor : peer_in_anchor_in_parent) {
      GE_CHK_STATUS_RET(af::GraphUtils::ReplaceEdgeSrc(parent_out_anchor, dst_in_anchor, output_anchor),
                        "[Replace][DataEdge] failed");
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::UnfoldAscbcNode(af::Node *const &ascbc_node, const af::AscGraph &asc_graph,
                                           const af::ComputeGraphPtr &target_computer_graph) {
  auto graph = af::AscGraphUtils::GetComputeGraph(asc_graph);
  GE_ASSERT_NOTNULL(graph);
  GE_CHK_STATUS_RET(MergeInputNodes(graph, ascbc_node),
                    "[Invoke][MergeInputNodes] Merge data nodes for graph %s failed", graph->GetName().c_str());
  GE_CHK_STATUS_RET(MergeOutputNodes(graph, ascbc_node),
                    "[Invoke][MergeNetOutputNode] Merge net output nodes for graph %s failed",
                    graph->GetName().c_str());
  GELOGI("[%s] Merging graph inputs and outputs successfully", graph->GetName().c_str());
  for (auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    (void)target_computer_graph->AddNode(node);
    GELOGI("[%s::%s] added to target graph: [%s].", graph->GetName().c_str(), node->GetName().c_str(),
           target_computer_graph->GetName().c_str());
    (void)node->SetOwnerComputeGraph(target_computer_graph);
  }

  GELOGI("[%s] Done merging graph. remove it from root graph", graph->GetName().c_str());
  GE_LOGW_IF(
      af::GraphUtils::RemoveNodeWithoutRelink(target_computer_graph, ascbc_node->shared_from_this()) != af::SUCCESS,
      "Remove node %s failed, graph:%s.", ascbc_node->GetName().c_str(), target_computer_graph->GetName().c_str())
  return af::SUCCESS;
}

Status FusedGraphUnfolder::ReAssembleOutputIndex(const af::ComputeGraphPtr &fused_graph) {
  for (const auto &node : fused_graph->GetAllNodes()) {
    if (node->GetType() == af::NETOUTPUT) {
      int64_t index = 0;
      for (auto &in_data_anchor : node->GetAllInDataAnchorsPtr()) {
        GE_ASSERT_NOTNULL(in_data_anchor);
        auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
        if (peer_out_anchor != nullptr) {
          auto asc_node = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
          GE_ASSERT_NOTNULL(asc_node, "In anchor [%ld]'s peer out anchor[%d] does have owner node.", index,
                            peer_out_anchor->GetIdx());
          GE_ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Output>(asc_node),
                         "Only output nodes can be directly connected to the netoutput.");
          GE_ASSERT_NOTNULL(asc_node->attr.ir_attr);
          auto ir_attr = asc_node->attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
          GE_ASSERT_NOTNULL(ir_attr);
          GE_ASSERT_SUCCESS(ir_attr->SetIndex(index));
          GELOGD("Output node [%s] has been set ir attr index [%ld].", node->GetNamePtr(), index);
          ++index;
        }
      }
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::ReAssembleDataIrAttr(const af::ComputeGraphPtr &fused_graph,
                                                const std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph) {
  for (const auto &node : fused_graph->GetAllNodes()) {
    if (node->GetType() != af::DATA) {
      continue;
    }
    auto output_anchor = node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(output_anchor);
    auto peer_in_anchor = output_anchor->GetPeerInDataAnchorsPtr();
    GE_ASSERT_TRUE(!peer_in_anchor.empty());
    auto peer_first_data_anchor = peer_in_anchor[0UL];
    auto iter = asc_backend_to_asc_graph.find(peer_first_data_anchor->GetOwnerNodeBarePtr());
    GE_ASSERT_TRUE(iter != asc_backend_to_asc_graph.end(), "Cannot find ascgraph for data [%s].", node->GetNamePtr());
    auto data_index = peer_first_data_anchor->GetIdx();
    // 存在geir和ascir构图两种可能性
    auto node_attr = node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
    GE_ASSERT_NOTNULL(node_attr);
    auto tensor_attr = af::AscTensorAttr::GetTensorAttrPtr(*output_anchor);
    GE_ASSERT_NOTNULL(tensor_attr);
    int64_t ir_index = -1;
    (void)af::AttrUtils::GetInt(node->GetOpDescBarePtr(), "_parent_node_index", ir_index);
    if (node_attr->ir_attr != nullptr) {
      (void)node_attr->ir_attr->GetAttrValue("index", ir_index);
    }
    GE_ASSERT_TRUE(ir_index >= 0, "Cannot find ir attr index from data node [%s].", node->GetNamePtr());

    for (const auto &sub_data : iter->second.GetAllNodes()) {
      if (!ScheduleUtils::IsDataInput(sub_data)) {
        continue;
      }
      int64_t sub_index = -1;
      (void)ScheduleUtils::GetNodeIrAttrIndex(sub_data, sub_index);
      GE_ASSERT_TRUE(sub_index >= 0, "Cannot find ir attr index for node [%s].", sub_data->GetNamePtr());
      if (sub_index == data_index) {
        *node_attr = sub_data->attr;
        *tensor_attr = sub_data->outputs[0].attr;
        GELOGD("Data node [%s] use attr from [%s].", node->GetNamePtr(), sub_data->GetNamePtr());
      }
    }
    GE_ASSERT_NOTNULL(node_attr->ir_attr);
    GELOGD("Data node [%s] has been set ir attr index [%ld].", node->GetNamePtr(), ir_index);
    auto ir_attr = node_attr->ir_attr->DownCastTo<af::AscDataIrAttrDef>();
    GE_ASSERT_NOTNULL(ir_attr);
    GE_ASSERT_SUCCESS(ir_attr->SetIndex(ir_index));
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::UnfoldFusedGraph(const af::ComputeGraphPtr &fused_graph,
                                            std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                            af::AscGraph &unfolded_asc_graph) {
  // step1 verify and choose loop
  std::vector<af::AxisPtr> new_loop_axes;
  GE_CHK_STATUS_RET(SelectCommonLoopAxis(asc_backend_to_asc_graph, new_loop_axes),
                    "The loop axis verification failed. Please confirm whether the fused graph [%s] is legitimate.",
                    fused_graph->GetName().c_str());
  // set loop and convert to ascgraph
  auto graph_attr = fused_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_CHECK_NOTNULL(graph_attr);
  graph_attr->axis = new_loop_axes;

  // reset data ir attr
  GE_ASSERT_SUCCESS(ReAssembleDataIrAttr(fused_graph, asc_backend_to_asc_graph),
                    "ReAssembleDataIrAttr failed, graph:[%s].", fused_graph->GetName().c_str());
  // step2 do graph unfold
  for (const auto &node : fused_graph->GetDirectNodePtr()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() == kAscGraphNodeType) {
      auto iter = asc_backend_to_asc_graph.find(node);
      GE_ASSERT_TRUE(iter != asc_backend_to_asc_graph.end());
      ascir::utils::DumpGraph(iter->second, "BeforeUnfoldAscBackend_" + iter->second.GetName());
      GE_CHK_STATUS_RET(UnfoldAscbcNode(node, iter->second, fused_graph),
                        "Unfold ascgraph node [%s] to fused graph [%s] failed.", node->GetNamePtr(),
                        fused_graph->GetName().c_str());
      ascir::utils::DumpGraph(iter->second, "AfterUnfoldAscBackend_" + iter->second.GetName());
    }
  }
  ascir::utils::DumpComputeGraph(fused_graph, "FusedGraphAfterUnfold");

  // step3 Do load cse
  GE_CHK_STATUS_RET(DoSameLoadCse(fused_graph),
                    "[Invoke][RemoveSameIndexData] Remove same index node for graph %s failed",
                    fused_graph->GetName().c_str());
  ascir::utils::DumpComputeGraph(fused_graph, "AfterDoSameLoadCse");
  // step4 reassemble io index by fused graph io nodes.
  GE_ASSERT_SUCCESS(ReAssembleOutputIndex(fused_graph), "ReAssembleOutputIndex failed, graph:[%s].",
                    fused_graph->GetName().c_str());

  // step5 Remove redundant Load, Output and Store nodes
  GE_CHK_STATUS_RET(RemoveRedundantLoads(fused_graph),
                    "[Invoke][RemoveRedundantLoads] Remove redundant Loads for graph %s failed",
                    fused_graph->GetName().c_str());
  ascir::utils::DumpComputeGraph(fused_graph, "AfterRemoveRedundantLoads");

  GE_CHK_STATUS_RET(fused_graph->TopologicalSorting(), "Failed to do topological sorting for graph:[%s].",
                    fused_graph->GetName().c_str());

  GE_ASSERT_GRAPH_SUCCESS(af::AscGraphUtils::ConvertComputeGraphToAscGraph(fused_graph, unfolded_asc_graph));

  return af::SUCCESS;
}

Status FusedGraphUnfolder::SelectCommonLoopAxis(std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                                std::vector<af::AxisPtr> &new_loop_axes) {
  GE_ASSERT_TRUE(!asc_backend_to_asc_graph.empty(),
                 "The map is empty after deserialization, which means the fused graph is valid.");
  size_t concat_dim = 0UL;
  bool has_concat = false;
  std::map<af::Node *, af::AscGraph> post_concat_node_to_asc_graph;
  std::vector<af::AxisId> loop_axis_ids;
  std::set<af::Node *> seen_nodes;
  for (auto &iter : asc_backend_to_asc_graph) {
    for (const auto &node : iter.second.GetAllNodes()) {
      if (!af::ops::IsOps<af::ascir_op::Concat>(node)) {
        continue;
      }
      GE_ASSERT_SUCCESS(ScheduleUtils::GetConcatDim(node, concat_dim));
      has_concat = true;
      auto loop_axis = iter.second.GetAllAxis();
      loop_axis_ids.resize(loop_axis.size());
      for (size_t i = 0UL; i < loop_axis.size(); ++i) {
        loop_axis_ids[i] = loop_axis[i]->id;
      }
      GE_ASSERT_SUCCESS(CollectPostConcatAscGraphs(iter.first, asc_backend_to_asc_graph, loop_axis, loop_axis_ids,
                                                   post_concat_node_to_asc_graph));
      new_loop_axes = iter.second.GetAllAxis();
      break;
    }
  }
  GE_ASSERT_TRUE(concat_dim < new_loop_axes.size(), "Concat dim [%zu] is greater than loop size:[%zu].", concat_dim,
                 new_loop_axes.size());
  GE_ASSERT_TRUE(has_concat, "Only subgraphs with concat currently support fused graphs.");

  // merge and check
  for (const auto &iter : asc_backend_to_asc_graph) {
    if (post_concat_node_to_asc_graph.count(iter.first) == 0UL) {
      GE_ASSERT_SUCCESS(ApplyMergedLoopAxis(iter.second, new_loop_axes, loop_axis_ids, concat_dim));
    }
  }

  return af::SUCCESS;
}

Status FusedGraphUnfolder::CollectPostConcatAscGraphs(
    af::Node *concat_ascbc_node, const std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
    const std::vector<af::AxisPtr> &new_loop_axes, const std::vector<af::AxisId> &loop_axis_ids,
    std::map<af::Node *, af::AscGraph> &post_concat_node_to_asc_graph) {
  std::queue<af::Node *> queue;
  std::set<af::Node *> seen_nodes;
  queue.push(concat_ascbc_node);
  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop();
    auto iter = asc_backend_to_asc_graph.find(node);
    if (seen_nodes.count(node) == 0UL && iter != asc_backend_to_asc_graph.end()) {
      GE_ASSERT_SUCCESS(DoAxisMappingForConstPostAscGraph(iter->second, new_loop_axes, loop_axis_ids),
                        "Failed to do axis mapping for graph [%s], asc_node:[%s].", iter->second.GetName().c_str(),
                        iter->first->GetNamePtr());
      post_concat_node_to_asc_graph.emplace(node, iter->second);
    }
    seen_nodes.insert(node);
    for (auto out_node : node->GetOutDataNodesPtr()) {
      queue.push(out_node);
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::MarkAllOutputAxisId(
    af::Node *concat_ascbc_node, std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
    const af::AxisId &axis_id, std::map<const af::AscGraph *, af::AxisId> &seen_graph_to_changed_axis_id,
    std::set<af::Node *> &seen_node) {
  std::queue<af::Node *> que;
  que.emplace(concat_ascbc_node);
  while (!que.empty()) {
    auto top = que.front();
    auto iter = asc_backend_to_asc_graph.find(top);
    GE_ASSERT_TRUE(iter != asc_backend_to_asc_graph.end(), "Cannot find ascgraph for node [%s].", top->GetNamePtr());
    seen_graph_to_changed_axis_id[&iter->second] = axis_id;
    GELOGD("Mark graph [%s] with id [%ld].", iter->second.GetName().c_str(), axis_id);
    seen_node.emplace(top);
    que.pop();
    for (auto &output_node : top->GetOutDataNodes()) {
      if (output_node->GetType() == kAscGraphNodeType && seen_node.count(output_node.get()) == 0UL) {
        que.emplace(output_node.get());
      }
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::MarkAllInputAxisId(af::Node *concat_input_node,
                                              std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                              const af::AxisId &axis_id,
                                              std::map<const af::AscGraph *, af::AxisId> &seen_graph_to_changed_axis_id,
                                              std::set<af::Node *> &seen_node) {
  std::queue<af::Node *> que;
  que.emplace(concat_input_node);
  while (!que.empty()) {
    auto top = que.front();
    if (top->GetType() == kAscGraphNodeType && seen_node.count(top) == 0U) {
      auto iter = asc_backend_to_asc_graph.find(top);
      GE_ASSERT_TRUE(iter != asc_backend_to_asc_graph.end(), "Cannot find ascgraph for node [%s].", top->GetNamePtr());
      seen_graph_to_changed_axis_id[&iter->second] = axis_id;
      GELOGD("Mark graph [%s] with id [%ld].", iter->second.GetName().c_str(), axis_id);
    }
    seen_node.emplace(top);
    que.pop();
    for (auto &in_node : top->GetInDataNodes()) {
      if (in_node->GetType() == kAscGraphNodeType && seen_node.count(in_node.get()) == 0UL) {
        que.emplace(in_node.get());
      }
    }
    for (auto &output_node : top->GetOutDataNodes()) {
      if (output_node->GetType() == kAscGraphNodeType && seen_node.count(output_node.get()) == 0UL) {
        que.emplace(output_node.get());
      }
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::ApplyMergedLoopAxis(const af::AscGraph &graph, const std::vector<af::AxisPtr> &new_loop_axes,
                                               const std::vector<af::AxisId> &loop_axis_ids, const size_t concat_dim) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  const auto graph_attr = compute_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  bool need_expand = false;
  auto old_axis = graph_attr->axis;
  if (old_axis.size() != loop_axis_ids.size()) {
    // 补轴场景只支持补concat_dim轴
    need_expand = true;
    GE_ASSERT_TRUE(old_axis.size() + 1UL == loop_axis_ids.size(), "Only expand concat_dim axis is supported.");
  }
  graph_attr->axis = new_loop_axes;

  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }

    node->attr.sched.axis = loop_axis_ids;
    for (auto &output : node->outputs()) {
      GE_ASSERT_NOTNULL(output);
      output->attr.axis = loop_axis_ids;
      if (!need_expand) {
        continue;
      }
      if (concat_dim == output->attr.repeats.size()) {
        output->attr.repeats.push_back(af::sym::kSymbolOne);
        output->attr.strides.push_back(af::sym::kSymbolZero);
      } else {
        GE_ASSERT_TRUE(concat_dim < output->attr.repeats.size(), "concat dim:[%zu] is invalid, node name:[%s]",
                       concat_dim, node->GetNamePtr());
        GE_ASSERT_TRUE(concat_dim < output->attr.strides.size());
        output->attr.repeats.insert(output->attr.repeats.begin() + static_cast<int64_t>(concat_dim),
                                    af::sym::kSymbolOne);
        output->attr.strides.insert(output->attr.strides.begin() + static_cast<int64_t>(concat_dim),
                                    af::sym::kSymbolZero);
      }
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::DoAxisMappingForConstPostAscGraph(const af::AscGraph &graph,
                                                             const std::vector<af::AxisPtr> &new_loop_axes,
                                                             const std::vector<af::AxisId> &loop_axis_ids) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  const auto graph_attr = compute_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  auto old_axis = graph_attr->axis;
  if (old_axis.size() == loop_axis_ids.size()) {
    return af::SUCCESS;
  }
  std::map<size_t, size_t> new_idx_to_old_idx;
  size_t old_idx = 0UL;
  const size_t old_size = old_axis.size();
  const size_t new_size = new_loop_axes.size();
  for (size_t new_idx = 0UL; new_idx < new_size; ++new_idx) {
    if (old_idx < old_size &&
        af::SymbolicUtils::StaticCheckEq(new_loop_axes[new_idx]->size, old_axis[old_idx]->size) == af::TriBool::kTrue) {
      new_idx_to_old_idx.emplace(new_idx, old_idx);
      old_idx++;
    }
  }

  GE_ASSERT_TRUE(old_idx == old_size, "Axes mapping failed, only expansion scenarios are supported, graph:[%s].",
                 graph.GetName().c_str());
  graph_attr->axis = new_loop_axes;
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    node->attr.sched.axis = loop_axis_ids;
    for (auto &output : node->outputs()) {
      output->attr.axis = loop_axis_ids;
      std::vector<af::Expression> new_repeats;
      std::vector<af::Expression> new_strides;
      const size_t axis_size = loop_axis_ids.size();
      for (size_t idx = 0UL; idx < axis_size; ++idx) {
        auto iter = new_idx_to_old_idx.find(idx);
        if (iter != new_idx_to_old_idx.end()) {
          const size_t old_index = iter->second;
          GE_ASSERT_TRUE(old_index < output->attr.repeats.size(), "Index out of bounds, node:[%s].",
                         node->GetNamePtr());
          GE_ASSERT_TRUE(old_index < output->attr.strides.size(), "Index out of bounds, node:[%s].",
                         node->GetNamePtr());
          new_repeats.push_back(output->attr.repeats[old_index]);
          new_strides.push_back(output->attr.strides[old_index]);
        } else {
          new_repeats.push_back(af::sym::kSymbolOne);
          new_strides.push_back(af::sym::kSymbolZero);
        }
      }
      output->attr.repeats = std::move(new_repeats);
      output->attr.strides = std::move(new_strides);
    }
  }

  return af::SUCCESS;
}
}  // namespace optimize
