/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "schedule_group_partitioner.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "schedule_utils.h"
#include "util/mem_utils.h"
#include <queue>
#include <set>

using namespace af;

namespace optimize {
namespace {
template<typename Heap>
size_t GetMinMergeCost(const Heap &heap) {
  auto temp = heap;
  const auto e1 = temp.top();
  temp.pop();
  const auto e2 = temp.top();
  return e1.first + e2.first;
}

bool IsAxisEqual(const af::AxisPtr &lhs, const af::AxisPtr &rhs) {
  if (lhs->id != rhs->id || lhs->name != rhs->name) {
    return false;
  }
  return af::SymbolicUtils::StaticCheckEq(lhs->size, rhs->size) == af::TriBool::kTrue;
}

bool IsAxisVecEqual(const std::vector<af::AxisPtr> &lhs, const std::vector<af::AxisPtr> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!IsAxisEqual(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}
}  // namespace

Status ScheduleGroupGraphPartitioner::PartitionByConnectivity(const ::ascir::ImplGraph &optimize_graph,
                                                              std::vector<::ascir::ImplGraph> &sub_optimize_graphs,
                                                              std::vector<AscNodePtr> node_order) {
  std::map<std::string, ::ascir::NodeView> name_to_output_nodes;
  size_t num_nodes = 0;
  for (const auto &node : optimize_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetOutDataNodes().empty()) {
      name_to_output_nodes.emplace(node->GetName(), node);
      GELOGI("Output node: %s[%s]", node->GetName().c_str(), node->GetType().c_str());
    }
    ++num_nodes;
  }
  if (node_order.empty()) {
    for (const auto &name_and_node : name_to_output_nodes) {
      node_order.emplace_back(name_and_node.second);
    }
  }
  int32_t index = 0;
  std::set<af::NodePtr> visited;
  for (const auto &node : node_order) {
    const auto &root_node = node;
    if (visited.find(root_node) == visited.cend()) {
      ::ascir::ImplGraph sub_graph((optimize_graph.GetName() + "_" + std::to_string(index)).c_str());
      GE_ASSERT_TRUE(sub_graph.CopyAttrFrom(optimize_graph));
      GE_CHK_STATUS_RET(AddConnectedNodes(root_node, sub_graph, visited),
                        "Failed to add connected nodes, root_node = %s", root_node->GetNamePtr());
      GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(sub_graph));
      sub_optimize_graphs.emplace_back(sub_graph);
      ++index;
    }
  }
  if (sub_optimize_graphs.size() > 1U) {  // 切图后会产生冗余轴信息
    for (auto &sub_optimize_graph : sub_optimize_graphs) {
      GE_CHK_STATUS_RET(ScheduleUtils::RemoveUnusedAxes(sub_optimize_graph), "Failed to remove unused axes");
    }
  }
  if (visited.size() != num_nodes) {
    for (const auto &node : optimize_graph.GetAllNodes()) {
      if (visited.find(node) == visited.cend()) {
        GELOGE(ge::FAILED, "node: %s[%s] not visited", node->GetName().c_str(), node->GetType().c_str());
      }
    }
    GELOGE(ge::FAILED, "number of visited nodes = %zu, num_nodes = %zu", visited.size(), num_nodes);
    return ge::FAILED;
  }

  GELOGI("Partition success, subgraph number = %zu", sub_optimize_graphs.size());
  return ge::SUCCESS;
}

Status ScheduleGroupGraphPartitioner::NeedRefreshAxisSize(const ::ascir::ImplGraph &optimize_graph,
                                                          bool &need_refresh) {
  const auto concat_node = ScheduleUtils::FindFirstNodeOfType<af::ascir_op::Concat>(optimize_graph);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(concat_node == nullptr, ge::SUCCESS, "no Concat node was found");
  need_refresh = true;
  return ge::SUCCESS;
}

Status ScheduleGroupGraphPartitioner::CollectConnectedNodes(const af::AscNodePtr &root_node,
                                                            std::set<af::NodePtr> &visited,
                                                            std::vector<af::AscNodePtr> &asc_nodes) {
  std::list<af::NodePtr> next_nodes{root_node};
  visited.emplace(root_node);
  while (!next_nodes.empty()) {
    auto node = next_nodes.front();
    next_nodes.pop_front();
    auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
    GE_CHECK_NOTNULL(asc_node);
    asc_nodes.emplace_back(std::move(asc_node));
    for (auto &in_node : node->GetInNodes()) {
      if (visited.find(in_node) == visited.cend()) {
        next_nodes.emplace_back(in_node);
        visited.emplace(in_node);
      }
    }
    for (auto &out_node : node->GetOutNodes()) {
      if (visited.find(out_node) == visited.cend()) {
        next_nodes.emplace_back(out_node);
        visited.emplace(out_node);
      }
    }
  }
  return ge::SUCCESS;
}

Status ScheduleGroupGraphPartitioner::AddConnectedNodes(const af::AscNodePtr &root_node, ::ascir::ImplGraph &sub_graph,
                                                        std::set<af::NodePtr> &all_visited) {
  GELOGI("AddConnectedNodes in, root_node = %s[%s]", root_node->GetName().c_str(), root_node->GetType().c_str());
  std::unordered_map<std::string, af::NodePtr> all_new_nodes;
  std::set<af::NodePtr> visited;
  std::vector<af::AscNodePtr> asc_nodes;
  GE_ASSERT_SUCCESS(CollectConnectedNodes(root_node, visited, asc_nodes));
  std::sort(asc_nodes.begin(), asc_nodes.end(), CompareByNodeId);
  for (const auto &asc_node : asc_nodes) {
    const auto &op_desc = af::GraphUtils::CopyOpDesc(asc_node->GetOpDesc(), nullptr);
    GE_CHECK_NOTNULL(op_desc);
    op_desc->SetName(asc_node->GetName());
    af::Operator op = af::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    auto dst_new_node = sub_graph.AddNode(op);
    all_new_nodes[dst_new_node->GetName()] = dst_new_node;
    GE_ASSERT_TRUE(AscGraph::CopyAscNodeTensorAttr(asc_node, dst_new_node),
                   "DoCopyAscNodeTensorAttr failed, node = %s[%s]", asc_node->GetNamePtr(), asc_node->GetTypePtr());
  }

  for (const auto &src_node : visited) {
    GE_CHK_STATUS_RET(af::GraphUtils::RelinkGraphEdges(src_node, "", all_new_nodes), "RelinkGraphEdges failed");
  }
  all_visited.insert(visited.cbegin(), visited.cend());
  return ge::SUCCESS;
}

bool ScheduleGroupGraphPartitioner::CompareByNodeId(const AscNodePtr &lhs, const AscNodePtr &rhs) {
  return lhs->GetOpDesc()->GetId() < rhs->GetOpDesc()->GetId();
}

Status ScheduleGroupGraphPartitioner::RecordAxisSizes(const std::vector<af::Expression> &repeats,
                                                      const std::vector<int64_t> &axis_ids,
                                                      std::map<af::AxisId, af::Expression> &axis_id_to_size) {
  for (size_t j = 0UL; j < repeats.size(); ++j) {
    if (af::SymbolicUtils::StaticCheckEq(repeats[j], ops::One) != af::TriBool::kTrue) {
      const auto axis_id = axis_ids[j];
      if (!axis_id_to_size.emplace(axis_id, repeats[j]).second) {
        GE_LOGW_IF(af::SymbolicUtils::StaticCheckEq(repeats[j], axis_id_to_size[axis_id]) != TriBool::kTrue,
                   "inconsistent axis size, id = %ld, size0 = %s, size1 = %s", axis_id,
                   SymbolicUtils::ToString(repeats[j]).c_str(),
                   SymbolicUtils::ToString(axis_id_to_size[axis_id]).c_str());
      }
    }
  }
  return SUCCESS;
}

Status ScheduleGroupGraphPartitioner::RefreshAxisSize(const ::ascir::ImplGraph &sub_graph) {
  GELOGI("RefreshAxisSize start, graph_name = %s", sub_graph.GetName().c_str());
  const auto graph_attr = af::AscGraphUtils::GetComputeGraph(sub_graph)->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  std::map<af::AxisId, af::Expression> axis_id_to_size;
  for (const auto &node : sub_graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    for (uint32_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
      const auto &repeats = node->outputs[i].attr.repeats;
      const auto &axis_ids = node->outputs[i].attr.axis;
      GE_ASSERT_TRUE(repeats.size() == axis_ids.size(), "node: %s:%u repeats.size() = %zu, but axis_ids.size() = %zu",
                     node->GetNamePtr(), i, repeats.size(), axis_ids.size());
      GE_ASSERT_SUCCESS(RecordAxisSizes(repeats, axis_ids, axis_id_to_size));
    }
  }
  for (const auto &axis : graph_attr->axis) {
    GE_ASSERT_NOTNULL(axis);
    const auto it = axis_id_to_size.find(axis->id);
    if ((it != axis_id_to_size.cend()) && (SymbolicUtils::StaticCheckEq(axis->size, it->second) != TriBool::kTrue)) {
      GELOGI("update axis, id = %ld, name = %s, from %s to %s", axis->id, axis->name.c_str(),
             SymbolicUtils::ToString(axis->size).c_str(), SymbolicUtils::ToString(it->second).c_str());
      axis->size = it->second;
    }
  }
  GELOGI("RefreshAxisSize end, graph_name = %s", sub_graph.GetName().c_str());
  return SUCCESS;
}

bool ScheduleGroupGraphPartitioner::IsSimpleComputeGraph(const ::ascir::ImplGraph &graph, size_t &node_count) {
  node_count = 0;
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    if (!ScheduleUtils::IsCompute(node)) {
      continue;
    }
    ++node_count;
    const auto compute_type = node->attr.api.compute_type;
    if (compute_type != af::ComputeType::kComputeLoad &&
        compute_type != af::ComputeType::kComputeStore &&
        compute_type != af::ComputeType::kComputeElewise &&
        compute_type != af::ComputeType::kComputeBroadcast) {
      return false;
    }
  }
  return true;
}

std::vector<MergeableGraphs> ScheduleGroupGraphPartitioner::FindMergeableGraphs(
    const std::vector<::ascir::ImplGraph> &grouped_graphs) {
  std::vector<MergeableGraphs> mergeable_groups;
  std::vector<bool> merged(grouped_graphs.size(), false);
  std::vector<size_t> node_counts(grouped_graphs.size());

  for (size_t i = 0; i < grouped_graphs.size(); ++i) {
    if (merged[i] || !IsSimpleComputeGraph(grouped_graphs[i], node_counts[i])) {
      continue;
    }

    MergeableGraphs group;
    group.graph_indices.push_back(i);
    group.node_counts.push_back(node_counts[i]);

    const auto graph_attr_i = AscGraphUtils::GetComputeGraph(grouped_graphs[i])
        ->GetOrCreateAttrsGroup<AscGraphAttr>();
    GE_ASSERT_NOTNULL(graph_attr_i);

    for (size_t j = i + 1; j < grouped_graphs.size(); ++j) {
      if (merged[j] || !IsSimpleComputeGraph(grouped_graphs[j], node_counts[j])) {
        continue;
      }

      const auto graph_attr_j = AscGraphUtils::GetComputeGraph(grouped_graphs[j])
          ->GetOrCreateAttrsGroup<AscGraphAttr>();
      GE_ASSERT_NOTNULL(graph_attr_j);

      if (IsAxisVecEqual(graph_attr_i->axis, graph_attr_j->axis)) {
        group.graph_indices.push_back(j);
        group.node_counts.push_back(node_counts[j]);
        merged[j] = true;
      }
    }

    if (group.graph_indices.size() > 1) {
      mergeable_groups.push_back(std::move(group));
    }
  }

  return mergeable_groups;
}

Status ScheduleGroupGraphPartitioner::MergeGraphs(::ascir::ImplGraph &dst,
                                                   const std::vector<const ::ascir::ImplGraph *> &srcs) {
  for (const auto *src : srcs) {
    GELOGI("MergeGraphs: merging %s into %s", src->GetName().c_str(), dst.GetName().c_str());

    std::unordered_map<std::string, af::NodePtr> all_new_nodes;
    for (const auto &node : src->GetAllNodes()) {
      const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
      GE_CHECK_NOTNULL(asc_node);

      const auto &op_desc = af::GraphUtils::CopyOpDesc(asc_node->GetOpDesc(), nullptr);
      GE_CHECK_NOTNULL(op_desc);
      af::Operator op = af::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
      auto dst_new_node = dst.AddNode(op);
      GE_ASSERT_TRUE(AscGraph::CopyAscNodeTensorAttr(asc_node, dst_new_node),
                     "CopyAscNodeTensorAttr failed, node = %s[%s]", asc_node->GetNamePtr(), asc_node->GetTypePtr());
      all_new_nodes.emplace(asc_node->GetName(), std::move(dst_new_node));
    }

    for (const auto &node : src->GetAllNodes()) {
      auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
      GE_CHK_STATUS_RET(af::GraphUtils::RelinkGraphEdges(asc_node, "", all_new_nodes), "RelinkGraphEdges failed");
    }
  }

  GE_CHK_STATUS_RET(ScheduleUtils::TopologicalSorting(dst), "Failed to sort merged graph");
  return ge::SUCCESS;
}

MergePlan ScheduleGroupGraphPartitioner::ResolveMergePlan(const std::vector<MergeableGraphs> &mergeable_groups,
                                                           size_t reductions_needed) {
  using NodeCountAndGraphIndex = std::pair<size_t, size_t>;
  using MinHeap = std::priority_queue<NodeCountAndGraphIndex, std::vector<NodeCountAndGraphIndex>, std::greater<> >;
  std::vector<MinHeap> heaps(mergeable_groups.size());
  std::set<std::pair<size_t, size_t> > global_heap;

  for (size_t grp_idx = 0UL; grp_idx < mergeable_groups.size(); ++grp_idx) {
    for (size_t i = 0UL; i < mergeable_groups[grp_idx].graph_indices.size(); ++i) {
      heaps[grp_idx].emplace(mergeable_groups[grp_idx].node_counts[i], mergeable_groups[grp_idx].graph_indices[i]);
    }
    global_heap.insert({GetMinMergeCost(heaps[grp_idx]), grp_idx});
  }

  std::unordered_map<size_t, size_t> dst_to_group;

  MergePlan plan;
  size_t reductions = reductions_needed;
  while (reductions > 0 && (!global_heap.empty())) {
    const auto grp_idx = global_heap.begin()->second;
    global_heap.erase(global_heap.begin());

    const auto e1 = heaps[grp_idx].top();
    heaps[grp_idx].pop();
    const auto e2 = heaps[grp_idx].top();
    heaps[grp_idx].pop();

    auto it = dst_to_group.find(e2.second);
    if (it != dst_to_group.end()) {
      plan.merge_groups[it->second].push_back(e1.second);
    } else {
      const size_t group_idx = plan.merge_groups.size();
      plan.merge_groups.push_back({e2.second, e1.second});
      dst_to_group[e2.second] = group_idx;
    }
    --reductions;

    heaps[grp_idx].emplace(e1.first + e2.first, e2.second);

    if (heaps[grp_idx].size() > 1) {
      global_heap.insert({GetMinMergeCost(heaps[grp_idx]), grp_idx});
    }
  }

  return plan;
}

Status ScheduleGroupGraphPartitioner::ReduceGraphCount(std::vector<::ascir::ImplGraph> &grouped_graphs,
                                                       size_t target_count) {
  if (grouped_graphs.size() <= target_count) {
    return ge::SUCCESS;
  }

  GELOGI("ReduceGraphCount: reducing from %zu to %zu graphs", grouped_graphs.size(), target_count);

  const auto mergeable_groups = FindMergeableGraphs(grouped_graphs);
  if (mergeable_groups.empty()) {
    GELOGI("No mergeable graphs found, cannot reduce");
    return ge::SUCCESS;
  }

  const auto plan = ResolveMergePlan(mergeable_groups, grouped_graphs.size() - target_count);
  if (plan.merge_groups.empty()) {
    GELOGI("No merge plan generated, cannot reduce");
    return ge::SUCCESS;
  }

  for (const auto &group : plan.merge_groups) {
    const size_t dst = group[0];
    std::vector<const ::ascir::ImplGraph *> src_ptrs;
    src_ptrs.reserve(group.size() - 1);
    for (size_t i = 1; i < group.size(); ++i) {
      src_ptrs.push_back(&grouped_graphs[group[i]]);
    }

    GE_CHK_STATUS_RET(MergeGraphs(grouped_graphs[dst], src_ptrs), "Failed to merge graphs into %zu", dst);
  }

  std::set<size_t> to_erase;
  for (const auto &group : plan.merge_groups) {
    for (size_t i = 1UL; i < group.size(); ++i) {
      to_erase.insert(group[i]);
    }
  }

  for (auto it = to_erase.rbegin(); it != to_erase.rend(); ++it) {
    grouped_graphs.erase(grouped_graphs.begin() + static_cast<int32_t>(*it));
  }

  GELOGI("ReduceGraphCount: final count is %zu graphs", grouped_graphs.size());
  return ge::SUCCESS;
}
}  // namespace optimize