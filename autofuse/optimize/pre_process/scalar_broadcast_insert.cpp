/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pre_process/scalar_broadcast_insert.h"

#include <string>
#include <vector>

#include "ascir_ops.h"
#include "graph/utils/graph_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "ascgen_log.h"

namespace af::pre_process {
namespace {
bool IsSkipDownstreamType(const af::NodePtr &node) {
  const auto &type = node->GetType();
  return type == af::ascir_op::Broadcast::Type || type == af::ascir_op::Data::Type ||
         type == af::ascir_op::Output::Type;
}

af::AscNodePtr BuildBroadcastNode(af::AscGraph &asc_graph, const af::AscNodePtr &scalar_node,
                                  const af::AscNodePtr &ref_node) {
  std::string brc_name = scalar_node->GetName() + "_broadcast";
  af::ascir_op::Broadcast brc(brc_name.c_str());
  auto b_node = asc_graph.AddNode(brc);
  b_node->attr.sched = ref_node->attr.sched;
  b_node->outputs[0].attr = ref_node->outputs[0].attr;
  b_node->outputs[0].attr.dtype = scalar_node->outputs[0].attr.dtype;
  return b_node;
}

Status InsertBroadcastAfterScalar(af::AscGraph &asc_graph, const af::AscNodePtr &scalar_node, bool &inserted) {
  auto out_anchor = scalar_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_anchor);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  if (peer_in_anchors.empty()) {
    return ge::SUCCESS;
  }

  std::vector<af::InDataAnchorPtr> compute_anchors;
  for (const auto &peer : peer_in_anchors) {
    GE_ASSERT_NOTNULL(peer);
    GE_ASSERT_NOTNULL(peer->GetOwnerNode());
    if (!IsSkipDownstreamType(peer->GetOwnerNode())) {
      compute_anchors.push_back(peer);
    }
  }
  if (compute_anchors.empty()) {
    return ge::SUCCESS;
  }

  auto ref_node = std::dynamic_pointer_cast<af::AscNode>(compute_anchors[0]->GetOwnerNode());
  GE_ASSERT_NOTNULL(ref_node);
  auto b_node = BuildBroadcastNode(asc_graph, scalar_node, ref_node);
  auto b_out = b_node->GetOutDataAnchor(0);
  for (const auto &dst : compute_anchors) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(out_anchor, dst));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(b_out, dst));
  }
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(out_anchor, b_node->GetInDataAnchor(0)));

  GELOGD("insert broadcast %s after scalar %s in graph %s.", b_node->GetName().c_str(), scalar_node->GetName().c_str(),
         asc_graph.GetName().c_str());
  inserted = true;
  return ge::SUCCESS;
}
}  // namespace

ge::Status InsertBroadcastAfterScalarForAscGraph(af::AscGraph &asc_graph) {
  std::vector<af::AscNodePtr> scalar_nodes;
  for (const auto &node : asc_graph.GetAllNodes()) {
    if (node->GetType() == af::ascir_op::Scalar::Type) {
      scalar_nodes.push_back(node);
    }
  }

  bool inserted = false;
  for (const auto &scalar_node : scalar_nodes) {
    GE_ASSERT_SUCCESS(InsertBroadcastAfterScalar(asc_graph, scalar_node, inserted));
  }

  if (inserted) {
    GE_ASSERT_GRAPH_SUCCESS(af::AscGraphUtils::GetComputeGraph(asc_graph)->TopologicalSorting());
  }
  return ge::SUCCESS;
}
}  // namespace af::pre_process
