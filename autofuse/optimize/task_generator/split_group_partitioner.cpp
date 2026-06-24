/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "optimize/task_generator/split_group_partitioner.h"

#include <queue>

#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/utils/graph_utils.h"

#include "ascir_ops.h"
#include "platform/platform_factory.h"

namespace optimize {
namespace {
constexpr uint32_t kMaxOutputNum = 48U;
constexpr int32_t kAlignment = 32;
constexpr int32_t kMinSizeForSmallTail = kAlignment * 2;
}  // namespace

SplitGroupPartitioner::SplitGroupPartitioner(af::AscNodePtr split_node, size_t split_dim)
    : split_node_(std::move(split_node)), split_dim_(split_dim) {}

Status SplitGroupPartitioner::Initialize() {
  constexpr uint32_t kLargeOutputNum = 512;
  // 防止group过小
  const auto &output_attr = split_node_->outputs[0].attr;
  dtype_size_ = ge::GetSizeByDataType(output_attr.dtype);
  GE_ASSERT_TRUE(dtype_size_ > 0, "unsupported dtype: %d, size = %ld", static_cast<int32_t>(output_attr.dtype),
                 dtype_size_);
  auto platform = PlatformFactory::GetInstance().GetPlatform();
  GE_ASSERT_NOTNULL(platform);
  group_type_to_limit_[kGroupTypeDefault] = kMaxBlockSize / dtype_size_;
  group_type_to_limit_[kGroupTypeAligned] = kMaxBlockSize / dtype_size_;
  group_type_to_limit_[kGroupTypeSmallTail] = kMaxBlockSizeForSmallTail;
  const uint32_t num_outputs = split_node_->outputs().size();
  const uint32_t kMinGroupNum = 2U;
  max_output_num_per_group_ = num_outputs / kMinGroupNum;
  if (num_outputs % kMinGroupNum != 0) {
    max_output_num_per_group_ += 1;
  }
  const uint32_t max_output_num = num_outputs >= kLargeOutputNum ? kMaxOutputNum : 16U;
  max_output_num_per_group_ = std::min(max_output_num_per_group_, max_output_num);
  GELOGI("input_num = %u, max_output_num_per_group_ = %u", num_outputs, max_output_num_per_group_);
  return ge::SUCCESS;
}

Status SplitGroupPartitioner::PartitionGroups([[maybe_unused]] const std::vector<SplitGroup> &groups) {
  GE_ASSERT_SUCCESS(Initialize());
  return ge::SUCCESS;
}

Status SplitGroupPartitioner::RecomputeDiffAxes() {
  // 1. 获取 Split 的总输出个数，并初始化默认 Group
  const auto num_outputs = split_node_->GetAllOutDataAnchors().size();
  for (uint32_t i = 0U; i < num_outputs; ++i) {
    groups_.emplace_back(SplitGroup{i, i + 1, kGroupTypeDefault, 0});
  }

  // 2. 触发跨组重计算
  GE_ASSERT_SUCCESS(RecomputeConsumersCrossGroups());
  return ge::SUCCESS;
}

Status SplitGroupPartitioner::RecomputeConsumersCrossGroups() {
  for (const auto &group : groups_) {
    std::set<af::InDataAnchor*> visited_in_anchors;
    std::map<std::string, af::AscNodePtr> name_to_new_node;

    for (size_t i = group.start; i < group.end; ++i) {
      GELOGD("output[%zu] check recompute start", i);
      auto const out_anchor = split_node_->GetAllOutDataAnchors().at(i);
      if (out_anchor == nullptr) {
        continue;
      }

      int32_t depth = 1024;
      while (--depth >= 0) {
        af::InDataAnchor *to_split = nullptr;
        // 找出需要拆分的节点
        GE_ASSERT_SUCCESS(FindFirstMultiRefAncestors(out_anchor, i, visited_in_anchors, to_split));

        if (to_split == nullptr) {
          break; // 找干净了，跳出当前输出端口的追溯
        }

        // 执行局部克隆与连线、拆分
        GE_ASSERT_SUCCESS(RecomputeUpstreamNodes(to_split, i, name_to_new_node));
      }
      GE_ASSERT_TRUE(depth >= 0);
    }
  }
  return ge::SUCCESS;
}

bool SplitGroupPartitioner::HasCrossBranchConflictAndSizeDiff(af::OutDataAnchor *start_out_anchor,
                                                              const af::OutDataAnchorPtr &out_anchor,
                                                              size_t branch_idx) const {
  if ((start_out_anchor == nullptr) || (out_anchor == nullptr) || (split_node_ == nullptr)) {
    return false;
  }

  std::queue<af::OutDataAnchor*> fw_queue({start_out_anchor});
  std::set<af::OutDataAnchor*> fw_visited({start_out_anchor});

  // 开始向下游执行前向 BFS 搜索，探查当前共享数据流的去向
  while (!fw_queue.empty()) {
    af::OutDataAnchor *fw_out = fw_queue.front();
    fw_queue.pop();

    for (const auto &fw_peer_in : fw_out->GetPeerInDataAnchors()) {
      if (fw_peer_in == nullptr) {
        continue;
      }
      auto ds_node = std::dynamic_pointer_cast<af::AscNode>(fw_peer_in->GetOwnerNode());
      if (ds_node == nullptr) {
        continue;
      }

      // 检查该下游算子的输入是否反向连接到了同一个 Split 的其他输出分支
      for (const auto &ds_in : ds_node->GetAllInDataAnchorsPtr()) {
        if (ds_in == nullptr) {
          continue;
        }
        auto split_peer_out = ds_in->GetPeerOutAnchor();
        if ((split_peer_out != nullptr) && 
            (split_peer_out->GetOwnerNode() == split_node_) && 
            (static_cast<size_t>(split_peer_out->GetIdx()) != branch_idx)) {
          const auto &cur_expr = split_node_->outputs[out_anchor->GetIdx()].attr.repeats[split_dim_];
          const auto &target_expr = split_node_->outputs[split_peer_out->GetIdx()].attr.repeats[split_dim_];

          // 若两个split的轴大小一致，可以不用拆分
          return af::SymbolicUtils::StaticCheckEq(cur_expr, target_expr) != af::TriBool::kTrue;
        }
      }

      // 继续向下游扩散前向搜索队列
      for (const auto &ds_out : ds_node->GetAllOutDataAnchorsPtr()) {
        if ((ds_out != nullptr) && (fw_visited.insert(ds_out).second)) {
          fw_queue.push(ds_out);
        }
      }
    }
  }
  return false;
}

Status SplitGroupPartitioner::FindFirstMultiRefAncestors(const af::OutDataAnchorPtr &out_anchor,
                                                         size_t branch_idx,
                                                         std::set<af::InDataAnchor*> &visited_in_anchors,
                                                         af::InDataAnchor* &to_split) const {
  if ((split_node_ == nullptr) || (out_anchor == nullptr)) {
    return ge::SUCCESS;
  }

  std::queue<af::InDataAnchor*> in_anchors;

  // 1. 初始化搜集：提取当前 Split 分支所有直接下游消费算子的“其他输入”
  for (const auto &peer : out_anchor->GetPeerInDataAnchors()) {
    if (peer == nullptr) {
      continue;
    }
    auto consumer = std::dynamic_pointer_cast<af::AscNode>(peer->GetOwnerNode());
    if (consumer == nullptr) {
      continue;
    }
    for (const auto &in_data : consumer->GetAllInDataAnchorsPtr()) {
      if ((in_data != nullptr) && (in_data != peer.get())) {
        in_anchors.push(in_data);
      }
    }
  }

  // 2. 逆向主循环：向计算图的源头深处（Backward BFS）回溯寻找公共祖先
  while (!in_anchors.empty()) {
    af::InDataAnchor *cur_in_anchor = in_anchors.front();
    in_anchors.pop();

    auto shared_out_anchor = cur_in_anchor->GetPeerOutAnchor();
    if (shared_out_anchor == nullptr) {
      continue;
    }
    auto shared_node = std::dynamic_pointer_cast<af::AscNode>(shared_out_anchor->GetOwnerNode());
    if (shared_node == nullptr) {
      continue;
    }

    // 去重判断
    if (visited_in_anchors.emplace(cur_in_anchor).second) {
      // 需要拆分节点判定：多输出&&split轴大小不一致，且下游节点输入反向连回到原来split的另一组输出
      if ((shared_out_anchor->GetPeerInDataAnchors().size() > 1) &&
          HasCrossBranchConflictAndSizeDiff(shared_out_anchor.get(), out_anchor, branch_idx)) {
        GELOGD("Found conflicting multi-ref ancestor at node: %s", shared_node->GetNamePtr());

        // 成功锁定引起冲突的数据截断点，返回给外层触发重计算
        to_split = cur_in_anchor;
        return ge::SUCCESS;
      }
    }

    // 若当前公共祖先无问题，则顺着它的所有输入端继续向更早期的祖先追溯
    for (const auto &upstream_in : shared_node->GetAllInDataAnchorsPtr()) {
      if (upstream_in != nullptr) {
        in_anchors.push(upstream_in);
      }
    }
  }
  return ge::SUCCESS;
}

Status SplitGroupPartitioner::RecomputeUpstreamNodes(af::InDataAnchor *to_split, size_t branch_idx,
                                                     std::map<std::string, af::AscNodePtr> &name_to_new_node) {
  auto shared_out_anchor = to_split->GetPeerOutAnchor();
  if (shared_out_anchor == nullptr) {
    return ge::SUCCESS;
  }
  auto shared_node = std::dynamic_pointer_cast<af::AscNode>(shared_out_anchor->GetOwnerNode());
  if (shared_node == nullptr) {
    return ge::SUCCESS;
  }

  ascir::ImplGraph impl_graph("");
  GE_ASSERT_SUCCESS(af::AscGraphUtils::FromComputeGraph(split_node_->GetOwnerComputeGraph(), impl_graph));

  // 保证同一分支内共享同一个克隆出来的上游链，避免无意义的二次克隆
  std::string unique_key = shared_node->GetName() + "_branch_" + std::to_string(branch_idx);
  af::AscNodePtr dst_new_node = name_to_new_node[unique_key];

  if (dst_new_node == nullptr) {
    const auto &op_desc = af::GraphUtils::CopyOpDesc(shared_node->GetOpDesc(), nullptr);
    if (op_desc == nullptr) {
      return af::FAILED;
    }

    op_desc->SetName(shared_node->GetName() + "_recompute_split_" + std::to_string(branch_idx));
    af::Operator op = af::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    dst_new_node = impl_graph.AddNode(op);

    GE_ASSERT_TRUE(af::AscGraph::CopyAscNodeTensorAttr(shared_node, dst_new_node),
                   "DoCopyAscNodeTensorAttr failed, node = %s[%s]", 
                   shared_node->GetNamePtr(), shared_node->GetTypePtr());

    // 联动恢复克隆节点自身的输入边
    for (const auto &shared_in_anchor : shared_node->GetAllInDataAnchorsPtr()) {
      if (shared_in_anchor != nullptr) {
        const auto upstream_out_anchor = shared_in_anchor->GetPeerOutAnchor();
        if (upstream_out_anchor != nullptr) {
          GE_ASSERT_GRAPH_SUCCESS(
              af::GraphUtils::AddEdge(upstream_out_anchor, dst_new_node->GetInDataAnchor(shared_in_anchor->GetIdx())));
        }
      }
    }
    name_to_new_node[unique_key] = dst_new_node;
    GELOGI("Successfully cloned shared node to [%s]", dst_new_node->GetNamePtr());
  }

  // 执行断线：将原本指向老 shared_node 的输入边，插到新克隆的独立的 dst_new_node 上
  auto curr_node = std::dynamic_pointer_cast<af::AscNode>(to_split->GetOwnerNode());
  if (curr_node != nullptr) {
    auto curr_in_anchor_ptr = curr_node->GetInDataAnchor(to_split->GetIdx());
    if (curr_in_anchor_ptr != nullptr) {
      GELOGI("Disconnecting [%s] -> [%s] and connecting [%s] -> [%s]",
            shared_node->GetNamePtr(), curr_node->GetNamePtr(), dst_new_node->GetNamePtr(), curr_node->GetNamePtr());

      curr_in_anchor_ptr->UnlinkAll();
      GE_ASSERT_GRAPH_SUCCESS(
          af::GraphUtils::AddEdge(dst_new_node->GetOutDataAnchor(shared_out_anchor->GetIdx()), curr_in_anchor_ptr));
    }
  }

  return ge::SUCCESS;
}

}  // namespace optimize
