/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_AUTOSCHEDULE_NODE_CACHE_MARKER_H
#define OPTIMIZE_AUTOSCHEDULE_NODE_CACHE_MARKER_H

#include "ascir.h"
#include "graph/node.h"

namespace af::optimize::autoschedule {
class NodeCacheMarker {
 public:
  NodeCacheMarker() = delete;
  ~NodeCacheMarker() = default;

  explicit NodeCacheMarker(::ascir::ImplGraph &graph) : graph_(graph) {};

  bool IsNodeVisited(const af::NodePtr &node) const;
  void VisitNode(const af::NodePtr &node);
  void AddToCacheStartSet(const af::NodePtr &node);

  af::ExecuteCondition DoesNodeNeedCache(const std::vector<int64_t> &in_axis, const std::vector<int64_t> &out_axis,
                                         const std::vector<af::Expression> &in_repeats,
                                         const std::vector<af::Expression> &out_repeats) const;
  af::ExecuteCondition DoesNodeNeedCache(const af::AscNodePtr &node) const;
  af::ExecuteCondition DoesNodeNeedCache(const af::NodePtr &node) const;
  af::ExecuteCondition DoesInlineNodeNeedCache(const af::NodePtr &node, int32_t brc_idx) const;

  static void MarkNodeCacheable(const af::NodePtr &node);

  void MarkNodesCacheableBottomUp(const af::AscNodePtr &node, const af::ExecuteCondition condition);
  void MarkNodesCacheableBottomUp(const af::NodePtr &node, const af::ExecuteCondition condition);
  static void MarkNodesCacheableUpBottom(const af::NodePtr &node);

  af::Status ReverseDfsCacheNode(const af::NodePtr &ge_node);
  af::Status MarkIfNodeNeedsCache();

  static af::Status GetAscNodeInputAttr(const af::NodePtr &node, int32_t idx, af::AscTensorAttr &attr);

 private:
  ::ascir::ImplGraph &graph_;
  std::set<af::NodePtr> cache_start_nodes_;
  std::set<af::NodePtr> visited_nodes_;
};
}  // namespace af::optimize::autoschedule

#endif  // OPTIMIZE_AUTOSCHEDULE_NODE_CACHE_MARKER_H
