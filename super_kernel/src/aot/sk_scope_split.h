/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sk_scope_split.h
 * \brief
 */

#ifndef __SK_SCOPE_SPLIT_H__
#define __SK_SCOPE_SPLIT_H__

#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

#include "sk_graph.h"
#include "sk_log.h"
#include "sk_lock_detector.h"
struct ScopeStreamInfo {
    uint32_t streamIdx = 0;
    uint64_t headNodeIdx = INVALID_TASK_ID;
    uint64_t tailNodeIdx = INVALID_TASK_ID;
    uint64_t nodeSize = 0;
};

struct SuperKernelScopeInfo{
    std::vector<ScopeStreamInfo> scopeStreamInfos;
    std::vector<SuperKernelBaseNode *> nodes;
    // skMainNodeId;
    // enterEventInfos: std::vector<std::pair<waitInfo, resetInfo>>;
    // exitEventInfos: std::vector<std::pair<notifyInfo, waitInfo>>;
};

// Stream state for multi-stream graph splitting
struct StreamState {
    uint64_t currentNodeIdx;    // Current node index being processed
    bool isSuspended;           // Whether the stream is suspended (waiting for notify)
    uint64_t waitingForNotify;  // Event ID that the stream is waiting for
    bool isTerminated;          // Whether the stream is terminated (due to deadlock or unfusible)

    StreamState()
        : currentNodeIdx(INVALID_TASK_ID),
          isSuspended(false),
          waitingForNotify(INVALID_TASK_ID),
          isTerminated(false) {}
};

class SuperKernelScopeSplitter {
public:
    SuperKernelScopeSplitter(SuperKernelGraph &graph) : graph(graph) { }
    ~SuperKernelScopeSplitter() = default;
    SuperKernelScopeSplitter(const SuperKernelScopeSplitter&) = delete;
    SuperKernelScopeSplitter& operator=(const SuperKernelScopeSplitter&) = delete;
    SuperKernelScopeSplitter(SuperKernelScopeSplitter&&) = default;
    SuperKernelScopeSplitter& operator=(SuperKernelScopeSplitter&&) = default;

    bool SplitGraph();
    std::vector<SuperKernelScopeInfo>& GetScopeInfos() noexcept { return scopeInfos; }

private:
    // Multi-stream splitting methods
    void InitNodeHeap(std::unordered_map<uint32_t, StreamState>& streamStates,
                     const std::set<uint64_t>& visitedNotifies,
                     const std::set<uint64_t>& processedNodes,
                     std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>>& nodeHeap,
                     LockDetector& lockDetector);

    void TryAddNodeToHeap(uint32_t streamIdx,
                         std::unordered_map<uint32_t, StreamState>& streamStates,
                         const std::set<uint64_t>& visitedNotifies,
                         const std::set<uint64_t>& processedNodes,
                         std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>>& nodeHeap,
                         LockDetector& lockDetector);

    void ProcessNotifyNode(SuperKernelBaseNode* notifyNode,
                         std::set<uint64_t>& visitedNotifies,
                         const std::set<uint64_t>& processedNodes,
                         std::unordered_map<uint32_t, StreamState>& streamStates,
                         std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>>& nodeHeap,
                         LockDetector& lockDetector);

    void ProcessResetNode(SuperKernelBaseNode* resetNode,
                        std::set<uint64_t>& visitedNotifies,
                        std::unordered_map<uint32_t, StreamState>& streamStates);

    void AddStreamInfoToScope(SuperKernelScopeInfo& scopeInfo, SuperKernelBaseNode* node);

    void SkipUnfusibleNodes(std::unordered_map<uint32_t, StreamState>& streamStates);

    void ResetStreamStates(std::unordered_map<uint32_t, StreamState>& streamStates);

    bool AllStreamsFinished(const std::unordered_map<uint32_t, StreamState>& streamStates) const;

    SuperKernelGraph& graph;
    std::vector<SuperKernelScopeInfo> scopeInfos;
};

#endif // __SK_SCOPE_SPLIT_H__