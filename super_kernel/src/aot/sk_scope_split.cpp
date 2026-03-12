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
 * \file sk_scope_split.cpp
 * \brief
 */

#include "sk_scope_split.h"
#include "sk_lock_detector.h"
#include "sk_common.h"
#include <queue>
#include <unordered_map>
#include <set>

void SuperKernelScopeSplitter::AddStreamInfoToScope(SuperKernelScopeInfo& scopeInfo, SuperKernelBaseNode* node) {
    uint32_t streamIdx = node->GetStreamIdxInGraph();
    SK_LOGD("AddStreamInfoToScope: node %lu (stream=%u, type=%s)", 
            node->GetNodeId(), streamIdx, to_string(node->GetNodeType()));
    auto it = std::find_if(scopeInfo.scopeStreamInfos.begin(), scopeInfo.scopeStreamInfos.end(),
                         [streamIdx](const ScopeStreamInfo& info) {
                             return info.streamIdx == streamIdx;
                         });
    if (it == scopeInfo.scopeStreamInfos.end()) {
        ScopeStreamInfo newInfo;
        newInfo.streamIdx = streamIdx;
        newInfo.headNodeIdx = node->GetNodeId();
        newInfo.tailNodeIdx = node->GetNodeId();
        newInfo.nodeSize = 1;
        scopeInfo.scopeStreamInfos.push_back(std::move(newInfo));
        SK_LOGD("Created new stream info for stream %u, headNode=%lu", streamIdx, node->GetNodeId());
    } else {
        it->tailNodeIdx = node->GetNodeId();
        it->nodeSize++;
        SK_LOGD("Updated stream info for stream %u, tailNode=%lu, nodeSize=%lu", 
                streamIdx, node->GetNodeId(), it->nodeSize);
    }
}

void SuperKernelScopeSplitter::SkipUnfusibleNodes() {
    for (auto& pair : streamStates_) {
        SK_LOGI("SkipUnfusibleNodes: stream %u, currentNodeIdx=%lu",
                 pair.first, pair.second.currentNodeIdx);
        while (pair.second.currentNodeIdx != INVALID_TASK_ID) {
            SuperKernelBaseNode* node = graph.GetNodeById(pair.second.currentNodeIdx);
            SK_LOGI("Checking node %lu, IsFusible=%d", pair.second.currentNodeIdx, node ? node->IsFusible() : -1);
            if (node != nullptr && !node->IsFusible()) {
                SK_LOGI("Skipping permanently unfusible node %lu (stream %u), nextNodeId=%lu", 
                        pair.second.currentNodeIdx, pair.first, node->GetNextNodeId());
                pair.second.currentNodeIdx = node->GetNextNodeId();
            } else {
                break;
            }
        }
        SK_LOGI("SkipUnfusibleNodes: stream %u, after skipping currentNodeIdx=%lu",
                 pair.first, pair.second.currentNodeIdx);
    }
}

void SuperKernelScopeSplitter::ResetStreamStates() {
    SK_LOGD("ResetStreamStates: resetting %zu stream states", streamStates_.size());
    for (auto& pair : streamStates_) {
        SK_LOGD("ResetStreamStates: stream %u, currentNodeIdx=%lu, isTerminated=%d, isSuspended=%d",
                pair.first, pair.second.currentNodeIdx, pair.second.isTerminated, pair.second.isSuspended);
        pair.second.isTerminated = false;
        pair.second.isSuspended = false;
        pair.second.waitingForNotify = INVALID_TASK_ID;
    }
    SkipUnfusibleNodes();
}

bool SuperKernelScopeSplitter::AllStreamsFinished() const {
    for (const auto& pair : streamStates_) {
        if (!pair.second.isTerminated && !pair.second.isSuspended &&
            pair.second.currentNodeIdx != INVALID_TASK_ID) {
            return false;
        }
    }
    return true;
}

void SuperKernelScopeSplitter::ResetSplittingState() {
    streamStates_.clear();
    visitedNotifies_.clear();
    processedNodes_.clear();
    while (!nodeHeap_.empty()) {
        nodeHeap_.pop();
    }
    lockDetector.Reset(graph);
    currentScopeBitFlags_.reset();
}

std::string SuperKernelScopeSplitter::GetScopeNamesFromBitFlags(
        const std::bitset<MAX_SCOPE_NUM>& scopeBitFlags) const {
    std::string scopeNames;
    for (size_t bit = 0; bit < MAX_SCOPE_NUM && bit < scopeBitFlags.size(); ++bit) {
        if (scopeBitFlags.test(bit)) {
            auto it = graph.scopeIdxToName.find(static_cast<uint32_t>(bit));
            if (it != graph.scopeIdxToName.end()) {
                if (!scopeNames.empty()) scopeNames += ", ";
                scopeNames += "'";
                scopeNames += it->second;
                scopeNames += "'";
            }
        }
    }
    return scopeNames.empty() ? "(none)" : scopeNames;
}

void SuperKernelScopeSplitter::PrintScopeNodes(size_t scopeIdx, const SuperKernelScopeInfo& scope) const {
    std::string nodeDetails;
    for (const auto* n : scope.nodes) {
        if (!nodeDetails.empty()) nodeDetails += ", ";
        nodeDetails += std::to_string(n->GetNodeId());
        nodeDetails += "(";
        nodeDetails += to_string(n->GetNodeType());
        nodeDetails += ",stream=";
        nodeDetails += std::to_string(n->GetStreamIdxInGraph());
        nodeDetails += ")";
    }
    SK_LOGI("  Scope %zu nodes: [%s]", scopeIdx, nodeDetails.c_str());
}

void SuperKernelScopeSplitter::PrintScopeStreamInfos(size_t scopeIdx, const SuperKernelScopeInfo& scope) const {
    for (size_t j = 0; j < scope.scopeStreamInfos.size(); ++j) {
        const auto& streamInfo = scope.scopeStreamInfos[j];
        SK_LOGI("  Scope %zu StreamInfo[%zu]: streamIdx=%u, headNode=%lu, tailNode=%lu, nodeSize=%lu",
                scopeIdx, j, streamInfo.streamIdx, streamInfo.headNodeIdx, 
                streamInfo.tailNodeIdx, streamInfo.nodeSize);
    }
}

void SuperKernelScopeSplitter::PrintScopeResults() const {
    SK_LOGI("========== Multi-stream graph splitting complete, total scopes: %zu ==========", 
            scopeInfos.size());
    for (size_t i = 0; i < scopeInfos.size(); ++i) {
        const auto& scope = scopeInfos[i];
        std::string scopeNames = GetScopeNamesFromBitFlags(scope.scopeBitFlags);
        SK_LOGI("Scope %zu: %zu nodes, %zu streams, scopeBitFlags=%s, scopeNames=[%s]",
                i, scope.nodes.size(), scope.scopeStreamInfos.size(),
                scope.scopeBitFlags.to_string().substr(0, MAX_SCOPE_NUM).c_str(),
                scopeNames.c_str());
        PrintScopeNodes(i, scope);
        PrintScopeStreamInfos(i, scope);
    }
}

bool SuperKernelScopeSplitter::DetermineCurrentScopeBitFlags() {
    uint64_t minNodeIdx = UINT64_MAX;
    SuperKernelBaseNode* minNode = nullptr;
    
    for (const auto& pair : streamStates_) {
        if (!pair.second.isTerminated && !pair.second.isSuspended && 
            pair.second.currentNodeIdx != INVALID_TASK_ID &&
            pair.second.currentNodeIdx < minNodeIdx) {
            SuperKernelBaseNode* node = graph.GetNodeById(pair.second.currentNodeIdx);
            if (node != nullptr && node->IsFusible()) {
                minNodeIdx = pair.second.currentNodeIdx;
                minNode = node;
            }
        }
    }
    
    if (minNode != nullptr) {
        currentScopeBitFlags_ = minNode->GetScopeBitFlags();
        SK_LOGI("Determined scopeBitFlags from min node %lu (idx=%lu): %s", 
                minNode->GetNodeId(), minNodeIdx,
                currentScopeBitFlags_.to_string().substr(0, MAX_SCOPE_NUM).c_str());
        return true;
    }
    
    SK_LOGI("No fusible node found to determine scopeBitFlags");
    return false;
}

void SuperKernelScopeSplitter::InitNodeHeap() {
    SK_LOGD("InitNodeHeap: iterating %zu streams", streamStates_.size());
    size_t heapSizeBefore = nodeHeap_.size();
    for (auto& pair : streamStates_) {
        TryAddNodeToHeap(pair.first);
    }
    SK_LOGD("InitNodeHeap: heap size changed from %zu to %zu", heapSizeBefore, nodeHeap_.size());
}

void SuperKernelScopeSplitter::TryAddNodeToHeap(uint32_t streamIdx) {
    StreamState& state = streamStates_[streamIdx];
    SK_LOGI("TryAddNodeToHeap: stream %u, currentNodeIdx=%lu, isTerminated=%d, isSuspended=%d",
            streamIdx, state.currentNodeIdx, state.isTerminated, state.isSuspended);

    // Condition 1: Stream is terminated, suspended, or finished
    if (state.isTerminated || state.isSuspended || state.currentNodeIdx == INVALID_TASK_ID) {
        return;
    }

    SuperKernelBaseNode* nextNode = graph.GetNodeById(state.currentNodeIdx);
    if (nextNode == nullptr) {
        state.currentNodeIdx = INVALID_TASK_ID;
        return;
    }

    // Condition 2: Node already processed
    if (processedNodes_.find(state.currentNodeIdx) != processedNodes_.end()) {
        SK_LOGW("Node %lu already processed, skip", state.currentNodeIdx);
        state.currentNodeIdx = nextNode->GetNextNodeId();
        return;
    }

    // Condition 3: Check scopeBitFlags match
    // Nodes can only fuse with nodes having the same scopeBitFlags
    if (nextNode->GetScopeBitFlags() != currentScopeBitFlags_) {
        state.isTerminated = true;
        SK_LOGI("Node %lu (stream=%u) has different scopeBitFlags, terminate stream in current scope",
                nextNode->GetNodeId(), streamIdx);
        return;
    }

    // Condition 4: Permanently unfusible node
    if (!nextNode->IsFusible()) {
        state.isTerminated = true;
        SK_LOGI("Node %lu (stream=%lu, type=%s) is permanently unfusible, terminate stream in current scope",
                nextNode->GetNodeId(), streamIdx, to_string(nextNode->GetNodeType()));
        return;
    }

    // Condition 5: Wait node - special handling
    if (nextNode->GetNodeType() == SkNodeType::NODE_WAIT) {
        HandleWaitNode(nextNode, streamIdx);
        return;
    }

    // Condition 6: Other node types - check deadlock
    if (lockDetector.IsFusible(*nextNode, graph)) {
        nodeHeap_.push(state.currentNodeIdx);
        SK_LOGD("Node %lu (stream=%lu, type=%s) fusible, add to heap", 
                nextNode->GetNodeId(), streamIdx, to_string(nextNode->GetNodeType()));
    } else {
        state.isTerminated = true;
        SK_LOGI("Node %lu (stream=%lu, type=%s) causes deadlock, terminate stream",
                nextNode->GetNodeId(), streamIdx, to_string(nextNode->GetNodeType()));
    }
}

void SuperKernelScopeSplitter::HandleWaitNode(SuperKernelBaseNode* waitNode, uint32_t streamIdx) {
    StreamState& state = streamStates_[streamIdx];
    uint64_t notifyId = waitNode->GetCorrespondingNotifyNodeId();
    SuperKernelBaseNode* notifyNode = graph.GetNodeById(notifyId);
    
    if (notifyNode == nullptr) {
        SK_LOGW("Notify node %lu not found", notifyId);
        return;
    }
    
    uint64_t eventId = notifyNode->GetEventId();
    if (visitedNotifies_.find(notifyId) != visitedNotifies_.end()) {
        // Notify already visited, check deadlock
        if (lockDetector.IsFusible(*waitNode, graph)) {
            nodeHeap_.push(state.currentNodeIdx);
            SK_LOGD("Wait node %lu (stream=%u, eventId=%lu): notify visited, no deadlock, add to heap", 
                    waitNode->GetNodeId(), streamIdx, eventId);
        } else {
            state.isTerminated = true;
            SK_LOGI("Wait node %lu (stream=%u, eventId=%lu): notify visited but deadlock, terminate",
                    waitNode->GetNodeId(), streamIdx, eventId);
        }
    } else {
        state.isSuspended = true;
        state.waitingForNotify = eventId;
        SK_LOGD("Wait node %lu (stream=%u, eventId=%lu): notify not visited, suspend stream", 
                waitNode->GetNodeId(), streamIdx, eventId);
    }
}

void SuperKernelScopeSplitter::ProcessNotifyNode(SuperKernelBaseNode* notifyNode) {
    uint64_t eventId = notifyNode->GetEventId();
    const auto& eventInfo = graph.eventToNodes.at(eventId);
    visitedNotifies_.insert(notifyNode->GetNodeId());
    SK_LOGD("Notify node %lu (stream=%lu, eventId=%lu) marked as visited", 
            notifyNode->GetNodeId(), notifyNode->GetStreamIdxInGraph(), eventId);

    for (uint64_t waitNodeId : eventInfo.waitNodeIdList) {
        SuperKernelBaseNode* waitNode = graph.GetNodeById(waitNodeId);
        if (waitNode == nullptr) {
            SK_LOGW("Wait node %lu not found", waitNodeId);
            continue;
        }

        uint32_t streamIdx = waitNode->GetStreamIdxInGraph();
        StreamState& state = streamStates_[streamIdx];

        if (state.isSuspended && state.waitingForNotify == eventId) {
            state.isSuspended = false;
            state.waitingForNotify = INVALID_TASK_ID;
            SK_LOGD("Resume stream %lu (was waiting for eventId=%lu)", streamIdx, eventId);
            TryAddNodeToHeap(streamIdx);
        }
    }
}

void SuperKernelScopeSplitter::ProcessResetNode(SuperKernelBaseNode* resetNode) {
    uint64_t eventId = resetNode->GetEventId();
    SK_LOGD("ProcessResetNode: resetNode=%lu, eventId=%lu", resetNode->GetNodeId(), eventId);
    const auto& eventInfo = graph.eventToNodes.at(eventId);

    size_t erasedCount = visitedNotifies_.erase(eventInfo.notifyNodeId);
    SK_LOGD("ProcessResetNode: erased notify %lu from visitedNotifies (erasedCount=%zu)", 
            eventInfo.notifyNodeId, erasedCount);

    for (uint64_t waitNodeId : eventInfo.waitNodeIdList) {
        SuperKernelBaseNode* waitNode = graph.GetNodeById(waitNodeId);
        if (waitNode == nullptr) {
            SK_LOGW("ProcessResetNode: waitNode %lu not found", waitNodeId);
            continue;
        }

        uint32_t streamIdx = waitNode->GetStreamIdxInGraph();
        if (streamStates_[streamIdx].waitingForNotify == eventId) {
            streamStates_[streamIdx].waitingForNotify = INVALID_TASK_ID;
            streamStates_[streamIdx].isSuspended = true;
            SK_LOGD("ProcessResetNode: suspended stream %u, waitNode=%lu", streamIdx, waitNodeId);
        }
    }
}

bool SuperKernelScopeSplitter::SplitGraph() {
    SK_LOGI("start splitting graph into scopes\n");
    const auto& streams = graph.GetStreams();
    const auto& headNodes = graph.GetHeadNodes();

    SK_LOGI("Start splitting multi-stream graph, stream count: %zu", streams.size());

    // Reset all splitting state before starting
    ResetSplittingState();

    // Initialize stream states
    for (size_t i = 0; i < streams.size(); ++i) {
        streamStates_[i] = StreamState();
        streamStates_[i].currentNodeIdx = headNodes[i];
    }

    // Skip unfusible nodes at initialization
    SkipUnfusibleNodes();

    while (true) {
        SuperKernelScopeInfo scopeInfo;
        
        // Determine scopeBitFlags from the node with minimum currentNodeIdx
        if (!DetermineCurrentScopeBitFlags()) {
            SK_LOGI("No fusible node found, splitting complete, total scopes: %zu", scopeInfos.size());
            break;
        }
        scopeInfo.scopeBitFlags = currentScopeBitFlags_;

        // Reset deadlock detector at the start of each scope
        lockDetector.Reset(graph);

        // Phase 1: Initialize heap
        InitNodeHeap();

        if (nodeHeap_.empty()) {
            SK_LOGI("No fusible nodes in current scope, splitting complete, total scopes: %zu", scopeInfos.size());
            break;
        }

        SK_LOGI("========== Start processing new scope %zu, heap size: %zu ==========", scopeInfos.size(), nodeHeap_.size());

        // Phase 2: Main fusion loop
        while (!nodeHeap_.empty()) {
            uint64_t nodeId = nodeHeap_.top();
            nodeHeap_.pop();

            SuperKernelBaseNode* node = graph.GetNodeById(nodeId);
            if (node == nullptr) {
                SK_LOGW("Node %lu not found in graph, skip", nodeId);
                continue;
            }

            if (processedNodes_.find(nodeId) != processedNodes_.end()) {
                SK_LOGW("Node %lu already processed, skip", nodeId);
                continue;
            }

            uint32_t streamIdx = node->GetStreamIdxInGraph();
            const char* nodeTypeStr = to_string(node->GetNodeType());

            // Double-check deadlock (already checked in TryAddNodeToHeap, but be safe)
            if (!lockDetector.IsFusible(*node, graph)) {
                streamStates_[streamIdx].isTerminated = true;
                SK_LOGI("Node %lu (type=%s, stream=%lu) causes deadlock, terminate current scope", nodeId, nodeTypeStr, streamIdx);
                continue;
            }

            // Add to scope
            scopeInfo.nodes.push_back(node);
            AddStreamInfoToScope(scopeInfo, node);
            processedNodes_.insert(nodeId);
            SK_LOGD("Added node %lu (type=%s, stream=%lu) to scope, total nodes: %zu", nodeId, nodeTypeStr, streamIdx, scopeInfo.nodes.size());

            // Update stream state: advance to next node
            streamStates_[streamIdx].currentNodeIdx = node->GetNextNodeId();

            // Special case: Notify node (may activate suspended streams)
            if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
                ProcessNotifyNode(node);
            }
            // Special case: Reset node
            else if (node->GetNodeType() == SkNodeType::NODE_RESET) {
                ProcessResetNode(node);
            }

            // Process next node of current stream
            TryAddNodeToHeap(streamIdx);
        }

        // Save scope
        if (!scopeInfo.nodes.empty()) {
            std::string nodeIdsStr;
            for (const auto* n : scopeInfo.nodes) {
                if (!nodeIdsStr.empty()) nodeIdsStr += ", ";
                nodeIdsStr += std::to_string(n->GetNodeId());
            }
            SK_LOGI("========== Creating scope %zu with %zu nodes, %zu streams ==========",
                     scopeInfos.size(), scopeInfo.nodes.size(), scopeInfo.scopeStreamInfos.size());
            SK_LOGI("Scope nodes: [%s]", nodeIdsStr.c_str());
            scopeInfos.emplace_back(std::move(scopeInfo));
        }

        // Reset stream states for next scope
        ResetStreamStates();
    }

    PrintScopeResults();

    return true;
}
