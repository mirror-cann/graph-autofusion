/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sk_scope_split.cpp
 * \brief Implementation of multi-pass scope splitting
 */

#include "sk_scope_split.h"
#include "sk_common.h"
#include <algorithm>

// ============ ScopeSplitPass Base Class Implementation ============

void ScopeSplitPass::PrintScopeResults(const std::vector<SuperKernelScopeInfo>& scopes, 
                                        const SuperKernelGraph& graph) {
    for (size_t i = 0; i < scopes.size(); ++i) {
        const auto& scope = scopes[i];
        std::string scopeNames = GetScopeNamesFromBitFlags(scope.scopeBitFlags, graph);
        SK_LOGI("Scope %zu: %zu nodes, %zu streams, scopeBitFlags=%s, scopeNames=[%s]",
                i, scope.nodes.size(), scope.scopeStreamInfos.size(),
                scope.scopeBitFlags.to_string().substr(0, MAX_SCOPE_NUM).c_str(),
                scopeNames.c_str());
        PrintScopeNodes(i, scope);
        PrintScopeStreamInfos(i, scope);
    }
}

std::string ScopeSplitPass::GetScopeNamesFromBitFlags(const std::bitset<MAX_SCOPE_NUM>& scopeBitFlags,
                                                       const SuperKernelGraph& graph) {
    std::string scopeNames;
    for (size_t bit = 0; bit < MAX_SCOPE_NUM && bit < scopeBitFlags.size(); ++bit) {
        if (scopeBitFlags.test(bit)) {
            std::string name;
            if (graph.GetScopeNameByIdx(static_cast<uint32_t>(bit), name)) {
                if (!scopeNames.empty()) scopeNames += ", ";
                scopeNames += "'";
                scopeNames += name;
                scopeNames += "'";
            }
        }
    }
    return scopeNames.empty() ? "(none)" : scopeNames;
}

void ScopeSplitPass::PrintScopeNodes(size_t scopeIdx, const SuperKernelScopeInfo& scope) {
    std::string nodeDetails;
    for (const auto* n : scope.nodes) {
        if (!nodeDetails.empty()) nodeDetails += ", ";
        nodeDetails += n->FormatNodeInfo();
    }
    SK_LOGI("  Scope %zu nodes: [%s]", scopeIdx, nodeDetails.c_str());
}

void ScopeSplitPass::PrintScopeStreamInfos(size_t scopeIdx, const SuperKernelScopeInfo& scope) {
    for (size_t j = 0; j < scope.scopeStreamInfos.size(); ++j) {
        const auto& streamInfo = scope.scopeStreamInfos[j];
        SK_LOGI("  Scope %zu StreamInfo[%zu]: streamIdx=%u, headNode=%lu, tailNode=%lu, nodeSize=%lu",
                scopeIdx, j, streamInfo.streamIdx, streamInfo.headNodeIdx, 
                streamInfo.tailNodeIdx, streamInfo.nodeSize);
    }
}

// ============ InitialScopeSplitPass Implementation ============

InitialScopeSplitPass::InitialScopeSplitPass(SuperKernelGraph& inputGraph)
    : ScopeSplitPass(inputGraph) {}

void InitialScopeSplitPass::InitStreamStates() {
    SK_LOGI("InitStreamStates: Initializing stream states for %s", GetName().c_str());
    const auto& streams = graph_.GetStreams();
    const auto& headNodes = graph_.GetHeadNodes();
    
    streamStates_.clear();
    for (size_t i = 0; i < streams.size(); ++i) {
        streamStates_[i] = StreamState();
        streamStates_[i].currentNodeIdx = headNodes[i];
        SK_LOGI("  Stream %u: initialized with head node %lu",
                static_cast<uint32_t>(i), headNodes[i]);
    }
    SK_LOGI("InitStreamStates: Completed, %zu streams initialized", streamStates_.size());
}

void InitialScopeSplitPass::ResetStreamStates() {
    SK_LOGI("ResetStreamStates: Resetting all stream states for next scope");
    for (auto& pair : streamStates_) {
        uint32_t streamIdx = pair.first;
        pair.second.isTerminated = false;
        pair.second.isSuspended = false;
        pair.second.waitingForNotify = INVALID_TASK_ID;
        SK_LOGD("  Stream %u: reset, currentNodeIdx=%lu",
                streamIdx, pair.second.currentNodeIdx);
    }
    SK_LOGI("ResetStreamStates: Starting to skip unfusible nodes");
    SkipUnfusibleNodes();
    SK_LOGI("ResetStreamStates: Completed reset, active streams will proceed to next scope");
}

/*!
 * \brief Skip all unfusible nodes for all streams
 * 
 * This function is called at the beginning of each scope to advance each stream's
 * current position past all unfusible nodes, so that we start processing from a
 * fusible node for the next scope.
 * 
 * Special handling for WAIT nodes:
 * - If the corresponding NOTIFY node has already been visited, skip the WAIT node
 * - If the NOTIFY node has not been visited, suspend the stream and keep the WAIT
 *   node as the current node (do not advance)
 * 
 * Special handling for NOTIFY nodes:
 * - Always add NOTIFY nodes to visitedNotifies_ even if they are unfusible, so that
 *   dependent WAIT nodes can be properly processed
 */
void InitialScopeSplitPass::SkipUnfusibleNodes() {
    SK_LOGI("SkipUnfusibleNodes: Starting to skip unfusible nodes for all streams");

    for (auto& pair : streamStates_) {
        uint32_t streamIdx = pair.first;
        SkipUnfusibleNodesForStream(streamIdx);
    }

    SK_LOGI("SkipUnfusibleNodes: Completed");
}

/*!
 * \brief Skip unfusible nodes for a single stream
 * 
 * \param streamIdx The stream index to process
 * 
 * \return true if the stream has more nodes to process (either fusible nodes or
 *         it's suspended on an unfusible WAIT node), false if the stream is terminated
 */
bool InitialScopeSplitPass::SkipUnfusibleNodesForStream(uint32_t streamIdx) {
    StreamState& state = streamStates_[streamIdx];
    uint32_t skipCount = 0;

    while (state.currentNodeIdx != INVALID_TASK_ID) {
        SuperKernelBaseNode* node = graph_.GetNodeById(state.currentNodeIdx);
        if (node == nullptr) {
            SK_LOGE("SkipUnfusibleNodesForStream: Stream %u: node %lu not found (graph integrity error)",
                    streamIdx, state.currentNodeIdx);
            return false;
        }

        // Fusible node: stop skipping and return
        if (node->IsFusible()) {
            SK_LOGD("SkipUnfusibleNodesForStream: Stream %u: Found fusible node %s, stop skipping",
                    streamIdx, node->FormatNodeInfo().c_str());
            break;
        }

        // Unfusible node: process special cases and decide whether to continue
        SkNodeType nodeType = node->GetNodeType();

        // Special case 1: Unfusible WAIT node
        if (nodeType == SkNodeType::NODE_WAIT) {
            bool shouldSkip = ProcessUnfusibleWaitNode(streamIdx, node);
            if (!shouldSkip) {
                // Wait node needs to wait, suspend stream and stop skipping
                SK_LOGI("SkipUnfusibleNodesForStream: Stream %u: Suspended on unfusible wait node %s",
                        streamIdx, node->FormatNodeInfo().c_str());
                break;
            }
            SK_LOGD("SkipUnfusibleNodesForStream: Stream %u: Skipping unfusible wait node %s",
                    streamIdx, node->FormatNodeInfo().c_str());
            // Wait node can be skipped, continue to next node
        }
        // Special case 2: Unfusible NOTIFY node
        else if (nodeType == SkNodeType::NODE_NOTIFY) {
            // Always record visited notifies for wait nodes to check
            visitedNotifies_.insert(node->GetNodeId());
            processedNodes_.insert(node->GetNodeId());  // Also add to processed nodes
            SK_LOGD("SkipUnfusibleNodesForStream: Stream %u: Unfusible notify node %s added to visitedNotifies and processedNodes",
                    streamIdx, node->FormatNodeInfo().c_str());
        }
        // Other unfusible nodes: skip and add to processed nodes
        else {
            processedNodes_.insert(node->GetNodeId());
            SK_LOGD("SkipUnfusibleNodesForStream: Stream %u: Skipping unfusible node %s (type=%d), added to processedNodes",
                    streamIdx, node->FormatNodeInfo().c_str(), static_cast<int>(nodeType));
        }

        // All unfusible nodes are skipped, advance to next node
        state.currentNodeIdx = node->GetNextNodeId();
        skipCount++;
    }

    if (skipCount > 0) {
        SK_LOGI("SkipUnfusibleNodesForStream: Stream %u: Skipped %u unfusible nodes, next node=%lu",
                streamIdx, skipCount, state.currentNodeIdx);
    }

    return state.currentNodeIdx != INVALID_TASK_ID;
}

/*!
 * \brief Process an unfusible WAIT node and determine if it should be skipped
 *
 * \param streamIdx The stream index
 * \param waitNode The unfusible WAIT node to process
 *
 * \return true if the WAIT node should be skipped (advance currentNodeIdx),
 *         false if the stream should be suspended (keep currentNodeIdx)
 *
 * Processing logic:
 * 1. Find the corresponding NOTIFY node
 * 2. If NOTIFY node has already been visited: skip the WAIT node and add to processedNodes_
 * 3. If NOTIFY node has not been visited: suspend the stream and keep the WAIT node
 *    as the current node, so it will be processed again later when the NOTIFY is visited
 */
bool InitialScopeSplitPass::ProcessUnfusibleWaitNode(uint32_t streamIdx, SuperKernelBaseNode* waitNode) {
    uint64_t notifyId = waitNode->GetCorrespondingNotifyNodeId();
    SuperKernelBaseNode* notifyNode = graph_.GetNodeById(notifyId);

    // Error handling: notify node not found
    if (notifyNode == nullptr) {
        SK_LOGE("ProcessUnfusibleWaitNode: Stream %u: Wait node %s's notify node %lu not found",
                streamIdx, waitNode->FormatNodeInfo().c_str(), notifyId);
        return true;  // Skip the wait node if notify not found to avoid hanging
    }

    // Check if notify node has been visited
    if (visitedNotifies_.find(notifyId) != visitedNotifies_.end()) {
        // Notify already visited, skip the wait node
        processedNodes_.insert(waitNode->GetNodeId());  // Add to processed nodes
        SK_LOGI("ProcessUnfusibleWaitNode: Stream %u: Unfusible wait node %s: notify %lu already visited, skipping and added to processedNodes_",
                streamIdx, waitNode->FormatNodeInfo().c_str(), notifyId);
        return true;  // Should skip (advance currentNodeIdx)
    } else {
        // Notify not visited, suspend the stream
        StreamState& state = streamStates_[streamIdx];
        uint64_t eventId = notifyNode->GetEventId();
        state.isSuspended = true;
        state.waitingForNotify = eventId;
        SK_LOGI("ProcessUnfusibleWaitNode: Stream %u: Unfusible wait node %s: notify %lu not visited, suspending stream (eventId=0x%lx)",
                streamIdx, waitNode->FormatNodeInfo().c_str(), notifyId, eventId);
        return false;  // Should NOT skip (keep currentNodeIdx for wait node)
    }
}


bool InitialScopeSplitPass::DetermineCurrentScopeBitFlags() {
    SK_LOGD("DetermineCurrentScopeBitFlags: Starting to determine scope bit flags");
    uint64_t minNodeIdx = UINT64_MAX;
    SuperKernelBaseNode* minNode = nullptr;
    uint32_t activeStreams = 0;
    
    for (const auto& pair : streamStates_) {
        uint32_t streamIdx = pair.first;
        if (!pair.second.isTerminated && !pair.second.isSuspended &&
            pair.second.currentNodeIdx != INVALID_TASK_ID) {
            activeStreams++;
            if (pair.second.currentNodeIdx < minNodeIdx) {
                SuperKernelBaseNode* node = graph_.GetNodeById(pair.second.currentNodeIdx);
                if (node != nullptr && node->IsFusible()) {
                    minNodeIdx = pair.second.currentNodeIdx;
                    minNode = node;
                    SK_LOGD("DetermineCurrentScopeBitFlags: Stream %u: candidate min node %s (nodeIdx=%lu)",
                            streamIdx, node->FormatNodeInfo().c_str(), minNodeIdx);
                }
            }
        }
    }
    
    if (minNode != nullptr) {
        currentScopeBitFlags_ = minNode->GetScopeBitFlags();
        std::string scopeNames = ScopeSplitPass::GetScopeNamesFromBitFlags(currentScopeBitFlags_, graph_);
        SK_LOGI("DetermineCurrentScopeBitFlags: Found min node %s, scopeBitFlags=%s, scopeNames=[%s], active streams=%u",
                minNode->FormatNodeInfo().c_str(),
                currentScopeBitFlags_.to_string().substr(0, MAX_SCOPE_NUM).c_str(),
                scopeNames.c_str(), activeStreams);
        return true;
    }
    SK_LOGD("DetermineCurrentScopeBitFlags: No active streams found, active streams=%u", activeStreams);
    return false;
}

void InitialScopeSplitPass::InitNodeHeap() {
    SK_LOGI("InitNodeHeap: Initializing node heap for current scope");
    while (!nodeHeap_.empty()) {
        nodeHeap_.pop();
    }
    size_t addedNodes = 0;
    for (auto& pair : streamStates_) {
        TryAddNodeToHeap(pair.first);
    }
    SK_LOGI("InitNodeHeap: Node heap initialized with %zu nodes", nodeHeap_.size());
}

void InitialScopeSplitPass::TryAddNodeToHeap(uint32_t streamIdx) {
    StreamState& state = streamStates_[streamIdx];

    if (state.isTerminated || state.isSuspended || state.currentNodeIdx == INVALID_TASK_ID) {
        SK_LOGD("TryAddNodeToHeap: Stream %u: skipped (terminated=%d, suspended=%d, currentNodeIdx=%lu)",
                streamIdx, state.isTerminated, state.isSuspended, state.currentNodeIdx);
        return;
    }

    SuperKernelBaseNode* node = graph_.GetNodeById(state.currentNodeIdx);
    if (node == nullptr) {
        SK_LOGE("TryAddNodeToHeap: Stream %u: node %lu not found, terminating stream (graph integrity error: nodeId not in graph)",
                streamIdx, state.currentNodeIdx);
        state.currentNodeIdx = INVALID_TASK_ID;
        return;
    }

    if (processedNodes_.find(state.currentNodeIdx) != processedNodes_.end()) {
        SK_LOGD("TryAddNodeToHeap: Stream %u: node %lu already processed, advancing",
                streamIdx, state.currentNodeIdx);
        state.currentNodeIdx = node->GetNextNodeId();
        TryAddNodeToHeap(streamIdx);
        return;
    }

    // Handle Wait nodes: check if corresponding notify has been visited
    if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
        HandleWaitNode(node, streamIdx);
        // HandleWaitNode will either:
        // 1. Add wait node to heap (if notify already visited)
        // 2. Suspend the stream (if notify not yet visited)
        return;  // Don't continue with normal processing
    }

    // Check scopeBitFlags match
    if (node->GetScopeBitFlags() != currentScopeBitFlags_) {
        SK_LOGD("TryAddNodeToHeap: Stream %u: node %s scopeBitFlags mismatch, terminating stream",
                streamIdx, node->FormatNodeInfo().c_str());
        state.isTerminated = true;
        return;
    }

    // Check fusibility
    if (!node->IsFusible()) {
        SK_LOGD("TryAddNodeToHeap: Stream %u: node %s is not fusible, terminating stream",
                streamIdx, node->FormatNodeInfo().c_str());
        state.isTerminated = true;
        return;
    }

    // Add to heap (no deadlock check in this pass)
    nodeHeap_.push(state.currentNodeIdx);
    SK_LOGD("TryAddNodeToHeap: Stream %u: added node %s to heap, heap size=%zu",
            streamIdx, node->FormatNodeInfo().c_str(), nodeHeap_.size());
}

void InitialScopeSplitPass::HandleWaitNode(SuperKernelBaseNode* waitNode, uint32_t streamIdx) {
    StreamState& state = streamStates_[streamIdx];
    uint64_t notifyId = waitNode->GetCorrespondingNotifyNodeId();
    SuperKernelBaseNode* notifyNode = graph_.GetNodeById(notifyId);

    if (notifyNode == nullptr) {
        SK_LOGE("HandleWaitNode: Stream %u: Wait node %s's notify node %lu not found",
                streamIdx, waitNode->FormatNodeInfo().c_str(), notifyId);
        return;
    }

    uint64_t eventId = notifyNode->GetEventId();
    if (visitedNotifies_.find(notifyId) != visitedNotifies_.end()) {
        // Notify already visited, add fusible wait node to heap
        nodeHeap_.push(state.currentNodeIdx);
        SK_LOGD("HandleWaitNode: Stream %u: Wait node %s: notify %lu already visited, adding to heap",
                streamIdx, waitNode->FormatNodeInfo().c_str(), notifyId);
    } else {
        // Suspend stream
        state.isSuspended = true;
        state.waitingForNotify = eventId;
        SK_LOGD("HandleWaitNode: Stream %u: Wait node %s: notify %lu not visited, suspending stream (eventId=0x%lx)",
                streamIdx, waitNode->FormatNodeInfo().c_str(), notifyId, eventId);
    }
}

void InitialScopeSplitPass::ProcessNotifyNode(SuperKernelBaseNode* notifyNode) {
    uint64_t eventId = notifyNode->GetEventId();
    uint64_t notifyNodeId = notifyNode->GetNodeId();
    const EventInfos* eventInfo = graph_.GetEventInfo(eventId);
    if (eventInfo == nullptr) {
        SK_LOGE("ProcessNotifyNode: Event info for eventId=0x%lx not found (notifyNode: %s)",
                 eventId, notifyNode->FormatNodeInfo().c_str());
        return;
    }
    visitedNotifies_.insert(notifyNodeId);
    SK_LOGD("ProcessNotifyNode: Notify node %s (eventId=0x%lx) added to visited, %zu wait nodes to check",
            notifyNode->FormatNodeInfo().c_str(), eventId, eventInfo->waitNodeIdList.size());

    uint32_t resumedCount = 0;
    for (uint64_t waitNodeId : eventInfo->waitNodeIdList) {
        SuperKernelBaseNode* waitNode = graph_.GetNodeById(waitNodeId);
        if (waitNode == nullptr) {
            SK_LOGE("ProcessNotifyNode: Wait node %lu not found (graph integrity error: eventId=0x%lx references non-existent node)",
                    waitNodeId, eventId);
            continue;
        }

        uint32_t streamIdx = waitNode->GetStreamIdxInGraph();
        StreamState& state = streamStates_[streamIdx];

        if (state.isSuspended && state.waitingForNotify == eventId) {
            state.isSuspended = false;
            state.waitingForNotify = INVALID_TASK_ID;
            TryAddNodeToHeap(streamIdx);
            resumedCount++;
            SK_LOGD("ProcessNotifyNode: Resumed stream %u for wait node %s",
                    streamIdx, waitNode->FormatNodeInfo().c_str());
        }
    }
    SK_LOGD("ProcessNotifyNode: Resumed %u suspended streams", resumedCount);
}

void InitialScopeSplitPass::ProcessResetNode(SuperKernelBaseNode* resetNode) {
    uint64_t eventId = resetNode->GetEventId();
    const EventInfos* eventInfo = graph_.GetEventInfo(eventId);
    if (eventInfo == nullptr) {
        SK_LOGE("ProcessResetNode: Event info for eventId=0x%lx not found (resetNode: %s)",
                 eventId, resetNode->FormatNodeInfo().c_str());
        return;
    }

    if (eventInfo->notifyNodeId != INVALID_TASK_ID) {
        visitedNotifies_.erase(eventInfo->notifyNodeId);
        SK_LOGD("ProcessResetNode: Erased notify %lu from visited (eventId=0x%lx)",
                eventInfo->notifyNodeId, eventId);
    }

    uint32_t suspendedCount = 0;
    for (uint64_t waitNodeId : eventInfo->waitNodeIdList) {
        SuperKernelBaseNode* waitNode = graph_.GetNodeById(waitNodeId);
        if (waitNode == nullptr) {
            SK_LOGE("ProcessResetNode: Wait node %lu not found (graph integrity error: eventId=0x%lx references non-existent node)",
                    waitNodeId, eventId);
            continue;
        }

        uint32_t streamIdx = waitNode->GetStreamIdxInGraph();
        if (streamStates_[streamIdx].waitingForNotify == eventId) {
            streamStates_[streamIdx].waitingForNotify = INVALID_TASK_ID;
            streamStates_[streamIdx].isSuspended = true;
            suspendedCount++;
            SK_LOGD("ProcessResetNode: Suspended stream %u for wait node %s (eventId=0x%lx)",
                    streamIdx, waitNode->FormatNodeInfo().c_str(), eventId);
        }
    }
    SK_LOGD("ProcessResetNode: Suspended %u streams due to reset", suspendedCount);
}

void InitialScopeSplitPass::AddStreamInfoToScope(SuperKernelScopeInfo& scopeInfo, SuperKernelBaseNode* node) {
    uint32_t streamIdx = node->GetStreamIdxInGraph();
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
    } else {
        it->tailNodeIdx = node->GetNodeId();
        it->nodeSize++;
    }
}

bool InitialScopeSplitPass::BuildCurrentScope(SuperKernelScopeInfo& scopeInfo) {
    SK_LOGI("BuildCurrentScope: Starting to build current scope");
    scopeInfo.scopeBitFlags = currentScopeBitFlags_;

    size_t nodeCount = 0;
    while (!nodeHeap_.empty()) {
        uint64_t nodeId = nodeHeap_.top();
        nodeHeap_.pop();

        SuperKernelBaseNode* node = graph_.GetNodeById(nodeId);
        if (node == nullptr) {
            SK_LOGE("BuildCurrentScope: Node %lu not found in graph (graph integrity error)", nodeId);
            return false;
        }
        if (processedNodes_.find(nodeId) != processedNodes_.end()) {
            continue;
        }

        uint32_t streamIdx = node->GetStreamIdxInGraph();

        // Add to scope
        if (!node->IsScopeNode()) {
            scopeInfo.nodes.push_back(node);
            nodeCount++;
            SK_LOGD("BuildCurrentScope: Added node %s to scope (count=%zu)",
                    node->FormatNodeInfo().c_str(), nodeCount);
        }
        AddStreamInfoToScope(scopeInfo, node);
        processedNodes_.insert(nodeId);

        // Advance stream
        uint64_t nextNodeId = node->GetNextNodeId();
        streamStates_[streamIdx].currentNodeIdx = nextNodeId;

        // Handle special nodes
        if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
            ProcessNotifyNode(node);
        } else if (node->GetNodeType() == SkNodeType::NODE_RESET) {
            ProcessResetNode(node);
        }

        // Try next node
        TryAddNodeToHeap(streamIdx);
    }

    SK_LOGI("BuildCurrentScope: Built scope with %zu nodes, %zu stream infos",
            nodeCount, scopeInfo.scopeStreamInfos.size());
    return true;  // Empty scope is not an error, allow it to be skipped in Run()
}

bool InitialScopeSplitPass::Run(std::vector<SuperKernelScopeInfo>& scopes) {
    SK_LOGI("split scope pass: %s start", GetName().c_str());

    scopes.clear();
    processedNodes_.clear();
    visitedNotifies_.clear();

    SK_LOGI("%s: Initializing stream states", GetName().c_str());
    InitStreamStates();
    SkipUnfusibleNodes();

    size_t scopeCount = 0;
    while (true) {
        if (!DetermineCurrentScopeBitFlags()) {
            SK_LOGI("%s: No more fusible nodes, initial scope split complete", GetName().c_str());
            break;
        }

        InitNodeHeap();
        if (nodeHeap_.empty()) {
            SK_LOGI("%s: Node heap is empty, initial scope split complete", GetName().c_str());
            break;
        }

        SuperKernelScopeInfo scopeInfo;
        if (BuildCurrentScope(scopeInfo)) {
            if (!scopeInfo.nodes.empty()) {
                std::string scopeNames = ScopeSplitPass::GetScopeNamesFromBitFlags(scopeInfo.scopeBitFlags, graph_);
                SK_LOGI("%s: Built scope %zu with %zu nodes, %zu streams, scopeNames=[%s]",
                        GetName().c_str(), scopeCount, scopeInfo.nodes.size(),
                        scopeInfo.scopeStreamInfos.size(), scopeNames.c_str());
                scopes.push_back(std::move(scopeInfo));
            } else {
                // Empty scope: skip it and continue to next iteration
                std::string scopeNames = ScopeSplitPass::GetScopeNamesFromBitFlags(scopeInfo.scopeBitFlags, graph_);
                SK_LOGD("%s: Empty scope %zu skipped, scopeNames=[%s], continuing",
                        GetName().c_str(), scopeCount, scopeNames.c_str());
            }
            scopeCount++;
        } else {
            SK_LOGE("%s: Failed to build scope %zu (graph integrity error), stopping scope split",
                    GetName().c_str(), scopeCount);
            return false;
        }

        SK_LOGI("%s: Resetting stream states for next scope", GetName().c_str());
        ResetStreamStates();
    }

    SK_LOGI("%s Complete: total scopes: %zu", GetName().c_str(), scopes.size());
    PrintScopeResults(scopes, graph_);
    return true;
}

// ============ DeadlockRefinePass Implementation ============

DeadlockRefinePass::DeadlockRefinePass(SuperKernelGraph& inputGraph)
    : ScopeSplitPass(inputGraph), lockDetector_() {}

bool DeadlockRefinePass::FindDeadlockInScope(const SuperKernelScopeInfo& scope,
                                              SuperKernelBaseNode** deadlockNode,
                                              SuperKernelBaseNode** deadlockWaitNode) {
    SK_LOGI("FindDeadlockInScope: Checking scope with %zu nodes for deadlock", scope.nodes.size());
    lockDetector_.Reset(graph_);

    // Track the most recent Wait node seen before each node
    SuperKernelBaseNode* lastWaitNode = nullptr;

    // Check each node for deadlock, keeping track of the nearest preceding Wait node
    for (size_t i = 0; i < scope.nodes.size(); ++i) {
        const auto* node = scope.nodes[i];
        // Update lastWaitNode when we encounter a Wait node
        if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
            lastWaitNode = const_cast<SuperKernelBaseNode*>(node);
            SK_LOGD("FindDeadlockInScope: Found Wait node %s at position %zu",
                    node->FormatNodeInfo().c_str(), i);
        }

        // Check for deadlock
        if (!lockDetector_.IsFusible(*const_cast<SuperKernelBaseNode*>(node), graph_)) {
            *deadlockNode = const_cast<SuperKernelBaseNode*>(node);
            *deadlockWaitNode = lastWaitNode;  // The nearest Wait node before deadlock point
            SK_LOGI("FindDeadlockInScope: Deadlock detected at node %s (position %zu)",
                    node->FormatNodeInfo().c_str(), i);
            if (lastWaitNode != nullptr) {
                SK_LOGI("  Nearest Wait node before deadlock: %s",
                        lastWaitNode->FormatNodeInfo().c_str());
            }
            return true;
        }
        SK_LOGD("FindDeadlockInScope: Node %s passed deadlock check", node->FormatNodeInfo().c_str());
    }

    SK_LOGI("FindDeadlockInScope: No deadlock found in scope");
    return false;
}

void DeadlockRefinePass::SplitScopeAtWaitNode(const SuperKernelScopeInfo& scope,
                                               SuperKernelBaseNode* waitNode,
                                               SuperKernelScopeInfo& scopeBefore,
                                               SuperKernelScopeInfo& scopeAfter) {
    scopeBefore.scopeBitFlags = scope.scopeBitFlags;
    scopeAfter.scopeBitFlags = scope.scopeBitFlags;
    
    bool foundWait = false;
    
    for (const auto* node : scope.nodes) {
        if (node == waitNode) {
            foundWait = true;
            continue;  // Don't include Wait node in either scope
        }
        
        if (!foundWait) {
            scopeBefore.nodes.push_back(const_cast<SuperKernelBaseNode*>(node));
        } else {
            scopeAfter.nodes.push_back(const_cast<SuperKernelBaseNode*>(node));
        }
    }
    
    RebuildStreamInfos(scopeBefore);
    RebuildStreamInfos(scopeAfter);
}

void DeadlockRefinePass::RebuildStreamInfos(SuperKernelScopeInfo& scope) {
    scope.scopeStreamInfos.clear();
    for (const auto* node : scope.nodes) {
        uint32_t streamIdx = node->GetStreamIdxInGraph();
        auto it = std::find_if(scope.scopeStreamInfos.begin(), scope.scopeStreamInfos.end(),
                              [streamIdx](const ScopeStreamInfo& info) {
                                  return info.streamIdx == streamIdx;
                              });
        if (it == scope.scopeStreamInfos.end()) {
            ScopeStreamInfo newInfo;
            newInfo.streamIdx = streamIdx;
            newInfo.headNodeIdx = node->GetNodeId();
            newInfo.tailNodeIdx = node->GetNodeId();
            newInfo.nodeSize = 1;
            scope.scopeStreamInfos.push_back(std::move(newInfo));
        } else {
            it->tailNodeIdx = node->GetNodeId();
            it->nodeSize++;
        }
    }
}

bool DeadlockRefinePass::Run(std::vector<SuperKernelScopeInfo>& scopes) {
    SK_LOGI("split scope pass: %s start", GetName().c_str());
    SK_LOGI("Input scopes count: %zu", scopes.size());
    
    std::vector<SuperKernelScopeInfo> refinedScopes;
    size_t splitCount = 0;
    
    for (size_t i = 0; i < scopes.size(); ++i) {
        auto& scope = scopes[i];
        SuperKernelBaseNode* deadlockNode = nullptr;
        SuperKernelBaseNode* deadlockWaitNode = nullptr;
        
        if (FindDeadlockInScope(scope, &deadlockNode, &deadlockWaitNode)) {
            if (deadlockWaitNode != nullptr) {
                // Split at Wait node
                SuperKernelScopeInfo scopeBefore, scopeAfter;
                SplitScopeAtWaitNode(scope, deadlockWaitNode, scopeBefore, scopeAfter);

                SK_LOGI("Scope %zu: Deadlock detected at node %s, splitting at Wait node %s",
                        i, deadlockNode->FormatNodeInfo().c_str(), deadlockWaitNode->FormatNodeInfo().c_str());
                SK_LOGI("  Before split: %zu nodes", scope.nodes.size());
                SK_LOGI("  After split: scopeBefore=%zu nodes, scopeAfter=%zu nodes",
                        scopeBefore.nodes.size(), scopeAfter.nodes.size());

                // Set Notify nodes expand numbers immediately after scope is generated
                if (!scopeBefore.nodes.empty()) {
                    SuperKernelScopeSplitter::SetNotifyNodesExpandNumForScope(scopeBefore);
                    refinedScopes.push_back(std::move(scopeBefore));
                }
                if (!scopeAfter.nodes.empty()) {
                    SuperKernelScopeSplitter::SetNotifyNodesExpandNumForScope(scopeAfter);
                    refinedScopes.push_back(std::move(scopeAfter));
                }
                splitCount++;
            } else {
                // No Wait node found, keep original scope
                SK_LOGW("Scope %zu: Deadlock at node %s but no Wait node found to split",
                        i, deadlockNode->FormatNodeInfo().c_str());
                // Set Notify nodes expand numbers for original scope
                SuperKernelScopeSplitter::SetNotifyNodesExpandNumForScope(scope);
                refinedScopes.push_back(std::move(scope));
            }
        } else {
            // Set Notify nodes expand numbers for unchanged scope
            SuperKernelScopeSplitter::SetNotifyNodesExpandNumForScope(scope);
            refinedScopes.push_back(std::move(scope));
        }
    }
    
    scopes = std::move(refinedScopes);
    
    SK_LOGI("%s complete: split %zu scopes, total scopes: %zu", 
            GetName().c_str(), splitCount, scopes.size());
    PrintScopeResults(scopes, graph_);
    return true;
}

// ============ SuperKernelScopeSplitter Implementation ============

SuperKernelScopeSplitter::SuperKernelScopeSplitter(SuperKernelGraph& inputGraph) 
    : graph_(inputGraph) {
    // Initialize passes
    passes_.push_back(std::make_unique<InitialScopeSplitPass>(inputGraph));
    passes_.push_back(std::make_unique<DeadlockRefinePass>(inputGraph));
}

bool SuperKernelScopeSplitter::SplitGraph() {
    SK_LOGI("Start scope splitting pipeline");
    
    scopeInfos_.clear();
    
    // Run all passes
    for (auto& pass : passes_) {
        SK_LOGI("Running pass: %s", pass->GetName().c_str());
        if (!pass->Run(scopeInfos_)) {
            SK_LOGE("Pass %s failed (input scopes: %zu)", pass->GetName().c_str(), scopeInfos_.size());
            return false;
        }
    }
    
    // SetNotifyNodesExpandNum is now called immediately after each scope is generated
    
    PrintFinalResults();
    
    return true;
}

void SuperKernelScopeSplitter::SetNotifyNodesExpandNumForScope(SuperKernelScopeInfo& scope) {
    uint32_t maxExpandVecNum = 0;
    uint32_t maxExpandCubeNum = 0;
    std::vector<SuperKernelBaseNode*> notifyNodes;
    
    // Find max vec/cube num and collect notify nodes
    for (const auto* node : scope.nodes) {
        if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
            maxExpandVecNum = std::max(maxExpandVecNum, node->GetVecNum());
            maxExpandCubeNum = std::max(maxExpandCubeNum, node->GetCubeNum());
        } else if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
            notifyNodes.push_back(const_cast<SuperKernelBaseNode*>(node));
        }
    }
    
    // Set expand numbers for all notify nodes
    for (auto* notifyNode : notifyNodes) {
        notifyNode->SetNotifyExpandVecNum(maxExpandVecNum);
        notifyNode->SetNotifyExpandCubeNum(maxExpandCubeNum);
        SK_LOGI("Set Notify node %lu expandVecNum=%u, expandCubeNum=%u", 
                notifyNode->GetNodeId(), maxExpandVecNum, maxExpandCubeNum);
    }
}

void SuperKernelScopeSplitter::PrintFinalResults() const {
    SK_LOGI("Scope splitting complete, total scopes: %zu", scopeInfos_.size());
    ScopeSplitPass::PrintScopeResults(scopeInfos_, graph_);
}
