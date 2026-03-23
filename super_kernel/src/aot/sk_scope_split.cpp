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
    SK_LOGI("[SplitScopeResult] printing scope split results, total scopes: %zu", scopes.size());
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
    SK_LOGI("[SplitScope] initializing stream states for %s", GetName().c_str());
    const auto& streams = graph_.GetStreams();
    const auto& headNodes = graph_.GetHeadNodes();
    
    streamStates_.clear();
    for (size_t i = 0; i < streams.size(); ++i) {
        streamStates_[i] = StreamState();
        streamStates_[i].currentNodeId = headNodes[i];
        SK_LOGI("  Stream %u: initialized with head node %lu",
                static_cast<uint32_t>(i), headNodes[i]);
    }
    SK_LOGI("[SplitScope] completed initializing %zu streams", streamStates_.size());
}

bool InitialScopeSplitPass::ResetStreamStates() {
    SK_LOGI("[SplitScope] resetting all stream states for next scope");
    for (auto& pair : streamStates_) {
        uint32_t streamIdx = pair.first;
        pair.second.isTerminated = false;
        pair.second.isSuspended = false;
        pair.second.waitingForNotify = INVALID_TASK_ID;
        SK_LOGD("  Stream %u: reset, currentNodeId=%lu",
                streamIdx, pair.second.currentNodeId);
    }
    SK_LOGI("[SplitScope] starting to skip unfusible nodes");
    if (!SkipUnfusibleNodes()) {
        SK_LOGE("[SplitScope] Failed to skip unfusible nodes during stream reset");
        return false;
    }
    SK_LOGI("[SplitScope] completed reset, active streams will proceed to next scope");
    return true;
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
bool InitialScopeSplitPass::SkipUnfusibleNodes() {
    SK_LOGI("[SkipUnfusible] starting to skip unfusible nodes for all streams");

    for (auto& pair : streamStates_) {
        uint32_t streamIdx = pair.first;
        if (!SkipUnfusibleNodesForStream(streamIdx)) {
            SK_LOGE("[SkipUnfusible] Failed to skip unfusible nodes for stream %u", streamIdx);
            return false;
        }
    }

    SK_LOGI("[SkipUnfusible] completed skipping unfusible nodes");
    return true;
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

    // If stream has already ended (currentNodeId == INVALID_TASK_ID), return success
    // This is a normal case and should not be considered an error
    if (state.currentNodeId == INVALID_TASK_ID) {
        SK_LOGD("Stream %u: Already at end (currentNodeId=INVALID), nothing to skip", streamIdx);
        return true;
    }

    while (state.currentNodeId != INVALID_TASK_ID) {
        SuperKernelBaseNode* node = graph_.GetNodeById(state.currentNodeId);
        if (node == nullptr) {
            SK_LOGE("Stream %u: node %lu not found (graph integrity error)",
                    streamIdx, state.currentNodeId);
            return false;
        }

        // Fusible node: stop skipping and return
        if (node->IsFusible()) {
            SK_LOGD("Stream %u: Found fusible node %s, stop skipping",
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
                SK_LOGI("Stream %u: Suspended on unfusible wait node %s",
                        streamIdx, node->FormatNodeInfo().c_str());
                break;
            }
            SK_LOGD("Stream %u: Skipping unfusible wait node %s",
                    streamIdx, node->FormatNodeInfo().c_str());
            // Wait node can be skipped, continue to next node
        }
        // Special case 2: Unfusible NOTIFY node
        else if (nodeType == SkNodeType::NODE_NOTIFY) {
            if (!HandleUnfusibleNotifyNode(node, streamIdx)) {
                SK_LOGE("Stream %u: Failed to handle unfusible notify node %s, abort processing",
                        streamIdx, node->FormatNodeInfo().c_str());
                return false;
            }
        }
        // Other unfusible nodes: skip and add to processed nodes
        else {
            processedNodes_.insert(node->GetNodeId());
            SK_LOGD("Stream %u: Skipping unfusible node %s (type=%d), added to processedNodes",
                    streamIdx, node->FormatNodeInfo().c_str(), static_cast<int>(nodeType));
        }

        // All unfusible nodes are skipped, advance to next node
        state.currentNodeId = node->GetNextNodeId();
        skipCount++;
    }

    if (skipCount > 0) {
        SK_LOGI("Stream %u: Skipped %u unfusible nodes, next node=%lu",
                streamIdx, skipCount, state.currentNodeId);
    }

    // Return true if stream is still active or has reached its end
    // Only return false if there was an error (node not found)
    return true;
}

/*!
 * \brief Process an unfusible WAIT node and determine if it should be skipped
 *
 * \param streamIdx The stream index
 * \param waitNode The unfusible WAIT node to process
 *
 * \return true if the WAIT node should be skipped (advance currentNodeId),
 *         false if the stream should be suspended (keep currentNodeId)
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
        SK_LOGE("Stream %u: Wait node %s's notify node %lu not found",
                streamIdx, waitNode->FormatNodeInfo().c_str(), notifyId);
        return true;  // Skip the wait node if notify not found to avoid hanging
    }

    // Check if notify node has been visited
    if (visitedNotifies_.find(notifyId) != visitedNotifies_.end()) {
        // Notify already visited, skip the wait node
        processedNodes_.insert(waitNode->GetNodeId());  // Add to processed nodes
        SK_LOGI("Stream %u: Unfusible wait node %s: notify %lu already visited, skipping and added to processedNodes_",
                streamIdx, waitNode->FormatNodeInfo().c_str(), notifyId);
        return true;  // Should skip (advance currentNodeId)
    } else {
        // Notify not visited, suspend the stream
        StreamState& state = streamStates_[streamIdx];
        uint64_t eventId = notifyNode->GetEventId();
        state.isSuspended = true;
        state.waitingForNotify = eventId;
        SK_LOGI("Stream %u: Unfusible wait node %s: notify %lu not visited, suspending stream (eventId=0x%lx)",
                streamIdx, waitNode->FormatNodeInfo().c_str(), notifyId, eventId);
        return false;  // Should NOT skip (keep currentNodeId for wait node)
    }
}


bool InitialScopeSplitPass::DetermineCurrentScopeBitFlags() {
    SK_LOGD("Starting to determine scope bit flags");
    uint64_t minNodeIdx = UINT64_MAX;
    SuperKernelBaseNode* minNode = nullptr;
    uint32_t activeStreams = 0;
    
    for (const auto& pair : streamStates_) {
        uint32_t streamIdx = pair.first;
        if (!pair.second.isTerminated && !pair.second.isSuspended &&
            pair.second.currentNodeId != INVALID_TASK_ID) {
            activeStreams++;
            if (pair.second.currentNodeId < minNodeIdx) {
                SuperKernelBaseNode* node = graph_.GetNodeById(pair.second.currentNodeId);
                if (node != nullptr && node->IsFusible()) {
                    minNodeIdx = pair.second.currentNodeId;
                    minNode = node;
                    SK_LOGD("Stream %u: candidate min node %s (nodeIdx=%lu)",
                            streamIdx, node->FormatNodeInfo().c_str(), minNodeIdx);
                }
            }
        }
    }
    
    if (minNode != nullptr) {
        currentScopeBitFlags_ = minNode->GetScopeBitFlags();
        std::string scopeNames = ScopeSplitPass::GetScopeNamesFromBitFlags(currentScopeBitFlags_, graph_);
        SK_LOGI("Found min node %s, scopeBitFlags=%s, scopeNames=[%s], active streams=%u",
                minNode->FormatNodeInfo().c_str(),
                currentScopeBitFlags_.to_string().substr(0, MAX_SCOPE_NUM).c_str(),
                scopeNames.c_str(), activeStreams);
        return true;
    }

    // No fusible nodes found - log diagnostic information
    LogFusibleNodeSearchResult();

    return false;
}

void InitialScopeSplitPass::InitNodeHeap() {
    SK_LOGI("[SplitScope] initializing node heap for current scope");
    while (!nodeHeap_.empty()) {
        nodeHeap_.pop();
    }
    size_t addedNodes = 0;
    for (auto& pair : streamStates_) {
        TryAddNodeToHeap(pair.first);
    }
    SK_LOGI("[SplitScope] node heap initialized with %zu nodes", nodeHeap_.size());
}

void InitialScopeSplitPass::TryAddNodeToHeap(uint32_t streamIdx) {
    StreamState& state = streamStates_[streamIdx];

    if (state.isTerminated || state.isSuspended || state.currentNodeId == INVALID_TASK_ID) {
        SK_LOGD("Stream %u: skipped (terminated=%d, suspended=%d, currentNodeId=%lu)",
                streamIdx, state.isTerminated, state.isSuspended, state.currentNodeId);
        return;
    }

    SuperKernelBaseNode* node = graph_.GetNodeById(state.currentNodeId);
    if (node == nullptr) {
        SK_LOGE("Stream %u: node %lu not found, terminating stream (graph integrity error: nodeId not in graph)",
                streamIdx, state.currentNodeId);
        state.currentNodeId = INVALID_TASK_ID;
        return;
    }

    if (processedNodes_.find(state.currentNodeId) != processedNodes_.end()) {
        SK_LOGD("Stream %u: node %lu already processed, advancing",
                streamIdx, state.currentNodeId);
        state.currentNodeId = node->GetNextNodeId();
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
        SK_LOGD("Stream %u: node %s scopeBitFlags mismatch, terminating stream",
                streamIdx, node->FormatNodeInfo().c_str());
        state.isTerminated = true;
        return;
    }

    // Check fusibility
    if (!node->IsFusible()) {
        SK_LOGD("Stream %u: node %s is not fusible, terminating stream",
                streamIdx, node->FormatNodeInfo().c_str());
        state.isTerminated = true;
        return;
    }

    // Add to heap (no deadlock check in this pass)
    nodeHeap_.push(state.currentNodeId);
    SK_LOGD("Stream %u: added node %s to heap, heap size=%zu",
            streamIdx, node->FormatNodeInfo().c_str(), nodeHeap_.size());
}

void InitialScopeSplitPass::HandleWaitNode(SuperKernelBaseNode* waitNode, uint32_t streamIdx) {
    StreamState& state = streamStates_[streamIdx];
    uint64_t notifyId = waitNode->GetCorrespondingNotifyNodeId();
    SuperKernelBaseNode* notifyNode = graph_.GetNodeById(notifyId);

    if (notifyNode == nullptr) {
        nodeHeap_.push(state.currentNodeId);
        SK_LOGI("Stream %u: Wait node %s's notify node %lu not found, adding wait node to heap to avoid hanging",
                streamIdx, waitNode->FormatNodeInfo().c_str(), notifyId);
        return;
    }

    uint64_t eventId = notifyNode->GetEventId();
    if (visitedNotifies_.find(notifyId) != visitedNotifies_.end()) {
        // Notify already visited, add fusible wait node to heap
        nodeHeap_.push(state.currentNodeId);
        SK_LOGD("Stream %u: Wait node %s: notify %lu already visited, adding to heap",
                streamIdx, waitNode->FormatNodeInfo().c_str(), notifyId);
    } else {
        // Suspend stream
        state.isSuspended = true;
        state.waitingForNotify = eventId;
        SK_LOGD("Stream %u: Wait node %s: notify %lu not visited, suspending stream (eventId=0x%lx)",
                streamIdx, waitNode->FormatNodeInfo().c_str(), notifyId, eventId);
    }
}

void InitialScopeSplitPass::ProcessNotifyNode(SuperKernelBaseNode* notifyNode) {
    uint64_t eventId = notifyNode->GetEventId();
    uint64_t notifyNodeId = notifyNode->GetNodeId();
    const EventInfos* eventInfo = graph_.GetEventInfo(eventId);
    if (eventInfo == nullptr) {
        SK_LOGE("Event info for eventId=0x%lx not found (notifyNode: %s)",
                 eventId, notifyNode->FormatNodeInfo().c_str());
        return;
    }
    visitedNotifies_.insert(notifyNodeId);
    SK_LOGD("Notify node %s (eventId=0x%lx) added to visited, %zu wait nodes to check",
            notifyNode->FormatNodeInfo().c_str(), eventId, eventInfo->waitNodeIdList.size());

    uint32_t resumedCount = 0;
    for (uint64_t waitNodeId : eventInfo->waitNodeIdList) {
        SuperKernelBaseNode* waitNode = graph_.GetNodeById(waitNodeId);
        if (waitNode == nullptr) {
            SK_LOGE("Wait node %lu not found (graph integrity error: eventId=0x%lx references non-existent node)",
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
            SK_LOGD("Resumed stream %u for wait node %s",
                    streamIdx, waitNode->FormatNodeInfo().c_str());
        }
    }
    SK_LOGD("Resumed %u suspended streams", resumedCount);
}

void InitialScopeSplitPass::ProcessResetNode(SuperKernelBaseNode* resetNode) {
    uint64_t eventId = resetNode->GetEventId();
    const EventInfos* eventInfo = graph_.GetEventInfo(eventId);
    if (eventInfo == nullptr) {
        SK_LOGE("Event info for eventId=0x%lx not found (resetNode: %s)",
                 eventId, resetNode->FormatNodeInfo().c_str());
        return;
    }

    if (eventInfo->notifyNodeId != INVALID_TASK_ID) {
        visitedNotifies_.erase(eventInfo->notifyNodeId);
        SK_LOGD("Erased notify %lu from visited (eventId=0x%lx)",
                eventInfo->notifyNodeId, eventId);
    }

    uint32_t suspendedCount = 0;
    for (uint64_t waitNodeId : eventInfo->waitNodeIdList) {
        SuperKernelBaseNode* waitNode = graph_.GetNodeById(waitNodeId);
        if (waitNode == nullptr) {
            SK_LOGE("Wait node %lu not found (graph integrity error: eventId=0x%lx references non-existent node)",
                    waitNodeId, eventId);
            continue;
        }

        uint32_t streamIdx = waitNode->GetStreamIdxInGraph();
        if (streamStates_[streamIdx].waitingForNotify == eventId) {
            streamStates_[streamIdx].waitingForNotify = INVALID_TASK_ID;
            streamStates_[streamIdx].isSuspended = true;
            suspendedCount++;
            SK_LOGE("Suspended stream %u for wait node %s (eventId=0x%lx)",
                    streamIdx, waitNode->FormatNodeInfo().c_str(), eventId);
        }
    }
    SK_LOGE("Suspended %u streams due to reset", suspendedCount);
}

bool InitialScopeSplitPass::HandleUnfusibleNotifyNode(SuperKernelBaseNode* notifyNode, uint32_t streamIdx) {
    // Always record visited notifies for wait nodes to check
    visitedNotifies_.insert(notifyNode->GetNodeId());
    processedNodes_.insert(notifyNode->GetNodeId());
    SK_LOGD("Stream %u: Unfusible notify node %s added to visitedNotifies and processedNodes",
            streamIdx, notifyNode->FormatNodeInfo().c_str());

    // Check if any corresponding wait nodes are suspended and need processing
    return ResumeSuspendedWaitStreams(notifyNode, streamIdx);
}

bool InitialScopeSplitPass::ResumeSuspendedWaitStreams(SuperKernelBaseNode* notifyNode, uint32_t notifyStreamIdx) {
    std::vector<uint64_t> waitNodeIds = notifyNode->GetCorrespondingWaitNodeIds();
    uint64_t eventId = notifyNode->GetEventId();

    for (uint64_t waitNodeId : waitNodeIds) {
        SuperKernelBaseNode* waitNode = graph_.GetNodeById(waitNodeId);
        if (waitNode == nullptr) {
            SK_LOGE("Stream %u: Notify node %s's wait node %lu not found",
                    notifyStreamIdx, notifyNode->FormatNodeInfo().c_str(), waitNodeId);
            return false;
        }

        uint32_t waitStreamIdx = waitNode->GetStreamIdxInGraph();
        const StreamState& waitStreamState = streamStates_[waitStreamIdx];

        // Check if the wait stream is suspended waiting for this notify's event
        if (waitStreamState.isSuspended && waitStreamState.waitingForNotify == eventId) {
            // The wait stream was suspended because this notify wasn't visited.
            // Now that we've visited the notify, we need to skip the unfusible wait node
            // and continue processing that stream.
            SK_LOGI("Stream %u: Notify node %s visited, processing suspended wait stream %u (wait node %s)",
                    notifyStreamIdx, notifyNode->FormatNodeInfo().c_str(), waitStreamIdx,
                    waitNode->FormatNodeInfo().c_str());

            // Resume the stream (clear suspend state)
            StreamState& mutableWaitState = streamStates_[waitStreamIdx];
            mutableWaitState.isSuspended = false;
            mutableWaitState.waitingForNotify = INVALID_TASK_ID;

            // Skip unfusible nodes on that stream (including the wait node)
            if (!SkipUnfusibleNodesForStream(waitStreamIdx)) {
                SK_LOGE("Stream %u: Failed to skip unfusible nodes for suspended wait stream %u (wait node %s)",
                        notifyStreamIdx, waitStreamIdx, waitNode->FormatNodeInfo().c_str());
                return false;
            }
        }
    }
    return true;
}

void InitialScopeSplitPass::LogFusibleNodeSearchResult() {
    SK_LOGI("[FindFusibleNodes] No fusible nodes found in any stream");

    // Check if all streams have finished (currentNodeId == INVALID_TASK_ID)
    bool allStreamsFinished = true;
    std::vector<uint32_t> nonFinishedStreamIdxs;

    for (const auto& pair : streamStates_) {
        uint32_t streamIdx = pair.first;
        const StreamState& state = pair.second;

        if (state.currentNodeId != INVALID_TASK_ID) {
            allStreamsFinished = false;
            nonFinishedStreamIdxs.push_back(streamIdx);
        }
    }

    if (allStreamsFinished) {
        SK_LOGI("[FindFusibleNodes] All streams have finished (all currentNodeId are INVALID_TASK_ID)");
    } else {
        // ERROR level when there are streams not finished but no fusible nodes found
        SK_LOGE("[FindFusibleNodes] Found %zu streams with non-INVALID currentNodeId but no fusible nodes",
                nonFinishedStreamIdxs.size());

        // Print detailed information about streams that haven't finished
        for (uint32_t streamIdx : nonFinishedStreamIdxs) {
            const StreamState& state = streamStates_.at(streamIdx);
            std::string stateInfo = state.FormatStreamStateInfo();
            SK_LOGE("[FindFusibleNodes] Stream %u state: %s", streamIdx, stateInfo.c_str());

            // If there's a current node, try to get more information
            if (state.currentNodeId != INVALID_TASK_ID) {
                SuperKernelBaseNode* node = graph_.GetNodeById(state.currentNodeId);
                if (node != nullptr) {
                    bool isFusible = node->IsFusible();
                    SkNodeType nodeType = node->GetNodeType();
                    std::string scopeBitFlagsStr = node->GetScopeBitFlags().to_string().substr(0, MAX_SCOPE_NUM);
                    SK_LOGE("[FindFusibleNodes] Stream %u current node: %s, isFusible=%d, nodeType=%d, scopeBitFlags=%s",
                            streamIdx, node->FormatNodeInfo().c_str(), isFusible,
                            static_cast<int>(nodeType), scopeBitFlagsStr.c_str());
                }
            }
        }
    }
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
    SK_LOGI("[SplitScope] starting to build current scope");
    scopeInfo.scopeBitFlags = currentScopeBitFlags_;

    size_t nodeCount = 0;
    while (!nodeHeap_.empty()) {
        uint64_t nodeId = nodeHeap_.top();
        nodeHeap_.pop();

        SuperKernelBaseNode* node = graph_.GetNodeById(nodeId);
        if (node == nullptr) {
            SK_LOGE("Node %lu not found in graph (graph integrity error)", nodeId);
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
            SK_LOGD("Added node %s to scope (count=%zu)",
                    node->FormatNodeInfo().c_str(), nodeCount);
        }
        AddStreamInfoToScope(scopeInfo, node);
        processedNodes_.insert(nodeId);

        // Advance stream
        uint64_t nextNodeId = node->GetNextNodeId();
        streamStates_[streamIdx].currentNodeId = nextNodeId;

        // Handle special nodes
        if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
            ProcessNotifyNode(node);
        } else if (node->GetNodeType() == SkNodeType::NODE_RESET) {
            ProcessResetNode(node);
        }

        // Try next node
        TryAddNodeToHeap(streamIdx);
    }

    SK_LOGI("[SplitScope] built scope with %zu nodes, %zu stream infos",
            nodeCount, scopeInfo.scopeStreamInfos.size());
    return true;  // Empty scope is not an error, allow it to be skipped in Run()
}

bool InitialScopeSplitPass::Run(std::vector<SuperKernelScopeInfo>& scopes) {
    SK_LOGI("[SplitScope] %s pass starting execution", GetName().c_str());

    scopes.clear();
    processedNodes_.clear();
    visitedNotifies_.clear();

    SK_LOGI("%s: Initializing stream states", GetName().c_str());
    InitStreamStates();
    if (!SkipUnfusibleNodes()) {
        SK_LOGE("%s: Failed to skip unfusible nodes during initialization (graph integrity error)",
                GetName().c_str());
        return false;
    }

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
        if (!ResetStreamStates()) {
            SK_LOGE("%s: Failed to reset stream states (graph integrity error), stopping scope split",
                    GetName().c_str());
            return false;
        }
    }

    SK_LOGI("[SplitScopeResult] %s pass completed, total scopes generated: %zu", GetName().c_str(), scopes.size());
    PrintScopeResults(scopes, graph_);
    return true;
}

// ============ DeadlockRefinePass Implementation ============

DeadlockRefinePass::DeadlockRefinePass(SuperKernelGraph& inputGraph)
    : ScopeSplitPass(inputGraph), lockDetector_(graph_) {}

bool DeadlockRefinePass::FindDeadlockInScope(const SuperKernelScopeInfo& scope,
                                              SuperKernelBaseNode** deadlockNode,
                                              SuperKernelBaseNode** deadlockWaitNode) {
    SK_LOGI("[DeadlockRefine] checking scope with %zu nodes for deadlock", scope.nodes.size());
    lockDetector_.Reset();

    // Track the most recent Wait node seen before each node
    SuperKernelBaseNode* lastWaitNode = nullptr;

    // Check each node for deadlock, keeping track of the nearest preceding Wait node
    for (size_t i = 0; i < scope.nodes.size(); ++i) {
        const auto* node = scope.nodes[i];
        // Update lastWaitNode when we encounter a Wait node
        if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
            lastWaitNode = const_cast<SuperKernelBaseNode*>(node);
            SK_LOGD("Found Wait node %s at position %zu",
                    node->FormatNodeInfo().c_str(), i);
        }

        // Check for deadlock
        if (!lockDetector_.IsFusible(*const_cast<SuperKernelBaseNode*>(node))) {
            *deadlockNode = const_cast<SuperKernelBaseNode*>(node);
            *deadlockWaitNode = lastWaitNode;  // The nearest Wait node before deadlock point
            SK_LOGI("Deadlock detected at node %s (position %zu)",
                    node->FormatNodeInfo().c_str(), i);
            if (lastWaitNode != nullptr) {
                SK_LOGI("  Nearest Wait node before deadlock: %s",
                        lastWaitNode->FormatNodeInfo().c_str());
            }
            return true;
        }
        SK_LOGD("Node %s passed deadlock check", node->FormatNodeInfo().c_str());
    }

    SK_LOGI("[DeadlockRefine] no deadlock found in scope");
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
    SK_LOGI("[DeadlockRefine] %s pass starting execution", GetName().c_str());
    SK_LOGI("[DeadlockRefine] input scopes count: %zu", scopes.size());
    
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

    SK_LOGI("[DeadlockRefine] %s pass completed, split %zu scopes, total scopes: %zu",
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
    SK_LOGI("[ScopeSplitPipeline] starting scope splitting pipeline");
    
    scopeInfos_.clear();

    // Run all passes
    for (auto& pass : passes_) {
        SK_LOGI("[ScopeSplitPipeline] running pass: %s", pass->GetName().c_str());
        if (!pass->Run(scopeInfos_)) {
            SK_LOGE("[ScopeSplitPipeline] pass %s failed (input scopes: %zu)", pass->GetName().c_str(), scopeInfos_.size());
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
    SK_LOGI("[ScopeSplitPipeline] scope splitting complete, total scopes: %zu", scopeInfos_.size());
    ScopeSplitPass::PrintScopeResults(scopeInfos_, graph_);
}
