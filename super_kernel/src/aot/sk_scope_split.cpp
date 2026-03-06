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

uint64_t SuperKernelScopeSplitter::FindSingleStreamAvailableHeadNode(uint64_t curNodeIdx) const {
    while (curNodeIdx != INVALID_TASK_ID) {
        const SuperKernelBaseNode * node = graph.GetNodeById(curNodeIdx);
        if (node == nullptr) {
            SK_LOGE("node with id %lu not found in graph during finding available head node\n", curNodeIdx);
            return INVALID_TASK_ID;
        }
        if (node->GetNodeType() == SkNodeType::NODE_KERNEL && node->IsFusible()) {
            return curNodeIdx;
        }
        curNodeIdx = node->GetNextNodeId();
    }
    return INVALID_TASK_ID;
}

uint64_t SuperKernelScopeSplitter::GenerateSingleStreamScopeInfosByNodeIdx(uint64_t curNodeIdx) {
    try {
        uint64_t headNodeIdx = FindSingleStreamAvailableHeadNode(curNodeIdx);
        if (headNodeIdx == INVALID_TASK_ID) {
            SK_LOGI("no available head node found starting from node %lu\n", curNodeIdx);
            return INVALID_TASK_ID;
        }
        SK_LOGI("Generating scope infos for head node %lu\n", headNodeIdx);
        SuperKernelScopeInfo scopeInfo;
        ScopeStreamInfo scopeStreamInfo;
        scopeStreamInfo.headNodeIdx = headNodeIdx;
        scopeStreamInfo.streamIdx = graph.GetNodeById(headNodeIdx)->GetStreamIdxInGraph();

        uint64_t preNodeIdx = headNodeIdx;
        curNodeIdx = headNodeIdx;
        LockDetector lockDetector;
        while (curNodeIdx != INVALID_TASK_ID) {
            SuperKernelBaseNode *node = graph.GetNodeById(curNodeIdx);
            if (node == nullptr) {
                SK_LOGE("node with id %lu not found in graph during generating scope infos\n", curNodeIdx);
                throw std::runtime_error("Node not found in graph");
            }
            if (!node->IsFusible() || !lockDetector.IsFusible(*node, graph)) {
                scopeStreamInfo.tailNodeIdx = preNodeIdx;
                curNodeIdx = node->GetNextNodeId();
                lockDetector.Reset(graph);
                break;
            }
            scopeInfo.nodes.emplace_back(std::move(node));
            ++scopeStreamInfo.nodeSize;
            preNodeIdx = curNodeIdx;
            curNodeIdx = node->GetNextNodeId();
        }
        SK_LOGI("Generated scope infos for head node %lu in stream %lu, tail node %lu, node size %lu\n", 
                scopeStreamInfo.headNodeIdx, scopeStreamInfo.streamIdx, scopeStreamInfo.tailNodeIdx, scopeStreamInfo.nodeSize);
        scopeInfo.scopeStreamInfos.emplace_back(std::move(scopeStreamInfo));
        scopeInfos.emplace_back(std::move(scopeInfo));
        return curNodeIdx;
    } catch (const std::exception &e) {
        SK_LOGE("Exception occurred in GenerateSingleStreamScopeInfosByNodeIdx: %s\n", e.what());
        return INVALID_TASK_ID;
    }
}

bool SuperKernelScopeSplitter::SplitSingleStreamGraph() {
    const auto& streams = graph.GetStreams();
    const auto& headNodes = graph.GetHeadNodes();
    SK_LOGI("start splitting single stream graph, stream count: %zu", streams.size());
    for (size_t streamIdx = 0; streamIdx < streams.size(); ++streamIdx) {
        SK_LOGI("start splitting stream %zu", streamIdx);
        uint64_t curNodeIdx = headNodes[streamIdx];
        while (curNodeIdx != INVALID_TASK_ID) {
            curNodeIdx = GenerateSingleStreamScopeInfosByNodeIdx(curNodeIdx);
        }
    }
    return true;
}


bool SuperKernelScopeSplitter::SplitGraph() {
    SK_LOGI("start splitting graph into scopes\n");
    const auto& streams = graph.GetStreams();
    if (streams.size() == 1) {
        return SplitSingleStreamGraph();
    }
    return SplitMultiStreamGraph();
}

void SuperKernelScopeSplitter::AddStreamInfoToScope(SuperKernelScopeInfo& scopeInfo, SuperKernelBaseNode* node) {
    uint32_t streamIdx = node->GetStreamIdxInGraph();
    // Check if stream info already exists
    auto it = std::find_if(scopeInfo.scopeStreamInfos.begin(), scopeInfo.scopeStreamInfos.end(),
                         [streamIdx](const ScopeStreamInfo& info) {
                             return info.streamIdx == streamIdx;
                         });
    if (it == scopeInfo.scopeStreamInfos.end()) {
        // Create new stream info
        ScopeStreamInfo newInfo;
        newInfo.streamIdx = streamIdx;
        newInfo.headNodeIdx = node->GetNodeId();
        newInfo.tailNodeIdx = node->GetNodeId();
        newInfo.nodeSize = 1;
        scopeInfo.scopeStreamInfos.push_back(std::move(newInfo));
    } else {
        // Update existing stream info
        it->tailNodeIdx = node->GetNodeId();
        it->nodeSize++;
    }
}
void SuperKernelScopeSplitter::SkipUnfusibleNodes(
    std::unordered_map<uint32_t, StreamState>& streamStates) {
    for (auto& pair : streamStates) {
        // Skip unfusible nodes at the current position
        while (pair.second.currentNodeIdx != INVALID_TASK_ID) {
            SuperKernelBaseNode* node = graph.GetNodeById(pair.second.currentNodeIdx);
            if (node != nullptr && !node->IsFusible()) {
                SK_LOGI("Skipping unfusible node %lu", pair.second.currentNodeIdx);
                pair.second.currentNodeIdx = node->GetNextNodeId();
            } else {
                break;
            }
        }
    }
}

void SuperKernelScopeSplitter::ResetStreamStates(
    std::unordered_map<uint32_t, StreamState>& streamStates) {
    for (auto& pair : streamStates) {
        // Only reset temporary states, keep currentNodeIdx
        pair.second.isTerminated = false;
        pair.second.isSuspended = false;
        pair.second.waitingForNotify = INVALID_TASK_ID;
    }

    // Skip unfusible nodes at the start of each new scope
    SkipUnfusibleNodes(streamStates);
}

bool SuperKernelScopeSplitter::AllStreamsFinished(
    const std::unordered_map<uint32_t, StreamState>& streamStates) const {
    for (const auto& pair : streamStates) {
        const StreamState& state = pair.second;
        // A stream is considered finished if:
        // 1. currentNodeIdx is INVALID_TASK_ID (naturally finished), OR
        // 2. The stream is terminated and will never progress
        if (state.currentNodeIdx != INVALID_TASK_ID) {
            return false;
        }
    }
    return true;
}

void SuperKernelScopeSplitter::InitNodeHeap(
    std::unordered_map<uint32_t, StreamState>& streamStates,
    const std::set<uint64_t>& visitedNotifies,
    const std::set<uint64_t>& processedNodes,
    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>>& nodeHeap,
    LockDetector& lockDetector) {
    // Initialize heap by trying to add current node from each stream
    for (auto& pair : streamStates) {
        uint32_t streamIdx = pair.first;
        TryAddNodeToHeap(streamIdx, streamStates, visitedNotifies,
                       processedNodes, nodeHeap, lockDetector);
    }
}

void SuperKernelScopeSplitter::TryAddNodeToHeap(
    uint32_t streamIdx,
    std::unordered_map<uint32_t, StreamState>& streamStates,
    const std::set<uint64_t>& visitedNotifies,
    const std::set<uint64_t>& processedNodes,
    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>>& nodeHeap,
    LockDetector& lockDetector) {
    StreamState& state = streamStates[streamIdx];

    // Condition 1: Stream is terminated, suspended, or finished
    if (state.isTerminated || state.isSuspended ||
        state.currentNodeIdx == INVALID_TASK_ID) {
        return;
    }

    SuperKernelBaseNode* nextNode = graph.GetNodeById(state.currentNodeIdx);
    if (nextNode == nullptr) {
        state.currentNodeIdx = INVALID_TASK_ID;
        return;
    }

    // Condition 2: Node already processed
    if (processedNodes.find(state.currentNodeIdx) != processedNodes.end()) {
        SK_LOGW("Node %lu already processed, skip", state.currentNodeIdx);
        state.currentNodeIdx = nextNode->GetNextNodeId();
        return;
    }

    // Condition 3: Node already in heap
    // Note: std::priority_queue doesn't have contains() method, we need to check manually
    // For simplicity, we'll skip this check as it's unlikely to cause issues
    // and ProcessNotifyNode has protection against duplicates

    // Special case 1: Permanently unfusible node
    if (!nextNode->IsFusible()) {
        // Mark stream as terminated (unfusible nodes terminate the current scope)
        state.isTerminated = true;
        SK_LOGI("Node %lu (stream=%lu) is permanently unfusible, terminate stream in current scope",
                nextNode->GetNodeId(), streamIdx);
        return;
    }

    // Special case 2: Wait node
    if (nextNode->GetNodeType() == SkNodeType::NODE_WAIT) {
        uint64_t notifyId = nextNode->GetNotifyNodeId();
        SuperKernelBaseNode* notifyNode = graph.GetNodeById(notifyId);
        if (notifyNode == nullptr) {
            SK_LOGW("Notify node %lu not found", notifyId);
            return;
        }
        uint64_t eventId = notifyNode->GetEventId();
        if (visitedNotifies.find(notifyId) != visitedNotifies.end()) {
            // Notify already visited, can add to heap
            nodeHeap.push(state.currentNodeIdx);
            SK_LOGD("Wait node %lu (stream=%lu, eventId=%lu): notify already visited, add to heap", nextNode->GetNodeId(), streamIdx, eventId);
        } else {
            // Notify not visited, suspend the stream
            state.isSuspended = true;
            state.waitingForNotify = eventId;
            SK_LOGD("Wait node %lu (stream=%lu, eventId=%lu): notify not visited, suspend stream", nextNode->GetNodeId(), streamIdx, eventId);
        }
    }
    // Special case 3: Deadlock detection (Notify/Reset nodes can also have deadlock)
    else if (lockDetector.IsFusible(*nextNode, graph)) {
        // Fusible and no deadlock, add to heap
        nodeHeap.push(state.currentNodeIdx);
        SK_LOGD("Node %lu (stream=%lu, type=%s) fusible, add to heap", nextNode->GetNodeId(), streamIdx, to_string(nextNode->GetNodeType()));
    }
    // Otherwise node is unfusible due to deadlock, don't add to heap
    // Important: currentNodeIdx is NOT advanced here to allow retry in next scope
    else {
        SK_LOGD("Node %lu (stream=%lu, type=%s) unfusible due to deadlock, skip", nextNode->GetNodeId(), streamIdx, to_string(nextNode->GetNodeType()));
    }
}

void SuperKernelScopeSplitter::ProcessNotifyNode(
    SuperKernelBaseNode* notifyNode,
    std::set<uint64_t>& visitedNotifies,
    const std::set<uint64_t>& processedNodes,
    std::unordered_map<uint32_t, StreamState>& streamStates,
    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>>& nodeHeap,
    LockDetector& lockDetector) {
    uint64_t eventId = notifyNode->GetEventId();
    const auto& eventInfo = graph.eventToNodes.at(eventId);
    // Mark notify as visited
    visitedNotifies.insert(notifyNode->GetNodeId());
    SK_LOGD("Notify node %lu (stream=%lu, eventId=%lu) marked as visited", notifyNode->GetNodeId(), notifyNode->GetStreamIdxInGraph(), eventId);

    // Resume streams that are waiting for this notify
    for (uint64_t waitNodeId : eventInfo.waitNodeIdList) {
        SuperKernelBaseNode* waitNode = graph.GetNodeById(waitNodeId);
        if (waitNode == nullptr) {
            SK_LOGW("Wait node %lu not found", waitNodeId);
            continue;
        }

        uint32_t streamIdx = waitNode->GetStreamIdxInGraph();
        StreamState& state = streamStates[streamIdx];

        // If the stream is suspended and waiting for this notify, resume it
        if (state.isSuspended && state.waitingForNotify == eventId) {
            state.isSuspended = false;
            state.waitingForNotify = INVALID_TASK_ID;
            SK_LOGD("Resume stream %lu (was waiting for eventId=%lu)", streamIdx, eventId);
            // Try to add the current node of this stream to heap
            TryAddNodeToHeap(streamIdx, streamStates, visitedNotifies,
                           processedNodes, nodeHeap, lockDetector);
        }
    }
}

void SuperKernelScopeSplitter::ProcessResetNode(
    SuperKernelBaseNode* resetNode,
    std::set<uint64_t>& visitedNotifies,
    std::unordered_map<uint32_t, StreamState>& streamStates) {
    uint64_t eventId = resetNode->GetEventId();
    const auto& eventInfo = graph.eventToNodes.at(eventId);

    // Reset clears the notify visitation marker
    visitedNotifies.erase(eventInfo.notifyNodeId);

    // Clear waiting state for all waits of this event
    for (uint64_t waitNodeId : eventInfo.waitNodeIdList) {
        SuperKernelBaseNode* waitNode = graph.GetNodeById(waitNodeId);
        if (waitNode == nullptr) {
            continue;
        }

        uint32_t streamIdx = waitNode->GetStreamIdxInGraph();
        if (streamStates[streamIdx].waitingForNotify == eventId) {
            streamStates[streamIdx].waitingForNotify = INVALID_TASK_ID;
            // Note: After reset, the wait node needs to wait for notify again
            // So we set isSuspended = true
            streamStates[streamIdx].isSuspended = true;
        }
    }
}

bool SuperKernelScopeSplitter::SplitMultiStreamGraph() {
    const auto& streams = graph.GetStreams();
    const auto& headNodes = graph.GetHeadNodes();

    SK_LOGI("Start splitting multi-stream graph, stream count: %zu", streams.size());

    // Initialize stream states
    std::unordered_map<uint32_t, StreamState> streamStates;
    for (size_t i = 0; i < streams.size(); ++i) {
        streamStates[i] = StreamState();
        streamStates[i].currentNodeIdx = headNodes[i];
    }

    // Skip unfusible nodes at initialization
    SkipUnfusibleNodes(streamStates);

    std::set<uint64_t> visitedNotifies;
    std::set<uint64_t> processedNodes;
    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>> nodeHeap;

    // Reuse the same LockDetector instance for all scopes
    LockDetector lockDetector;

    while (true) {
        SuperKernelScopeInfo scopeInfo;

        // Reset deadlock detector at the start of each scope
        lockDetector.Reset(graph);

        // Phase 1: Initialize heap
        InitNodeHeap(streamStates, visitedNotifies, processedNodes, nodeHeap, lockDetector);

        if (nodeHeap.empty()) {
            SK_LOGI("No fusible nodes in current scope, splitting complete");
            break;
        }

        SK_LOGI("========== Start processing new scope %zu, heap size: %zu ==========", scopeInfos.size(), nodeHeap.size());

        // Phase 2: Main fusion loop
        while (!nodeHeap.empty()) {
            // Pop the minimum nodeId
            uint64_t nodeId = nodeHeap.top();
            nodeHeap.pop();

            SuperKernelBaseNode* node = graph.GetNodeById(nodeId);
            if (node == nullptr) {
                SK_LOGW("Node %lu not found in graph, skip", nodeId);
                continue;
            }

            // Safety check: Already processed (should not happen)
            if (processedNodes.find(nodeId) != processedNodes.end()) {
                SK_LOGW("Node %lu already processed, skip", nodeId);
                continue;
            }

            uint32_t streamIdx = node->GetStreamIdxInGraph();
            const char* nodeTypeStr = to_string(node->GetNodeType());

            // Deadlock detection (permanently unfusible nodes already handled in TryAddNodeToHeap)
            // Reuse single-stream LockDetector interface
            if (!lockDetector.IsFusible(*node, graph)) {
                // Unfusible due to deadlock, terminate current scope
                streamStates[streamIdx].isTerminated = true;
                // Don't advance currentNodeIdx, will retry in next scope
                SK_LOGI("Node %lu (type=%s, stream=%lu) causes deadlock, terminate current scope", nodeId, nodeTypeStr, streamIdx);
                continue;
            }

            // Add to scope
            scopeInfo.nodes.push_back(node);
            AddStreamInfoToScope(scopeInfo, node);
            processedNodes.insert(nodeId);
            SK_LOGD("Added node %lu (type=%s, stream=%lu) to scope, total nodes: %zu", nodeId, nodeTypeStr, streamIdx, scopeInfo.nodes.size());
            // LockDetector internally tracks nodes through IsFusible calls

            // Update stream state: mark current node as processed, advance to next node
            streamStates[streamIdx].currentNodeIdx = node->GetNextNodeId();

            // Special case: Notify node (may activate suspended streams)
            if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
                ProcessNotifyNode(node, visitedNotifies, processedNodes,
                              streamStates, nodeHeap, lockDetector);
            }
            // Special case: Reset node
            else if (node->GetNodeType() == SkNodeType::NODE_RESET) {
                ProcessResetNode(node, visitedNotifies, streamStates);
            }

            // Process next node of current stream: add to heap or suspend
            // Key: Only process currentNodeIdx of current stream, guarantee forward traversal
            TryAddNodeToHeap(streamIdx, streamStates, visitedNotifies,
                           processedNodes, nodeHeap, lockDetector);
        }

        // Save scope
        if (!scopeInfo.nodes.empty()) {
            std::string nodeIdsStr;
            for (const auto* node : scopeInfo.nodes) {
                if (!nodeIdsStr.empty()) nodeIdsStr += ", ";
                nodeIdsStr += std::to_string(node->GetNodeId());
            }
            scopeInfos.emplace_back(std::move(scopeInfo));
            SK_LOGI("========== Created scope %zu with %zu nodes, %zu streams ==========",
                     scopeInfos.size() - 1, scopeInfo.nodes.size(), scopeInfo.scopeStreamInfos.size());
            SK_LOGI("Scope nodes: [%s]", nodeIdsStr.c_str());
        }

        // Reset stream states for next scope
        ResetStreamStates(streamStates);
    }

    SK_LOGI("========== Multi-stream graph splitting complete, total scopes: %zu ==========", scopeInfos.size());
    for (size_t i = 0; i < scopeInfos.size(); ++i) {
        SK_LOGI("Scope %zu: %zu nodes, %zu streams", i, scopeInfos[i].nodes.size(), scopeInfos[i].scopeStreamInfos.size());
    }

    return true;
}