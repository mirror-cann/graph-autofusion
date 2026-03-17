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
        nodeDetails += n->GetNodeName();
        nodeDetails += " ";
        nodeDetails += std::to_string(n->GetNodeId());
        nodeDetails += "(";
        nodeDetails += to_string(n->GetNodeType());
        nodeDetails += ",stream=";
        nodeDetails += std::to_string(n->GetStreamIdxInGraph());
        nodeDetails += ",nodeIdxInStream=";
        nodeDetails += std::to_string(n->GetNodeIdxInStream());
        nodeDetails += ")";
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
    const auto& streams = graph_.GetStreams();
    const auto& headNodes = graph_.GetHeadNodes();
    
    streamStates_.clear();
    for (size_t i = 0; i < streams.size(); ++i) {
        streamStates_[i] = StreamState();
        streamStates_[i].currentNodeIdx = headNodes[i];
    }
}

void InitialScopeSplitPass::ResetStreamStates() {
    for (auto& pair : streamStates_) {
        pair.second.isTerminated = false;
        pair.second.isSuspended = false;
        pair.second.waitingForNotify = INVALID_TASK_ID;
    }
    SkipUnfusibleNodes();
}

void InitialScopeSplitPass::SkipUnfusibleNodes() {
    for (auto& pair : streamStates_) {
        while (pair.second.currentNodeIdx != INVALID_TASK_ID) {
            SuperKernelBaseNode* node = graph_.GetNodeById(pair.second.currentNodeIdx);
            if (node == nullptr) {
                break;
            }
            SkNodeType nodeType = node->GetNodeType();
            // Skip permanently unfusible nodes
            if (!node->IsFusible()) {
                pair.second.currentNodeIdx = node->GetNextNodeId();
            } else {
                break;
            }
        }
    }
}


bool InitialScopeSplitPass::DetermineCurrentScopeBitFlags() {
    uint64_t minNodeIdx = UINT64_MAX;
    SuperKernelBaseNode* minNode = nullptr;
    
    for (const auto& pair : streamStates_) {
        if (!pair.second.isTerminated && !pair.second.isSuspended && 
            pair.second.currentNodeIdx != INVALID_TASK_ID &&
            pair.second.currentNodeIdx < minNodeIdx) {
            SuperKernelBaseNode* node = graph_.GetNodeById(pair.second.currentNodeIdx);
            if (node != nullptr && node->IsFusible()) {
                minNodeIdx = pair.second.currentNodeIdx;
                minNode = node;
            }
        }
    }
    
    if (minNode != nullptr) {
        currentScopeBitFlags_ = minNode->GetScopeBitFlags();
        return true;
    }
    return false;
}

void InitialScopeSplitPass::InitNodeHeap() {
    while (!nodeHeap_.empty()) {
        nodeHeap_.pop();
    }
    for (auto& pair : streamStates_) {
        TryAddNodeToHeap(pair.first);
    }
}

void InitialScopeSplitPass::TryAddNodeToHeap(uint32_t streamIdx) {
    StreamState& state = streamStates_[streamIdx];
    
    if (state.isTerminated || state.isSuspended || state.currentNodeIdx == INVALID_TASK_ID) {
        return;
    }
    
    SuperKernelBaseNode* node = graph_.GetNodeById(state.currentNodeIdx);
    if (node == nullptr) {
        state.currentNodeIdx = INVALID_TASK_ID;
        return;
    }
    
    if (processedNodes_.find(state.currentNodeIdx) != processedNodes_.end()) {
        state.currentNodeIdx = node->GetNextNodeId();
        TryAddNodeToHeap(streamIdx);
        return;
    }
    
    // Check scopeBitFlags match
    if (node->GetScopeBitFlags() != currentScopeBitFlags_) {
        state.isTerminated = true;
        return;
    }
    
    // Check fusibility
    if (!node->IsFusible()) {
        state.isTerminated = true;
        return;
    }
    
    // Special handling for Wait nodes
    if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
        HandleWaitNode(node, streamIdx);
        return;
    }
    
    // Add to heap (no deadlock check in this pass)
    nodeHeap_.push(state.currentNodeIdx);
}

void InitialScopeSplitPass::HandleWaitNode(SuperKernelBaseNode* waitNode, uint32_t streamIdx) {
    StreamState& state = streamStates_[streamIdx];
    uint64_t notifyId = waitNode->GetCorrespondingNotifyNodeId();
    SuperKernelBaseNode* notifyNode = graph_.GetNodeById(notifyId);
    
    if (notifyNode == nullptr) {
        return;
    }
    
    uint64_t eventId = notifyNode->GetEventId();
    if (visitedNotifies_.find(notifyId) != visitedNotifies_.end()) {
        // Notify already visited, add to heap
        nodeHeap_.push(state.currentNodeIdx);
    } else {
        // Suspend stream
        state.isSuspended = true;
        state.waitingForNotify = eventId;
    }
}

void InitialScopeSplitPass::ProcessNotifyNode(SuperKernelBaseNode* notifyNode) {
    uint64_t eventId = notifyNode->GetEventId();
    const EventInfos* eventInfo = graph_.GetEventInfo(eventId);
    if (eventInfo == nullptr) {
        return;
    }
    visitedNotifies_.insert(notifyNode->GetNodeId());
    
    for (uint64_t waitNodeId : eventInfo->waitNodeIdList) {
        SuperKernelBaseNode* waitNode = graph_.GetNodeById(waitNodeId);
        if (waitNode == nullptr) {
            continue;
        }
        
        uint32_t streamIdx = waitNode->GetStreamIdxInGraph();
        StreamState& state = streamStates_[streamIdx];
        
        if (state.isSuspended && state.waitingForNotify == eventId) {
            state.isSuspended = false;
            state.waitingForNotify = INVALID_TASK_ID;
            TryAddNodeToHeap(streamIdx);
        }
    }
}

void InitialScopeSplitPass::ProcessResetNode(SuperKernelBaseNode* resetNode) {
    uint64_t eventId = resetNode->GetEventId();
    const EventInfos* eventInfo = graph_.GetEventInfo(eventId);
    if (eventInfo == nullptr) {
        return;
    }
    
    visitedNotifies_.erase(eventInfo->notifyNodeId);
    
    for (uint64_t waitNodeId : eventInfo->waitNodeIdList) {
        SuperKernelBaseNode* waitNode = graph_.GetNodeById(waitNodeId);
        if (waitNode == nullptr) {
            continue;
        }
        
        uint32_t streamIdx = waitNode->GetStreamIdxInGraph();
        if (streamStates_[streamIdx].waitingForNotify == eventId) {
            streamStates_[streamIdx].waitingForNotify = INVALID_TASK_ID;
            streamStates_[streamIdx].isSuspended = true;
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
    scopeInfo.scopeBitFlags = currentScopeBitFlags_;
    
    while (!nodeHeap_.empty()) {
        uint64_t nodeId = nodeHeap_.top();
        nodeHeap_.pop();
        
        SuperKernelBaseNode* node = graph_.GetNodeById(nodeId);
        if (node == nullptr || processedNodes_.find(nodeId) != processedNodes_.end()) {
            continue;
        }
        
        uint32_t streamIdx = node->GetStreamIdxInGraph();
        
        // Add to scope
        if (!node->IsScopeNode()) {
            scopeInfo.nodes.push_back(node);
        }
        AddStreamInfoToScope(scopeInfo, node);
        processedNodes_.insert(nodeId);
        
        // Advance stream
        streamStates_[streamIdx].currentNodeIdx = node->GetNextNodeId();
        
        // Handle special nodes
        if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
            ProcessNotifyNode(node);
        } else if (node->GetNodeType() == SkNodeType::NODE_RESET) {
            ProcessResetNode(node);
        }
        
        // Try next node
        TryAddNodeToHeap(streamIdx);
    }
    
    return !scopeInfo.nodes.empty();
}

bool InitialScopeSplitPass::Run(std::vector<SuperKernelScopeInfo>& scopes) {
    SK_LOGI("=== %s Start ===", GetName().c_str());
    
    scopes.clear();
    processedNodes_.clear();
    visitedNotifies_.clear();
    
    InitStreamStates();
    SkipUnfusibleNodes();
    
    size_t scopeCount = 0;
    while (true) {
        if (!DetermineCurrentScopeBitFlags()) {
            SK_LOGI("No more fusible nodes, initial scope split complete");
            break;
        }
        
        InitNodeHeap();
        if (nodeHeap_.empty()) {
            SK_LOGI("Node heap is empty, initial scope split complete");
            break;
        }
        
        SuperKernelScopeInfo scopeInfo;
        if (BuildCurrentScope(scopeInfo)) {
            SK_LOGI("Built scope %zu with %zu nodes, scopeBitFlags=%s",
                    scopeCount, scopeInfo.nodes.size(),
                    scopeInfo.scopeBitFlags.to_string().substr(0, MAX_SCOPE_NUM).c_str());
            scopes.push_back(std::move(scopeInfo));
            scopeCount++;
        }
        
        ResetStreamStates();
    }
    
    SK_LOGI("%s complete, total scopes: %zu", GetName().c_str(), scopes.size());
    PrintScopeResults(scopes, graph_);
    return true;
}

// ============ DeadlockRefinePass Implementation ============

DeadlockRefinePass::DeadlockRefinePass(SuperKernelGraph& inputGraph)
    : ScopeSplitPass(inputGraph), lockDetector_() {}

bool DeadlockRefinePass::FindDeadlockInScope(const SuperKernelScopeInfo& scope,
                                              SuperKernelBaseNode** deadlockNode,
                                              SuperKernelBaseNode** deadlockWaitNode) {
    lockDetector_.Reset(graph_);
    
    // Track the most recent Wait node seen before each node
    SuperKernelBaseNode* lastWaitNode = nullptr;
    
    // Check each node for deadlock, keeping track of the nearest preceding Wait node
    for (const auto* node : scope.nodes) {
        // Update lastWaitNode when we encounter a Wait node
        if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
            lastWaitNode = const_cast<SuperKernelBaseNode*>(node);
        }
        
        // Check for deadlock
        if (!lockDetector_.IsFusible(*const_cast<SuperKernelBaseNode*>(node), graph_)) {
            *deadlockNode = const_cast<SuperKernelBaseNode*>(node);
            *deadlockWaitNode = lastWaitNode;  // The nearest Wait node before deadlock point
            return true;
        }
    }
    
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
    SK_LOGI("=== %s Start ===", GetName().c_str());
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
                
                SK_LOGI("Scope %zu: Deadlock detected at node %s, id: %lu, splitting at Wait node %lu",
                        i, deadlockNode->GetNodeName().c_str(), deadlockNode->GetNodeId(), deadlockWaitNode->GetNodeId());
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
                SK_LOGW("Scope %zu: Deadlock at node %s, id: %lu but no Wait node found to split",
                        i, deadlockNode->GetNodeName().c_str(), deadlockNode->GetNodeId());
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
            SK_LOGE("Pass %s failed", pass->GetName().c_str());
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
