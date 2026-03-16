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
 * \file sk_graph.cpp
 * \brief
 */

#include "sk_graph.h"

#include <stdexcept>
#include <vector>
#include <cstdint>
#include <bitset>

aclError SuperKernelGraph::Update() {
    aclError ret = aclmdlRIUpdate(modelRI);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to update modelRI");
    }
    return ret;
}

SuperKernelBaseNode* SuperKernelGraph::GetNodeById(uint64_t nodeId) const {
    auto it = graphMap.find(nodeId);
    if (it != graphMap.end()) {
        return it->second.get();
    }
    SK_LOGE("Node with id %lu not found in graph", nodeId);
    return nullptr;
}

bool SuperKernelGraph::AddNode(std::unique_ptr<SuperKernelBaseNode> node) {
    uint64_t nodeId = node->GetNodeId();
    uint64_t eventId = node->GetEventId();
    switch (node->GetNodeType()) {
        case SkNodeType::NODE_NOTIFY:
            if (!AddEventAssociateNotify(eventId, nodeId)) {
                SK_LOGE("Failed to associate notify event %lu with node %lu", eventId, nodeId);
                return false;
            }
            break;
        case SkNodeType::NODE_WAIT:
            if (!AddEventAssociateWait(eventId, nodeId)) {
                SK_LOGE("Failed to associate wait event %lu with node %lu", eventId, nodeId);
                return false;
            }
            break;
        case SkNodeType::NODE_RESET:
            if (!AddEventAssociateReset(eventId, nodeId)) {
                SK_LOGE("Failed to associate reset event %lu with node %lu", eventId, nodeId);
                return false;
            }
            break;
        default:
            break;
    }

    if (graphMap.find(nodeId) != graphMap.end()) {
        SK_LOGE("Node with id %lu already exists in graph", nodeId);
        return false;
    }
    graphMap[nodeId] = std::move(node);
    return true;
}

bool SuperKernelGraph::AddEventAssociateNotify(uint64_t eventId, uint64_t nodeId) {
    auto &eventInfo = eventToNodes[eventId];
    if (eventInfo.notifyNodeId != INVALID_TASK_ID) {
        SK_LOGE("Notify event already associated with node %lu", eventInfo.notifyNodeId);
        return false;
    }
    eventInfo.notifyNodeId = nodeId;
    return true;
}

bool SuperKernelGraph::AddEventAssociateWait(uint64_t eventId, uint64_t nodeId) {
    auto &eventInfo = eventToNodes[eventId];
    if (eventInfo.waitNodeIdList.find(nodeId) != eventInfo.waitNodeIdList.end()) {
        SK_LOGE("Wait event already associated with node %lu", nodeId);
        return false;
    }
    eventInfo.waitNodeIdList.insert(nodeId);

    return true;
}

bool SuperKernelGraph::AddEventAssociateReset(uint64_t eventId, uint64_t nodeId) {
    auto &eventInfo = eventToNodes[eventId];

    if (eventInfo.resetNodeId != INVALID_TASK_ID) {
        SK_LOGE("Reset event already associated with node %lu", eventInfo.resetNodeId);
        return false;
    }
    eventInfo.resetNodeId = nodeId;

    return true;
}

void SuperKernelGraph::BuildWaitNodeAssociations() {
    for (const auto& it : eventToNodes) {
        const uint64_t eventId = it.first;
        const EventInfos& eventInfo = it.second;
        if (eventInfo.notifyNodeId != INVALID_TASK_ID && !eventInfo.waitNodeIdList.empty()) {
            auto* notifyNode = GetNodeById(eventInfo.notifyNodeId);
            if (notifyNode != nullptr &&
                (notifyNode->GetNodeType() == SkNodeType::NODE_NOTIFY)) {
                std::vector<uint64_t> waitNodeIds(
                    eventInfo.waitNodeIdList.begin(),
                    eventInfo.waitNodeIdList.end());
                notifyNode->SetCorrespondingWaitNodeIds(waitNodeIds);

                // Set corresponding notify node ID for each wait node
                for (uint64_t waitNodeId : waitNodeIds) {
                    auto* waitNode = GetNodeById(waitNodeId);
                    if (waitNode != nullptr &&
                        (waitNode->GetNodeType() == SkNodeType::NODE_WAIT)) {
                        waitNode->SetCorrespondingNotifyNodeId(eventInfo.notifyNodeId);
                    }
                }

                SK_LOGI("Built wait node associations for notify node %lu with %zu wait nodes",
                         eventInfo.notifyNodeId, eventInfo.waitNodeIdList.size());
            }
        }
    }
}

namespace {

// Scope stack entry for tracking nested scope contexts
struct ScopeStackEntry {
    uint32_t scopeIdx = INVALID_SCOPE_ID;
    std::string scopeName;
    bool isFusible = true;
};

// Compute current scope bit flags from the scope stack
// Only marks fusible scopes to determine which fusible scopes a node belongs to
std::bitset<MAX_SCOPE_NUM> ComputeScopeBitFlags(const std::vector<ScopeStackEntry>& scopeStack) {
    std::bitset<MAX_SCOPE_NUM> flags;
    for (const auto& entry : scopeStack) {
        if (entry.isFusible) {
            flags.set(entry.scopeIdx);
        }
    }
    return flags;
}

// Check if current scope stack contains any unfusible scope
// If a node is inside an unfusible scope, the node is also unfusible
bool HasUnfusibleScope(const std::vector<ScopeStackEntry>& scopeStack) {
    for (const auto& entry : scopeStack) {
        if (!entry.isFusible) {
            return true;
        }
    }
    return false;
}

// Parse scope node information to extract scope index, name, and fusible attribute
// Get scope index for a scope node
// For fusible scopes, look up index from scopeNameToIdx; for unfusible scopes, use MAX_SCOPE_NUM
uint32_t GetScopeIdx(SuperKernelBaseNode* node,
                    const std::unordered_map<std::string, uint32_t>& scopeNameToIdx) {
    bool isFusible = node->IsFusible();

    if (isFusible) {
        const std::string& scopeName = node->GetScopeName();
        auto it = scopeNameToIdx.find(scopeName);
        if (it == scopeNameToIdx.end()) {
            SK_LOGW("Fusible scope name '%s' not registered for node %lu",
                    scopeName.c_str(), node->GetNodeId());
            return MAX_SCOPE_NUM;
        }
        return it->second;
    }

    // Unfusible scopes use MAX_SCOPE_NUM to avoid conflict with fusible scope indices [0, MAX_SCOPE_NUM-1]
    // The actual index value is not used since unfusible scopes are excluded from scope bit flags
    return MAX_SCOPE_NUM;
}

// Pop the scope with matching name from the stack
// Searches from top to bottom to find the matching scope and removes only that scope
// This supports interleaved scope begin/end patterns (e.g., A-B-A-B)
bool PopScopeByName(std::vector<ScopeStackEntry>& scopeStack, const std::string& scopeName) {
    for (int i = static_cast<int>(scopeStack.size()) - 1; i >= 0; --i) {
        if (scopeStack[i].scopeName == scopeName) {
            scopeStack.erase(scopeStack.begin() + i);
            SK_LOGI("Popped scope '%s' at position %d, stack_size=%zu",
                    scopeName.c_str(), i, scopeStack.size());
            return true;
        }
    }
    SK_LOGW("Scope end without matching begin: name='%s'", scopeName.c_str());
    return false;
}

// Process scope begin node: parse scope info, push to stack, and mark associated event nodes
void ProcessScopeBegin(SuperKernelGraph* graph,
                      SuperKernelBaseNode* node,
                      std::vector<ScopeStackEntry>& scopeStack,
                      const std::unordered_map<std::string, uint32_t>& scopeNameToIdx) {
    std::string scopeName = node->GetScopeName();
    bool isFusible = node->IsFusible();
    uint32_t scopeIdx = GetScopeIdx(node, scopeNameToIdx);

    scopeStack.push_back({scopeIdx, scopeName, isFusible});
    SK_LOGI("Scope begin: name='%s' idx=%u fusible=%d stack_size=%zu",
            scopeName.c_str(), scopeIdx, isFusible, scopeStack.size());
}

// Process scope end node: parse scope info, pop from stack, and mark associated event nodes
void ProcessScopeEnd(SuperKernelGraph* graph,
                    SuperKernelBaseNode* node,
                    std::vector<ScopeStackEntry>& scopeStack,
                    const std::unordered_map<std::string, uint32_t>& scopeNameToIdx) {
    std::string scopeName = node->GetScopeName();
    uint32_t scopeIdx = GetScopeIdx(node, scopeNameToIdx);

    SK_LOGI("Scope end: name='%s' idx=%u", scopeName.c_str(), scopeIdx);
    PopScopeByName(scopeStack, scopeName);
}

// Log warning if there are unclosed scopes remaining in the stack at the end of graph processing
void LogUnclosedScopes(const std::vector<ScopeStackEntry>& scopeStack) {
    if (!scopeStack.empty()) {
        SK_LOGW("Found %zu unclosed scope(s) at end of graph:", scopeStack.size());
        for (const auto& entry : scopeStack) {
            SK_LOGW("  - Scope '%s' (idx=%u, fusible=%d)", entry.scopeName.c_str(), entry.scopeIdx, entry.isFusible);
        }
    }
}

} // namespace

// Update scope bit flags for all nodes based on scope contexts
//
// The algorithm processes nodes in sorted order and maintains a scope stack to track nested scopes:
// 1. When encountering a scope begin node, push scope info to stack
// 2. For scope end nodes, compute flags BEFORE popping (node belongs to the scope it closes), then pop
// 3. For all other nodes (scope begin, notify, wait, regular nodes), compute flags from current stack
//
// Scope bit flags indicate which fusible scopes a node belongs to, enabling scope-based fusion decisions
//
// Node types:
// - Scope begin kernel node: marks start of a scope, also belongs to its parent scopes
// - Scope end kernel node: marks end of a scope, belongs to its parent scopes
// - Notify/Wait nodes: scope-related synchronization nodes, not scope begin/end
// - Regular kernel nodes: normal computation nodes
//
// Note: Scopes with the same name follow stack semantics (LIFO), allowing nested and sequential scopes
void SuperKernelGraph::UpdateNodeScopeBitFlags() {
    std::vector<ScopeStackEntry> scopeStack;
    std::vector<uint64_t> orderedNodeIds = GetSortedNodeIds();

    SK_LOGI("Starting UpdateNodeScopeBitFlags, total nodes: %zu", orderedNodeIds.size());

    for (uint64_t nodeId : orderedNodeIds) {
        SuperKernelBaseNode* node = GetNodeById(nodeId);
        if (node == nullptr) {
            SK_LOGE("Node with id %lu not found", nodeId);
            continue;
        }

        if (node->IsScopeBegin()) {
            ProcessScopeBegin(this, node, scopeStack, scopeNameToIdx);
        } else if (node->IsScopeEnd()) {
            // Scope end nodes belong to their parent scopes, compute flags before popping
            std::bitset<MAX_SCOPE_NUM> currentScopeFlags = ComputeScopeBitFlags(scopeStack);
            node->SetScopeBitFlags(currentScopeFlags);
            SK_LOGD("Set scope flags for scope end node %lu, flags=%s", nodeId,
                    currentScopeFlags.to_string().substr(0, MAX_SCOPE_NUM).c_str());
            ProcessScopeEnd(this, node, scopeStack, scopeNameToIdx);
        }

        // Update flags for all nodes except scope end nodes (already handled above)
        // This includes: scope begin nodes, notify nodes, wait nodes, and regular kernel nodes
        if (!node->IsScopeEnd()) {
            std::bitset<MAX_SCOPE_NUM> flags = ComputeScopeBitFlags(scopeStack);
            node->SetScopeBitFlags(flags);

            // Log flags for regular nodes (non-scope nodes)
            if (!node->IsScopeNode()) {
                SK_LOGD("Set scope flags for node %lu, flags=%s, fusible=%d",
                        nodeId, flags.to_string().substr(0, MAX_SCOPE_NUM).c_str(), node->IsFusible());
            }

            // Mark regular nodes as unfusible if inside any unfusible scope
            if (!node->IsScopeNode() && HasUnfusibleScope(scopeStack)) {
                node->SetIsFusible(false);
                SK_LOGD("Marked node %lu as unfusible (inside unfusible scope)", nodeId);
            }
        }
    }
    LogUnclosedScopes(scopeStack);
    SK_LOGI("UpdateNodeScopeBitFlags completed");
}

std::unique_ptr<SuperKernelBaseNode> SuperKernelNodeFactory::CreateNode(std::unique_ptr<aclmdlRITask> task, aclmdlRITaskType taskType, uint64_t nodeIdx, uint64_t streamId, uint64_t preNodeId) {
    switch (taskType) {
        case ACL_MODEL_RI_TASK_KERNEL:
            return std::make_unique<SuperKernelKernelNode>(std::move(task), taskType, nodeIdx, streamId, preNodeId);
        case ACL_MODEL_RI_TASK_EVENT_RECORD:
        case ACL_MODEL_RI_TASK_EVENT_WAIT:
        case ACL_MODEL_RI_TASK_EVENT_RESET:
        case ACL_MODEL_RI_TASK_VALUE_WRITE:
        case ACL_MODEL_RI_TASK_VALUE_WAIT:
            return std::make_unique<SuperKernelMemoryNode>(std::move(task), taskType, nodeIdx, streamId, preNodeId);
        default:
            return std::make_unique<SuperKernelDefaultNode>(std::move(task), taskType, nodeIdx, streamId, preNodeId);
    }
}

bool SuperKernelGraph::InitSKGraph() {
    uint32_t streamNum = 0;
    aclError ret = aclmdlRIGetStreams(modelRI, nullptr, &streamNum);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to get number of streams in model RI");
        return false;
    }
    streams.clear();
    streams.resize(streamNum);
    ret = aclmdlRIGetStreams(modelRI, streams.data(), &streamNum);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to get streams in model RI");
        return false;
    }

    auto tasks = std::make_unique<aclmdlRITask[]>(MAX_TASK_NUM);
    for (uint32_t streamIdx = 0; streamIdx < streamNum; ++streamIdx) {
        uint32_t taskNum = 0;
        ret = aclmdlRIGetTasksByStream(streams[streamIdx], nullptr, &taskNum);
        if (ret != ACL_SUCCESS) {
            SK_LOGE("Failed to get number of tasks in stream %u", streamIdx);
            return false;
        }
        if (taskNum > MAX_TASK_NUM) {
            tasks = std::make_unique<aclmdlRITask[]>(taskNum);
        }
        ret = aclmdlRIGetTasksByStream(streams[streamIdx], tasks.get(), &taskNum);
        if (ret != ACL_SUCCESS) {
            SK_LOGE("Failed to get tasks in stream %u", streamIdx);
            return false;
        }
        nodeSizeInStream.emplace_back(taskNum);
        SK_LOGI("Stream %u has %u tasks", streamIdx, taskNum);
        uint64_t preNodeId = INVALID_TASK_ID;
        for (uint32_t taskIdx = 0; taskIdx < taskNum; ++taskIdx) {
            aclmdlRITaskType taskType;
            ret = aclmdlRITaskGetType(tasks[taskIdx], &taskType);
            if (ret != ACL_SUCCESS) {
                SK_LOGE("Failed to get task type for task %u in stream %u", taskIdx, streamIdx);
                return false;
            }
            auto node = SuperKernelNodeFactory::CreateNode(std::make_unique<aclmdlRITask>(tasks[taskIdx]), taskType, taskIdx, streamIdx, preNodeId);
            if (!node->InitNode()) {
                SK_LOGE("Failed to initialize node for task %u in stream %u", taskIdx, streamIdx);
                return false;
            }
            if (node->GetNodeType() == SkNodeType::NODE_KERNEL && node->IsScopeNode()){
                // Register fusible scopes with scope names to scopeNameToIdx for index assignment
                // This ensures consistent scope indices across the graph for scope bit flag computation
                if (node->GetScopeName().length() > 0 && node->IsFusible()){
                    if(scopeNameToIdx.size() >= MAX_SCOPE_NUM){
                        SK_LOGW("The number of scope names is greater than the maximum allowed: %u", MAX_SCOPE_NUM);
                    } else {
                        if (scopeNameToIdx.find(node->GetScopeName()) == scopeNameToIdx.end()) {
                            uint32_t scopeIdx = static_cast<uint32_t>(scopeNameToIdx.size());
                            scopeNameToIdx[node->GetScopeName()] = scopeIdx;
                            scopeIdxToName[scopeIdx] = node->GetScopeName();
                            SK_LOGI("Registered fusible scope '%s' with index %u",
                                    node->GetScopeName().c_str(), scopeIdx);
                        }
                    }
                }
            }
            uint64_t nodeId = node->GetNodeId();
            if (!AddNode(std::move(node))) {
                SK_LOGE("Failed to add node for task %u in stream %u to graph", taskIdx, streamIdx);
                return false;
            }

            if (taskIdx == 0) {
                headNodes.push_back(nodeId);
            }
            if (preNodeId != INVALID_TASK_ID) {
                graphMap[preNodeId]->SetNextNodeId(nodeId);
            }
            preNodeId = nodeId;
        }
    }

    UpdateNodeScopeBitFlags();
    BuildWaitNodeAssociations();

    return true;
}

aclrtStream SuperKernelGraph::GetStreamByIndex(uint32_t streamIdx) const {
    if (streamIdx >= streams.size()) {
        SK_LOGE("Stream index %u out of bounds, total streams: %zu", streamIdx, streams.size());
        return nullptr;
    }
    return streams[streamIdx];
}

std::vector<uint64_t> SuperKernelGraph::GetSortedNodeIds() const {
    std::vector<uint64_t> nodeIds;
    nodeIds.reserve(graphMap.size());
    for (const auto& pair : graphMap) {
        nodeIds.push_back(pair.first);
    }
    std::sort(nodeIds.begin(), nodeIds.end());
    return nodeIds;
}
