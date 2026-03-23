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

#include <optional>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <bitset>

aclError SuperKernelGraph::Update() {
    for(auto node : needUpdateNodes) {
        if (node == nullptr) {
            SK_LOGE("Null node found in needUpdateNodes");
            return ACL_ERROR_FAILURE;
        }
        if (!node->IsUpdated()) {
            aclmdlRITaskParams customParams = {};
            auto addrValue = node->nodeInfos.syncInfos.addrValue;
            switch (node->GetNodeType()) {
                case SkNodeType::NODE_NOTIFY: {
                    customParams.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
                    customParams.valueWriteTaskParams.value = 1;
                    customParams.valueWriteTaskParams.devAddr = addrValue;
                    break;
                }
                case SkNodeType::NODE_WAIT: {
                    customParams.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
                    customParams.valueWaitTaskParams.value = 1;
                    customParams.valueWaitTaskParams.flag = 1;
                    customParams.valueWaitTaskParams.devAddr = addrValue;
                    break;
                }
                case SkNodeType::NODE_RESET: {
                    customParams.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
                    customParams.valueWriteTaskParams.value = 0;
                    customParams.valueWriteTaskParams.devAddr = addrValue;
                    break;
                }
                default:
                    SK_LOGE("Unsupported node type for event update: %u", static_cast<uint32_t>(node->GetNodeType()));
                    return ACL_ERROR_FAILURE;
            }
            UpdateContext ctx;
            ctx.customParams = &customParams;
            if(!node->Update(ctx)){
                SK_LOGE("Failed to update event node %lu in stream %u", node->GetNodeId(), node->GetStreamIdxInGraph());
                return ACL_ERROR_FAILURE;
            }
        }
    }
    aclError ret = aclmdlRIUpdate(modelRI);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to update modelRI");
    }
    // Clear shape info memory after update completes
    ClearShapeInfoPtrList();
    return ret;
}

bool SuperKernelGraph::ExpandUpdateNodes(std::vector<SuperKernelBaseNode*>& customNodes) {
    for (auto* node : customNodes) {
        if (node != nullptr && needUpdateNodes.find(node) == needUpdateNodes.end()) {
            needUpdateNodes.insert(node);
        }
    }
    return true;
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
            if (!AddEventAssociateNotify(eventId, node.get())) {
                SK_LOGE("Failed to associate notify event %lu with node %lu", eventId, nodeId);
                return false;
            }
            break;
        case SkNodeType::NODE_WAIT:
            if (!AddEventAssociateWait(eventId, node.get())) {
                SK_LOGE("Failed to associate wait event %lu with node %lu", eventId, nodeId);
                return false;
            }
            break;
        case SkNodeType::NODE_RESET:
            if (!AddEventAssociateReset(eventId, node.get())) {
                SK_LOGE("Failed to associate reset event %lu with node %lu", eventId, nodeId);
                return false;
            }
            break;
        case SkNodeType::NODE_MEMORY_WRITE:
            if (!AddMemoryAssociateWrite(eventId, node.get())) {
                SK_LOGE("Failed to associate memory write event %lu with node %lu", eventId, nodeId);
                return false;
            }
            break;
        case SkNodeType::NODE_MEMORY_WAIT:
            if (!AddMemoryAssociateWait(eventId, node.get())) {
                SK_LOGE("Failed to associate memory wait event %lu with node %lu", eventId, nodeId);
                return false;
            }
            break;
        default:
            break;
    }

    if (graphMap.find(nodeId) != graphMap.end()) {
        SuperKernelBaseNode* existingNode = graphMap[nodeId].get();
        SK_LOGE("Duplicate node ID detected! Node with id %lu already exists in graph", nodeId);
        SK_LOGE("  Existing node: %s", existingNode->FormatNodeInfo().c_str());
        SK_LOGE("  New node to add: %s", node->FormatNodeInfo().c_str());
        SK_LOGE("  Duplicate nodeId=%lu, Please check for duplicate node assignments in stream %lu",
                 nodeId, node->GetStreamIdxInGraph());
        return false;
    }
    graphMap[nodeId] = std::move(node);
    return true;
}

bool SuperKernelGraph::AddMemoryAssociateWrite(uint64_t eventId, SuperKernelBaseNode* node) {
    auto &memoryInfo = memoryToNodes[eventId];
    uint64_t nodeId = node->GetNodeId();
    if (memoryInfo.writeNodeIdList.find(nodeId) != memoryInfo.waitNodeIdList.end()) {
        // Get the node information for duplicate memory write node binding
        SuperKernelBaseNode* existingNode = GetNodeById(nodeId);
        SK_LOGE("memory event 0x%lx already associated with this node, cannot reassociate!", eventId);
        SK_LOGE("  Duplicate memory write node: node_id=%lu, details=%s",
                 nodeId,
                 existingNode ? existingNode->FormatNodeInfo().c_str() : "NOT_FOUND");
        SK_LOGE("  Please check for duplicate memory write node bindings in the graph.");
        return false;
    }
    memoryInfo.writeNodeIdList.insert(nodeId);
    return true;
}

bool SuperKernelGraph::AddMemoryAssociateWait(uint64_t eventId, SuperKernelBaseNode* node) {
    auto &memoryInfo = memoryToNodes[eventId];
    uint64_t nodeId = node->GetNodeId();
    if (memoryInfo.waitNodeIdList.find(nodeId) != memoryInfo.waitNodeIdList.end()) {
        // Get the node information for duplicate memory wait node binding
        SuperKernelBaseNode* existingNode = GetNodeById(nodeId);
        SK_LOGE("memory event 0x%lx already associated with this node, cannot reassociate!", eventId);
        SK_LOGE("  Duplicate memory wait node: node_id=%lu, details=%s",
                 nodeId,
                 existingNode ? existingNode->FormatNodeInfo().c_str() : "NOT_FOUND");
        SK_LOGE("  Please check for duplicate memory wait node bindings in the graph.");
        return false;
    }
    memoryInfo.waitNodeIdList.insert(nodeId);
    return true;
}

bool SuperKernelGraph::AddEventAssociateNotify(uint64_t eventId, SuperKernelBaseNode* node) {
    auto &eventInfo = eventToNodes[eventId];
    if (eventInfo.notifyNodeId != INVALID_TASK_ID) {
        // Get the already bound node information
        SuperKernelBaseNode* existingNode = GetNodeById(eventInfo.notifyNodeId);
        SK_LOGE("Notify event 0x%lx already associated, cannot reassociate!", eventId);
        SK_LOGE("  Existing bound node: node_id=%lu, details=%s",
                 eventInfo.notifyNodeId,
                 existingNode ? existingNode->FormatNodeInfo().c_str() : "NOT_FOUND");
        SK_LOGE("  Attempting to bind: node_id=%lu, details=%s",
                 node->GetNodeId(),
                 node->FormatNodeInfo().c_str());
        SK_LOGE("  Please check for duplicate event bindings in the graph.");
        return false;
    }
    eventInfo.notifyNodeId = node->GetNodeId();
    return true;
}

bool SuperKernelGraph::AddEventAssociateWait(uint64_t eventId, SuperKernelBaseNode* node) {
    auto &eventInfo = eventToNodes[eventId];
    uint64_t nodeId = node->GetNodeId();
    if (eventInfo.waitNodeIdList.find(nodeId) != eventInfo.waitNodeIdList.end()) {
        // Get the node information for duplicate WAIT binding
        SuperKernelBaseNode* existingNode = GetNodeById(nodeId);

        SK_LOGE("Wait event 0x%lx already associated with this node, cannot reassociate!", eventId);
        SK_LOGE("  Duplicate WAIT node: node_id=%lu, details=%s",
                 nodeId,
                 existingNode ? existingNode->FormatNodeInfo().c_str() : "NOT_FOUND");
        SK_LOGE("  Please check for duplicate WAIT event bindings in the graph.");
        return false;
    }
    eventInfo.waitNodeIdList.insert(nodeId);

    return true;
}

bool SuperKernelGraph::AddEventAssociateReset(uint64_t eventId, SuperKernelBaseNode* node) {
    auto &eventInfo = eventToNodes[eventId];
    uint64_t nodeId = node->GetNodeId();
    if (eventInfo.resetNodeIdList.find(nodeId) != eventInfo.resetNodeIdList.end()) {
        SK_LOGE("Reset event 0x%lx already associated, cannot reassociate!", eventId);
        SK_LOGE("  Existing bound node: node_id=%lu, details=%s",
                 nodeId, node ? node->FormatNodeInfo().c_str() : "NOT_FOUND");
        SK_LOGE("  Attempting to bind: node_id=%lu, details=%s",
                nodeId, node->FormatNodeInfo().c_str());
        SK_LOGE("  Please check for duplicate RESET event bindings in the graph.");
        return false;
    }
    eventInfo.resetNodeIdList.insert(nodeId);
    return true;
}

/**
 * @brief Build event associations and establish send-receive relationships between nodes
 *
 * This function traverses all event information to establish direct send-receive relationships
 * between NOTIFY nodes and WAIT nodes associated with each event:
 * 1. For each event, find the nearest KERNEL node before the NOTIFY node as the sender node
 * 2. For each WAIT node, find the nearest KERNEL node after it as the receiver node
 * 3. Establish bidirectional association between sender and receiver nodes
 *
 * @return true Association establishment successful
 * @return false Association establishment failed (current implementation always returns true)
 *
 * @note This function is mainly used to build the task dependency graph within super kernel,
 *       facilitating subsequent graph optimization and execution
 */
bool SuperKernelGraph::AddEventAssociate() {
    SK_LOGI("Start to build event associations, total events: %zu", eventToNodes.size());

    uint32_t totalProcessed = 0;
    uint32_t skipDueToNoNotify = 0;
    uint32_t skipDueToNoSendNode = 0;
    uint32_t skipDueToNoWaitNode = 0;
    uint32_t successAssociations = 0;

    // Traverse all event information
    for (auto iter = eventToNodes.begin(); iter != eventToNodes.end(); iter++) {
        uint64_t eventId = iter->first;                     // Event ID
        uint64_t notifyId = iter->second.notifyNodeId;      // NOTIFY node ID
        std::unordered_set<uint64_t> waitIdSet = iter->second.waitNodeIdList;  // WAIT node ID set

        SK_LOGD("Processing event 0x%lx: notify=%lu, wait_count=%zu",
                eventId, notifyId, waitIdSet.size());

        // Skip if the event has no associated NOTIFY node
        if (INVALID_TASK_ID == notifyId) {
            SK_LOGD("Event 0x%lx has no notify node, skip", eventId);
            skipDueToNoNotify++;
            continue;
        }

        // Search backward from NOTIFY node to find the first KERNEL node as sender node
        // This ensures the sender is the actual compute kernel rather than event notification node
        auto nodeId = graphMap[notifyId]->GetPreNodeId();
        while (nodeId != INVALID_TASK_ID && graphMap[nodeId]->GetNodeType() != SkNodeType::NODE_KERNEL) {
            nodeId = graphMap[nodeId]->GetPreNodeId();
        }

        // Skip this event if no KERNEL node found before NOTIFY node
        if (nodeId == INVALID_TASK_ID) {
            SK_LOGW("Event 0x%lx: cannot find KERNEL node before notify node %lu, skip",
                    eventId, notifyId);
            skipDueToNoSendNode++;
            continue;
        }

        auto sendNode = graphMap[nodeId].get();
        if (sendNode == nullptr) {
            SK_LOGE("Failed to cast node %lu to SuperKernelSendNode", nodeId);
            continue;
        }

        SK_LOGD("Event 0x%lx: send node found (id=%lu), starting to process %zu wait nodes",
                eventId, nodeId, waitIdSet.size());

        // Traverse all WAIT nodes associated with this event
        for (auto waitId : waitIdSet) {
            // Search forward from WAIT node to find the first KERNEL node as receiver node
            // This ensures the receiver is the actual compute kernel that needs synchronization
            while (waitId != INVALID_TASK_ID && graphMap[waitId]->GetNodeType() != SkNodeType::NODE_KERNEL) {
                waitId = graphMap[waitId]->GetNextNodeId();
            }

            // Skip this WAIT node if no KERNEL node found after it
            if (waitId == INVALID_TASK_ID) {
                SK_LOGW("Event 0x%lx: cannot find KERNEL node after wait node, skip one wait association",
                        eventId);
                skipDueToNoWaitNode++;
                continue;
            }

            auto recvNode = graphMap[waitId].get();
            if (recvNode == nullptr) {
                SK_LOGE("Failed to cast node %lu to SuperKernelSendNode", waitId);
                continue;
            }

            // Establish bidirectional association:
            // 1. Receiver node records which sender nodes it needs to wait for
            // 2. Sender node records which receiver nodes it needs to notify
            recvNode->receiveNodeId.insert(nodeId);
            sendNode->sendToNodeId.insert(waitId);

            SK_LOGD("Event 0x%lx: built association - send(node %lu) -> wait(node %lu)",
                    eventId, nodeId, waitId);
            successAssociations++;
        }

        totalProcessed++;
    }

    SK_LOGI("Event association build completed: processed=%u, success=%u, "
            "skip(no_notify)=%u, skip(no_send)=%u, skip(no_wait)=%u",
            totalProcessed, successAssociations, skipDueToNoNotify,
            skipDueToNoSendNode, skipDueToNoWaitNode);

    return true;
}

void SuperKernelGraph::BuildEventNodeAssociations() {
    SK_LOGI("Starting to build wait node associations, total events: %zu", eventToNodes.size());
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

                SK_LOGI("Processing event 0x%lx: notify node %lu has %zu wait nodes",
                        eventId, eventInfo.notifyNodeId, waitNodeIds.size());

                // Set corresponding notify node ID for each wait node
                for (uint64_t waitNodeId : waitNodeIds) {
                    auto* waitNode = GetNodeById(waitNodeId);
                    if (waitNode != nullptr &&
                        (waitNode->GetNodeType() == SkNodeType::NODE_WAIT)) {
                        waitNode->SetCorrespondingNotifyNodeId(eventInfo.notifyNodeId);
                        SK_LOGD("Associated wait node %lu with notify node %lu",
                                waitNodeId, eventInfo.notifyNodeId);
                    }
                }

                SK_LOGI("Built wait node associations for notify node %lu with %zu wait nodes",
                         eventInfo.notifyNodeId, eventInfo.waitNodeIdList.size());
            } else {
                SK_LOGE("Event 0x%lx: notify node %lu is invalid or not a notify node",
                        eventId, eventInfo.notifyNodeId);
            }
        }
        if (!eventInfo.resetNodeIdList.empty()) {
            for (auto resetNodeId : eventInfo.resetNodeIdList) {
                auto* resetNode = GetNodeById(resetNodeId);
                if (resetNode != nullptr &&
                    (resetNode->GetNodeType() == SkNodeType::NODE_RESET)) {
                    resetNode->nodeInfos.syncInfos.correspondingResetNodeIds.assign(
                        eventInfo.resetNodeIdList.begin(), eventInfo.resetNodeIdList.end());
                    SK_LOGD("Associated reset node %lu has %lu reset nodes",
                        resetNodeId, eventInfo.resetNodeIdList.size());
                }
            }
        }
    }
    SK_LOGI("Completed building all event node associations");
}

bool SuperKernelGraph::ProcessMemoryWriteNodes(const uint64_t eventId, const MemoryInfos& memoryInfo,
    const uint64_t memoryWaitValue, const uint32_t waitFlag)
{
    std::vector<uint64_t> notifyIdVec;
    std::vector<uint64_t> resetIdVec;

    // Iterate through all memory write nodes and classify them as Notify or Reset nodes based on waitFlag
    // waitFlag definitions:
    //   0x0: >= comparison mode
    //   0x1: == equality mode
    //   0x2: & bitwise AND mode
    //   0x3: ~| bitwise OR with NOT mode
    for (const auto writeNodeId : memoryInfo.writeNodeIdList) {
        auto* writeNode = GetNodeById(writeNodeId);
        // Only process valid MEMORY_WRITE type nodes
        if (writeNode == nullptr || writeNode->GetNodeType() != SkNodeType::NODE_MEMORY_WRITE) {
            continue;
        }
        const uint64_t writeMemoryValue = writeNode->nodeInfos.syncInfos.memoryValue;
        bool isNotify = false;
        // Calculate whether notification condition is satisfied based on different waitFlag values
        switch (waitFlag) {
            case 0x0:
                // Greater than or equal mode: trigger notification when written value >= wait value
                isNotify = (writeMemoryValue >= memoryWaitValue);
                break;
            case 0x1:
                // Equal mode: trigger notification when written value == wait value
                isNotify = (writeMemoryValue == memoryWaitValue);
                break;
            case 0x2:
                // Bitwise AND mode: trigger notification when (written value & wait value) != 0
                isNotify = ((writeMemoryValue & memoryWaitValue) != 0);
                break;
            case 0x3:
                // Bitwise OR with NOT mode: trigger notification when ~(written value | wait value) != 0
                isNotify = ((~(writeMemoryValue | memoryWaitValue)) != 0);
                break;
            default:
                // Undefined waitFlag, do not trigger notification by default
                isNotify = false;
                SK_LOGW("Unknown waitFlag %u for event %lu, treated as reset node",
                    waitFlag, eventId);
                break;
        }

        // Classify nodes into corresponding vectors based on judgment result
        if (isNotify) {
            notifyIdVec.push_back(writeNodeId);
        } else {
            resetIdVec.push_back(writeNodeId);
        }
    }
    if (notifyIdVec.size() > 1) {
        SK_LOGE("there exits multi memory write node which is notify, it is illegal, eventId: %lu",
            eventId);
        return false;
    } else if (notifyIdVec.size() == 1) {
        auto* writeNode = GetNodeById(notifyIdVec[0]);
        SK_LOGD("there exits only one memory write node which is notify, it may cause dead lock, details=%s",
            writeNode->FormatNodeInfo().c_str());
        writeNode->SetNodeType(SkNodeType::NODE_NOTIFY);
        writeNode->SetIsFusible(true);
        if (!AddEventAssociateNotify(eventId, writeNode)) {
            SK_LOGE("Failed to associate notify event %lu with node %lu", eventId, notifyIdVec[0]);
            return false;
        }
        for (auto waitNodeId : memoryInfo.waitNodeIdList) {
            auto* waitNode = GetNodeById(waitNodeId);
            if (waitNode != nullptr &&
                (waitNode->GetNodeType() == SkNodeType::NODE_MEMORY_WAIT)) {
                waitNode->SetNodeType(SkNodeType::NODE_WAIT);
                waitNode->SetIsFusible(true);
                if (!AddEventAssociateWait(eventId, waitNode)) {
                    SK_LOGE("Failed to associate wait event %lu with node %lu", eventId, waitNodeId);
                    return false;
                }
            }
        }
    }
    if (!resetIdVec.empty()) {
        for (auto resetId: resetIdVec) {
            auto* resetNode = GetNodeById(resetId);
            resetNode->SetNodeType(SkNodeType::NODE_RESET);
            resetNode->SetIsFusible(false);
            if (!AddEventAssociateReset(eventId, resetNode)) {
                SK_LOGE("Failed to associate reset event %lu with node %lu", eventId, resetIdVec[0]);
                return false;
            }
        }
    }
    return true;
}

bool SuperKernelGraph::PostProcessMemoryNode() {
    SK_LOGI("Starting to build memory node associations, total events: %zu", memoryToNodes.size());
    for (const auto& it : memoryToNodes) {
        const uint64_t eventId = it.first;
        const MemoryInfos& memoryInfo = it.second;
        if (memoryInfo.writeNodeIdList.empty() && !memoryInfo.waitNodeIdList.empty()) {
            // No memory write exists, meaning the memory write is outside modelRI,
            // therefore change all waits to event semantics, but they cannot be fused.
            for (auto waitNodeId : memoryInfo.waitNodeIdList) {
                auto* waitNode = GetNodeById(waitNodeId);
                if (waitNode != nullptr &&
                    (waitNode->GetNodeType() == SkNodeType::NODE_MEMORY_WAIT)) {
                    waitNode->SetNodeType(SkNodeType::NODE_WAIT);
                    waitNode->SetIsFusible(false);
                }
            }
        } else if (!memoryInfo.writeNodeIdList.empty() && memoryInfo.waitNodeIdList.empty()) {
            // only exists memory write nodes, mask it as unfusible
            for (auto writeNodeId : memoryInfo.writeNodeIdList) {
                auto* writeNode = GetNodeById(writeNodeId);
                if (writeNode != nullptr &&
                    (writeNode->GetNodeType() == SkNodeType::NODE_MEMORY_WRITE)) {
                    writeNode->SetIsFusible(false);
                }
            }
        } else {
            // Check consistency of memoryValue and flag in waitNode list
            std::optional<std::pair<uint64_t, uint32_t>> firstWaitInfo;

            for (auto waitNodeId : memoryInfo.waitNodeIdList) {
                auto* waitNode = GetNodeById(waitNodeId);
                if (waitNode == nullptr || 
                    waitNode->GetNodeType() != SkNodeType::NODE_MEMORY_WAIT) {
                    continue;
                }

                const uint64_t memoryValue = waitNode->nodeInfos.syncInfos.memoryValue;
                const uint32_t flag = waitNode->nodeInfos.syncInfos.flag;

                if (!firstWaitInfo.has_value()) {
                    firstWaitInfo = {memoryValue, flag};
                } else if (memoryValue != firstWaitInfo->first || flag != firstWaitInfo->second) {
                    // Inconsistent wait parameters found, log all waitNode details
                    SK_LOGE("waitNode list contains inconsistent memoryValue/flag combinations, eventId: %lu",
                        eventId);
                    for (auto waitNodeId : memoryInfo.waitNodeIdList) {
                        auto* waitNode = GetNodeById(waitNodeId);
                        if (waitNode != nullptr && (waitNode->GetNodeType() == SkNodeType::NODE_MEMORY_WAIT)) {
                            SK_LOGE("eventId: %lu, waitNodeId: %lu, memoryValue: %lu, flag: %u",
                            eventId,
                            waitNodeId,
                            waitNode->nodeInfos.syncInfos.memoryValue,
                            waitNode->nodeInfos.syncInfos.flag);
                        }
                    }
                    return false;
                }
            }

            uint64_t memoryWaitValue = firstWaitInfo->first;
            uint32_t waitFlag = firstWaitInfo->second;

            bool ret = ProcessMemoryWriteNodes(eventId, memoryInfo, memoryWaitValue, waitFlag);
            if (!ret) {
                return ret;
            }
        }
    }
    SK_LOGI("Completed building all memory node associations");
    return true;
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
            SK_LOGE("Fusible scope name '%s' not registered for node %lu (node marked fusible but scope not in registry), graph structure error",
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
    SK_LOGE("Scope end without matching begin: name='%s', graph structure error (missing corresponding scope begin)",
            scopeName.c_str());
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
        SK_LOGE("Found %zu unclosed scope(s) at end of graph: graph structure error (missing scope end nodes)",
                scopeStack.size());
        for (const auto& entry : scopeStack) {
            SK_LOGE("  - Scope '%s' (idx=%u, fusible=%d)", entry.scopeName.c_str(), entry.scopeIdx, entry.isFusible);
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
    bool outOfScopeFusible = true;
    if (!scopeNameToIdx.empty()) {
        outOfScopeFusible = false;
        SK_LOGI("Super Kernel has named scopes, out-of-scope nodes will be marked as unfusible");
    }
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
            ProcessScopeEnd(this, node, scopeStack, scopeNameToIdx);
        }

        // Update flags for all nodes except scope end nodes (already handled above)
        // This includes: scope begin nodes, notify nodes, wait nodes, and regular kernel nodes
        if (!node->IsScopeEnd()) {
            std::bitset<MAX_SCOPE_NUM> flags = ComputeScopeBitFlags(scopeStack);
            node->SetScopeBitFlags(flags);

            if (!outOfScopeFusible && scopeStack.empty()) {
                // If there are named scopes, mark nodes outside of any scope as unfusible
                node->SetIsFusible(false);
                SK_LOGI("Marked node %s as unfusible (outside of any named scope)", node->FormatNodeInfo().c_str());
            }
            // Mark regular nodes as unfusible if inside any unfusible scope
            if (!node->IsScopeNode() && HasUnfusibleScope(scopeStack)) {
                node->SetIsFusible(false);
                SK_LOGI("Marked node %s as unfusible (inside unfusible scope)", node->FormatNodeInfo().c_str());
            }
        }

        // scope nodes are always marked as fusible
        if (node->IsScopeNode()) {
            node->SetIsFusible(true);
        }
        SK_LOGI("Processed node %s: type=%d, scopeFlags=%s, isFusible=%d, stackSize=%zu",
                node->FormatNodeInfo().c_str(), static_cast<int>(node->GetNodeType()),
                node->GetScopeBitFlags().to_string().substr(0, MAX_SCOPE_NUM).c_str(),
                node->IsFusible(), scopeStack.size());
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
    SK_LOGI("Starting to initialize SuperKernel graph");
    
    uint32_t streamNum = 0;
    aclError ret = aclmdlRIGetStreams(modelRI, nullptr, &streamNum);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to get number of streams in model RI, ret=%d", ret);
        return false;
    }
    SK_LOGI("Get %u streams from model RI", streamNum);
    
    streams.clear();
    streams.resize(streamNum);
    ret = aclmdlRIGetStreams(modelRI, streams.data(), &streamNum);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to get streams in model RI, ret=%d", ret);
        return false;
    }

    auto tasks = std::make_unique<aclmdlRITask[]>(MAX_TASK_NUM);
    for (uint32_t streamIdx = 0; streamIdx < streamNum; ++streamIdx) {
        uint32_t taskNum = 0;
        ret = aclmdlRIGetTasksByStream(streams[streamIdx], nullptr, &taskNum);
        if (ret != ACL_SUCCESS) {
            SK_LOGE("Failed to get number of tasks in stream %u, ret=%d", streamIdx, ret);
            return false;
        }
        if (taskNum > MAX_TASK_NUM) {
            tasks = std::make_unique<aclmdlRITask[]>(taskNum);
            SK_LOGI("Reallocated task array to %u tasks for stream %u", taskNum, streamIdx);
        }
        ret = aclmdlRIGetTasksByStream(streams[streamIdx], tasks.get(), &taskNum);
        if (ret != ACL_SUCCESS) {
            SK_LOGE("Failed to get tasks in stream %u, ret=%d", streamIdx, ret);
            return false;
        }
        nodeSizeInStream.emplace_back(taskNum);
        SK_LOGI("Stream %u has %u tasks", streamIdx, taskNum);
        
        uint64_t preNodeId = INVALID_TASK_ID;
        for (uint32_t taskIdx = 0; taskIdx < taskNum; ++taskIdx) {
            aclmdlRITaskType taskType;
            ret = aclmdlRITaskGetType(tasks[taskIdx], &taskType);
            if (ret != ACL_SUCCESS) {
                SK_LOGE("Failed to get task type for task %u in stream %u, ret=%d", taskIdx, streamIdx, ret);
                return false;
            }
            auto node = SuperKernelNodeFactory::CreateNode(std::make_unique<aclmdlRITask>(tasks[taskIdx]), taskType, taskIdx, streamIdx, preNodeId);
            if (!node->InitNode()) {
                SK_LOGE("Failed to initialize node for task %u in stream %u (taskType=%u, nodeId=%lu)",
                         taskIdx, streamIdx, taskType, node->GetNodeId());
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
                            SK_LOGI("Registered fusible scope '%s' with index %u at stream %u task %u",
                                    node->GetScopeName().c_str(), scopeIdx, streamIdx, taskIdx);
                        }
                    }
                }
            }
            uint64_t nodeId = node->GetNodeId();
            if (!AddNode(std::move(node))) {
                SK_LOGE("Failed to add node for task %u in stream %u to graph (likely duplicate node ID %lu)",
                         taskIdx, streamIdx, nodeId);
                return false;
            }

            if (taskIdx == 0) {
                headNodes.push_back(nodeId);
                SK_LOGI("Stream %u: Added head node %lu", streamIdx, nodeId);
            }
            if (preNodeId != INVALID_TASK_ID) {
                graphMap[preNodeId]->SetNextNodeId(nodeId);
            }
            preNodeId = nodeId;
        }
    }
    
    SK_LOGI("Total nodes added: %zu, total streams: %u", graphMap.size(), streamNum);
    SK_LOGI("Starting UpdateNodeScopeBitFlags");
    UpdateNodeScopeBitFlags();
    SK_LOGI("UpdateNodeScopeBitFlags completed");

    SK_LOGI("Starting PostProcessMemoryNode");
    bool flag = PostProcessMemoryNode();
    SK_LOGI("PostProcessMemoryNode completed");
    if (!flag) {
        return flag;
    }

    SK_LOGI("Starting BuildEventNodeAssociations");
    BuildEventNodeAssociations();
    SK_LOGI("BuildEventNodeAssociations completed");

    SK_LOGI("Starting AddEventAssociate");
    AddEventAssociate();
    SK_LOGI("AddEventAssociate completed");
    
    SK_LOGI("Successfully initialized SuperKernel graph with %zu nodes and %zu streams",
            graphMap.size(), streams.size());
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
