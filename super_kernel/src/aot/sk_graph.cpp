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

void SuperKernelGraph::Update() {
    aclmdlRIUpdate(nullptr);
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

void SuperKernelGraph::MarkEventNodeToScopeBegin(SuperKernelBaseNode* node){
    uint64_t recordId = node->GetNextNodeId();
    SuperKernelBaseNode* recordNode = GetNodeById(recordId);
    if (recordNode == nullptr || recordNode->GetNodeType() != SkNodeType::NODE_NOTIFY){
        SK_LOGW("Failed to find valid notify (record) node after scope begin node %lu, recordNodeId=%lu", 
                node->GetNodeId(), recordId);
        return;
    }
    uint64_t waitId = recordNode->GetNextNodeId();
    SuperKernelBaseNode* waitNode = GetNodeById(waitId);
    if (waitNode == nullptr || waitNode->GetNodeType() != SkNodeType::NODE_WAIT){
        SK_LOGW("Failed to find valid wait node after notify node %lu, waitNodeId=%lu", 
                recordNode->GetNodeId(), waitId);
        return;
    }
    recordNode->SetNodeToScope(true);
    waitNode->SetNodeToScope(true);
}

void SuperKernelGraph::MarkEventNodeToScopeEnd(SuperKernelBaseNode* node){
    uint64_t waitId = node->GetPreNodeId();
    SuperKernelBaseNode* waitNode = GetNodeById(waitId);
    if (waitNode == nullptr && waitNode->GetNodeType() != SkNodeType::NODE_WAIT){
        SK_LOGW("Failed to find valid wait node before scope end node %lu, waitNodeId=%lu", 
                node->GetNodeId(), waitId);
        return;
    }
    uint64_t notifyId = waitNode->GetPreNodeId();
    SuperKernelBaseNode* notifyNode = GetNodeById(notifyId);
    if (notifyNode == nullptr || notifyNode->GetNodeType() != SkNodeType::NODE_NOTIFY){
        SK_LOGW("Failed to find valid notify (record) node before wait node %lu, notifyNodeId=%lu", 
                waitNode->GetNodeId(), notifyId);
        return;
    }
    recordNode->SetNodeToScope(true);
    waitNode->SetNodeToScope(true);
}


void SuperKernelGraph::UpdateNodeScopeBitFlags(){
    std::bitset<MAX_SCOPE_NUM> curScopeBitFlags;  
    bool curIsFusibleByScope = true;
    std::vector<uint64_t> orderedNodeIds = GetSortedNodeIds();
    for (auto nodeId : orderedNodeIds){ 
        SuperKernelBaseNode* node = GetNodeById(nodeId);
        if (!node->IsScopeNode()){
            node->SetIsFusible(curIsFusibleByScope);
            node->SetScopeBitFlags(curScopeBitFlags);
        } else { // 是scope类型的kernel node
            curIsFusibleByScope = node->IsFusible(); // 当前可融合的状态
            if (curIsFusibleByScope){
                if (unique_scopeNames.size() >= MAX_SCOPE_NUM && unique_scopeNames.find(node->GetScopeName()) == unique_scopeNames.end()){ 
                    SK_LOGW("scope num is more than %d, current scope will not take effect", MAX_SCOPE_NUM);
                    continue;
                }
                uint32_t scopeIdx = unique_scopeNames[node->GetScopeName()];
                if (node->IsScopeBegin()){
                    curScopeBitFlags.set(scopeIdx);
                    MarkEventNodeToScopeBegin(node);
                } else {
                    node->SetScopeBitFlags(curScopeBitFlags);
                    curScopeBitFlags.reset(scopeIdx);
                    MarkEventNodeToScopeEnd(node);
                    continue;
                }
            } else {
                if (node->IsScopeBegin()){
                    MarkEventNodeToScopeBegin(node);
                } else {
                    MarkEventNodeToScopeEnd(node);
                    curIsFusibleByScope = true;
                    continue;
                }
            }
        }
    }
}

std::unique_ptr<SuperKernelBaseNode> SuperKernelNodeFactory::CreateNode(std::unique_ptr<aclrtTask> task, aclrtTaskType taskType, uint64_t nodeIdx, uint64_t streamId, uint64_t preNodeId) {
    switch (taskType) {
        case ACL_RT_TASK_KERNEL:
            return std::make_unique<SuperKernelKernelNode>(std::move(task), SkNodeType::NODE_KERNEL, nodeIdx, streamId, preNodeId);
        case ACL_RT_TASK_EVENT_RECORD:
            return std::make_unique<SuperKernelEventNotifyNode>(std::move(task), SkNodeType::NODE_NOTIFY, nodeIdx, streamId, preNodeId);
        case ACL_RT_TASK_EVENT_WAIT:
            return std::make_unique<SuperKernelEventWaitNode>(std::move(task), SkNodeType::NODE_WAIT, nodeIdx, streamId, preNodeId);
        case ACL_RT_TASK_VALUE_WRITE:
            return std::make_unique<SuperKernelMemoryNotifyNode>(std::move(task), SkNodeType::NODE_NOTIFY, nodeIdx, streamId, preNodeId);
        case ACL_RT_TASK_VALUE_WAIT:
            return std::make_unique<SuperKernelMemoryWaitNode>(std::move(task), SkNodeType::NODE_WAIT, nodeIdx, streamId, preNodeId);
        default:
            return std::make_unique<SuperKernelDefaultNode>(std::move(task), SkNodeType::NODE_DEFAULT, nodeIdx, streamId, preNodeId);
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

    auto tasks = std::make_unique<aclrtTask[]>(MAX_TASK_NUM);
    for (uint32_t streamIdx = 0; streamIdx < streamNum; ++streamIdx) {
        uint32_t taskNum = 0;
        ret = aclrtStreamGetTasks(streams[streamIdx], nullptr, &taskNum);
        if (ret != ACL_SUCCESS) {
            SK_LOGE("Failed to get number of tasks in stream %u", streamIdx);
            return false;
        }
        if (taskNum > MAX_TASK_NUM) {
            tasks = std::make_unique<aclrtTask[]>(taskNum);
        }
        ret = aclrtStreamGetTasks(streams[streamIdx], tasks.get(), &taskNum);
        if (ret != ACL_SUCCESS) {
            SK_LOGE("Failed to get tasks in stream %u", streamIdx);
            return false;
        }
        nodeSizeInStream.emplace_back(taskNum);
        SK_LOGI("Stream %u has %u tasks", streamIdx, taskNum);
        uint64_t preNodeId = INVALID_TASK_ID;
        for (uint32_t taskIdx = 0; taskIdx < taskNum; ++taskIdx) {
            aclrtTaskType taskType;
            ret = aclrtTaskGetType(tasks[taskIdx], &taskType);
            if (ret != ACL_SUCCESS) {
                SK_LOGE("Failed to get task type for task %u in stream %u", taskIdx, streamIdx);
                return false;
            }
            auto node = SuperKernelNodeFactory::CreateNode(std::make_unique<aclrtTask>(tasks[taskIdx]), taskType, taskIdx, streamIdx, preNodeId);
            if (!node->InitNode()) {
                SK_LOGE("Failed to initialize node for task %u in stream %u", taskIdx, streamIdx);
                return false;
            }
            if (node->GetNodeType() == SkNodeType::NODE_KERNEL && node->GetScopeName().length() > 0){
                node->SetNodeToScope(true);
                if(unique_scopeNames.size() >= MAX_SCOPE_NUM){
                    SK_LOGW("The number of scope names is greater than the maximum allowed: %u", MAX_SCOPE_NUM);
                } else {
                    if (unique_scopeNames.find(node->GetScopeName()) == unique_scopeNames.end()) {
                        unique_scopeNames[node->GetScopeName()] = unique_scopeNames.size();
                    }
                }
            }
            uint64_t nodeId = AllocateNodeId(); // todo need init by aclrt
            node->SetNodeId(nodeId);
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

    
    // update scopeBitFlags
    if (unique_scopeNames.empty()){
        splitByScope = true;
    }
    UpdateNodeScopeBitFlags();
    // Build wait node associations for notify nodes
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
