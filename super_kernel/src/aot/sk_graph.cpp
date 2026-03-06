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
