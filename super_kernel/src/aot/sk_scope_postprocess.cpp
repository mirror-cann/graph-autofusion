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
 * \file sk_scope_postprocess.cpp
 * \brief Post-processing functions for SK scope
 */

#include "acl/acl.h"
#include "securec.h"
#include "sk_resource_manager.h"
#include "sk_scope_postprocess.h"

namespace {

SuperKernelBaseNode* AdvanceNodeWithinScope(SuperKernelGraph& graph, SuperKernelBaseNode* node, uint64_t tailNodeId,
                                            uint32_t stepCount)
{
    if (node == nullptr) {
        SK_LOGI("advance node skipped: start node is nullptr, tailNodeId=%lu, stepCount=%u", tailNodeId, stepCount);
        return nullptr;
    }

    SK_LOGI("advance node begin: startNodeId=%lu, tailNodeId=%lu, stepCount=%u", node->GetNodeId(), tailNodeId,
            stepCount);
    while (node != nullptr && stepCount > 0) {
        if (node->GetNodeId() == tailNodeId || node->GetNextNodeId() == INVALID_TASK_ID) {
            SK_LOGI("advance node stopped at boundary: currentNodeId=%lu, tailNodeId=%lu, remainStep=%u",
                    node->GetNodeId(), tailNodeId, stepCount);
            return nullptr;
        }
        node = graph.GetNodeById(node->GetNextNodeId());
        if (node == nullptr) {
            SK_LOGI("advance node failed: next node not found in graph");
            return nullptr;
        }
        --stepCount;
    }
    SK_LOGI("advance node end: resultNodeId=%lu, remainStep=%u", node->GetNodeId(), stepCount);
    return node;
}

uint64_t FindKernelNodeWithFrontReserve(SuperKernelGraph& graph, SuperKernelBaseNode* headNode, uint64_t tailNodeId,
                                        uint32_t frontReserveCount)
{
    // Search range is inclusive: after skipping reserved prefix nodes, the
    // remaining candidate window is [curNode, tailNodeId]. Therefore curNode
    // may legally be the tail node when the stream has only one remaining node.
    SK_LOGI("find kernel with reserve begin: headNodeId=%lu, tailNodeId=%lu, frontReserveCount=%u",
            headNode == nullptr ? INVALID_TASK_ID : headNode->GetNodeId(), tailNodeId, frontReserveCount);
    auto* curNode = AdvanceNodeWithinScope(graph, headNode, tailNodeId, frontReserveCount);
    while (curNode != nullptr) {
        if (curNode->GetNodeType() == SkNodeType::NODE_KERNEL) {
            SK_LOGI("find kernel with reserve success: kernelNodeId=%lu", curNode->GetNodeId());
            return curNode->GetNodeId();
        }
        if (curNode->GetNodeId() == tailNodeId || curNode->GetNextNodeId() == INVALID_TASK_ID) {
            break;
        }
        curNode = graph.GetNodeById(curNode->GetNextNodeId());
    }
    SK_LOGI("find kernel with reserve failed: headNodeId=%lu, tailNodeId=%lu, frontReserveCount=%u",
            headNode == nullptr ? INVALID_TASK_ID : headNode->GetNodeId(), tailNodeId, frontReserveCount);
    return INVALID_TASK_ID;
}

struct StreamPostPlan {
    bool needFrontWait = false;
    bool needBackBlock = false;
    uint64_t candidateNodeId = INVALID_TASK_ID; // candidate node id for sk main node
};

std::vector<SuperKernelBaseNode*> FilterCancelledTasks(const std::vector<SuperKernelBaseNode*>& tasks)
{
    // Constraints:
    // 1) Within one scope, each event has at most one NOTIFY and may have multiple WAIT nodes.
    // 2) All WAIT nodes for that event are recorded in syncInfos.correspondingWaitNodeIds.

    std::vector<SuperKernelBaseNode*> filteredTasks;
    filteredTasks.reserve(tasks.size());
    // Core invariant per eventId after step 1:
    // eventCounts[eventId] = expected_wait_count_from_notify - observed_wait_count.
    // A value of 0 means notify-side expectation matches observed waits exactly.
    std::unordered_map<uint64_t, int64_t> eventCounts;

    // step 1: accumulate expected-vs-observed WAIT balance per eventId.
    for (size_t i = 0; i < tasks.size(); i++) {
        if (tasks[i]->GetNodeType() == SkNodeType::NODE_NOTIFY) {
            eventCounts[tasks[i]->GetEventId()] += tasks[i]->GetNodeInfos().syncInfos.correspondingWaitNodeIds.size();
        } else if (tasks[i]->GetNodeType() == SkNodeType::NODE_WAIT) {
            eventCounts[tasks[i]->GetEventId()]--;
        }
    }

    // step 2:
    // WAIT cancellation condition: corresponding NOTIFY exists.
    // NOTIFY cancellation condition: all WAITs of this event are cancellable.
    for (size_t i = 0; i < tasks.size(); i++) {
        const uint64_t eventId = tasks[i]->GetEventId();

        // Notify is removable only when expected and observed waits are fully matched.
        if (tasks[i]->GetNodeType() == SkNodeType::NODE_NOTIFY && eventCounts[tasks[i]->GetEventId()] == 0) {
            SK_LOGI("Event[%lu] cancelled in post-process: NOTIFY task[%zu]", tasks[i]->GetEventId(), i);
            continue;
        // WAIT is removable while the per-event balance remains non-negative.
        // Under the constraints above, a negative value means this scope has WAITs
        // for this event but no corresponding NOTIFY.
        } else if (tasks[i]->GetNodeType() == SkNodeType::NODE_WAIT && eventCounts[tasks[i]->GetEventId()] >= 0) {
            SK_LOGI("Event[%lu] cancelled in post-process: WAIT task[%zu]", tasks[i]->GetEventId(), i);
            continue;
        }
        if (!tasks[i]->IsScopeNode()) {
            filteredTasks.push_back(tasks[i]);
        }
    }
    SK_LOGI("scope post-process filtered tasks: %zu -> %zu (%zu cancelled)", tasks.size(), filteredTasks.size(),
            tasks.size() - filteredTasks.size());
    return filteredTasks;
}

bool EnsureStreamCapacity(const UpdateStreamInfo& streamInfo)
{
    SK_LOGI("stream capacity check: streamId=%u, nodeSize=%lu, customParamSize=%zu",
            streamInfo.streamIdx, streamInfo.nodeSize, streamInfo.customParams.size());
    if (streamInfo.customParams.size() > streamInfo.nodeSize) {
        SK_LOGE("scope post-process overflow: streamId=%u, nodeSize=%lu, customParamSize=%zu", streamInfo.streamIdx,
                streamInfo.nodeSize, streamInfo.customParams.size());
        return false;
    }
    return true;
}

bool ProcessFrontWaitForStream(SuperKernelGraph& graph, const SuperKernelScopeInfo& scopeInfo,
                               std::vector<StreamPostPlan>& plans, SuperKernelProcessedScopeInfo& processedScopeInfo,
                               uint32_t curStreamIdx, uint64_t lastNodeId, uint32_t& needFrontWaitCount,
                               uint32_t& prevWaitStreamIdx)
{
    auto& scopeStreamInfo = scopeInfo.scopeStreamInfos[curStreamIdx];
    SK_LOGI("front-wait process begin: streamId=%u, prevWaitStreamIdx=%u, remainFrontWait=%u",
            scopeStreamInfo.streamIdx, prevWaitStreamIdx, needFrontWaitCount);
    needFrontWaitCount--;
    SK_LOGI("front-wait process state updated: streamId=%u, remainFrontWaitAfterDec=%u",
            scopeStreamInfo.streamIdx, needFrontWaitCount);

    // apply value memory addr
    void* addr = nullptr;
    aclError allocRet = SkResourceManager::ValueMemory(&addr);
    if (allocRet != ACL_SUCCESS || addr == nullptr) {
        SK_LOGE("front-wait value memory alloc failed: streamId=%u, ret=%d",
                scopeStreamInfo.streamIdx, allocRet);
        return false;
    }

    // create resetNode for sk optimize
    auto resetNode =
        SuperKernelNodeFactory::CreateNode(std::make_unique<aclmdlRITask>(nullptr), ACL_MODEL_RI_TASK_EVENT_RESET,
                                           INVALID_TASK_ID, scopeStreamInfo.streamIdx, INVALID_STREAM_ID, lastNodeId);
    resetNode->SetNodeType(SkNodeType::NODE_RESET);
    resetNode->nodeInfos.syncInfos.eventId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(addr));
    resetNode->nodeInfos.syncInfos.addrValue = addr;
    // record resetNode for sk optimize
    processedScopeInfo.eventNodes.emplace_back(std::move(resetNode));

    // cur stream add record event task
    aclmdlRITaskParams notifyParams = {};
    notifyParams.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
    notifyParams.valueWriteTaskParams.devAddr = addr;
    notifyParams.valueWriteTaskParams.value = 1;
    processedScopeInfo.updateStreamInfos[curStreamIdx].customParams.emplace_back(notifyParams);

    // prev stream add wait event task
    aclmdlRITaskParams waitParams = {};
    waitParams.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
    waitParams.valueWaitTaskParams.devAddr = addr;
    waitParams.valueWaitTaskParams.value = 1;
    waitParams.valueWaitTaskParams.flag = 1;
    processedScopeInfo.updateStreamInfos[prevWaitStreamIdx].customParams.emplace(
        processedScopeInfo.updateStreamInfos[prevWaitStreamIdx].customParams.begin(), waitParams);
    if (!EnsureStreamCapacity(processedScopeInfo.updateStreamInfos[curStreamIdx])
        || !EnsureStreamCapacity(processedScopeInfo.updateStreamInfos[prevWaitStreamIdx])) {
        return false;
    }

    // update info for next step
    prevWaitStreamIdx = curStreamIdx;
    return true;
}

bool ProcessBackBlockForStream(const SuperKernelScopeInfo& scopeInfo, std::vector<StreamPostPlan>& plans,
                               SuperKernelProcessedScopeInfo& processedScopeInfo, uint32_t curStreamIdx,
                               uint64_t lastNodeId)
{
    auto& scopeStreamInfo = scopeInfo.scopeStreamInfos[curStreamIdx];
    SK_LOGI("back-block process begin: streamId=%u", scopeStreamInfo.streamIdx);

    // apply value memory addr
    void* addr = nullptr;
    aclError allocRet = SkResourceManager::ValueMemory(&addr);
    if (allocRet != ACL_SUCCESS || addr == nullptr) {
        SK_LOGE("back-block value memory alloc failed: streamId=%u, ret=%d",
                scopeStreamInfo.streamIdx, allocRet);
        return false;
    }

    // create notifyNode for sk optimize
    auto notifyNode =
        SuperKernelNodeFactory::CreateNode(std::make_unique<aclmdlRITask>(nullptr), ACL_MODEL_RI_TASK_EVENT_RECORD,
                                           INVALID_TASK_ID, scopeStreamInfo.streamIdx, INVALID_STREAM_ID, lastNodeId);
    notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
    notifyNode->nodeInfos.syncInfos.eventId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(addr));
    notifyNode->nodeInfos.syncInfos.addrValue = addr;

    // record notifyNode for sk optimize
    processedScopeInfo.eventNodes.emplace_back(std::move(notifyNode));

    // cur stream add wait event task
    aclmdlRITaskParams waitParams = {};
    waitParams.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
    waitParams.valueWaitTaskParams.devAddr = addr;
    waitParams.valueWaitTaskParams.value = 1;
    waitParams.valueWaitTaskParams.flag = 1;
    processedScopeInfo.updateStreamInfos[curStreamIdx].customParams.emplace_back(waitParams);

    // cur stream add reset event task
    aclmdlRITaskParams resetParams = {};
    resetParams.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
    resetParams.valueWriteTaskParams.devAddr = addr;
    resetParams.valueWriteTaskParams.value = 0;
    processedScopeInfo.updateStreamInfos[curStreamIdx].customParams.emplace_back(resetParams);
    if (!EnsureStreamCapacity(processedScopeInfo.updateStreamInfos[curStreamIdx])) {
        return false;
    }
    return true;
}

bool GetMainAndSubStreamOrder(SuperKernelGraph& graph, std::vector<StreamPostPlan>& plans,
                                            SuperKernelProcessedScopeInfo& processedScopeInfo,
                                            uint32_t needFrontWaitCount, uint32_t& mainStreamIdx,
                                            std::vector<uint32_t>& subStreamOrder)
{
    const uint32_t streamCount = processedScopeInfo.updateStreamInfos.size();
    std::vector<uint32_t> subStreamCandidate;
    std::vector<uint32_t> mainStreamCandidate;
    std::vector<uint32_t> subStreamEntryCandidate;
    uint32_t entrySubStreamIdx = INVALID_STREAM_ID;
    subStreamOrder.clear();
    subStreamOrder.reserve(streamCount);

    for (uint32_t curStreamIdx = 0; curStreamIdx < streamCount; ++curStreamIdx) {
        bool isMainStreamCandidate = false;
        bool isSubStreamCandidate = false;
        bool isSubStreamEntryCandidate = false;

        const uint32_t otherFrontWaitCount = needFrontWaitCount - (plans[curStreamIdx].needFrontWait ? 1U : 0U);
        const uint32_t mainFrontReserveCount = otherFrontWaitCount > 0 ? 1U : 0U;
        uint64_t candidateNodeId = FindKernelNodeWithFrontReserve(
            graph, graph.GetNodeById(processedScopeInfo.updateStreamInfos[curStreamIdx].headNodeIdx),
            processedScopeInfo.updateStreamInfos[curStreamIdx].tailNodeIdx, mainFrontReserveCount);
        if (candidateNodeId != INVALID_TASK_ID) {
            mainStreamCandidate.push_back(curStreamIdx);
            plans[curStreamIdx].candidateNodeId = candidateNodeId;
            isMainStreamCandidate = true;
            SK_LOGI("main stream candidate added: streamId=%u, candidateNodeId=%lu", 
                    processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx, candidateNodeId);
        }

        uint32_t frontWaitNodeCount = 2 * (plans[curStreamIdx].needFrontWait ? 1U : 0U);
        uint32_t backBlockNodeCount = 2 * (plans[curStreamIdx].needBackBlock ? 1U : 0U);
        uint32_t needNodeCount = frontWaitNodeCount + backBlockNodeCount;
        if (needNodeCount <= processedScopeInfo.updateStreamInfos[curStreamIdx].nodeSize) {
            subStreamCandidate.push_back(curStreamIdx);
            isSubStreamCandidate = true;
            SK_LOGI("sub stream candidate added: streamId=%u, needNodeCount=%u", 
                    processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx, needNodeCount);
        }

        needNodeCount = needNodeCount - (plans[curStreamIdx].needFrontWait ? 1U : 0U);
        if (needNodeCount <= processedScopeInfo.updateStreamInfos[curStreamIdx].nodeSize) {
            subStreamEntryCandidate.push_back(curStreamIdx);
            isSubStreamEntryCandidate = true;
            SK_LOGI("sub stream entry candidate added: streamId=%u, needNodeCount=%u", 
                    processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx, needNodeCount);
        }

        if(!isMainStreamCandidate && !isSubStreamCandidate && !isSubStreamEntryCandidate) {
            SK_LOGI("streamId=%u does not meet the requirements for being mainStream and subStream.",
                    processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx);
            SK_LOGI("   1.stream have not candidate node, reserve num=%u", mainFrontReserveCount);
            SK_LOGI("   2.stream capacity insufficient for sub stream: nodeSize=%lu, but minimun required=%u",
                    processedScopeInfo.updateStreamInfos[curStreamIdx].nodeSize, needNodeCount);
            return false;
        }
    }

    if (streamCount == 1) {
        if (plans[0].candidateNodeId != INVALID_TASK_ID) {
            processedScopeInfo.skMainNodeId = plans[0].candidateNodeId;
            return true;
        }
        return false;
    }

    for (auto candidateMainStreamIdx : mainStreamCandidate) {
        for (auto candidateEntrySubStreamIdx : subStreamEntryCandidate) {
            size_t localStreamCnt = subStreamCandidate.size();
            if (candidateEntrySubStreamIdx == candidateMainStreamIdx) {
                continue;
            }
            auto it = find(subStreamCandidate.begin(), subStreamCandidate.end(), candidateMainStreamIdx);
            if (it == subStreamCandidate.end()) {
                ++localStreamCnt;
            }
            it = find(subStreamCandidate.begin(), subStreamCandidate.end(), candidateEntrySubStreamIdx);
            if (it == subStreamCandidate.end()) {
                ++localStreamCnt;
            }
            if (localStreamCnt == streamCount) {
                mainStreamIdx = candidateMainStreamIdx;
                processedScopeInfo.skMainNodeId = plans[candidateMainStreamIdx].candidateNodeId;
                entrySubStreamIdx = candidateEntrySubStreamIdx;
                break;
            }
        }
        if (mainStreamIdx != INVALID_STREAM_ID && processedScopeInfo.skMainNodeId != INVALID_TASK_ID) {
            SK_LOGI("main stream and entry sub stream selected: mainStreamIdx=%u, entrySubStreamIdx=%u",
                        mainStreamIdx, entrySubStreamIdx);
            break;
        }
    }

    if (mainStreamIdx == INVALID_STREAM_ID ||processedScopeInfo.skMainNodeId == INVALID_TASK_ID) {
        SK_LOGE("failed to find main SK node in scope during post-process, skip update");
        return false;
    }
    if (entrySubStreamIdx == INVALID_STREAM_ID) {
        SK_LOGE("failed to find entry sub stream in scope during post-process, skip update");
        return false;
    }

    for (auto curStreamIdx : subStreamCandidate) {
        if (curStreamIdx == mainStreamIdx || curStreamIdx == entrySubStreamIdx) {
            continue;
        }
        subStreamOrder.push_back(curStreamIdx);
    }
    subStreamOrder.push_back(entrySubStreamIdx);
    return true;
}

bool ApplyEventMemoryResource(SuperKernelGraph& graph, SuperKernelBaseNode* eventNode,
                              std::vector<SuperKernelBaseNode*>& needUpdateNodes)
{
    // Constraint : eventNode.syncInfos have all associated node
    auto& syncInfos = eventNode->nodeInfos.syncInfos;
    // Determine whether the event has already been allocated memory. If not, then request memory allocation.
    if (syncInfos.addrValue != nullptr) {
        // already applied
        SK_LOGI("event memory already applied: eventId=%lu, addr=%p", syncInfos.eventId, syncInfos.addrValue);
        return true;
    } else {
        auto eventId = eventNode->GetEventId();
        auto eventInfos = graph.GetEventInfo(eventId);
        if (eventInfos == nullptr) {
            SK_LOGE("event not found in graph: event infos=%s", eventNode->FormatNodeInfo().c_str());
            return false;
        }
        // check syncInfos
        if (eventInfos->notifyNodeId == INVALID_TASK_ID || eventInfos->waitNodeIdList.empty()
            || eventInfos->resetNodeIdList.empty()) {
            SK_LOGE("event syncInfos invalid: eventId=%lu, NotifyNodeId=%lu, WaitNodeIdsSize=%zu, ResetNodeId=%lu",
                    syncInfos.eventId, eventInfos->notifyNodeId, eventInfos->waitNodeIdList.size(),
                    eventInfos->resetNodeIdList);
            return false;
        }
        SK_LOGI("event memory allocated start ...");
        void* addr = nullptr;
        aclError allocRet = SkResourceManager::ValueMemory(&addr);
        if (allocRet != ACL_SUCCESS || addr == nullptr) {
            SK_LOGE("event memory alloc failed: eventId=%lu, ret=%d", syncInfos.eventId, allocRet);
            return false;
        }
        // notify sync info update
        auto notifyNode = graph.GetNodeById(eventInfos->notifyNodeId);
        if (notifyNode == nullptr) {
            SK_LOGE("notify event node not found in graph during event memory apply: notifyNodeId=%lu",
                    eventInfos->notifyNodeId);
            return false;
        }
        // wait sync info update
        notifyNode->nodeInfos.syncInfos.addrValue = addr;
        needUpdateNodes.emplace_back(notifyNode);
        SK_LOGI("Updated notify node addrValue: nodeId=%lu, addr=%p", notifyNode->GetNodeId(), addr);
        for (auto waitNodeId : eventInfos->waitNodeIdList) {
            auto waitNode = graph.GetNodeById(waitNodeId);
            if (waitNode == nullptr) {
                SK_LOGE("wait event node not found in graph during event memory apply: waitNodeId=%lu", waitNodeId);
                return false;
            }
            waitNode->nodeInfos.syncInfos.addrValue = addr;
            needUpdateNodes.emplace_back(waitNode);
            SK_LOGI("Updated wait node addrValue: nodeId=%lu, addr=%p", waitNode->GetNodeId(), addr);
        }
        // reset sync info update
        for (auto resetNodeId: eventInfos->resetNodeIdList) {
            auto resetNode = graph.GetNodeById(resetNodeId);
            if (resetNode == nullptr) {
                SK_LOGE("reset event node not found in graph during event memory apply: resetNodeId=%lu",
                    resetNodeId);
                return false;
            }
            resetNode->nodeInfos.syncInfos.addrValue = addr;
            needUpdateNodes.emplace_back(resetNode);
            SK_LOGI("Updated reset node addrValue: nodeId=%lu, addr=%p", resetNode->GetNodeId(), addr);
            SK_LOGI("event memory allocated end: eventId=%lu, addr=%p", syncInfos.eventId, syncInfos.addrValue);
        }
    }

    syncInfos = eventNode->nodeInfos.syncInfos;
    if (syncInfos.addrValue == nullptr) {
        SK_LOGE("event memory apply failed: eventId=%lu, addr is nullptr", syncInfos.eventId);
        return false;
    }
    return true;
}

} // namespace

SuperKernelProcessedScopeInfo SuperKernelScopePostProcessor::PostProcess(SuperKernelScopeInfo& scopeInfo)
{
    // Implementation for post-processing scopes
    // Constraint : directed acyclic graph

    for (auto& streamInfo : scopeInfo.scopeStreamInfos) {
        uint64_t curNodeId = streamInfo.headNodeIdx;
        while (curNodeId != INVALID_TASK_ID) {
            auto* curNode = graph.GetNodeById(curNodeId);
            if (curNode == nullptr) {
                SK_LOGE("node not found during scope post process: nodeId=%lu, streamId=%u", curNodeId,
                        streamInfo.streamIdx);
                return {};
            }
            if (!curNode->IsFusible()) {
                SK_LOGE("node is not fusible during scope post process: nodeId=%lu, streamId=%u", curNodeId,
                        streamInfo.streamIdx);
                return {};
            }
            if (curNodeId == streamInfo.tailNodeIdx) {
                break;
            }
            curNodeId = curNode->GetNextNodeId();
        }
    }

    uint32_t streamCount = scopeInfo.scopeStreamInfos.size();
    SK_LOGI("scope post-process begin: streamCount=%u, nodeCount=%zu", streamCount, scopeInfo.nodes.size());

    std::vector<SuperKernelBaseNode*> filteredTasks = FilterCancelledTasks(scopeInfo.nodes);
    if (filteredTasks.empty()) {
        SK_LOGE("scope post-process failed: no task remains after cancelling notify/wait pairs");
        return {};
    }

    // init processedScopeInfo
    SuperKernelProcessedScopeInfo processedScopeInfo;
    processedScopeInfo.nodes = std::move(filteredTasks);
    processedScopeInfo.skMainNodeId = INVALID_TASK_ID;
    processedScopeInfo.updateStreamInfos.resize(streamCount);

    std::vector<StreamPostPlan> plans(streamCount);
    uint32_t needFrontWaitCount = 0;
    std::vector<SuperKernelBaseNode*> needUpdateNodes;

    // step 0: apply event memory resource for filtered task
    for(auto node: processedScopeInfo.nodes) {
        auto nodeType = node->GetNodeType();
        switch (nodeType) {
            case SkNodeType::NODE_NOTIFY:
            case SkNodeType::NODE_WAIT:
            case SkNodeType::NODE_RESET:
                if (!ApplyEventMemoryResource(graph, node, needUpdateNodes)) {
                    SK_LOGE("event memory resource apply failed during post-process: nodeId=%lu, nodeType=%d",
                            node->GetNodeId(), static_cast<int>(nodeType));
                    return {};
                }
                break;
            default:
                break;
        }
    }
    SK_LOGI("applied event memory resource for filtered tasks, needUpdateNodesSize=%zu", needUpdateNodes.size());

    // step 1: collect per-stream info and boundary plans
    for (uint32_t curStreamIdx = 0; curStreamIdx < streamCount; ++curStreamIdx) {
        SK_LOGI("collect stream info: streamIdx=%u, nodeSize=%lu", curStreamIdx,
                scopeInfo.scopeStreamInfos[curStreamIdx].nodeSize);
        auto& scopeStreamInfo = scopeInfo.scopeStreamInfos[curStreamIdx];
        processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx = scopeStreamInfo.streamIdx;
        processedScopeInfo.updateStreamInfos[curStreamIdx].nodeSize = scopeStreamInfo.nodeSize;
        processedScopeInfo.updateStreamInfos[curStreamIdx].headNodeIdx = scopeStreamInfo.headNodeIdx;
        processedScopeInfo.updateStreamInfos[curStreamIdx].tailNodeIdx = scopeStreamInfo.tailNodeIdx;

        auto* headNode = graph.GetNodeById(scopeStreamInfo.headNodeIdx);
        auto* tailNode = graph.GetNodeById(scopeStreamInfo.tailNodeIdx);
        if (headNode == nullptr) {
            SK_LOGE("head node not found in graph during post-process: head=%lu", scopeStreamInfo.headNodeIdx);
            return {};
        }
        if (tailNode == nullptr) {
            SK_LOGE("tail node not found in graph during post-process: tail=%lu", scopeStreamInfo.tailNodeIdx);
            return {};
        }

        if (headNode != nullptr && headNode->GetPreNodeId() != INVALID_TASK_ID) {
            // judge whether front need wait
            plans[curStreamIdx].needFrontWait = true;
            needFrontWaitCount++;
        }
        if (tailNode != nullptr && tailNode->GetNextNodeId() != INVALID_TASK_ID) {
            // judge whether back need block
            plans[curStreamIdx].needBackBlock = true;
        }
        SK_LOGI("stream plan collected: streamId=%u, headNodeId=%lu, tailNodeId=%lu, nodeSize=%lu, needFrontWait=%d,\
             needBackBlock=%d",
                scopeStreamInfo.streamIdx, scopeStreamInfo.headNodeIdx, scopeStreamInfo.tailNodeIdx,
                scopeStreamInfo.nodeSize, plans[curStreamIdx].needFrontWait, plans[curStreamIdx].needBackBlock);
    }
    SK_LOGI("collect stream done: streamCount=%u, needFrontWaitCount=%u", streamCount, needFrontWaitCount);

    // step 2: select main stream and entry sub stream.
    // For each candidate stream, only other streams can force the future main
    // stream to reserve its first original node for an inserted wait event.
    uint32_t mainStreamIdx = INVALID_STREAM_ID;
    std::vector<uint32_t> subStreamOrder;
    if (!GetMainAndSubStreamOrder(graph, plans, processedScopeInfo, needFrontWaitCount, mainStreamIdx,
                                                subStreamOrder)) {
        return {};
    }
    SK_LOGI("select main stream and sub stream done.");

    // step 3: process front wait and back block
    uint32_t prevWaitStreamIdx = mainStreamIdx;
    uint64_t lastNodeId = processedScopeInfo.nodes.back()->GetNodeId();
    for (uint32_t curStreamIdx : subStreamOrder) {
        if (plans[curStreamIdx].needFrontWait
            && !ProcessFrontWaitForStream(graph, scopeInfo, plans, processedScopeInfo, curStreamIdx, lastNodeId,
                                          needFrontWaitCount, prevWaitStreamIdx)) {
            SK_LOGE("process front-wait failed streamId=%u, nodeSize=%lu, FrontWait=%u",
                    processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx,
                    processedScopeInfo.updateStreamInfos[curStreamIdx].nodeSize, plans[curStreamIdx].needFrontWait);
            return {};
        }

        if (plans[curStreamIdx].needBackBlock
            && !ProcessBackBlockForStream(scopeInfo, plans, processedScopeInfo, curStreamIdx, lastNodeId)) {
            SK_LOGE("process back-block failed streamId=%u, nodeSize=%lu, BackBlock=%u",
                    processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx,
                    processedScopeInfo.updateStreamInfos[curStreamIdx].nodeSize, plans[curStreamIdx].needBackBlock);
            return {};
        }
    }
    SK_LOGI("scope post-process front-wait and back-block processing done for all sub streams");

    // step 4: record nodes that need update for graph update
    if (!graph.ExpandUpdateNodes(needUpdateNodes)) {
        SK_LOGE("expand update node failed to record update nodes for graph update");
        return {};
    }
    SK_LOGI("expand update node done, needUpdateNodesSize=%zu", needUpdateNodes.size());

    size_t totalCustomParamSize = 0;
    for (const auto& streamInfo : processedScopeInfo.updateStreamInfos) {
        totalCustomParamSize += streamInfo.customParams.size();
    }
    SK_LOGI("scope post-process end: streamCount=%u, skMainNodeId=%lu, eventNodeCount=%zu, totalCustomParamSize=%zu",
            streamCount, processedScopeInfo.skMainNodeId, processedScopeInfo.eventNodes.size(), totalCustomParamSize);
    return processedScopeInfo;
}