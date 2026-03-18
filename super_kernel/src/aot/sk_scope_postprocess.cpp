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
    // Core invariant per eventId after pass-1:
    // eventCounts[eventId] = expected_wait_count_from_notify - observed_wait_count.
    // A value of 0 means notify-side expectation matches observed waits exactly.
    std::unordered_map<uint64_t, int64_t> eventCounts;

    // First pass: accumulate expected-vs-observed WAIT balance per eventId.
    for (size_t i = 0; i < tasks.size(); i++) {
        if (tasks[i]->GetNodeType() == SkNodeType::NODE_NOTIFY) {
            eventCounts[tasks[i]->GetEventId()] += tasks[i]->GetNodeInfos().syncInfos.correspondingWaitNodeIds.size();
        } else if (tasks[i]->GetNodeType() == SkNodeType::NODE_WAIT) {
            eventCounts[tasks[i]->GetEventId()]--;
        }
    }

    // Second pass:
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
        filteredTasks.push_back(tasks[i]);
    }

    SK_LOGI("scope post-process filtered tasks: %zu -> %zu (%zu cancelled)", tasks.size(), filteredTasks.size(),
            tasks.size() - filteredTasks.size());
    return filteredTasks;
}

bool EnsureStreamCapacity(const SuperKernelProcessedScopeInfo& processedScopeInfo, uint32_t checkStreamIdx)
{
    const auto& streamInfo = processedScopeInfo.updateStreamInfos[checkStreamIdx];
    SK_LOGI("stream capacity check: streamIdx=%u, streamId=%u, nodeSize=%lu, customParamSize=%zu", checkStreamIdx,
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
    SK_LOGI("front-wait process begin: streamIdx=%u, streamId=%u, prevWaitStreamIdx=%u, remainFrontWait=%u",
            curStreamIdx, scopeStreamInfo.streamIdx, prevWaitStreamIdx, needFrontWaitCount);
    needFrontWaitCount--;
    SK_LOGI("front-wait process state updated: streamIdx=%u, streamId=%u, remainFrontWaitAfterDec=%u", curStreamIdx,
            scopeStreamInfo.streamIdx, needFrontWaitCount);

    // apply value memory addr
    void* addr = nullptr;
    aclError allocRet = SkResourceManager::ValueMemory(&addr);
    if (allocRet != ACL_SUCCESS || addr == nullptr) {
        SK_LOGE("front-wait value memory alloc failed: streamIdx=%u, streamId=%u, ret=%d", curStreamIdx,
                scopeStreamInfo.streamIdx, allocRet);
        return false;
    }

    // create resetNode for sk optimize
    auto resetNode =
        SuperKernelNodeFactory::CreateNode(std::make_unique<aclmdlRITask>(nullptr), ACL_MODEL_RI_TASK_EVENT_RESET,
                                           INVALID_TASK_ID, scopeStreamInfo.streamIdx, lastNodeId);
    resetNode->SetNodeType(SkNodeType::NODE_RESET);
    resetNode->nodeInfos.syncInfos.eventId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(addr));
    resetNode->nodeInfos.syncInfos.addrValue = addr;
    // record resetNode for sk optimize
    processedScopeInfo.eventNodes.emplace_back(std::move(resetNode));

    // cur stream add record event task
    aclmdlRITaskParams notifyParams;
    errno_t err = memset_s(&notifyParams, sizeof(notifyParams), 0, sizeof(notifyParams));
    if (err != 0) {
        SK_LOGE("front-wait memset_s notify params failed, ret=%d", static_cast<int>(err));
        return false;
    }
    notifyParams.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
    notifyParams.valueWriteTaskParams.devAddr = addr;
    notifyParams.valueWriteTaskParams.value = 1;
    processedScopeInfo.updateStreamInfos[curStreamIdx].customParams.emplace_back(notifyParams);

    // prev stream add wait event task
    aclmdlRITaskParams waitParams;
    err = memset_s(&waitParams, sizeof(waitParams), 0, sizeof(waitParams));
    if (err != 0) {
        SK_LOGE("front-wait memset_s wait params failed, ret=%d", static_cast<int>(err));
        return false;
    }
    waitParams.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
    waitParams.valueWaitTaskParams.devAddr = addr;
    waitParams.valueWaitTaskParams.value = 1;
    waitParams.valueWaitTaskParams.flag = 0;
    processedScopeInfo.updateStreamInfos[prevWaitStreamIdx].customParams.emplace(
        processedScopeInfo.updateStreamInfos[prevWaitStreamIdx].customParams.begin(), waitParams);
    if (!EnsureStreamCapacity(processedScopeInfo, curStreamIdx)
        || !EnsureStreamCapacity(processedScopeInfo, prevWaitStreamIdx)) {
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
    SK_LOGI("back-block process begin: streamIdx=%u, streamId=%u", curStreamIdx, scopeStreamInfo.streamIdx);

    // apply value memory addr
    void* addr = nullptr;
    aclError allocRet = SkResourceManager::ValueMemory(&addr);
    if (allocRet != ACL_SUCCESS || addr == nullptr) {
        SK_LOGE("back-block value memory alloc failed: streamIdx=%u, streamId=%u, ret=%d", curStreamIdx,
                scopeStreamInfo.streamIdx, allocRet);
        return false;
    }

    // create notifyNode for sk optimize
    auto notifyNode =
        SuperKernelNodeFactory::CreateNode(std::make_unique<aclmdlRITask>(nullptr), ACL_MODEL_RI_TASK_EVENT_RECORD,
                                           INVALID_TASK_ID, scopeStreamInfo.streamIdx, lastNodeId);
    notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
    notifyNode->nodeInfos.syncInfos.eventId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(addr));
    notifyNode->nodeInfos.syncInfos.addrValue = addr;

    // record notifyNode for sk optimize
    processedScopeInfo.eventNodes.emplace_back(std::move(notifyNode));

    // cur stream add wait event task
    aclmdlRITaskParams waitParams;
    errno_t err = memset_s(&waitParams, sizeof(waitParams), 0, sizeof(waitParams));
    if (err != 0) {
        SK_LOGE("back-block memset_s wait params failed, ret=%d", static_cast<int>(err));
        return false;
    }
    waitParams.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
    waitParams.valueWaitTaskParams.devAddr = addr;
    waitParams.valueWaitTaskParams.value = 1;
    waitParams.valueWaitTaskParams.flag = 0;
    processedScopeInfo.updateStreamInfos[curStreamIdx].customParams.emplace_back(waitParams);

    // cur stream add reset event task
    aclmdlRITaskParams resetParams;
    err = memset_s(&resetParams, sizeof(resetParams), 0, sizeof(resetParams));
    if (err != 0) {
        SK_LOGE("back-block memset_s reset params failed, ret=%d", static_cast<int>(err));
        return false;
    }
    resetParams.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
    resetParams.valueWriteTaskParams.devAddr = addr;
    resetParams.valueWriteTaskParams.value = 0;
    processedScopeInfo.updateStreamInfos[curStreamIdx].customParams.emplace_back(resetParams);
    if (!EnsureStreamCapacity(processedScopeInfo, curStreamIdx)) {
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
        const uint32_t otherFrontWaitCount = needFrontWaitCount - (plans[curStreamIdx].needFrontWait ? 1U : 0U);
        const uint32_t mainFrontReserveCount = otherFrontWaitCount > 0 ? 1U : 0U;
        uint64_t candidateNodeId = FindKernelNodeWithFrontReserve(
            graph, graph.GetNodeById(processedScopeInfo.updateStreamInfos[curStreamIdx].headNodeIdx),
            processedScopeInfo.updateStreamInfos[curStreamIdx].tailNodeIdx, mainFrontReserveCount);
        if (candidateNodeId != INVALID_TASK_ID) {
            mainStreamCandidate.push_back(curStreamIdx);
            plans[curStreamIdx].candidateNodeId = candidateNodeId;
            SK_LOGI("main stream candidate added: streamIdx=%u, streamId=%u, candidateNodeId=%lu", curStreamIdx,
                    processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx, candidateNodeId);
        }

        uint32_t frontWaitNodeCount = 2 * (plans[curStreamIdx].needFrontWait ? 1U : 0U);
        uint32_t backBlockNodeCount = 2 * (plans[curStreamIdx].needBackBlock ? 1U : 0U);
        uint32_t needNodeCount = frontWaitNodeCount + backBlockNodeCount;
        if (needNodeCount <= processedScopeInfo.updateStreamInfos[curStreamIdx].nodeSize) {
            subStreamCandidate.push_back(curStreamIdx);
            SK_LOGI("sub stream candidate added: streamIdx=%u, streamId=%u, needNodeCount=%u", curStreamIdx,
                    processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx, needNodeCount);
        }

        needNodeCount = needNodeCount - (plans[curStreamIdx].needFrontWait ? 1U : 0U);
        if (needNodeCount <= processedScopeInfo.updateStreamInfos[curStreamIdx].nodeSize) {
            SK_LOGI("sub stream entry candidate added: streamIdx=%u, streamId=%u, needNodeCount=%u", curStreamIdx,
                    processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx, needNodeCount);
            subStreamEntryCandidate.push_back(curStreamIdx);
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
                SK_LOGI("main stream and entry sub stream selected: mainStreamIdx=%u, entrySubStreamIdx=%u",
                        mainStreamIdx, entrySubStreamIdx);
                break;
            }
        }
    }

    if (processedScopeInfo.skMainNodeId == INVALID_TASK_ID || mainStreamIdx == INVALID_STREAM_ID) {
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

} // namespace

SuperKernelProcessedScopeInfo SuperKernelScopePostProcessor::PostProcess(SuperKernelScopeInfo& scopeInfo)
{
    // Implementation for post-processing scopes

    uint32_t streamCount = scopeInfo.scopeStreamInfos.size();
    SK_LOGI("scope post-process begin: streamCount=%u, nodeCount=%zu", streamCount, scopeInfo.nodes.size());

    std::vector<SuperKernelBaseNode*> filteredTasks = FilterCancelledTasks(scopeInfo.nodes);
    if (filteredTasks.empty()) {
        SK_LOGE("scope post-process failed: no task remains after cancelling notify/wait pairs");
        return {};
    }

    // init
    SuperKernelProcessedScopeInfo processedScopeInfo;
    processedScopeInfo.nodes = std::move(filteredTasks);
    processedScopeInfo.skMainNodeId = INVALID_TASK_ID;
    processedScopeInfo.updateStreamInfos.resize(streamCount);

    std::vector<StreamPostPlan> plans(streamCount);
    uint32_t needFrontWaitCount = 0;

    // pass1: collect per-stream info and boundary plans
    for (uint32_t curStreamIdx = 0; curStreamIdx < streamCount; ++curStreamIdx) {
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
        SK_LOGI(
            "stream plan collected: streamIdx=%u, streamId=%u, head=%lu, tail=%lu, nodeSize=%lu, needFrontWait=%d, needBackBlock=%d",
            curStreamIdx, scopeStreamInfo.streamIdx, scopeStreamInfo.headNodeIdx, scopeStreamInfo.tailNodeIdx,
            scopeStreamInfo.nodeSize, plans[curStreamIdx].needFrontWait, plans[curStreamIdx].needBackBlock);
    }
    SK_LOGI("scope pass1 done: streamCount=%u, needFrontWaitCount=%u", streamCount, needFrontWaitCount);

    // pass2: select main stream and entry sub stream.
    // For each candidate stream, only other streams can force the future main
    // stream to reserve its first original node for an inserted wait event.
    uint32_t mainStreamIdx = INVALID_STREAM_ID;
    std::vector<uint32_t> subStreamOrder;
    if (!GetMainAndSubStreamOrder(graph, plans, processedScopeInfo, needFrontWaitCount, mainStreamIdx,
                                                subStreamOrder)) {
        return {};
    }

    // pass3: one traversal for front-chain and back block/release
    uint32_t prevWaitStreamIdx = mainStreamIdx;
    uint64_t lastNodeId = processedScopeInfo.nodes.back()->GetNodeId();
    for (uint32_t curStreamIdx : subStreamOrder) {
        SK_LOGI("scope pass3 stream begin: streamIdx=%u, streamId=%u, needFrontWait=%d, needBackBlock=%d", curStreamIdx,
                processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx, plans[curStreamIdx].needFrontWait,
                plans[curStreamIdx].needBackBlock);

        if (plans[curStreamIdx].needFrontWait
            && !ProcessFrontWaitForStream(graph, scopeInfo, plans, processedScopeInfo, curStreamIdx, lastNodeId,
                                          needFrontWaitCount, prevWaitStreamIdx)) {
            SK_LOGE("scope pass3 front-wait failed due to runtime-dependent path: streamIdx=%u, streamId=%u",
                    curStreamIdx, processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx);
            return {};
        }

        if (plans[curStreamIdx].needBackBlock
            && !ProcessBackBlockForStream(scopeInfo, plans, processedScopeInfo, curStreamIdx, lastNodeId)) {
            SK_LOGE("scope pass3 back-block failed due to runtime-dependent path: streamIdx=%u, streamId=%u",
                    curStreamIdx, processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx);
            return {};
        }

        SK_LOGI("scope pass3 stream end: streamIdx=%u, streamId=%u, remainFrontWait=%u", curStreamIdx,
                processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx, needFrontWaitCount);
    }

    size_t totalCustomParamSize = 0;
    for (const auto& streamInfo : processedScopeInfo.updateStreamInfos) {
        totalCustomParamSize += streamInfo.customParams.size();
    }
    SK_LOGI("scope post-process done: streamCount=%u, skMainNodeId=%lu, eventNodeCount=%zu, totalCustomParamSize=%zu",
            streamCount, processedScopeInfo.skMainNodeId, processedScopeInfo.eventNodes.size(), totalCustomParamSize);
    return processedScopeInfo;
}