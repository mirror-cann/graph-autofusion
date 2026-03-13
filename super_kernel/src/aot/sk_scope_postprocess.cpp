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

#include "sk_scope_postprocess.h"

namespace {

SuperKernelBaseNode* AdvanceNodeWithinScope(SuperKernelGraph& graph, SuperKernelBaseNode* node, uint64_t tailNodeId,
                                            uint32_t stepCount)
{
    if (node == nullptr) {
        SK_LOGW("advance node skipped: start node is nullptr, tailNodeId=%lu, stepCount=%u", tailNodeId, stepCount);
        return nullptr;
    }

    SK_LOGI("advance node begin: startNodeId=%lu, tailNodeId=%lu, stepCount=%u", node->GetNodeId(), tailNodeId,
            stepCount);
    while (node != nullptr && stepCount > 0) {
        if (node->GetNodeId() == tailNodeId || node->GetNextNodeId() == INVALID_TASK_ID) {
            SK_LOGW("advance node stopped at boundary: currentNodeId=%lu, tailNodeId=%lu, remainStep=%u",
                    node->GetNodeId(), tailNodeId, stepCount);
            return nullptr;
        }
        node = graph.GetNodeById(node->GetNextNodeId());
        if (node == nullptr) {
            SK_LOGE("advance node failed: next node not found in graph");
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
    SK_LOGW("find kernel with reserve failed: headNodeId=%lu, tailNodeId=%lu, frontReserveCount=%u",
            headNode == nullptr ? INVALID_TASK_ID : headNode->GetNodeId(), tailNodeId, frontReserveCount);
    return INVALID_TASK_ID;
}

struct StreamPostPlan {
    bool needFrontWait = false;
    bool needBackBlock = false;
    SuperKernelBaseNode* workNode = nullptr;
};

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
    auto* workNode = plans[curStreamIdx].workNode;
    if (needFrontWaitCount != 0 && workNode != nullptr && workNode->GetNextNodeId() != INVALID_TASK_ID) {
        // use workNode next node when curStream is not last sub curStream which need front wait
        auto* nextNode = graph.GetNodeById(workNode->GetNextNodeId());
        if (nextNode != nullptr) {
            workNode = nextNode;
            SK_LOGI("front-wait process moved work node: streamIdx=%u, streamId=%u, workNodeId=%lu", curStreamIdx,
                    scopeStreamInfo.streamIdx, workNode->GetNodeId());
        } else {
            SK_LOGE("head next node not found in graph during post-process: next=%lu",
                    plans[curStreamIdx].workNode->GetNextNodeId());
        }
    }

    if (workNode == nullptr) {
        SK_LOGE("front work node is nullptr.");
        return true;
    }

    // Feature(aclmdIRITaskParams): This branch still depends on runtime event APIs.
    // Migrate to IR-task params after introducing unified event-memory allocator.
    // Feature(event-memory): Replace per-branch local event creation with central
    // event memory apply/release flow owned by post-process context.
    // Feature : feature: sk apply memory resource, and update with memory task
    SK_LOGI("front-wait process hit path: streamIdx=%u, streamId=%u", curStreamIdx, scopeStreamInfo.streamIdx);
    SK_LOGE("This branch still depends on runtime event APIs, which should be migrated to IR-task params \
        after introducing unified event-memory allocator. Skip front wait process for stream %u.",
            curStreamIdx);
    return false;
    // Feature : apply memory addr
    // apply event for front wait
    aclrtEvent event;
    // EventManager::CreateEvent(&event);
    aclrtTaskEventParams customParams;
    customParams.type = ACL_RT_TASK_EVENT_RECORD;
    customParams.flag = ACL_RT_TASK_VALID;
    customParams.event = event;

    // apply event mermory and get event info
    aclrtTaskEventParams eventGetParams; // Feature : feature use aclmdIRITaskParams
    auto notifyTask = workNode->originTask.get();
    CHECK_ACL(aclrtTaskSetEventParams(notifyTask, &customParams));
    CHECK_ACL(aclrtTaskGetEventParams(notifyTask, &eventGetParams));

    // Feature : create resetNode for sk optimize
    auto resetNode = SuperKernelNodeFactory::CreateNode(std::make_unique<aclrtTask>(nullptr), ACL_RT_TASK_EVENT_RESET,
                                                        INVALID_TASK_ID, scopeStreamInfo.streamIdx, lastNodeId);
    resetNode->nodeInfos.syncInfos.eventId = eventGetParams.sequenceId;
    resetNode->nodeInfos.syncInfos.addrValue = eventGetParams.u.memoryEventInfo.eventAddr;

    // Feature cur stream add record event task
    aclrtTaskEventParams recordParams;
    recordParams.type = ACL_RT_TASK_EVENT_RECORD;
    recordParams.flag = ACL_RT_TASK_VALID;
    recordParams.event = event;
    processedScopeInfo.updateStreamInfos[curStreamIdx].customParams.emplace_back(recordParams);

    // Feature prev stream add wait event task
    aclrtTaskEventParams waitParams;
    waitParams.type = ACL_RT_TASK_EVENT_WAIT;
    waitParams.flag = ACL_RT_TASK_VALID;
    waitParams.event = event;
    processedScopeInfo.updateStreamInfos[prevWaitStreamIdx].customParams.emplace(
        processedScopeInfo.updateStreamInfos[prevWaitStreamIdx].customParams.begin(), waitParams);
    if (!EnsureStreamCapacity(processedScopeInfo, curStreamIdx)
        || !EnsureStreamCapacity(processedScopeInfo, prevWaitStreamIdx)) {
        return false;
    }

    // record resetNode for sk optimize
    processedScopeInfo.eventNodes.emplace_back(std::move(resetNode));

    // update info for next step, workNode move
    plans[curStreamIdx].workNode = graph.GetNodeById(workNode->GetNextNodeId());
    prevWaitStreamIdx = curStreamIdx;
    return true;
}

bool ProcessBackBlockForStream(const SuperKernelScopeInfo& scopeInfo, std::vector<StreamPostPlan>& plans,
                               SuperKernelProcessedScopeInfo& processedScopeInfo, uint32_t curStreamIdx,
                               uint64_t lastNodeId)
{
    auto& scopeStreamInfo = scopeInfo.scopeStreamInfos[curStreamIdx];
    SK_LOGI("back-block process begin: streamIdx=%u, streamId=%u", curStreamIdx, scopeStreamInfo.streamIdx);
    // direct use workNode, when need front wait, it will move to next node.
    auto* workNode = plans[curStreamIdx].workNode;
    if (workNode == nullptr) {
        SK_LOGE("back work node is nullptr.");
        return true;
    }

    // Feature(aclmdIRITaskParams): This branch still depends on runtime event APIs.
    // Migrate to IR-task params after introducing unified event-memory allocator.
    // Feature(event-memory): Replace per-branch local event creation with central
    // event memory apply/release flow owned by post-process context.
    // Feature : feature: sk apply memory resource, and update with memory task
    SK_LOGI("back-block process hit Feature path: streamIdx=%u, streamId=%u", curStreamIdx, scopeStreamInfo.streamIdx);
    SK_LOGE("This branch still depends on runtime event APIs, which should be migrated to IR-task params \
        after introducing unified event-memory allocator. Skip back block process for stream %u.",
            curStreamIdx);
    SK_LOGE("back-block process aborted: streamIdx=%u, streamId=%u, lastNodeId=%lu", curStreamIdx,
            scopeStreamInfo.streamIdx, lastNodeId);
    return false;
    // apply event for back block
    aclrtEvent event;
    // EventManager::CreateEvent(&event);
    aclrtTaskEventParams customParams;
    customParams.type = ACL_RT_TASK_EVENT_WAIT;
    customParams.flag = ACL_RT_TASK_VALID;
    customParams.event = event;

    // apply event memory and get event info
    aclrtTaskEventParams eventGetParams; // Feature : feature use aclmdIRITaskParams
    auto waitTask = workNode->originTask.get();
    CHECK_ACL(aclrtTaskSetEventParams(waitTask, &customParams));
    CHECK_ACL(aclrtTaskGetEventParams(waitTask, &eventGetParams));

    // create notifyNode for sk optimize
    auto notifyNode = SuperKernelNodeFactory::CreateNode(std::make_unique<aclrtTask>(nullptr), ACL_RT_TASK_EVENT_RECORD,
                                                         INVALID_TASK_ID, scopeStreamInfo.streamIdx, lastNodeId);
    notifyNode->nodeInfos.syncInfos.eventId = eventGetParams.sequenceId;
    notifyNode->nodeInfos.syncInfos.addrValue = eventGetParams.u.memoryEventInfo.eventAddr;

    // record notifyNode for sk optimize
    processedScopeInfo.eventNodes.emplace_back(std::move(notifyNode));

    // cur stream add wait event task
    aclrtTaskEventParams waitParams;
    waitParams.type = ACL_RT_TASK_EVENT_WAIT;
    waitParams.flag = ACL_RT_TASK_VALID;
    waitParams.event = event;
    processedScopeInfo.updateStreamInfos[curStreamIdx].customParams.emplace_back(waitParams);

    // cur stream add reset event task
    aclrtTaskEventParams resetParams;
    resetParams.type = ACL_RT_TASK_EVENT_RESET;
    resetParams.flag = ACL_RT_TASK_VALID;
    resetParams.event = event;
    processedScopeInfo.updateStreamInfos[curStreamIdx].customParams.emplace_back(resetParams);
    if (!EnsureStreamCapacity(processedScopeInfo, curStreamIdx)) {
        return false;
    }
    return true;
}

} // namespace

SuperKernelProcessedScopeInfo SuperKernelScopePostProcessor::PostProcess(SuperKernelScopeInfo& scopeInfo)
{
    // Implementation for post-processing scopes

    SuperKernelProcessedScopeInfo processedScopeInfo;
    if (scopeInfo.nodes.empty()) {
        SK_LOGI("scope post-process skipped: empty scope nodes");
        return processedScopeInfo;
    }
    uint32_t streamCount = scopeInfo.scopeStreamInfos.size();
    // init
    processedScopeInfo.nodes = scopeInfo.nodes;
    processedScopeInfo.skMainNodeId = INVALID_TASK_ID;
    processedScopeInfo.updateStreamInfos.resize(streamCount);

    uint64_t lastNodeId = scopeInfo.nodes.back()->GetNodeId();
    uint32_t mainStreamIdx = INVALID_STREAM_ID;

    std::vector<StreamPostPlan> plans(streamCount);
    std::vector<uint32_t> streamOrder;
    streamOrder.reserve(streamCount);
    uint32_t needFrontWaitCount = 0;
    SK_LOGI("scope post-process begin: streamCount=%u, nodeCount=%zu", streamCount, scopeInfo.nodes.size());

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
        }
        if (tailNode == nullptr) {
            SK_LOGE("tail node not found in graph during post-process: tail=%lu", scopeStreamInfo.tailNodeIdx);
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
        plans[curStreamIdx].workNode = headNode;
        SK_LOGI(
            "stream plan collected: streamIdx=%u, streamId=%u, head=%lu, tail=%lu, nodeSize=%lu, needFrontWait=%d, needBackBlock=%d",
            curStreamIdx, scopeStreamInfo.streamIdx, scopeStreamInfo.headNodeIdx, scopeStreamInfo.tailNodeIdx,
            scopeStreamInfo.nodeSize, plans[curStreamIdx].needFrontWait, plans[curStreamIdx].needBackBlock);
    }
    SK_LOGI("scope pass1 done: streamCount=%u, needFrontWaitCount=%u", streamCount, needFrontWaitCount);

    // pass2: select main stream.
    // For each candidate stream, only other streams can force the future main
    // stream to reserve its first original node for an inserted wait event.
    for (uint32_t curStreamIdx = 0; curStreamIdx < streamCount; ++curStreamIdx) {
        if (plans[curStreamIdx].workNode == nullptr) {
            SK_LOGE("head work node is nullptr during main stream selection: streamIdx=%u", curStreamIdx);
            return {};
        }
        if (processedScopeInfo.skMainNodeId == INVALID_TASK_ID) {
            const uint32_t otherFrontWaitCount = needFrontWaitCount - (plans[curStreamIdx].needFrontWait ? 1U : 0U);
            const uint32_t mainFrontReserveCount = otherFrontWaitCount > 0 ? 1U : 0U;
            uint64_t candidateNodeId = FindKernelNodeWithFrontReserve(
                graph, plans[curStreamIdx].workNode, processedScopeInfo.updateStreamInfos[curStreamIdx].tailNodeIdx,
                mainFrontReserveCount);
            SK_LOGI("main stream candidate checked: streamIdx=%u, streamId=%u, reserveCount=%u, candidateNodeId=%lu",
                    curStreamIdx, processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx, mainFrontReserveCount,
                    candidateNodeId);
            if (candidateNodeId != INVALID_TASK_ID) {
                processedScopeInfo.skMainNodeId = candidateNodeId;
                mainStreamIdx = curStreamIdx;
                SK_LOGI("main stream selected: streamIdx=%u, streamId=%u, skMainNodeId=%lu", mainStreamIdx,
                        processedScopeInfo.updateStreamInfos[mainStreamIdx].streamIdx, processedScopeInfo.skMainNodeId);
            }
        }
        if (mainStreamIdx != curStreamIdx) {
            streamOrder.emplace_back(curStreamIdx);
            SK_LOGI("stream added to post order: streamIdx=%u, streamId=%u, orderSize=%zu", curStreamIdx,
                    processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx, streamOrder.size());
        }
    }

    if (processedScopeInfo.skMainNodeId == INVALID_TASK_ID || mainStreamIdx == INVALID_STREAM_ID) {
        SK_LOGE("failed to find main SK node in scope during post-process, skip update");
        return {};
    }

    SK_LOGI("scope pass2 done: mainStreamIdx=%u, mainStreamId=%u, streamOrderSize=%zu", mainStreamIdx,
            processedScopeInfo.updateStreamInfos[mainStreamIdx].streamIdx, streamOrder.size());

    // pass3: one traversal for front-chain and back block/release
    uint32_t prevWaitStreamIdx = mainStreamIdx;
    for (uint32_t curStreamIdx : streamOrder) {
        SK_LOGI("scope pass3 stream begin: streamIdx=%u, streamId=%u, needFrontWait=%d, needBackBlock=%d", curStreamIdx,
                processedScopeInfo.updateStreamInfos[curStreamIdx].streamIdx, plans[curStreamIdx].needFrontWait,
                plans[curStreamIdx].needBackBlock);
        // Feature(aclmdIRITaskParams): Keep stream traversal logic unchanged, but swap
        // helper internals to IR-task params once memory resource flow is available.
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
    SK_LOGI(
        "scope post-process done: streamCount=%u, streamOrderSize=%zu, skMainNodeId=%lu, eventNodeCount=%zu, totalCustomParamSize=%zu",
        streamCount, streamOrder.size(), processedScopeInfo.skMainNodeId, processedScopeInfo.eventNodes.size(),
        totalCustomParamSize);
    return processedScopeInfo;
}