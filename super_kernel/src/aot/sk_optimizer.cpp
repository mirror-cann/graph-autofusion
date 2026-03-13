/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>
#include "sk_optimizer.h"
#include "sk_scope_split.h"
#include "sk_scope_postprocess.h"
#include "sk_task_builder.h"
#include "sk_log.h"

extern "C" aclrtBinHandle AscendGetEntryBinHandle();
namespace {

aclrtFuncHandle ResolveSkEntryFunc(const char* funcName)
{
    aclrtBinHandle bhdl = nullptr;
    bhdl = AscendGetEntryBinHandle();
    if (bhdl == nullptr) {
        SK_LOGE("failed to get entry bin handle: AscendGetEntryBinHandle() returned null");
        return nullptr;
    }
    aclrtFuncHandle fhdl = nullptr;
    CHECK_ACL(aclrtBinaryGetFunction(bhdl, funcName, &fhdl));
    if (fhdl == nullptr) {
        SK_LOGE("failed to resolve entry func handle: funcName=%s, binHandle=%p", funcName, bhdl);
        return nullptr;
    }
    return fhdl;
}

} // namespace

// Schedule task-stream nodes for super-kernel launch.
void SuperKernelOptimizer::Schedule(SuperKernelProcessedScopeInfo& processedScopeInfo, SuperKernelGraph& graph,
                                    SkTaskBuilder& builder)
{
    const auto& taskNodes = processedScopeInfo.nodes;
    std::vector<SuperKernelBaseNode*> customTasks;
    customTasks.reserve(processedScopeInfo.eventNodes.size());
    for (const auto& eventNode : processedScopeInfo.eventNodes) {
        customTasks.emplace_back(eventNode.get());
    }
    if (taskNodes.empty()) {
        SK_LOGW("no tasks for super kernel optimization: scope has 0 nodes");
        return;
    }

    SK_LOGI("schedule scope: taskCount=%zu, customTaskCount=%zu, updateStreamCount=%zu", taskNodes.size(),
            customTasks.size(), processedScopeInfo.updateStreamInfos.size());

    SkLaunchInfo launchInfo = builder.Build(taskNodes, customTasks);
    SK_LOGI("schedule scope: build finished, entryFuncName=%s", launchInfo.entryInfo.skEntryFuncName);

    aclrtFuncHandle skEntryFunc = ResolveSkEntryFunc(launchInfo.entryInfo.skEntryFuncName);
    if (skEntryFunc == nullptr) {
        SK_LOGE("failed to resolve sk entry function: entryFuncName=%s", launchInfo.entryInfo.skEntryFuncName);
        return;
    }

    Update(processedScopeInfo, graph, launchInfo, skEntryFunc);
}

void SuperKernelOptimizer::Update(SuperKernelProcessedScopeInfo& processedScopeInfo, SuperKernelGraph& graph,
                                  const SkLaunchInfo& launchInfo, aclrtFuncHandle skEntryFunc)
{
    SK_LOGI("scope update begin: streamCount=%zu, nodeCount=%zu", processedScopeInfo.updateStreamInfos.size(),
            processedScopeInfo.nodes.size());
    bool skMainNodeUpdated = false;
    size_t updateFailCount = 0;
    size_t updateTotalCount = 0;
    for (auto& streamInfo : processedScopeInfo.updateStreamInfos) {
        SK_LOGI("update stream begin: streamId=%u, headNodeId=%lu, tailNodeId=%lu, nodeSize=%lu, customParamSize=%zu",
                streamInfo.streamIdx, streamInfo.headNodeIdx, streamInfo.tailNodeIdx, streamInfo.nodeSize,
                streamInfo.customParams.size());
        size_t customParamSize = streamInfo.customParams.size();
        if (streamInfo.nodeSize < customParamSize) {
            SK_LOGE("node size is less than custom params size: nodeSize=%lu, customParamSize=%zu", streamInfo.nodeSize,
                    customParamSize);
            continue;
        }

        uint64_t curNodeId = streamInfo.headNodeIdx;
        size_t eventCnt = 0;

        while (curNodeId != INVALID_TASK_ID) {
            auto* node = graph.GetNodeById(curNodeId);
            if (node == nullptr) {
                SK_LOGE("node not found during stream-based update: nodeId=%lu, streamIdx=%u", curNodeId,
                        streamInfo.streamIdx);
                // Continue with next stream when this stream chain is broken.
                break;
            }
            UpdateContext ctx;
            if (eventCnt < customParamSize) {
                // Feature(aclmdIRITaskParams): customParams source type will change
                // after post-process migrates from aclrtTaskEventParams to IR-task params.
                // set front node for stream sync
                auto& customParams = streamInfo.customParams[eventCnt++];
                ctx.customParams = &customParams;
            } else if (curNodeId == processedScopeInfo.skMainNodeId) {
                // search node for sk launch after front node
                if (!skMainNodeUpdated) {
                    skMainNodeUpdated = true;
                    ctx.launchInfo = const_cast<SkLaunchInfo*>(&launchInfo);
                    ctx.skEntryFunc = skEntryFunc;
                } else {
                    SK_LOGE("repeat find sk launch node, skip update kernel and set invalid node");
                }
            }

            ++updateTotalCount;
            if (!node->Update(ctx)) { // default invalid
                ++updateFailCount;
                SK_LOGW("node update failed: nodeId=%lu, streamIdx=%u", curNodeId, streamInfo.streamIdx);
            }

            if (curNodeId == streamInfo.tailNodeIdx) {
                break;
            }
            curNodeId = node->GetNextNodeId();
        }
        SK_LOGI("update stream end: streamId=%u, visitedNodes=%zu", streamInfo.streamIdx, eventCnt);
    }

    if (!skMainNodeUpdated) {
        SK_LOGE("not find sk launch node, sk optimize faild");
    }

    if (updateFailCount > 0) {
        SK_LOGW("scope update finished with failures: failed=%zu, total=%zu", updateFailCount, updateTotalCount);
    } else {
        SK_LOGI("scope update finished: failed=0, total=%zu", updateTotalCount);
    }

    if (updateTotalCount != processedScopeInfo.nodes.size()) {
        SK_LOGE("update node count mismatch: expected=%zu, actual=%zu", processedScopeInfo.nodes.size(),
                updateTotalCount);
    }
}

void SuperKernelOptimizer::Process(SuperKernelGraph& graph)
{
    // Split graph into multiple scopes.
    SuperKernelScopeSplitter splitter(graph);
    if (splitter.SplitGraph()) {
        SK_LOGI("graph split into %zu scopes", splitter.GetScopeInfos().size());
    } else {
        SK_LOGW("graph split failed or no scopes found: cannot proceed with super kernel optimization");
        return;
    }
    auto& scopeInfos = splitter.GetScopeInfos();

    SkTaskBuilder builder(opts, graph);
    SuperKernelScopePostProcessor postProcessor(graph);

    // Process each scope sequentially.
    size_t scopeIndex = 0;
    for (auto& scopeInfo : scopeInfos) {
        SK_LOGI("process scope begin: scopeIndex=%zu", scopeIndex);
        SuperKernelProcessedScopeInfo processedScopeInfo = postProcessor.PostProcess(scopeInfo);
        Schedule(processedScopeInfo, graph, builder);
        SK_LOGI("process scope end: scopeIndex=%zu, processedNodeCount=%zu", scopeIndex, processedScopeInfo.nodes.size());
        ++scopeIndex;
    }
    SK_LOGI("super kernel process finished: scopeCount=%zu", scopeInfos.size());
}
