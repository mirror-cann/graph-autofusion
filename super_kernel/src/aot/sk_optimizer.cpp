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

// Schedule task-stream nodes for super-kernel launch.
bool SuperKernelOptimizer::Schedule(SuperKernelProcessedScopeInfo& processedScopeInfo, SuperKernelGraph& graph,
                                    SkTaskBuilder& builder)
{
    const auto& taskNodes = processedScopeInfo.nodes;
    if (taskNodes.empty()) {
        SK_LOGW("no tasks for super kernel optimization: scope has 0 nodes");
        return true;
    }

    std::vector<SuperKernelBaseNode*> customTasks;
    customTasks.reserve(processedScopeInfo.eventNodes.size());
    for (const auto& eventNode : processedScopeInfo.eventNodes) {
        customTasks.emplace_back(eventNode.get());
    }

    SK_LOGI("schedule scope: taskCount=%zu, customTaskCount=%zu, updateStreamCount=%zu", taskNodes.size(),
            customTasks.size(), processedScopeInfo.updateStreamInfos.size());

    SkLaunchInfo launchInfo = builder.Build(taskNodes, customTasks);
    if (launchInfo.entryInfo.skEntryFunc == nullptr || launchInfo.devArgs.Get() == nullptr) {
        SK_LOGE("schedule failed: build launch info failed");
        return false;
    }
    SK_LOGI("schedule scope: build finished, entryType=%s, entryFuncHandle=%p",
            to_string(launchInfo.entryInfo.entryType), launchInfo.entryInfo.skEntryFunc);

    if (!Update(processedScopeInfo, graph, launchInfo)) {
        SK_LOGE("schedule failed: scope update failed");
        return false;
    }
    return true;
}

bool SuperKernelOptimizer::Update(SuperKernelProcessedScopeInfo& processedScopeInfo, SuperKernelGraph& graph,
                                  const SkLaunchInfo& launchInfo)
{
    SK_LOGI("scope update begin: streamCount=%zu", processedScopeInfo.updateStreamInfos.size());
    bool skMainNodeUpdated = false;
    size_t updateTotalCount = 0;

    for (auto& streamInfo : processedScopeInfo.updateStreamInfos) {
        SK_LOGI("update stream begin: streamId=%u, headNodeId=%lu, tailNodeId=%lu, nodeSize=%lu, customParamSize=%zu",
                streamInfo.streamIdx, streamInfo.headNodeIdx, streamInfo.tailNodeIdx, streamInfo.nodeSize,
                streamInfo.customParams.size());
        size_t customParamSize = streamInfo.customParams.size();
        if (streamInfo.nodeSize < customParamSize) {
            SK_LOGE("node size is less than custom params size: nodeSize=%lu, customParamSize=%zu", streamInfo.nodeSize,
                    customParamSize);
            return false;
        }

        uint64_t curNodeId = streamInfo.headNodeIdx;
        size_t eventCnt = 0;

        while (curNodeId != INVALID_TASK_ID) {
            auto* node = graph.GetNodeById(curNodeId);
            if (node == nullptr) {
                SK_LOGE("node not found during stream-based update: nodeId=%lu, streamIdx=%u", curNodeId,
                        streamInfo.streamIdx);
                return false;
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
                } else {
                    SK_LOGW("repeat find sk launch node, skip update kernel and set invalid node");
                }
            }

            ++updateTotalCount;
            if (!node->Update(ctx)) { // default invalid
                SK_LOGE("node update failed: nodeId=%lu, streamIdx=%u", curNodeId, streamInfo.streamIdx);
                return false;
            }

            if (curNodeId == streamInfo.tailNodeIdx) {
                break;
            }
            curNodeId = node->GetNextNodeId();
        }
        SK_LOGI("update stream end: streamId=%u, visitedNodes=%zu", streamInfo.streamIdx, eventCnt);
    }

    if (!skMainNodeUpdated) {
        SK_LOGE("not find sk launch node, sk optimize failed");
        return false;
    }

    SK_LOGI("scope update finished: update total nodes=%zu", updateTotalCount);

    return true;
}

bool SuperKernelOptimizer::Process(SuperKernelGraph& graph)
{
    // Split graph into multiple scopes.
    SuperKernelScopeSplitter splitter(graph);
    if (splitter.SplitGraph()) {
        SK_LOGI("graph split into %zu scopes", splitter.GetScopeInfos().size());
    } else {
        SK_LOGE("graph split failed or no scopes found: cannot proceed with super kernel optimization");
        return false;
    }
    auto& scopeInfos = splitter.GetScopeInfos();

    SkTaskBuilder builder(opts, graph);
    SuperKernelScopePostProcessor postProcessor(graph);

    // Process each scope sequentially.
    size_t scopeIndex = 0;
    for (auto& scopeInfo : scopeInfos) {
        SK_LOGI("process scope begin: scopeIndex=%zu", scopeIndex);
        SuperKernelProcessedScopeInfo processedScopeInfo = postProcessor.PostProcess(scopeInfo);
        if (!Schedule(processedScopeInfo, graph, builder)) {
            SK_LOGE("process scope failed: scopeIndex=%zu, schedule/update returned false", scopeIndex);
            return false;
        }
        ++scopeIndex;
    }
    SK_LOGI("super kernel process finished: scopeCount=%zu", scopeInfos.size());
    return true;
}
