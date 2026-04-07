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
#include "aprof_pub.h"
#include "securec.h"
#include "sk_event_recorder.h"

namespace {
std::string GetSkFuncName(const std::vector<SuperKernelBaseNode*>& nodes, uint32_t scopeIdx)
{
    size_t startNodeIdx = nodes.size() - 1;
    size_t endNodeIdx = 0;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->GetNodeType() == SkNodeType::NODE_KERNEL) {
            if (i < startNodeIdx) {
                startNodeIdx = i;
            }
            if (i > endNodeIdx) {
                endNodeIdx = i;
            }
        }
    }
    const NodeInfos& startNodeInfos = nodes[startNodeIdx]->GetNodeInfos();
    const NodeInfos& endNodeInfos = nodes[endNodeIdx]->GetNodeInfos();

    std::string skName = "skId: " + std::to_string(scopeIdx) + "__startNodeName: " + startNodeInfos.kernelInfos.funcName 
                            + "__endNodeName: " + endNodeInfos.kernelInfos.funcName;
    return skName;
}

void PrintSKNodesDetail(std::string skFuncName, SuperKernelProcessedScopeInfo& processedScopeInfo)
{
    auto& scopeIdx = processedScopeInfo.scopeIdx;
    auto& nodes = processedScopeInfo.nodes;
    SK_LOGI("  SK Function: %s, scope id: %zu, Node Count: %zu", skFuncName.c_str(), scopeIdx, nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        SK_LOGI("    [%zu] %s", i, nodes[i]->Format().c_str());
    }
}

void PrintSKNodes(std::string skFuncName, SuperKernelProcessedScopeInfo& processedScopeInfo)
{
    {
        SK_LOG_CONTEXT_SIMPLE("sk_fused_nodes.log");
        PrintSKNodesDetail(skFuncName, processedScopeInfo);
    }
    PrintSKNodesDetail(skFuncName, processedScopeInfo);
}
} // namespace

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
                SK_LOGE("node not found during stream-based update: nodeId=%lu, streamId=%u", curNodeId,
                        streamInfo.streamIdx);
                return false;
            }
            UpdateContext ctx;
            if (eventCnt < customParamSize) {
                // set front node for stream sync
                auto& customParams = streamInfo.customParams[eventCnt++];
                ctx.customParams = &customParams;
            } else if (curNodeId == processedScopeInfo.skMainNodeId) {
                // search node for sk launch after front node
                if (!skMainNodeUpdated) {
                    skMainNodeUpdated = true;
                    ctx.launchInfo = const_cast<SkLaunchInfo*>(&launchInfo);
                } else {
                    SK_LOGI("repeat find sk launch node, skip update kernel and set invalid node");
                }
            }

            ++updateTotalCount;
            if (!node->Update(ctx)) { // default invalid
                SK_LOGE("node update failed: nodeId=%lu, streamId=%u", curNodeId, streamInfo.streamIdx);
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

    graph.SetUpdateFlag(true); // set graph update flag
    SK_LOGI("scope update finished: update total nodes=%zu", updateTotalCount);

    return true;
}

bool SuperKernelOptimizer::ExpandScopeNodes(SuperKernelScopeInfo& scopeInfo, SuperKernelGraph& graph)
{
    std::vector<SuperKernelBaseNode*> needUpdateNodes;
    for (auto& streamInfo : scopeInfo.scopeStreamInfos) {
        uint64_t curNodeId = streamInfo.headNodeIdx;
        while (curNodeId != INVALID_TASK_ID) {
            auto* curNode = graph.GetNodeById(curNodeId);
            if (curNode == nullptr) {
                SK_LOGE("node not found during scope node update: nodeId=%lu, streamId=%u", curNodeId,
                        streamInfo.streamIdx);
                return false;
            }
            if (curNode->IsScopeNode()) {
                needUpdateNodes.emplace_back(curNode);
            }
            if (curNodeId == streamInfo.tailNodeIdx) {
                break;
            }
            curNodeId = curNode->GetNextNodeId();
        }
    }
    graph.ExpandUpdateNodes(needUpdateNodes);
    SK_LOGI("expand scope nodes finished: expandedNodeCount=%zu", needUpdateNodes.size());
    return true;
}

// Schedule task-stream nodes for super-kernel launch.
bool SuperKernelOptimizer::Schedule(SuperKernelProcessedScopeInfo& processedScopeInfo, SuperKernelGraph& graph,
                                    SkTaskBuilder& builder)
{
    const auto& taskNodes = processedScopeInfo.nodes;
    if (taskNodes.empty()) {
        SK_LOGE("no tasks for super kernel optimization: scope has 0 nodes for optimization");
        return false;
    }

    std::string skFuncName = GetSkFuncName(taskNodes, processedScopeInfo.scopeIdx);
    PrintSKNodes(skFuncName, processedScopeInfo);

    std::vector<SuperKernelBaseNode*> customTasks;
    customTasks.reserve(processedScopeInfo.eventNodes.size());
    for (const auto& eventNode : processedScopeInfo.eventNodes) {
        customTasks.emplace_back(eventNode.get());
    }

    SK_LOGI("schedule scope: taskCount=%zu, customTaskCount=%zu, updateStreamCount=%zu", taskNodes.size(),
            customTasks.size(), processedScopeInfo.updateStreamInfos.size());

    SkLaunchInfo launchInfo = builder.Build(skFuncName, taskNodes, customTasks);

    if (!SkProfiling(processedScopeInfo, launchInfo, graph)) {
        SK_LOGE("SkProfiling failed");
        return false;
    }
    // DFX 支持profiling功能，在SkEventRecorder::Instance().Init()里根据环境变量开关
    if (!DumpProfilingDetail(taskNodes, launchInfo, processedScopeInfo, graph.modelRI)) {
            SK_LOGE("Dump sk time profiling detail failed");
            return false;
    }

    if (launchInfo.entryInfo.skEntryFunc == nullptr || launchInfo.devArgs.Get() == nullptr) {
        SK_LOGE("schedule failed: build launch info failed");
        return false;
    }
    SK_LOGI("schedule scope: build finished, entryType=%s, entryFuncHandle=%p, skFuncName=%s",
            to_string(launchInfo.entryInfo.entryType), launchInfo.entryInfo.skEntryFunc, launchInfo.skFuncName.c_str());

    if (!Update(processedScopeInfo, graph, launchInfo)) {
        SK_LOGE("schedule failed: scope update failed");
        return false;
    }
    return true;
}

bool SuperKernelOptimizer::Process(SuperKernelGraph& graph)
{
    // Split graph into multiple scopes.
    SuperKernelScopeSplitter splitter(graph, opts);
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
    for (size_t scopeIdx = 0; scopeIdx < scopeInfos.size(); ++scopeIdx) {
        auto& scopeInfo = scopeInfos[scopeIdx];
        SK_LOGI("process scope begin: scopeIdx=%zu", scopeIdx);
        SuperKernelProcessedScopeInfo processedScopeInfo = postProcessor.PostProcess(scopeInfo);
        if (processedScopeInfo.nodes.empty()) {
            SK_LOGI("scope has no nodes after post-process, skipping scheduling: scopeIdx=%zu", scopeIdx);
            if (!ExpandScopeNodes(scopeInfo, graph)) {
                SK_LOGE("failed to expand scope nodes: scopeIdx=%zu", scopeIdx);
                return false;
            }
            continue;
        }
        processedScopeInfo.scopeIdx = static_cast<uint32_t>(scopeIdx);
        if (!Schedule(processedScopeInfo, graph, builder)) {
            SK_LOGE("process scope failed: scopeIdx=%zu, schedule/update returned false", scopeIdx);
            return false;
        }
    }
    SK_LOGI("super kernel process finished: scopeCount=%zu", scopeInfos.size());
    return true;
}
