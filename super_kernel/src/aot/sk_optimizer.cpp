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
std::string GetSkFuncName(const std::vector<SuperKernelBaseNode*>& nodes, uint16_t scopeId, const std::string& scopeName)
{
    const SuperKernelBaseNode* startKernelNode = nullptr;
    const SuperKernelBaseNode* endKernelNode = nullptr;
    for (const auto* node : nodes) {
        if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
            if (startKernelNode == nullptr) {
                startKernelNode = node;
            }
            endKernelNode = node;
        }
    }

    std::string scopePrefix = scopeName.empty() ? "" : "scopeName: " + scopeName + "__";
    if (startKernelNode == nullptr || endKernelNode == nullptr) {
        return scopePrefix + "skId: " + std::to_string(scopeId) + "__no_kernel_scope";
    }

    const NodeInfos& startNodeInfos = startKernelNode->GetNodeInfos();
    const NodeInfos& endNodeInfos = endKernelNode->GetNodeInfos();
    return scopePrefix + "skId: " + std::to_string(scopeId) + "__startNodeName: "
        + startNodeInfos.kernelInfos.funcName + "__endNodeName: " + endNodeInfos.kernelInfos.funcName;
}

void PrintSKNodesDetail(std::string skFuncName, SuperKernelScopeInfo& scopeInfo)
{
    uint16_t scopeId = scopeInfo.GetScopeId();
    auto& nodes = scopeInfo.GetExtInfo().filteredNodes;
    SK_LOGI("  SK Function: %s, scope id: %u, Node Count: %zu", skFuncName.c_str(), scopeId, nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        SK_LOGI("    [%zu] %s", i, nodes[i]->Format().c_str());
    }
}

void PrintSKNodes(std::string skFuncName, SuperKernelScopeInfo& scopeInfo)
{
    {
        SK_LOG_CONTEXT_SIMPLE("sk_fused_nodes.log");
        PrintSKNodesDetail(skFuncName, scopeInfo);
    }
    PrintSKNodesDetail(skFuncName, scopeInfo);
}
} // namespace

bool SuperKernelOptimizer::Update(SuperKernelScopeInfo& scopeInfo, SuperKernelGraph& graph,
                                  const SkLaunchInfo& launchInfo)
{
    SK_LOGI("scope update begin: streamCount=%zu", scopeInfo.GetScopeStreamInfos().size());
    bool skMainNodeUpdated = false;
    size_t updateTotalCount = 0;

    auto& scopeStreamInfos = scopeInfo.GetScopeStreamInfos();
    auto& extInfo = scopeInfo.MutableExtInfo();
    for (size_t streamIdx = 0; streamIdx < scopeStreamInfos.size(); ++streamIdx) {
        auto& streamInfo = scopeStreamInfos[streamIdx];
        auto& customParams = extInfo.customParamsList[streamIdx];
        SK_LOGI("update stream begin: streamId=%u, headNodeId=%lu, tailNodeId=%lu, nodeSize=%lu, customParamSize=%zu",
                streamInfo.streamIdx, streamInfo.headNodeIdx, streamInfo.tailNodeIdx, streamInfo.nodeSize,
                customParams.size());
        size_t customParamSize = customParams.size();
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
                auto& curCustomParams = customParams[eventCnt++];
                ctx.customParams = &curCustomParams;
            } else if (curNodeId == extInfo.skMainNodeId) {
                if (!skMainNodeUpdated) {
                    skMainNodeUpdated = true;
                    ctx.launchInfo = const_cast<SkLaunchInfo*>(&launchInfo);
                } else {
                    SK_LOGI("repeat find sk launch node, skip update kernel and set invalid node");
                }
            }

            ++updateTotalCount;
            if (!node->Update(ctx)) {
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

    graph.SetUpdateFlag(true);
    SK_LOGI("scope update finished: update total nodes=%zu", updateTotalCount);
    return true;
}

// Schedule task-stream nodes for super-kernel launch.
bool SuperKernelOptimizer::Schedule(SuperKernelScopeInfo& scopeInfo, SuperKernelGraph& graph,
                                    SkTaskBuilder& builder)
{
    const auto& taskNodes = scopeInfo.GetExtInfo().filteredNodes;
    if (taskNodes.empty()) {
        SK_LOGE("no tasks for super kernel optimization: scope has 0 nodes for optimization");
        return false;
    }

    std::string skFuncName = GetSkFuncName(taskNodes, scopeInfo.GetScopeId(), scopeInfo.GetExtInfo().scopeName);
    PrintSKNodes(skFuncName, scopeInfo);

    std::vector<SuperKernelBaseNode*> customTasks;
    customTasks.reserve(scopeInfo.GetExtInfo().eventNodes.size());
    for (const auto& eventNode : scopeInfo.GetExtInfo().eventNodes) {
        customTasks.emplace_back(eventNode.get());
    }

    SK_LOGI("schedule scope: taskCount=%zu, customTaskCount=%zu, updateStreamCount=%zu", taskNodes.size(),
            customTasks.size(), scopeInfo.GetScopeStreamInfos().size());

    SkLaunchInfo launchInfo = builder.Build(skFuncName, taskNodes, customTasks);

    if (!SkProfiling(scopeInfo, launchInfo, graph)) {
        SK_LOGE("SkProfiling failed");
        return false;
    }
    if (!DumpProfilingDetail(taskNodes, launchInfo, scopeInfo, graph.modelRI)) {
        SK_LOGE("Dump sk time profiling detail failed");
        return false;
    }

    if (launchInfo.entryInfo.skEntryFunc == nullptr || launchInfo.devArgs.Get() == nullptr) {
        SK_LOGE("schedule failed: build launch info failed");
        return false;
    }
    SK_LOGI("schedule scope: build finished, entryType=%s, entryFuncHandle=%p, skFuncName=%s",
            to_string(launchInfo.entryInfo.entryType), launchInfo.entryInfo.skEntryFunc, launchInfo.skFuncName.c_str());

    if (!Update(scopeInfo, graph, launchInfo)) {
        SK_LOGE("schedule failed: scope update failed");
        return false;
    }
    return true;
}

bool SuperKernelOptimizer::Process(SuperKernelGraph& graph)
{
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

    for (size_t i = 0; i < scopeInfos.size(); ++i) {
        auto& scopeInfo = scopeInfos[i];
        SK_LOGI("process scope begin: scopeId=%u", scopeInfo.GetScopeId());
        if (!postProcessor.PostProcess(scopeInfo)) {
            SK_LOGI("scope unprocessable after post-process, skip schedule/update: scopeId=%u, reason=%s",
                    scopeInfo.GetScopeId(), to_string(scopeInfo.GetExtInfo().failReason));
            continue;
        }
        if (scopeInfo.GetExtInfo().filteredNodes.empty()) {
            SK_LOGI("scope has no nodes after post-process, skipping schedule/update: scopeId=%u", scopeInfo.GetScopeId());
            continue;
        }

        scopeInfo.MutableExtInfo().scopeName = ScopeSplitPass::GetScopeNamesFromBitFlags(scopeInfo.GetScopeBitFlags(), graph);
        if (!Schedule(scopeInfo, graph, builder)) {
            SK_LOGE("process scope failed: scopeId=%u, schedule/update returned false", scopeInfo.GetScopeId());
            return false;
        }
    }
    SK_LOGI("super kernel process finished: scopeCount=%zu", scopeInfos.size());
    return true;
}
