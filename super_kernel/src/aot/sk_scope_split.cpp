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
 * \file sk_scope_split.cpp
 * \brief
 */

#include "sk_scope_split.h"
#include "sk_lock_detector.h"

uint64_t SuperKernelScopeSplitter::FindAvailableHeadNode(uint64_t curNodeIdx) const {
    while (curNodeIdx != INVALID_TASK_ID) {
        const SuperKernelBaseNode * node = graph.GetNodeById(curNodeIdx);
        if (node == nullptr) {
            SK_LOGE("node with id %lu not found in graph during finding available head node\n", curNodeIdx);
            return INVALID_TASK_ID;
        }
        if (node->GetNodeType() == SkNodeType::NODE_KERNEL && node->IsFusible()) {
            return curNodeIdx;
        }
        curNodeIdx = node->GetNextNodeId();
    }
    return INVALID_TASK_ID;
}

uint64_t SuperKernelScopeSplitter::GenerateScopeInfosByNodeIdx(uint64_t curNodeIdx) {
    try {
        uint64_t headNodeIdx = FindAvailableHeadNode(curNodeIdx);
        if (headNodeIdx == INVALID_TASK_ID) {
            SK_LOGI("no available head node found starting from node %lu\n", curNodeIdx);
            return INVALID_TASK_ID;
        }
        SK_LOGI("Generating scope infos for head node %lu\n", headNodeIdx);
        SuperKernelScopeInfo scopeInfo;
        ScopeStreamInfo scopeStreamInfo;
        scopeStreamInfo.headNodeIdx = headNodeIdx;
        scopeStreamInfo.streamIdx = graph.GetNodeById(headNodeIdx)->GetStreamIdxInGraph();

        uint64_t preNodeIdx = headNodeIdx;
        curNodeIdx = headNodeIdx;
        LockDetector lockDetector;
        while (curNodeIdx != INVALID_TASK_ID) {
            SuperKernelBaseNode *node = graph.GetNodeById(curNodeIdx);
            if (node == nullptr) {
                SK_LOGE("node with id %lu not found in graph during generating scope infos\n", curNodeIdx);
                throw std::runtime_error("Node not found in graph");
            }
            if (!node->IsFusible() || !lockDetector.IsFusible(*node, graph)) {
                scopeStreamInfo.tailNodeIdx = preNodeIdx;
                curNodeIdx = node->GetNextNodeId();
                lockDetector.Reset(graph);
                break;
            }
            scopeInfo.nodes.emplace_back(std::move(node));
            ++scopeStreamInfo.nodeSize;
            preNodeIdx = curNodeIdx;
            curNodeIdx = node->GetNextNodeId();
        }
        SK_LOGI("Generated scope infos for head node %lu in stream %lu, tail node %lu, node size %lu\n", 
                scopeStreamInfo.headNodeIdx, scopeStreamInfo.streamIdx, scopeStreamInfo.tailNodeIdx, scopeStreamInfo.nodeSize);
        scopeInfo.scopeStreamInfos.emplace_back(std::move(scopeStreamInfo));
        scopeInfos.emplace_back(std::move(scopeInfo));
        return curNodeIdx;
    } catch (const std::exception &e) {
        SK_LOGE("Exception occurred in GenerateScopeInfosByNodeIdx: %s\n", e.what());
        return INVALID_TASK_ID;
    }
}

bool SuperKernelScopeSplitter::SplitSingleStreamGraph() {
    const auto& streams = graph.GetStreams();
    const auto& headNodes = graph.GetHeadNodes();
    SK_LOGI("start splitting single stream graph, stream count: %zu", streams.size());

    for (size_t streamIdx = 0; streamIdx < streams.size(); ++streamIdx) {
        SK_LOGI("start splitting stream %zu", streamIdx);
        uint64_t curNodeIdx = headNodes[streamIdx];
        while (curNodeIdx != INVALID_TASK_ID) {
            curNodeIdx = GenerateScopeInfosByNodeIdx(curNodeIdx);
        }
    }
    return true;
}