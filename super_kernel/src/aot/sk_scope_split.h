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
 * \file sk_scope_split.h
 * \brief
 */

#pragma once


#include "sk_graph.h"
#include "sk_log.h"

struct ScopeStreamInfo {
    uint32_t streamIdx = 0;
    uint64_t headNodeIdx = INVALID_TASK_ID;
    uint64_t tailNodeIdx = INVALID_TASK_ID;
    uint64_t nodeSize = 0;
};

struct SuperKernelScopeInfo{
    std::vector<ScopeStreamInfo> scopeStreamInfos;
    std::vector<SuperKernelBaseNode *> nodes;
};

class SuperKernelScopeSplitter {
public:
    SuperKernelScopeSplitter(SuperKernelGraph &graph) : graph(graph) { }
    ~SuperKernelScopeSplitter() = default;
    SuperKernelScopeSplitter(const SuperKernelScopeSplitter&) = delete;
    SuperKernelScopeSplitter& operator=(const SuperKernelScopeSplitter&) = delete;
    SuperKernelScopeSplitter(SuperKernelScopeSplitter&&) = default;
    SuperKernelScopeSplitter& operator=(SuperKernelScopeSplitter&&) = default;
    bool SplitSingleStreamGraph();
    std::vector<SuperKernelScopeInfo>& GetScopeInfos() noexcept { return scopeInfos; }

private:

    uint64_t FindAvailableHeadNode(uint64_t curNodeIdx) const;
    uint64_t GenerateSingleStreamScopeInfosByNodeIdx(uint64_t curNodeIdx);

    SuperKernelGraph& graph;
    std::vector<SuperKernelScopeInfo> scopeInfos;
};