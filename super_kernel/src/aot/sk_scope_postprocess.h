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
 * \file sk_scope_postprocess.h
 * \brief
 */

#ifndef __SK_SCOPE_POSTPROCESS_H__
#define __SK_SCOPE_POSTPROCESS_H__

#include "sk_graph.h"
#include "sk_scope_split.h"
#include <memory>
#include <vector>
#include <deque>
#include <utility>

struct UpdateStreamInfo {
    uint32_t streamIdx = 0;
    uint64_t headNodeIdx = INVALID_TASK_ID;
    uint64_t tailNodeIdx = INVALID_TASK_ID;
    uint64_t nodeSize = 0;
    // Feature(aclmdIRITaskParams): Replace aclrtTaskEventParams with aclmdIRITaskParams
    // once event memory allocation is switched from runtime task APIs to IR-task APIs.
    std::vector<aclrtTaskEventParams> customParams; // Feature : feature use aclmdIRITaskParams
};

struct SuperKernelProcessedScopeInfo {
    std::vector<UpdateStreamInfo> updateStreamInfos;
    std::vector<SuperKernelBaseNode*> nodes;
    // Own synthesized event nodes created in post-process.
    std::vector<std::unique_ptr<SuperKernelBaseNode>> eventNodes;
    uint64_t skMainNodeId = INVALID_TASK_ID;
};

class SuperKernelScopePostProcessor {
public:
    SuperKernelScopePostProcessor() = default;
    SuperKernelScopePostProcessor(SuperKernelGraph& graph) : graph(graph) {}
    ~SuperKernelScopePostProcessor() = default;
    SuperKernelProcessedScopeInfo PostProcess(SuperKernelScopeInfo& scopeInfo);

private:
    SuperKernelGraph& graph;
};

#endif // __SK_SCOPE_POSTPROCESS_H__