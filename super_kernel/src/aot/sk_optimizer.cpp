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
#include "sk_task_builder.h"
#include "sk_log.h"

extern "C" aclrtBinHandle AscendGetEntryBinHandle();
namespace {

aclrtFuncHandle ResolveSkEntryFunc(const char *funcName) {
    aclrtBinHandle bhdl = nullptr;
    bhdl = AscendGetEntryBinHandle();
    if (bhdl == nullptr) {
        SK_LOGE("[sk error] failed to get entry bin handle");
        return nullptr;
    }

    aclrtFuncHandle fhdl = nullptr;
    CHECK_ACL(aclrtBinaryGetFunction(bhdl, funcName, &fhdl));
    if (fhdl == nullptr) {
        SK_LOGE("[sk error] failed to resolve entry func handle");
        return nullptr;
    }
    return fhdl;
}

} // namespace

// 调度执行任务流节点
void SuperKernelOptimizer::Schedule(const SuperKernelScopeInfo &scopeInfo,
                                    const SuperKernelGraph &graph) {
    auto taskNodes = scopeInfo.nodes;
    uint32_t streamIdx = scopeInfo.scopeStreamInfos[0].streamIdx;
    if (taskNodes.empty()) {
        printf("[sk warning] no task for super kernel optimization\n");
        return;
    }
    printf("[sk info] total task count : %zu, streamIdx=%u\n", taskNodes.size(), streamIdx);

    SkTaskBuilder builder(opts, graph);
    SkLaunchInfo launchInfo = builder.Build(taskNodes);

    aclrtFuncHandle skEntryFunc = ResolveSkEntryFunc(launchInfo.entryInfo.skEntryFuncName);
    if (skEntryFunc == nullptr) {
        printf("[sk error] failed to resolve sk entry function\n");
        return;
    }

    Update(scopeInfo, launchInfo, skEntryFunc);
}

void SuperKernelOptimizer::Update(const SuperKernelScopeInfo &scopeInfo,
                                   const SkLaunchInfo &launchInfo,
                                   aclrtFuncHandle skEntryFunc) {
    bool foundMain = false;
    for (auto* node : scopeInfo.nodes) {
        UpdateContext ctx;
        if (node->GetNodeType() == SkNodeType::NODE_KERNEL && !foundMain) {
            ctx.launchInfo = const_cast<SkLaunchInfo*>(&launchInfo);
            ctx.skEntryFunc = skEntryFunc;
            foundMain = true;
        }
        node->Update(ctx); // 根据不同的ctx执行不同的Update行为，顶层屏蔽细节
    }
}

void SuperKernelOptimizer::Process(SuperKernelGraph &graph) {
    // 切分图为多个子图
    SuperKernelScopeSplitter splitter(graph);
    if(splitter.SplitSingleStreamGraph()) {
        SK_LOGI("graph split into %zu scopes\n", splitter.GetScopeInfos().size());
    } else {
        SK_LOGW("graph split failed or no scopes found\n");
        return;
    }

    auto scopeInfos = splitter.GetScopeInfos();

    // 逐个处理每个子图
    for (auto scopeInfo : scopeInfos) {
        Schedule(scopeInfo, graph);
    }
}
