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

// 调度执行任务流节点
void SuperKernelOptimizer::Schedule(const SuperKernelScopeInfo &scopeInfo,
                                    const SuperKernelGraph &graph) {
    auto taskNodes = scopeInfo.nodes;
    if (taskNodes.empty()) {
        SK_LOGW("no tasks for super kernel optimization: scope has 0 nodes");
        return;
    }

    SK_LOGI("total task count for optimization: %zu", taskNodes.size());

    SkTaskBuilder builder(opts, graph);
    SkLaunchInfo launchInfo = builder.Build(taskNodes);

    aclrtFuncHandle skEntryFunc = ResolveSkEntryFunc(launchInfo.entryInfo.skEntryFuncName);
    if (skEntryFunc == nullptr) {
        SK_LOGE("failed to resolve sk entry function: entryFuncName=%s", launchInfo.entryInfo.skEntryFuncName);
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
        SK_LOGI("graph split into %zu scopes", splitter.GetScopeInfos().size());
    } else {
        SK_LOGW("graph split failed or no scopes found: cannot proceed with super kernel optimization");
        return;
    }

    auto scopeInfos = splitter.GetScopeInfos();

    // 逐个处理每个子图
    for (auto scopeInfo : scopeInfos) {
        Schedule(scopeInfo, graph);
    }
}
