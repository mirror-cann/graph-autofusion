/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include "sk_types.h"
#include "sk_options_manager.h"
#include "sk_scope_split.h"
#include "sk_graph.h"

/**
 * @brief Super Kernel 优化器 - 核心编排类 (Facade Pattern)
 * 
 * 职责：
 * 1. 解析运行时选项为 OptimizeOptions
 * 2. 从模型中捕获任务信息
 * 3. 使用 SkTaskBuilder 构建优化任务
 * 4. 使用 KernelLauncher 启动内核
 * 
 * 设计：
 * - 直接调用 KernelLauncher（只保留 V2）
 */
class SuperKernelOptimizer {
public:
    SuperKernelOptimizer(SuperKernelOptionsManager &opts) : opts(opts) {}
    virtual ~SuperKernelOptimizer() = default;
    void Process(const SuperKernelGraph &graph);

private:
    SuperKernelOptionsManager &opts;
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>> SplitGraph(const SuperKernelGraph &graph) const;
    std::vector<SuperKernelBaseNode*> SerializeSubGraph(const SuperKernelGraph &graph,
                                                     const std::vector<uint64_t> &headIds,
                                                     const std::vector<uint64_t> &tailIds) const;
    void Schedule(const SuperKernelScopeInfo &scopeInfo, const SuperKernelGraph &graph);
    static void DumpLaunchDebug(const SkHostEntryInfo &entryInfo, const SkDeviceEntryArgs *skEntryArgs);
};
