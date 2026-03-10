/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "super_kernel.h"
#include "sk_log.h"
#include "sk_options_manager.h"
#include "sk_optimizer.h"
#include "sk_graph.h"
#include "sk_lock_detector.h"
#include "sk_scope_launch.h"

#ifdef __cplusplus
extern "C" {
#endif

aclError aclskOptimize(aclmdlRI model, aclskOptions *options) {

    SK_LOGI("Begin aclskOptimize");

    // 使用默认策略（通过编译配置自动选择 V1 或 V2）
    SuperKernelGraph graph(model);
    if (!graph.InitSKGraph()) {
        return ACL_ERROR_FAILURE;
    }
    if (!LockDetector::GetDeviceCores()) {
        return ACL_ERROR_FAILURE;
    }

    SuperKernelOptionsManager opts;
    opts.ParseOptions(options);

    // 使用新的 SuperKernelOptimizer（支持切图和序列化）
    SuperKernelOptimizer optimizer(opts);
    optimizer.Process(graph);
    graph.Update();
    SK_LOGI("End aclskOptimize");
    
    return ACL_SUCCESS;
}


aclError aclskScopeBegin(const char* scopeName, aclrtStream stream) {
    if (scopeName[0] == '\0') {
        SK_LOGE("Invalid scopeName: name is empty.");
        return ACL_ERROR_INVALID_PARAM;
    }
    return LaunchScopeKernel(scopeName, stream, true);
}

aclError aclskScopeEnd(const char* scopeName, aclrtStream stream) {
    if (scopeName[0] == '\0') {
        SK_LOGE("Invalid scopeName: name is empty.");
        return ACL_ERROR_INVALID_PARAM;
    }
    return LaunchScopeKernel(scopeName, stream, false);
}

#ifdef __cplusplus
}
#endif
