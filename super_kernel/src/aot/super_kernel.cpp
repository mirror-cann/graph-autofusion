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
#include "sk_dfx_exception_handler.h"
#include "sk_lock_detector.h"
#include "sk_resource_manager.h"
#include "sk_scope_launch.h"

namespace {
class CurrentModelGuard {
public:
    explicit CurrentModelGuard(aclmdlRI model)
    {
        SkResourceManager::SetCurrentModel(model);
    }

    ~CurrentModelGuard()
    {
        SkResourceManager::SetCurrentModel(nullptr);
    }
};
}

#ifdef __cplusplus
extern "C" {
#endif

aclError aclskOptimize(aclmdlRI model, aclskOptions *options) {
    CurrentModelGuard modelGuard(model);
    aclError ret = aclrtSetExceptionInfoCallback(SuperKernelExceptionCallBackFunc);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to set exception callback.");
        return ACL_ERROR_FAILURE;
    }

    SK_LOGI("Begin aclskOptimize");
    SK_LOGI("Start init sk graph...");
    SuperKernelGraph graph(model);
    if (!graph.InitSKGraph()) {
        return ACL_ERROR_FAILURE;
    }
    ret = LockDetector::GetDeviceCores();
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    SK_LOGI("End init sk graph");

    SK_LOGI("Start parse sk options...");
    SuperKernelOptionsManager opts;
    opts.ParseOptions(options);
    SK_LOGI("End parse sk options");

    SK_LOGI("Start optimize sk graph...");
    SuperKernelOptimizer optimizer(opts);
    if (!optimizer.Process(graph)) {
        SK_LOGE("aclskOptimize failed: optimize sk graph failed");
        return ACL_ERROR_FAILURE;
    }
    SK_LOGI("End optimize sk graph");

    SK_LOGI("Start update graph...");
    ret = graph.Update();
    SK_LOGI("End update graph");

    SK_LOGI("End aclskOptimize");
    return ret;
}

aclError aclskScopeBegin(const char* scopeName, aclrtStream stream) {
    if (scopeName != nullptr && scopeName[0] == '\0') {
        SK_LOGE("Invalid scopeName: name is empty.");
        return ACL_ERROR_INVALID_PARAM;
    }
    return LaunchScopeKernel(scopeName, stream, true);
}

aclError aclskScopeEnd(const char* scopeName, aclrtStream stream) {
    if (scopeName != nullptr && scopeName[0] == '\0') {
        SK_LOGE("Invalid scopeName: name is empty.");
        return ACL_ERROR_INVALID_PARAM;
    }
    return LaunchScopeKernel(scopeName, stream, false);
}

#ifdef __cplusplus
}
#endif
