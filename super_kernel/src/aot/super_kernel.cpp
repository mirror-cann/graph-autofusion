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

aclError aclskOptimize(aclmdlRI model, aclskOptions *options) {

    SK_LOGI("enter aclskOptimize");
#if 0
    aclskPreloadOption preloadingMode = {1};
    aclskSplitModeOption splitMode = {2};
    char *kernelName[3] = {"test1", "test2", "test3"};
    aclskDcciOption disableDcciOption;
    disableDcciOption.kernelNames = (char**)malloc(3 * sizeof(char*));
    for (int i = 0; i < 3; i++) {
        disableDcciOption.kernelNames[i] = kernelName[i];
    }
    disableDcciOption.kernelCnt = 3;

    aclskDebugSyncAllOption debugSync = {0};

    aclskOption tmp;
    tmp.splitMode = splitMode;
    tmp.optionType = aclskOtionType::SPLIT_MODE;

    aclskOption tmp1;
    tmp1.preload = preloadingMode;
    tmp1.optionType = aclskOtionType::PRELOAD_CODE;

    aclskOption tmp2;
    tmp2.disableKernelDcci = disableDcciOption;
    tmp2.optionType = aclskOtionType::DEBUG_DCCI_DISABLE_ON_KERNEL;

    aclskOption tmp3;
    tmp3.debugSync = debugSync;
    tmp3.optionType = aclskOtionType::DEBUG_SYNC_ALL;


    aclskOption subOptions[5] = {tmp, tmp1, tmp2, tmp3};
    aclskOptions options_test = {subOptions, 5};

    SuperKernelOptionsManager opts_test;
    opts_test.ParseOptions(&options_test);
#endif

    // 使用默认策略（通过编译配置自动选择 V1 或 V2）
    SuperKernelGraph graph(model);
    if (!graph.InitSKGraph()) {
        return ACL_ERROR_FAILURE;
    }

    SuperKernelOptionsManager opts;
    opts.ParseOptions(options);

    // 使用新的 SuperKernelOptimizer（支持切图和序列化）
    SuperKernelOptimizer optimizer(opts);
    optimizer.Process(graph);
    return ACL_SUCCESS;
}
