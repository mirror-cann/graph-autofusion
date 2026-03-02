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
 * \file superkernel_opt_options.h
 * \brief
 */

#ifndef ACL_SUPERKERNEL_H
#define ACL_SUPERKERNEL_H

#include <cstdint>
#include <cstddef>
#include "acl/acl.h"

enum class aclskOtionType : uint32_t {
    PRELOAD_CODE = 0,
    SPLIT_MODE = 1,
    STREAM_FUSION = 2,
    DEBUG_DCCI_DISABLE_ON_KERNEL = 3,
    DEBUG_SYNC_ALL = 4,
    KERNEL_MAP = 5,
    SK_OPTION_MAX = 0xFFFFFFFF
};

typedef struct aclskPreloadOption {
    uint32_t preloadMode;
} aclskPreloadOption;

typedef struct aclskSplitModeOption {
    uint32_t splitCnt;
} aclskSplitModeOption;

typedef struct aclskStreamFusionOption {
    uint32_t streamFusion;
} aclskStreamFusionOption;

typedef struct aclskDcciOption {
    char** kernelNames;
    size_t kernelCnt;
} aclskDcciOption;

typedef struct aclskDebugSyncAllOption {
    uint32_t debugSyncAll;
} aclskDebugSyncAllOption;

typedef struct aclskKernelMap {
    char* globalName;
    char* sknlNames[4];
} aclskKernelMap;

typedef struct aclskKernelMapOption {
    aclskKernelMap* kernelMaps;
    size_t numKernels;
} aclskKernelMapOption;

struct aclskOption {
    aclskOtionType optionType;
    union {
        aclskPreloadOption preload;
        aclskSplitModeOption splitMode;
        aclskStreamFusionOption streamFusion;
        aclskDcciOption disableKernelDcci;
        aclskDebugSyncAllOption debugSync;
        aclskKernelMapOption kernelMap;
    };
};

typedef struct aclskOptions {
    aclskOption *options;
    size_t numOptions;
} aclskOptions;

/*
model -- aclGraph capture model
options -- super kernel optimization options
return -- ACL_ERROR_NONE success, other value fail
*/
aclError aclskOptimize(aclmdlRI model, aclskOptions *options);

#endif
