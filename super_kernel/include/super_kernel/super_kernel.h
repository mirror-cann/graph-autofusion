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

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY _declspec(dllexport)
#else
#define ACL_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define ACL_FUNC_VISIBILITY
#endif
#endif

enum class aclskOptionType : uint32_t {
    PRELOAD_CODE = 0,
    SPLIT_MODE = 1,
    STREAM_FUSION = 2,
    DCCI_DISABLE_ON_KERNEL = 3,
    DEBUG_SYNC_ALL = 4,
    KERNEL_MAP = 5,
    CONSTANT_CODEGEN = 6,  // 常量化代码生成选项
    AUTO_OP_PARALLEL = 7,  // 优化多流算子排布
    DCCI_BEFORE_KERNEL_START = 8,
    DEBUG_OP_EXEC_TRACE = 9,
    DEBUG_CROSS_CORE_SYNC_CHECK = 10,
    OPT_EXTEND_OPTION = 11,   // 扩展选项，预留后续使用
    DEBUG_EXTEND_OPTION = 12, // 扩展选项，预留后续使用
    DCCI_AFTER_KERNEL_END = 13,
    AGGRESSIVE_OPT_STRATEGIES = 14, // aggressive fusion strategy options
    SK_OPTION_MAX
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

typedef struct aclskAutoOpParallelOption {
    uint32_t enableAutoOpParallel;
} aclskAutoOpParallelOption;

typedef struct aclskDebugOpExecTraceOption {
    uint32_t enableOpExecTrace;
} aclskDebugOpExecTraceOption;

typedef struct aclskDebugCrossCoreSyncCheckOption {
    uint32_t enableCrossCoreSyncCheck;
} aclskDebugCrossCoreSyncCheckOption;

typedef struct aclskExtendOption {
    char* value;
} aclskExtendOption;

/**
 * aggressiveOpts carries aggressive fusion strategy switches:
 *  - eventBreakerBypass: reserved for event breaker policy
 *  - taskBreakerBypass: enable default-node bypass when set to 1
 *
 * aggressiveOpts.valueBreakerBypass is a bitmask-style value-memory wait policy:
 *  - ACLSK_VALUE_BREAKER_BYPASS_NONE (0b00): reject notify/wait pairing relation, keep wait unfusible
 *  - ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT (0b01): for paired notify+wait, existing rule must pass or SK exits
 *  - ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT (0b10): for waits without notify after rule check, allow wait fusion
 */
enum aclskValueBreakerBypassFlag : uint32_t {
    ACLSK_VALUE_BREAKER_BYPASS_NONE = 0b00U,
    ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT = 0b01U,
    ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT = 0b10U
};

typedef struct aclskAggressiveOptStrategies {
    uint32_t eventBreakerBypass;
    uint32_t valueBreakerBypass;
    uint32_t taskBreakerBypass;
    uint32_t ubufLockIgnoreKernelCnt;
    char** ubufLockIgnoreKernel;
} aclskAggressiveOptStrategies;

/**
 * 常量化代码生成选项
 * enableConstant: 1 启用常量化, 0 禁用常量化
 */
typedef struct aclskConstantCodegenOption {
    uint32_t enableConstant;
} aclskConstantCodegenOption;

struct aclskOption {
    aclskOptionType optionType;
    union {
        aclskPreloadOption preload;
        aclskSplitModeOption splitMode;
        aclskStreamFusionOption streamFusion;
        aclskDcciOption disableKernelDcci;
        aclskDebugSyncAllOption debugSync;
        aclskKernelMapOption kernelMap;
        aclskConstantCodegenOption constantCodegen;
        aclskAutoOpParallelOption autoOpParallel;
        aclskDcciOption dcciBeforeKernelStart;
        aclskDebugOpExecTraceOption debugOpExecTrace;
        aclskDebugCrossCoreSyncCheckOption debugCrossCoreSyncCheck;
        aclskExtendOption optExtend;
        aclskExtendOption debugExtend;
        aclskDcciOption dcciAfterKernelEnd;
        aclskAggressiveOptStrategies aggressiveOpts;
    };
};

typedef struct aclskOptions {
    aclskOption *options;
    size_t numOptions;
} aclskOptions;

/**
 * @ingroup AscendCL
 * @brief Optimize model with super kernel
 *
 * @param model [IN]    Model handle to be optimized
 * @param options [IN]  Pointer to optimization options
 *
 * @retval ACL_ERROR_SUCCESS Optimization succeeded
 * @retval ACL_ERROR_INVALID_PARAM Invalid parameters
 * @retval ACL_ERROR_FAILURE Optimization failed
 *
 * @see aclskOptions
 */
ACL_FUNC_VISIBILITY aclError aclskOptimize(aclmdlRI model, aclskOptions *options);
ACL_FUNC_VISIBILITY aclError aclskScopeBegin(const char* scopeName, aclrtStream stream);
ACL_FUNC_VISIBILITY aclError aclskScopeEnd(const char* scopeName, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
