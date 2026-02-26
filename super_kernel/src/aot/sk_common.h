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
 * \file sk_common.h
 * \brief
 */

#ifndef SK_COMMON_H
#define SK_COMMON_H

enum class SkNodeType: uint32_t {
    NODE_KERNEL = 0,
    NODE_NOTIFY = 1,
    NODE_WAIT = 2,
    NODE_RESET = 3,
    NODE_DEFAULT = 4,
};

enum class SkKernelType: uint32_t {
    AIC_ONLY = 1,
    AIV_ONLY = 2,
    MIX_AIV_1_0 = 3,
    MIX_AIC_1_0 = 4,
    MIX_AIC_1_1 = 5,
    MIX_AIC_1_2 = 6,
    DEFAULT = 0xFFFFFFFF,
};

enum class SkTaskType : uint8_t {
    TYPE_FUNC,
    TYPE_SYNC,
    TYPE_PRELOAD,
    TYPE_EVENT_NOTIFY,  // notify <---> wait
    TYPE_EVENT_WAIT,    // notify <---> wait
    TYPE_MAX = 0xFF,
};

enum class SkCoreSyncType : uint8_t {
    ALL_SYNC_DEBUG = 0,
    CROSS_SYNC_AIC_TO_AIC,
    CROSS_SYNC_AIV_TO_AIV,
    INTER_SYNC_SET_AIC_TO_AIV,
    INTER_SYNC_SET_AIV_TO_AIC,
    INTER_SYNC_WAIT_AIC_TO_AIV,
    INTER_SYNC_WAIT_AIV_TO_AIC,
    SYNC_NONE,
};



struct TaskInfo {
    uint32_t index;
    SkTaskType type;
    SkKernelType originType = SkKernelType::DEFAULT;
    uint8_t blocks;
    uint8_t entryCnt;
    uint64_t args;
    uint64_t entry[4];
    // 根据bit位确定option选项
    // 1： debug_dcci_disable_on_kernel
    // 2： debug_sync_all
    uint64_t debugOptions;
};

struct TaskQue {
    uint32_t taskCnt;
    uint32_t cap;
    uint64_t fftsAddr;
    TaskInfo taskInfos[0];
};

struct SkHeaderInfo {
    uint32_t aicQueOffset;
    uint32_t aivQueOffset;
    uint32_t counterOffset;
    uint32_t wsOffset;
    uint32_t dfxOffset;
    uint32_t nodeCnt;
    uint64_t totalSize;
};

struct SkCounterInfo {
    uint32_t index;
    uint8_t launch; // todo：似乎用1个变量就够了， 1表示launch，0表示complete or not launch
    uint8_t exit;
    uint8_t reserve[2];
};

struct SkWorkSpace {
    uint32_t workspace[8 * 1024];
};

struct SkDfxInfo {
    uint64_t binHdl;
    uint64_t funcHdl;
    uint64_t funcHdlOri;
};

struct SkDeviceEntryArgs {
    SkHeaderInfo skHeader;
    uint8_t data[0];
};
#endif
