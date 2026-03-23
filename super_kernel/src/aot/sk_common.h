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
#include <string>
#include <cstddef>
enum class SkNodeType : uint32_t {
    NODE_KERNEL = 0,
    NODE_NOTIFY = 1,
    NODE_WAIT = 2,
    NODE_RESET = 3,
    NODE_MEMORY_WRITE = 4,
    NODE_MEMORY_WAIT = 5,
    NODE_DEFAULT = 6,
};

inline const char* to_string(SkNodeType type)
{
    switch (type) {
    case SkNodeType::NODE_KERNEL:
        return "KERNEL";
    case SkNodeType::NODE_NOTIFY:
        return "NOTIFY";
    case SkNodeType::NODE_WAIT:
        return "WAIT";
    case SkNodeType::NODE_RESET:
        return "RESET";
    case SkNodeType::NODE_DEFAULT:
        return "DEFAULT";
    default:
        return "UNKNOWN";
    }
}

enum class SkKernelType : uint8_t {
    AIC_ONLY = 1,
    AIV_ONLY = 2,
    MIX_AIV_1_0 = 3,
    MIX_AIC_1_0 = 4,
    MIX_AIC_1_1 = 5,
    MIX_AIC_1_2 = 6,
    DEFAULT = 0xFF,
};

constexpr size_t SK_KERNEL_TYPE_COUNT = 6;

inline const char* to_string(SkKernelType type)
{
    switch (type) {
    case SkKernelType::AIC_ONLY:
        return "AIC_ONLY";
    case SkKernelType::AIV_ONLY:
        return "AIV_ONLY";
    case SkKernelType::MIX_AIV_1_0:
        return "MIX_AIV_1_0";
    case SkKernelType::MIX_AIC_1_0:
        return "MIX_AIC_1_0";
    case SkKernelType::MIX_AIC_1_1:
        return "MIX_1_1";
    case SkKernelType::MIX_AIC_1_2:
        return "MIX_1_2";
    case SkKernelType::DEFAULT:
        return "DEFAULT";
    default:
        return "UNKNOWN";
    }
}

enum class SkTaskType : uint8_t {
    TYPE_FUNC,
    TYPE_SYNC,
    TYPE_PRELOAD,
    TYPE_EVENT_NOTIFY,
    TYPE_EVENT_WAIT,
    TYPE_EVENT_RESET,
    TYPE_MAX,
};

inline const char* to_string(SkTaskType type)
{
    switch (type) {
    case SkTaskType::TYPE_FUNC:
        return "FUNC";
    case SkTaskType::TYPE_SYNC:
        return "SYNC";
    case SkTaskType::TYPE_PRELOAD:
        return "PRELOAD";
    case SkTaskType::TYPE_EVENT_NOTIFY:
        return "EVENT_NOTIFY";
    case SkTaskType::TYPE_EVENT_WAIT:
        return "EVENT_WAIT";
    case SkTaskType::TYPE_EVENT_RESET:
        return "EVENT_RESET";
    default:
        return "UNKNOWN";
    }
}

enum class SkCoreSyncType : uint8_t {
    ALL_SYNC = 0,
    CROSS_SYNC_AIC_TO_AIC,
    CROSS_SYNC_AIV_TO_AIV,
    INTER_SYNC_SET_AIC_TO_AIV,
    INTER_SYNC_SET_AIV_TO_AIC,
    INTER_SYNC_WAIT_AIC_TO_AIV,
    INTER_SYNC_WAIT_AIV_TO_AIC,
    SYNC_NONE,
};

inline const char* to_string(SkCoreSyncType type)
{
    switch (type) {
    case SkCoreSyncType::ALL_SYNC:
        return "ALL_SYNC";
    case SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC:
        return "AIC_TO_AIC";
    case SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV:
        return "AIV_TO_AIV";
    case SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV:
        return "SET_AIC_TO_AIV";
    case SkCoreSyncType::INTER_SYNC_SET_AIV_TO_AIC:
        return "SET_AIV_TO_AIC";
    case SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV:
        return "WAIT_AIC_TO_AIV";
    case SkCoreSyncType::INTER_SYNC_WAIT_AIV_TO_AIC:
        return "WAIT_AIV_TO_AIC";
    case SkCoreSyncType::SYNC_NONE:
        return "SYNC_NONE";
    default:
        return "UNKNOWN";
    }
}

struct TaskInfo {
    uint32_t index;
    SkTaskType type;
    SkKernelType originType = SkKernelType::DEFAULT;
    uint8_t numBlocks;
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
    TaskInfo taskInfos[0];
};

struct SkHeaderInfo {
    uint32_t aicQueOffset;
    uint32_t aivQueOffset;
    uint32_t counterOffset;
    uint32_t wsOffset;
    uint32_t dfxOffset;
    uint32_t eventConfigOffset;  // 算子打印事件配置偏移量
    uint32_t nodeCnt;
    uint64_t totalSize;
};

struct SkCounterInfo {
    uint32_t index;
    uint8_t launch; // todo：似乎用1个变量就够了， 1表示launch，0表示complete or not launch
    uint8_t exit;
    // dcci single cacheline size is 64bytes
    uint8_t reserve[58];
};

struct SkWorkSpace {
    uint32_t workspace[8 * 1024];
};

struct SkDfxInfo {
    uint64_t binHdl;
    uint64_t funcHdlOri;
};

struct SkDeviceEntryArgs {
    SkHeaderInfo skHeader;
    uint8_t data[0];
};
// ==================== 事件记录相关结构体 ====================
// Kernel 侧的时间记录结构体
struct SkKernelEventRecord {
    uint64_t modelRI;     // modelRI 标识
    uint32_t skId;        // SK 标识
    uint32_t nodeId;      // 算子节点 ID
    uint8_t blockIdx;      // block 索引
    uint8_t blockNum;
    uint64_t startTime;   // 开始时间戳
    uint64_t endTime;     // 结束时间戳
};


// 每个 core 的缓冲区头部（Kernel 侧）
struct SkKernelEventCoreBuf {
    uint32_t offset;  // 当前写入偏移
    uint32_t reserved;         // 保留字段
};

// 事件记录配置信息（放在 SkHeaderInfo 的 dfxOffset 位置）
struct SkEventConfig {
    uint64_t eventGmAddr;   // 事件记录 GM 基地址
    uint64_t modelRI;       // modelRI 标识
    uint32_t skId;          // SK 标识
    uint8_t enabled;       // 是否启用
    uint32_t reserved;      // 保留字段
};

bool GetFuncSymbolInfo(const char* binAddr, size_t binSize, uint64_t funcAddr, std::string& symbolName,
                       uint64_t& funcSize);

namespace sk {
/*
ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_AIC : 0b00;
ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_AIV : 0b01;
ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_MIX : 0b10;
ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_AIC : 0b0100;
ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_AIV : 0b0101;
ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_MIX : 0b0110;
ASCENDC_SUPER_KERNEL_EARLY_START_MIX_TO_AIC : 0b1000;
ASCENDC_SUPER_KERNEL_EARLY_START_MIX_TO_AIV : 0b1001;
ASCENDC_SUPER_KERNEL_EARLY_START_MIX_TO_MIX : 0b1010;
*/

constexpr uint16_t SYNC_COMBINATION_TABLE[SK_KERNEL_TYPE_COUNT][SK_KERNEL_TYPE_COUNT] = {
//                  AIC_ONLY | AIV_ONLY | MIX_AIV_1_0 | MIX_AIC_1_0 | MIX_AIC_1_1 | MIX_AIC_1_2
/* AIC_ONLY */    { 0b00,      0b01,      0b10,         0b10,         0b10,         0b10   },
/* AIV_ONLY */    { 0b0100,    0b0101,    0b0110,       0b0110,       0b0110,       0b0110 },
/* MIX_AIV_1_0 */ { 0b1000,    0b1001,    0b1010,       0b1010,       0b1010,       0b1010 },
/* MIX_AIC_1_0 */ { 0b1000,    0b1001,    0b1010,       0b1010,       0b1010,       0b1010 },
/* MIX_AIC_1_1 */ { 0b1000,    0b1001,    0b1010,       0b1010,       0b1010,       0b1010 },
/* MIX_AIC_1_2 */ { 0b1000,    0b1001,    0b1010,       0b1010,       0b1010,       0b1010 },
};

constexpr uint16_t INVALID_SYNC_COMBINATION = 0xFFFF;
} // namespace sk

#endif
