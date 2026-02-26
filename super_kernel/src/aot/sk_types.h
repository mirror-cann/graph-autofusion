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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <cstring>
#include <vector>

#include "securec.h"
#include "acl/acl.h"
#include "sk_common.h"

constexpr uint32_t K_TYPE_AICORE = 1;
constexpr uint32_t K_TYPE_AIC = 2;
constexpr uint32_t K_TYPE_AIV = 3;
constexpr uint32_t K_TYPE_MIX_AIC_MAIN = 4;
constexpr uint32_t K_TYPE_MIX_AIV_MAIN = 5;
constexpr uint32_t K_TYPE_AIC_ROLLBACK = 6;
constexpr uint32_t K_TYPE_AIV_ROLLBACK = 7;

// Super Kernel Configuration Constants
constexpr uint32_t DEFAULT_COUNTER_COUNT = 75;       // 默认计数器数量（对应平台AIC/AIV核心数）
constexpr uint32_t TASK_QUE_EXPAND_FACTOR = 2;       // TaskQue扩容因子

constexpr uint32_t MAX_TASK_NUM = 1000;
constexpr uint64_t INVALID_TASK_ID = 0xFFFFFFFFFFFFFFFF;

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

inline size_t GetTaskQueSize(const TaskQue *que) {
    return sizeof(TaskQue) + que->taskCnt * sizeof(TaskInfo);
}

struct SkTask {
    uint32_t blockDim;
    TaskQue *taskQue;
    uint32_t funcCnt = 0;
    SkKernelType nodeType = SkKernelType::DEFAULT;

    SkTask()
        : blockDim(0), taskQue(nullptr), funcCnt(0), nodeType(SkKernelType::DEFAULT) {}

    SkTask(const SkTask &) = delete;
    SkTask &operator=(const SkTask &) = delete;

    SkTask(SkTask &&other) noexcept
        : blockDim(other.blockDim),
          taskQue(other.taskQue),
          funcCnt(other.funcCnt),
          nodeType(other.nodeType) {
        other.taskQue = nullptr;
        other.blockDim = 0;
        other.funcCnt = 0;
        other.nodeType = SkKernelType::DEFAULT;
    }

    SkTask &operator=(SkTask &&other) noexcept {
        if (this != &other) {
            this->~SkTask();
            blockDim = other.blockDim;
            taskQue = other.taskQue;
            funcCnt = other.funcCnt;
            nodeType = other.nodeType;
            other.taskQue = nullptr;
            other.blockDim = 0;
            other.funcCnt = 0;
            other.nodeType = SkKernelType::DEFAULT;
        }
        return *this;
    }

    ~SkTask() {
        TaskQue *taskQue = this->taskQue;
        if (taskQue) {
            taskQue->taskCnt = 0;
            free(taskQue);
        }
    }
};

struct SkHostEntryInfo {
    uint32_t blockDim;
    uint32_t nodeCnt;
    const char *funcName;
    SkDfxInfo *dfxInfos;

    SkHostEntryInfo()
        : blockDim(0), nodeCnt(0), funcName(nullptr), dfxInfos(nullptr) {}

    SkHostEntryInfo(const SkHostEntryInfo &) = delete;
    SkHostEntryInfo &operator=(const SkHostEntryInfo &) = delete;

    SkHostEntryInfo(SkHostEntryInfo &&other) noexcept
        : blockDim(other.blockDim),
          nodeCnt(other.nodeCnt),
          funcName(other.funcName),
          dfxInfos(other.dfxInfos) {
        other.funcName = nullptr;
        other.dfxInfos = nullptr;
        other.blockDim = 0;
        other.nodeCnt = 0;
    }

    SkHostEntryInfo &operator=(SkHostEntryInfo &&other) noexcept {
        if (this != &other) {
            this->~SkHostEntryInfo();
            blockDim = other.blockDim;
            nodeCnt = other.nodeCnt;
            funcName = other.funcName;
            dfxInfos = other.dfxInfos;
            other.funcName = nullptr;
            other.dfxInfos = nullptr;
            other.blockDim = 0;
            other.nodeCnt = 0;
        }
        return *this;
    }

    ~SkHostEntryInfo() {
        if (this->dfxInfos) {
            free(this->dfxInfos);
            this->dfxInfos = nullptr;
        }
    }
};

struct BuildResult {
    SkTask aicTask;
    SkTask aivTask;
    SkHostEntryInfo entryInfo;
    SkDeviceEntryArgs *devArgs;
};
