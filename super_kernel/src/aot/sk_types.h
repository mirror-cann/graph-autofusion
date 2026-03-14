/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_TYPES_H__
#define __SK_TYPES_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <cstring>
#include <limits>
#include <vector>
#include <memory>

#include "securec.h"
#include "acl/acl.h"
#include "sk_common.h"
#include "sk_log.h"

constexpr uint32_t K_TYPE_AICORE = 1;
constexpr uint32_t K_TYPE_AIC = 2;
constexpr uint32_t K_TYPE_AIV = 3;
constexpr uint32_t K_TYPE_MIX_AIC_MAIN = 4;
constexpr uint32_t K_TYPE_MIX_AIV_MAIN = 5;
constexpr uint32_t K_TYPE_AIC_ROLLBACK = 6;
constexpr uint32_t K_TYPE_AIV_ROLLBACK = 7;

// Super Kernel Configuration Constants
constexpr uint32_t DEFAULT_COUNTER_COUNT = 75; // 默认计数器数量（对应平台AIC/AIV核心数）
constexpr uint32_t TASK_QUE_EXPAND_FACTOR = 2; // TaskQue扩容因子

constexpr uint32_t MAX_TASK_NUM = 1000;
constexpr uint32_t MAX_SCOPE_NUM = 64;
constexpr uint32_t INVALID_SCOPE_ID = std::numeric_limits<uint32_t>::max();
constexpr uint32_t INVALID_STREAM_ID = std::numeric_limits<uint32_t>::max();
constexpr uint64_t INVALID_TASK_ID = std::numeric_limits<uint64_t>::max();
inline size_t GetTaskQueSize(const TaskQue* que)
{
    if (que == nullptr) {
        return 0;
    }
    return sizeof(TaskQue) + que->taskCnt * sizeof(TaskInfo);
}

class TaskQuePtr {
    std::unique_ptr<uint8_t[]> data_;
    TaskQue* taskQue_;
    size_t capacity_;

public:
    TaskQuePtr(size_t cap)
    {
        capacity_ = 0;
        taskQue_ = nullptr;
        if (cap > (std::numeric_limits<size_t>::max() - sizeof(TaskQue)) / sizeof(TaskInfo)) {
            SK_LOGE("TaskQuePtr size overflow: requested cap=%zu, max size=%zu, sizeof(TaskInfo)=%zu", cap,
                    std::numeric_limits<size_t>::max(), sizeof(TaskInfo));
            return;
        }
        size_t size = sizeof(TaskQue) + sizeof(TaskInfo) * cap;
        data_ = std::make_unique<uint8_t[]>(size);
        taskQue_ = reinterpret_cast<TaskQue*>(data_.get());
        taskQue_->cap = static_cast<uint32_t>(cap);
        taskQue_->taskCnt = 0;
        taskQue_->fftsAddr = 0;
        capacity_ = cap;
    }

    void expand()
    {
        if (taskQue_ == nullptr) {
            return;
        }

        size_t extendSize = static_cast<size_t>(taskQue_->cap) * static_cast<size_t>(TASK_QUE_EXPAND_FACTOR);
        if (extendSize <= taskQue_->cap || extendSize > std::numeric_limits<uint32_t>::max()) {
            SK_LOGE("TaskQuePtr expand cap overflow: current cap=%u, expansion factor=%u, max uint32=%u", taskQue_->cap,
                    TASK_QUE_EXPAND_FACTOR, std::numeric_limits<uint32_t>::max());
            return;
        }
        if (extendSize > (std::numeric_limits<size_t>::max() - sizeof(TaskQue)) / sizeof(TaskInfo)) {
            SK_LOGE("TaskQuePtr expand size overflow: extendSize=%zu, max size=%zu, sizeof(TaskInfo)=%zu", extendSize,
                    std::numeric_limits<size_t>::max(), sizeof(TaskInfo));
            return;
        }
        size_t newSize = sizeof(TaskQue) + sizeof(TaskInfo) * extendSize;

        std::unique_ptr<uint8_t[]> newData = std::make_unique<uint8_t[]>(newSize);

        TaskQue* newTaskQue = reinterpret_cast<TaskQue*>(newData.get());

        size_t oldSize = sizeof(TaskQue) + sizeof(TaskInfo) * taskQue_->cap;
        errno_t err = memcpy_s(newData.get(), newSize, data_.get(), oldSize);
        if (err != 0) {
            SK_LOGE("TaskQuePtr expand memcpy failed: errno=%d, srcSize=%zu, dstSize=%zu", err, oldSize, newSize);
            return;
        }

        newTaskQue->cap = static_cast<uint32_t>(extendSize);
        data_ = std::move(newData);
        taskQue_ = newTaskQue;
        capacity_ = extendSize;
    }

    TaskQue* get() const
    {
        return taskQue_;
    }

    TaskQuePtr() noexcept : data_(nullptr), taskQue_(nullptr), capacity_(0) {}

    TaskQuePtr(TaskQuePtr&& other) noexcept :
        data_(std::move(other.data_)), taskQue_(other.taskQue_), capacity_(other.capacity_)
    {
        other.taskQue_ = nullptr;
        other.capacity_ = 0;
    }

    TaskQuePtr& operator=(TaskQuePtr&& other) noexcept
    {
        if (this != &other) {
            data_ = std::move(other.data_);
            taskQue_ = other.taskQue_;
            capacity_ = other.capacity_;
            other.taskQue_ = nullptr;
            other.capacity_ = 0;
        }
        return *this;
    }

    TaskQuePtr(const TaskQuePtr&) = delete;
    TaskQuePtr& operator=(const TaskQuePtr&) = delete;
};

class DeviceArgsPtr {
    std::unique_ptr<uint8_t[]> data_;
    SkDeviceEntryArgs* args_;

public:
    DeviceArgsPtr(size_t size)
    {
        args_ = nullptr;
        if (size == 0) {
            return;
        }
        data_ = std::make_unique<uint8_t[]>(size);
        args_ = reinterpret_cast<SkDeviceEntryArgs*>(data_.get());
    }

    SkDeviceEntryArgs* get() const
    {
        return args_;
    }

    DeviceArgsPtr() noexcept : data_(nullptr), args_(nullptr) {}

    DeviceArgsPtr(DeviceArgsPtr&& other) noexcept : data_(std::move(other.data_)), args_(other.args_)
    {
        other.args_ = nullptr;
    }

    DeviceArgsPtr& operator=(DeviceArgsPtr&& other) noexcept
    {
        if (this != &other) {
            data_ = std::move(other.data_);
            args_ = other.args_;
            other.args_ = nullptr;
        }
        return *this;
    }

    DeviceArgsPtr(const DeviceArgsPtr&) = delete;
    DeviceArgsPtr& operator=(const DeviceArgsPtr&) = delete;
};

struct SkTask {
    uint32_t numBlocks;
    TaskQuePtr taskQue;
    uint32_t funcCnt;
    SkKernelType nodeType = SkKernelType::DEFAULT;

    SkTask() : numBlocks(0), funcCnt(0){};
    SkTask(const SkTask&) = delete;
    SkTask& operator=(const SkTask&) = delete;
    SkTask(SkTask&&) = default;
    SkTask& operator=(SkTask&&) = default;
};

struct SkHostEntryInfo {
    uint32_t numBlocks;
    uint32_t nodeCnt;
    const char* skEntryFuncName;

    SkHostEntryInfo() : numBlocks(0), nodeCnt(0), skEntryFuncName(nullptr) {}

    SkHostEntryInfo(const SkHostEntryInfo&) = delete;
    SkHostEntryInfo& operator=(const SkHostEntryInfo&) = delete;

    SkHostEntryInfo(SkHostEntryInfo&& other) noexcept :
        numBlocks(other.numBlocks), nodeCnt(other.nodeCnt), skEntryFuncName(other.skEntryFuncName)
    {
        other.skEntryFuncName = nullptr;
        other.numBlocks = 0;
        other.nodeCnt = 0;
    }

    SkHostEntryInfo& operator=(SkHostEntryInfo&& other) noexcept
    {
        if (this != &other) {
            numBlocks = other.numBlocks;
            nodeCnt = other.nodeCnt;
            skEntryFuncName = other.skEntryFuncName;
            other.skEntryFuncName = nullptr;
            other.numBlocks = 0;
            other.nodeCnt = 0;
        }
        return *this;
    }

    ~SkHostEntryInfo() = default;
};

struct SkLaunchInfo {
    SkHostEntryInfo entryInfo;
    DeviceArgsPtr devArgs;
};

#endif // __SK_TYPES_H__
