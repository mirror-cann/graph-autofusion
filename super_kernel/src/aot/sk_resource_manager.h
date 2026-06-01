/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SK_RESOURCE_MANAGER_H
#define SK_RESOURCE_MANAGER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "acl/acl.h"

class SkResourceManager {
public:
    static constexpr size_t kDefaultValueMemoryBytes = sizeof(uint64_t);

    enum class ResourceKind : uint8_t {
        kDeviceMemory = 0,
    };

    struct ResourceRecord {
        ResourceKind kind = ResourceKind::kDeviceMemory;
        void* addr = nullptr;
        size_t bytes = 0U;
    };

    struct ModelDestroyContext {
        aclmdlRI model = nullptr;
        std::string modelId;
        std::string modelLabel;
        bool resourcesReleased = false;
    };

    static SkResourceManager& GetInstance();

    static void SetCurrentModel(aclmdlRI model);
    static aclError CallbackRegister(aclmdlRI model);
    static aclError ValueMemory(void** addr, size_t bytes = kDefaultValueMemoryBytes);

    SkResourceManager(const SkResourceManager&) = delete;
    SkResourceManager& operator=(const SkResourceManager&) = delete;

private:
    SkResourceManager() = default;
    ~SkResourceManager() = default;

    static std::mutex resourceMutex_;
    static std::unordered_map<std::string, std::vector<ResourceRecord>> modelResources_;
    static std::unordered_set<std::string> registeredModelLabels_;
    static std::unordered_map<std::string, std::unique_ptr<ModelDestroyContext>> destroyContexts_;
    static std::unordered_map<void*, std::string> callbackDataLabels_;
    static thread_local aclmdlRI currentModel_;

    aclError AllocForModel(aclmdlRI model, void** addr, size_t bytes);
    static aclError CheckCallbackRegistered(const std::string& modelLabel);
    static aclError ReleaseRecord(const ResourceRecord& record);
    static bool TakeCallbackModelLabel(void* userData, std::string& modelLabel);
    static bool BeginModelResourceRelease(const std::string& modelLabel, std::string& modelId,
                                          std::vector<ResourceRecord>& resources);
    static void FinishModelDestroy(const std::string& modelLabel);
    static void ReleaseModelResources(const std::string& modelLabel, const std::string& modelId,
                                      const std::vector<ResourceRecord>& resources);
    static void OnModelDestroy(void* userData);
};

#endif
