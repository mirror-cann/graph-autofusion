/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_resource_manager.h"

#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sk_log.h"

std::mutex SkResourceManager::resourceMutex_;
std::unordered_map<aclmdlRI, std::vector<SkResourceManager::ResourceRecord>> SkResourceManager::modelResources_;
std::unordered_set<aclmdlRI> SkResourceManager::registeredModels_;
thread_local aclmdlRI SkResourceManager::currentModel_ = nullptr;

SkResourceManager& SkResourceManager::GetInstance()
{
    static SkResourceManager instance;
    return instance;
}

void SkResourceManager::SetCurrentModel(aclmdlRI model)
{
    currentModel_ = model;
}

aclError SkResourceManager::ValueMemory(void** addr, size_t bytes)
{
    return GetInstance().AllocForModel(currentModel_, addr, bytes);
}

aclError SkResourceManager::EnsureDestroyCallbackRegistered(aclmdlRI model)
{
    if (model == nullptr) {
        SK_LOGE("ensure destroy callback failed: current model is null");
        return ACL_ERROR_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(resourceMutex_);
    if (registeredModels_.count(model) != 0U) {
        return ACL_SUCCESS;
    }

    aclError ret = aclmdlRIDestroyRegisterCallback(model, OnModelDestroy, model);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("register model destroy callback failed: model=%p, ret=%d", model, ret);
        return ret;
    }

    registeredModels_.insert(model);
    return ACL_SUCCESS;
}

aclError SkResourceManager::AllocForModel(aclmdlRI model, void** addr, size_t bytes)
{
    if (addr == nullptr || bytes == 0U) {
        SK_LOGE("resource alloc invalid param: model=%p, addr=%p, bytes=%zu", model, addr, bytes);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (model == nullptr) {
        SK_LOGE("resource alloc failed: current model is null, bytes=%zu", bytes);
        return ACL_ERROR_INVALID_PARAM;
    }

    aclError ret = EnsureDestroyCallbackRegistered(model);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    ret = aclrtMalloc(addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS || *addr == nullptr) {
        SK_LOGE("resource alloc by aclrtMalloc failed: model=%p, bytes=%zu, ret=%d", model, bytes, ret);
        return ret == ACL_SUCCESS ? ACL_ERROR_FAILURE : ret;
    }

    std::lock_guard<std::mutex> lock(resourceMutex_);
    modelResources_[model].push_back(ResourceRecord{ResourceKind::kDeviceMemory, *addr, bytes});
    SK_LOGI("resource alloc success: model=%p, addr=%p, bytes=%zu", model, *addr, bytes);
    return ACL_SUCCESS;
}

aclError SkResourceManager::ReleaseRecord(const ResourceRecord& record, aclmdlRI model)
{
    if (record.addr == nullptr) {
        return ACL_SUCCESS;
    }

    switch (record.kind) {
    case ResourceKind::kDeviceMemory: {
        aclError ret = aclrtFree(record.addr);
        if (ret != ACL_SUCCESS) {
            SK_LOGE("resource free failed in model destroy callback: model=%p, addr=%p, bytes=%zu, ret=%d", model,
                    record.addr, record.bytes, ret);
        }
        return ret;
    }
    default:
        SK_LOGE("unknown resource kind in model destroy callback: model=%p, addr=%p", model, record.addr);
        return ACL_ERROR_FAILURE;
    }
}

void SkResourceManager::OnModelDestroy(void* userData)
{
    aclmdlRI model = reinterpret_cast<aclmdlRI>(userData);
    std::vector<ResourceRecord> resources;

    {
        std::lock_guard<std::mutex> lock(resourceMutex_);
        auto it = modelResources_.find(model);
        if (it != modelResources_.end()) {
            resources.swap(it->second);
            modelResources_.erase(it);
        }
        registeredModels_.erase(model);
    }

    for (const auto& record : resources) {
        aclError ret = ReleaseRecord(record, model);
        if (ret != ACL_SUCCESS) {
            SK_LOGE("Failed to release some resources during model destroy: model=%p, ret=%d", model, ret);
        }
    }
}