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
    SK_LOGI("register model destroy callback success: model=%p", model);
    return ACL_SUCCESS;
}

aclError SkResourceManager::AllocForModel(aclmdlRI model, void** addr, size_t bytes)
{
    if (addr == nullptr || bytes == 0U || model == nullptr) {
        SK_LOGE("resource alloc invalid param: model=%p, addr=%p, bytes=%zu", model, addr, bytes);
        return ACL_ERROR_INVALID_PARAM;
    }

    aclError ret = EnsureDestroyCallbackRegistered(model);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    ret = aclrtMalloc(addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("resource alloc by aclrtMalloc failed: model=%p, bytes=%zu, ret=%d", model, bytes, ret);
        return ret;
    }
    ret = aclrtMemset(*addr, bytes, 0, bytes);
    if(ret != ACL_SUCCESS) {
        SK_LOGE("resource memset by aclrtMemset failed: model=%p, addr=%p, bytes=%zu, ret=%d", model, *addr, bytes, ret);
        aclrtFree(*addr);
        *addr = nullptr;
        return ret;
    }

    std::lock_guard<std::mutex> lock(resourceMutex_);
    modelResources_[model].push_back(ResourceRecord{ResourceKind::kDeviceMemory, *addr, bytes});
    SK_LOGI("resource alloc success: model=%p, addr=%p, bytes=%zu", model, *addr, bytes);
    return ACL_SUCCESS;
}

aclError SkResourceManager::ReleaseRecord(const ResourceRecord& record)
{
    SK_LOGI("release resource record: addr=%p, bytes=%zu", record.addr, record.bytes);
    if (record.addr == nullptr) {
        return ACL_SUCCESS;
    }

    switch (record.kind) {
    case ResourceKind::kDeviceMemory: {
        aclError ret = aclrtFree(record.addr);
        if (ret != ACL_SUCCESS) {
            SK_LOGE("resource free failed: addr=%p, bytes=%zu, ret=%d",
                    record.addr, record.bytes, ret);
        }
        return ret;
    }
    default:
        SK_LOGE("unknown resource kind: addr=%p", record.addr);
        return ACL_ERROR_FAILURE;
    }
    SK_LOGI("resource free success: addr=%p, bytes=%zu", record.addr, record.bytes);
    return ACL_SUCCESS;
}

void SkResourceManager::OnModelDestroy(void* userData)
{
    SK_LOGI("sk resource manager OnModelDestroy called: model=%p", userData);
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
        SK_LOGI("release resource record: model=%p, addr=%p, bytes=%zu", model, record.addr, record.bytes);
        aclError ret = ReleaseRecord(record);
        if (ret != ACL_SUCCESS) {
            SK_LOGE("Failed to release some resources during model destroy: model=%p, ret=%d", model, ret);
        }
    }
    SK_LOGI("sk resource manager OnModelDestroy completed: model=%p", model);
}