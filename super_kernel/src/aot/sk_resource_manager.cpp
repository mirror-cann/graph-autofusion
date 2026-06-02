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
#include "sk_event_recorder.h"
#include "sk_model_context.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {
class ScopedModelLogContext {
public:
    explicit ScopedModelLogContext(const std::string& modelLabel)
        : previousModelLabel_(sk::logger::FileLogger::GetCurrentThreadModelLabel()),
          previousHandle_(sk::logger::FileHandleManager::Instance().GetCurrentHandle())
    {
        sk::logger::FileLogger::SetCurrentModelLabel(modelLabel);
        if (!modelLabel.empty()) {
            std::string handleName = "model_" + SanitizePathComponent(modelLabel);
            sk::logger::FileHandleManager::Instance().SwitchToFile(handleName);
        } else {
            sk::logger::FileHandleManager::Instance().SwitchToDefault();
        }
    }

    ~ScopedModelLogContext()
    {
        sk::logger::FileLogger::SetCurrentModelLabel(previousModelLabel_);
        if (previousHandle_ == "default") {
            sk::logger::FileHandleManager::Instance().SwitchToDefault();
            return;
        }
        sk::logger::FileHandleManager::Instance().SwitchToFile(previousHandle_);
    }

private:
    std::string previousModelLabel_;
    std::string previousHandle_;
};
}  // namespace

std::mutex SkResourceManager::resourceMutex_;
std::unordered_map<std::string, std::vector<SkResourceManager::ResourceRecord>> SkResourceManager::modelResources_;
std::unordered_set<std::string> SkResourceManager::registeredModelLabels_;
std::unordered_map<void*, std::string> SkResourceManager::callbackDataLabels_;
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

aclError SkResourceManager::CallbackRegister(aclmdlRI model)
{
    if (model == nullptr) {
        SK_LOGE("ensure destroy callback failed: current model is null");
        return ACL_ERROR_INVALID_PARAM;
    }

    const std::string modelId = GetCurrentModelId();
    const std::string modelLabel = GetCurrentModelLabel();
    if (modelId.empty() || modelLabel.empty()) {
        SK_LOGE("ensure destroy callback failed: no active model context, model=%p", model);
        return ACL_ERROR_FAILURE;
    }

    std::lock_guard<std::mutex> lock(resourceMutex_);
    if (registeredModelLabels_.count(modelLabel) != 0U) {
        SK_LOGI("model destroy callback already registered: modelLabel=%s, modelId=%s",
                modelLabel.c_str(), modelId.c_str());
        return ACL_SUCCESS;
    }

    char* labelCopy = strdup(modelLabel.c_str());
    if (labelCopy == nullptr) {
        SK_LOGE("alloc model destroy callback label failed: modelLabel=%s, modelId=%s",
                modelLabel.c_str(), modelId.c_str());
        return ACL_ERROR_FAILURE;
    }

    callbackDataLabels_.emplace(labelCopy, modelLabel);

    aclError ret = aclmdlRIDestroyRegisterCallback(model, OnModelDestroy, labelCopy);
    if (ret != ACL_SUCCESS) {
        callbackDataLabels_.erase(labelCopy);
        free(labelCopy);
        SK_LOGE("register model destroy callback failed: modelLabel=%s, modelId=%s, ret=%d",
                modelLabel.c_str(), modelId.c_str(), ret);
        return ret;
    }

    registeredModelLabels_.insert(modelLabel);
    SK_LOGI("register model destroy callback success: modelLabel=%s, modelId=%s",
            modelLabel.c_str(), modelId.c_str());
    return ACL_SUCCESS;
}

aclError SkResourceManager::CheckCallbackRegistered(const std::string& modelLabel)
{
    std::lock_guard<std::mutex> lock(resourceMutex_);
    return registeredModelLabels_.count(modelLabel) != 0U ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

aclError SkResourceManager::AllocForModel(aclmdlRI model, void** addr, size_t bytes)
{
    if (addr == nullptr || bytes == 0U || model == nullptr) {
        SK_LOGE("resource alloc invalid param: model=%p, addr=%p, bytes=%zu", model, addr, bytes);
        return ACL_ERROR_INVALID_PARAM;
    }

    const std::string modelId = GetCurrentModelId();
    const std::string modelLabel = GetCurrentModelLabel();
    if (modelId.empty() || modelLabel.empty()) {
        SK_LOGE("resource alloc failed: no active model context, model=%p", model);
        return ACL_ERROR_FAILURE;
    }

    aclError ret = CheckCallbackRegistered(modelLabel);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("resource alloc failed: model destroy callback is not registered, modelLabel=%s, modelId=%s",
                modelLabel.c_str(), modelId.c_str());
        return ret;
    }

    ret = aclrtMalloc(addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("resource alloc by aclrtMalloc failed: modelLabel=%s, modelId=%s, bytes=%zu, ret=%d",
                modelLabel.c_str(), modelId.c_str(), bytes, ret);
        return ret;
    }
    ret = aclrtMemset(*addr, bytes, 0, bytes);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("resource memset by aclrtMemset failed: modelLabel=%s, modelId=%s, addr=%p, bytes=%zu, ret=%d",
                modelLabel.c_str(), modelId.c_str(), *addr, bytes, ret);
        aclrtFree(*addr);
        *addr = nullptr;
        return ret;
    }

    std::lock_guard<std::mutex> lock(resourceMutex_);
    modelResources_[modelLabel].push_back(ResourceRecord{ResourceKind::kDeviceMemory, *addr, bytes});
    SK_LOGI("resource alloc success: modelLabel=%s, modelId=%s, addr=%p, bytes=%zu",
            modelLabel.c_str(), modelId.c_str(), *addr, bytes);
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
            } else {
                SK_LOGI("resource free success: addr=%p, bytes=%zu", record.addr, record.bytes);
            }
            return ret;
        }
        default:
            SK_LOGE("unknown resource kind: addr=%p", record.addr);
            return ACL_ERROR_FAILURE;
    }
}

bool SkResourceManager::TakeCallbackModelLabel(void* userData, std::string& modelLabel)
{
    std::lock_guard<std::mutex> lock(resourceMutex_);
    auto it = callbackDataLabels_.find(userData);
    if (it == callbackDataLabels_.end()) {
        return false;
    }

    modelLabel = it->second;
    callbackDataLabels_.erase(it);
    return true;
}

bool SkResourceManager::BeginModelResourceRelease(const std::string& modelLabel, std::string& modelId,
                                                  std::vector<ResourceRecord>& resources)
{
    std::lock_guard<std::mutex> lock(resourceMutex_);
    if (registeredModelLabels_.count(modelLabel) == 0U) {
        return false;
    }

    modelId = GetCurrentModelId();
    auto resourceIt = modelResources_.find(modelLabel);
    if (resourceIt != modelResources_.end()) {
        resources.swap(resourceIt->second);
        modelResources_.erase(resourceIt);
    }
    return true;
}

void SkResourceManager::FinishModelDestroy(const std::string& modelLabel, void* labelCopy)
{
    std::lock_guard<std::mutex> lock(resourceMutex_);
    registeredModelLabels_.erase(modelLabel);
    free(labelCopy);
}

void SkResourceManager::ReleaseModelResources(const std::string& modelLabel, const std::string& modelId,
                                              const std::vector<ResourceRecord>& resources)
{
    SkEventRecorder::Instance().RemoveModelMappings(modelId);

    for (const auto& record : resources) {
        SK_LOGI("release resource record: modelLabel=%s, modelId=%s, addr=%p, bytes=%zu",
                modelLabel.c_str(), modelId.c_str(), record.addr, record.bytes);
        aclError ret = ReleaseRecord(record);
        if (ret != ACL_SUCCESS) {
            SK_LOGE("Failed to release some resources during model destroy: modelLabel=%s, modelId=%s, ret=%d",
                    modelLabel.c_str(), modelId.c_str(), ret);
        }
    }
}

void SkResourceManager::OnModelDestroy(void* userData)
{
    char* labelCopy = static_cast<char*>(userData);
    if (labelCopy == nullptr) {
        SK_LOGE("sk resource manager OnModelDestroy invalid userData=%p", userData);
        return;
    }

    std::string modelLabel;
    if (!TakeCallbackModelLabel(userData, modelLabel)) {
        SK_LOGW("sk resource manager OnModelDestroy callback data invalid: userData=%p", userData);
        return;
    }

    std::string modelId;
    std::vector<ResourceRecord> resources;
    if (!BeginModelResourceRelease(modelLabel, modelId, resources)) {
        SK_LOGW("sk resource manager OnModelDestroy invalid model: modelLabel=%s", modelLabel.c_str());
        FinishModelDestroy(modelLabel, labelCopy);
        return;
    }

    ScopedModelLogContext logContext(modelLabel);
    SK_LOGI("sk resource manager OnModelDestroy called: modelLabel=%s, modelId=%s",
            modelLabel.c_str(), modelId.c_str());

    ReleaseModelResources(modelLabel, modelId, resources);
    FinishModelDestroy(modelLabel, labelCopy);

    SK_LOGI("sk resource manager OnModelDestroy completed: modelLabel=%s, modelId=%s",
            modelLabel.c_str(), modelId.c_str());
}