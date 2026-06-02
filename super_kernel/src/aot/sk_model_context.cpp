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
 * \file sk_model_context.cpp
 * \brief Host-only per-model identity implementation.
 */

#include "sk_model_context.h"

#include <cstdint>
#include <map>
#include <mutex>

#include "sk_log.h"

namespace {

struct ModelIdentity {
    std::string modelId;
    std::string modelLabel;
};

class ModelContextState {
public:
    // aclskOptimize treats every invocation as a distinct model instance. The RTS
    // model id is the stable prefix; this counter disambiguates repeated appearances
    // of the same RTS model id, regardless of whether the aclmdlRI address is reused.
    inline static std::map<uint32_t, uint64_t> modelCallCounter_{};
    inline static std::mutex modelCounterMutex_{};
    inline static thread_local ModelIdentity currentIdentity_{};
};

uint32_t GetRtsModelId(aclmdlRI model)
{
    if (model == nullptr) {
        SK_LOGE("Failed to get model id: model is nullptr");
        return 0;
    }
    uint32_t rtsModelId = 0;
    aclError ret = aclmdlRIGetId(model, &rtsModelId);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to get model id, ret=%d", ret);
        return 0;
    }
    return rtsModelId;
}

ModelIdentity BuildModelIdentity(aclmdlRI model, bool bumpCounter)
{
    if (model == nullptr) {
        SK_LOGE("Failed to make model identity: model is nullptr");
        return {"0_0", "model_nullptr"};
    }

    uint32_t rtsModelId = GetRtsModelId(model);
    uint64_t callCount = 0U;
    if (bumpCounter) {
        std::lock_guard<std::mutex> lock(ModelContextState::modelCounterMutex_);
        callCount = ++ModelContextState::modelCallCounter_[rtsModelId];
    }

    std::string uniqueModelId = std::to_string(rtsModelId) + "_" + std::to_string(callCount);
    return {uniqueModelId, "model_" + uniqueModelId};
}

}  // namespace

const std::string& GetCurrentModelId()
{
    if (ModelContextState::currentIdentity_.modelId.empty()) {
        SK_LOGE("Failed to get current model id: no active SkModelContext");
    }
    return ModelContextState::currentIdentity_.modelId;
}

const std::string& GetCurrentModelLabel()
{
    if (ModelContextState::currentIdentity_.modelLabel.empty()) {
        SK_LOGE("Failed to get current model label: no active SkModelContext");
    }
    return ModelContextState::currentIdentity_.modelLabel;
}

SkModelContext::SkModelContext(aclmdlRI model)
    : previousModelId_(ModelContextState::currentIdentity_.modelId),
      previousModelLabel_(ModelContextState::currentIdentity_.modelLabel)
{
    ModelContextState::currentIdentity_ = BuildModelIdentity(model, true);
}

SkModelContext::~SkModelContext()
{
    ModelContextState::currentIdentity_ = {previousModelId_, previousModelLabel_};
}
