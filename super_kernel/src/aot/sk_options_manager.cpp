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
 * \file sk_options_manager.cpp
 * \brief
 */

#include "sk_options_manager.h"
#include "sk_log.h"

void NumberOptOption::SetValue(const uint32_t value) {
    if (value < optValueMin || value > optValueMax) {
        SK_LOGE("OptionName: %s, set value is invalid, value is %u, valid range is [%u, %u]",
            optionName.c_str(), value, optValueMin, optValueMax);
        return;
    }
    SK_LOGI("OptionName: %s, set value: %u", optionName.c_str(), value);
    optValue = value;
}

uint32_t NumberOptOption::GetIntValue() const {
    return optValue;
}


void StringOptOption::SetValue(const std::string& value) {
    if (value.empty()) {
        SK_LOGE("OptionName: %s, set value is invalid, value is empty", optionName.c_str());
        return;
    }
    SK_LOGI("OptionName: %s, set value: %s", optionName.c_str(), value.c_str());
    optValue = value;
}

std::string StringOptOption::GetStringValue() const {
    return optValue;
}


void StringListOptOption::SetValue(const std::vector<std::string>& value) {
    if (value.empty()) {
        SK_LOGE("OptionName: %s, set value is invalid, value is empty", optionName.c_str());
        return;
    }
    for (int i = 0; i < value.size(); i++) {
        SK_LOGI("OptionName:%s, value[%d]: %s", optionName.c_str(), i, value[i].c_str());
    }
    optValue = value;
}

std::vector<std::string> StringListOptOption::GetStringListValue() const {
    return optValue;
}



void MapOptOption::SetValue(const std::unordered_map<std::string, std::vector<std::string>>& value) {
    if (value.empty()) {
        SK_LOGE("OptionName:%s, set value is invalid, value is empty", optionName.c_str());
        return;
    }
    optValue = value;
}

std::unordered_map<std::string, std::vector<std::string>> MapOptOption::GetMapValue() const {
    return optValue;
}


void SuperKernelOptionsManager::AddOption(std::unique_ptr<OptOptionBase> option) {
    if (option == nullptr) {
        SK_LOGI("option is nullptr");
        return;
    }
    auto iter = optionMap.find(option->GetType());
    if (iter != optionMap.end()) {
        SK_LOGE("OptionName:%s, already exists", option->GetName().c_str());
        return;
    }
    optionMap[option->GetType()] = std::move(option);
}

OptOptionBase* SuperKernelOptionsManager::GetOption(const aclskOtionType optType) const {
    auto iter = optionMap.find(optType);
    if (iter == optionMap.end()) {
        return nullptr;
    }
    return iter->second.get();
}

bool SuperKernelOptionsManager::JudgeDisableKernelDcci(std::vector<std::string>& dcciOps, const std::string& opName) const {
    for (size_t i = 0; i < dcciOps.size(); i++) {
        try {
            std::regex pattern(dcciOps[i]);
            if (std::regex_match(opName, pattern)) {
                SK_LOGE("op: %s match disable dcci option: %s, op's dcci will be disabled",
                    opName.c_str(), dcciOps[i].c_str());
                return true;
            }
        } catch (const std::regex_error& e) {
            SK_LOGE("regex error: %s, error code: %d", e.what(), e.code());
            return false;
        }
    }
    return false;
}

bool SuperKernelOptionsManager::EnableDebug() const {
    auto iter = optionMap.find(aclskOtionType::DEBUG_SYNC_ALL);
    auto iter1= optionMap.find(aclskOtionType::DEBUG_DCCI_DISABLE_ON_KERNEL);
    if (iter != optionMap.end() || iter1 != optionMap.end()) {
        SK_LOGI("debug mode enabled");
        return true;
    }
    return false;
}

void SuperKernelOptionsManager::SetOptOptionValue(aclskOption* option) {
    switch (option->optionType) {
        case aclskOtionType::PRELOAD_CODE:
            {
                AddOption(std::make_unique<NumberOptOption>("preload_code", option->optionType, 1, 0, 2));
                auto subOption = GetOption(option->optionType);
                subOption->SetValue((option->preload.preloadMode));
                break;
            }
        case aclskOtionType::SPLIT_MODE:
            {
                AddOption(std::make_unique<NumberOptOption>("split_mode", option->optionType, 4, 1, 4));
                auto subOption = GetOption(option->optionType);
                subOption->SetValue(option->splitMode.splitCnt);
                break;
            }
        case aclskOtionType::DEBUG_DCCI_DISABLE_ON_KERNEL:
            {
                AddOption(std::make_unique<StringListOptOption>("dcci_disable_on_kernel", option->optionType));
                auto subOption = GetOption(option->optionType);
                std::vector<std::string> vecValue;
                for (int i = 0; i < option->disableKernelDcci.kernelCnt; i++) {
                    vecValue.push_back(std::string(option->disableKernelDcci.kernelNames[i]));
                }
                subOption->SetValue(vecValue);
                break;
            }
        case aclskOtionType::DEBUG_SYNC_ALL:
            {
                AddOption(std::make_unique<NumberOptOption>("debug_sync_all", option->optionType, 0, 0, 1));
                auto subOption = GetOption(option->optionType);
                subOption->SetValue(option->debugSync.debugSyncAll);
                break;
            }
        default:
            SK_LOGI("Optiontype: %d is not support now", static_cast<int>(option->optionType));
            break;
    }
}

void SuperKernelOptionsManager::ParseOptions(aclskOptions* options) {
    if (options == nullptr) {
        return;
    }
    for (int i = 0; i < options->numOptions; i++) {
        auto iter = optionMap.find(options->options[i].optionType);
        if (iter != optionMap.end()) {
            SK_LOGW("OptionName %s already exists", iter->second->GetName().c_str());
            continue;
        }
        SetOptOptionValue(&options->options[i]);
    }
}
