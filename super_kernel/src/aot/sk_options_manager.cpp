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
 * \brief Implementation of SuperKernelOptionsManager and option value classes
 */

#include <vector>
#include <cstdlib>
#include <string>

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
    for (size_t i = 0; i < value.size(); i++) {
        SK_LOGI("OptionName:%s, value[%zu]: %s", optionName.c_str(), i, value[i].c_str());
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
    for (const auto& pair : value) {
        const std::string& key = pair.first;
        const std::vector<std::string>& valList = pair.second;
        SK_LOGI("OptionName:%s, key: %s, values:", optionName.c_str(), key.c_str());
        for (const auto& val : valList) {
            SK_LOGI("  %s", val.c_str());
        }
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

OptOptionBase* SuperKernelOptionsManager::GetOption(aclskOtionType optType) {
    auto iter = optionMap.find(optType);
    if (iter == optionMap.end()) {
        return nullptr;
    }
    return iter->second.get();
}

const OptOptionBase* SuperKernelOptionsManager::GetOption(aclskOtionType optType) const {
    auto iter = optionMap.find(optType);
    if (iter == optionMap.end()) {
        return nullptr;
    }
    return iter->second.get();
}

bool SuperKernelOptionsManager::JudgeDisableKernelDcci(
    const std::vector<std::string>& dcciOps, const std::string& opName) const {
    for (size_t i = 0; i < dcciOps.size(); i++) {
        if (MatchRegex(dcciOps[i], opName)) {
            SK_LOGI("op: %s match disable dcci option: %s, op's dcci will be disabled",
                opName.c_str(), dcciOps[i].c_str());
            return true;
        }
    }
    return false;
}

bool SuperKernelOptionsManager::MatchRegex(const std::string& pattern, const std::string& opName) {
    size_t m = opName.size();
    size_t n = pattern.size();
    if (n > 0 && pattern[0] == '*') {
        SK_LOGW("invalid pattern starts with '*': %s", pattern.c_str());
        return false;
    }

    auto matches = [&](size_t i, size_t j) {
        if (i == 0 || j == 0) {
            return false;
        }
        if (pattern[j - 1] == '.') {
            return true;
        }
        return opName[i - 1] == pattern[j - 1];
    };

    std::vector<std::vector<size_t>> matchFlag(m + 1, std::vector<size_t>(n + 1));
    matchFlag[0][0] = true;
    for (size_t i = 0; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            if (pattern[j - 1] == '*') {
                if (j >= 2) {
                    matchFlag[i][j] |= matchFlag[i][j - 2];
                    if (matches(i, j - 1)) {
                        matchFlag[i][j] |= matchFlag[i - 1][j];
                    }
                }
            }
            else {
                if (matches(i, j)) {
                    matchFlag[i][j] |= matchFlag[i - 1][j - 1];
                }
            }
        }
    }
    return matchFlag[m][n];
}

bool SuperKernelOptionsManager::EnableDebug() const {
    auto iterSyncAll = optionMap.find(aclskOtionType::DEBUG_SYNC_ALL);
    auto iterDcci = optionMap.find(aclskOtionType::DEBUG_DCCI_DISABLE_ON_KERNEL);
    const bool enableSyncAll =
        (iterSyncAll != optionMap.end() && iterSyncAll->second != nullptr && iterSyncAll->second->GetIntValue() == 1);
    const bool enableDcciDisable = (iterDcci != optionMap.end() && iterDcci->second != nullptr);
    if (enableSyncAll || enableDcciDisable) {
        SK_LOGI("debug mode enabled");
        return true;
    }
    return false;
}

void SuperKernelOptionsManager::SetOptOptionValue(const aclskOption* option) {
    if (option == nullptr) {
        SK_LOGE("sub aclskOption is nullptr");
        return;
    }
    switch (option->optionType) {
        case aclskOtionType::PRELOAD_CODE:
            {
                AddOption(std::make_unique<NumberOptOption>("preload_code", option->optionType, 1, 0, 2));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->preload.preloadMode);
                }
                break;
            }
        case aclskOtionType::SPLIT_MODE:
            {
                AddOption(std::make_unique<NumberOptOption>("split_mode", option->optionType, 4, 1, 4));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->splitMode.splitCnt);
                }
                break;
            }
        case aclskOtionType::DEBUG_DCCI_DISABLE_ON_KERNEL:
            {
                AddOption(std::make_unique<StringListOptOption>("dcci_disable_on_kernel", option->optionType));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    std::vector<std::string> vecValue;
                    const size_t kernelCnt = static_cast<size_t>(option->disableKernelDcci.kernelCnt);
                    if (kernelCnt > 0 && option->disableKernelDcci.kernelNames == nullptr) {
                        SK_LOGE("OptionName:%s, kernelNames is nullptr while kernelCnt is %zu",
                            subOption->GetName().c_str(), kernelCnt);
                        break;
                    }
                    vecValue.reserve(kernelCnt);
                    for (size_t i = 0; i < kernelCnt; i++) {
                        if (option->disableKernelDcci.kernelNames[i] == nullptr) {
                            SK_LOGW("OptionName:%s, kernelNames[%zu] is nullptr, skip",
                                subOption->GetName().c_str(), i);
                            continue;
                        }
                        vecValue.push_back(std::string(option->disableKernelDcci.kernelNames[i]));
                    }
                    subOption->SetValue(vecValue);
                }
                break;
            }
        case aclskOtionType::DEBUG_SYNC_ALL:
            {
                AddOption(std::make_unique<NumberOptOption>("debug_sync_all", option->optionType, 0, 0, 1));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->debugSync.debugSyncAll);
                }
                break;
            }
        case aclskOtionType::STREAM_FUSION:
            {
                AddOption(std::make_unique<NumberOptOption>("stream_fusion", option->optionType, 0, 0, 1));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->streamFusion.streamFusion);
                }
                break;
            }
        case aclskOtionType::CONSTANT_CODEGEN:
            {
                // 默认关闭常量化代码生成（值为1启用，0禁用）
                AddOption(std::make_unique<NumberOptOption>("constant_codegen", option->optionType, 0, 0, 1));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->constantCodegen.enableConstant);
                }
                SK_LOGI("Constant codegen option set: enable=%u", option->constantCodegen.enableConstant);
                break;
            }
        case aclskOtionType::OPS_LAYOUT_OPTIMIZE:
            {
                AddOption(std::make_unique<NumberOptOption>("ops_layout_optimize", option->optionType, 0, 0, 1));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->layoutOptimize.enableOpsLayoutOptimize);
                }
                SK_LOGI("Ops layout optimize option set: enable=%u", option->layoutOptimize.enableOpsLayoutOptimize);
                break;
            }
        default:
            SK_LOGI("Optiontype: %d is not support now", static_cast<int>(option->optionType));
            break;
    }
}

void SuperKernelOptionsManager::ParseOptions(const aclskOptions* options) {
    if (options == nullptr) {
        SK_LOGI("aclskOption is nullptr");
        return;
    }
    SK_LOGI("Options nums: %d\n", static_cast<int>(options->numOptions));
    for (size_t i = 0; i < static_cast<size_t>(options->numOptions); i++) {
        auto iter = optionMap.find(options->options[i].optionType);
        if (iter != optionMap.end()) {
            SK_LOGW("OptionName %s already exists", iter->second->GetName().c_str());
            continue;
        }
        SetOptOptionValue(&options->options[i]);
    }
}
