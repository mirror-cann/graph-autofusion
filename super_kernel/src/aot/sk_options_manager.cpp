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

#include <cctype>
#include <vector>
#include <cstdlib>
#include <limits>
#include <string>

#include "sk_options_manager.h"
#include "sk_log.h"
#include <nlohmann/json.hpp>

namespace {
constexpr size_t kMaxExtendOptionLength = 1024;

uint32_t GetValidatedUintValue(const std::string& optionName, uint32_t value, uint32_t defaultValue,
                               uint32_t minValue, uint32_t maxValue)
{
    if (value < minValue || value > maxValue) {
        SK_LOGW("OptionName: %s, set value is invalid, value is %u, valid range is [%u, %u],"
            " the process will use default value: %u",
            optionName.c_str(), value, minValue, maxValue, defaultValue);
        return defaultValue;
    }
    SK_LOGI("OptionName: %s, set value: %u", optionName.c_str(), value);
    return value;
}

std::string TrimString(const std::string& input)
{
    size_t start = 0;
    while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start])) != 0) {
        ++start;
    }
    size_t end = input.size();
    while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
        --end;
    }
    return input.substr(start, end - start);
}

std::vector<std::string> SplitString(const std::string& input, char delimiter)
{
    std::vector<std::string> tokens;
    size_t start = 0;
    while (start <= input.size()) {
        size_t end = input.find(delimiter, start);
        if (end == std::string::npos) {
            tokens.push_back(input.substr(start));
            break;
        }
        tokens.push_back(input.substr(start, end - start));
        start = end + 1;
    }
    return tokens;
}

bool IsValidExtendOptionToken(const std::string& token, bool allowSlash)
{
    if (token.empty()) {
        return false;
    }
    for (char ch : token) {
        const unsigned char uchar = static_cast<unsigned char>(ch);
        if (std::isalnum(uchar) != 0 || ch == '_' || ch == '-' || ch == '.') {
            continue;
        }
        if (allowSlash && ch == '/') {
            continue;
        }
        return false;
    }
    return true;
}

bool ParseAndValidateExtendOptionValue(const char* rawValue, const std::string& optionName,
    std::unordered_map<std::string, std::vector<std::string>>& parsedValue)
{
    if (rawValue == nullptr) {
        SK_LOGW("OptionName:%s, raw extend value is nullptr", optionName.c_str());
        return false;
    }

    const std::string input(rawValue);
    if (input.size() > kMaxExtendOptionLength) {
        SK_LOGW("OptionName:%s, raw extend value is too long: %zu", optionName.c_str(), input.size());
        return false;
    }

    const std::string trimmedInput = TrimString(input);
    if (trimmedInput.empty()) {
        SK_LOGW("OptionName:%s, raw extend value is empty after trim", optionName.c_str());
        return false;
    }

    std::unordered_map<std::string, std::vector<std::string>> tmpResult;
    const std::vector<std::string> pairs = SplitString(trimmedInput, ':');
    for (const std::string& rawPair : pairs) {
        const std::string pair = TrimString(rawPair);
        if (pair.empty()) {
            SK_LOGW("OptionName:%s, extend pair is empty", optionName.c_str());
            return false;
        }

        const size_t eqPos = pair.find('=');
        if (eqPos == std::string::npos || eqPos == 0 || eqPos == pair.size() - 1 ||
            pair.find('=', eqPos + 1) != std::string::npos) {
            SK_LOGW("OptionName:%s, extend pair format is invalid: %s", optionName.c_str(), pair.c_str());
            return false;
        }

        const std::string key = TrimString(pair.substr(0, eqPos));
        if (!IsValidExtendOptionToken(key, false)) {
            SK_LOGW("OptionName:%s, extend key is invalid: %s", optionName.c_str(), key.c_str());
            return false;
        }
        if (tmpResult.find(key) != tmpResult.end()) {
            SK_LOGW("OptionName:%s, extend key is duplicated: %s", optionName.c_str(), key.c_str());
            return false;
        }

        const std::vector<std::string> rawValues = SplitString(pair.substr(eqPos + 1), ',');
        std::vector<std::string> valueList;
        valueList.reserve(rawValues.size());
        for (const std::string& rawSubValue : rawValues) {
            const std::string value = TrimString(rawSubValue);
            if (!IsValidExtendOptionToken(value, true)) {
                SK_LOGW("OptionName:%s, extend value is invalid: %s", optionName.c_str(), value.c_str());
                return false;
            }
            valueList.push_back(value);
        }
        tmpResult.emplace(key, std::move(valueList));
    }

    parsedValue = std::move(tmpResult);
    return true;
}
}

void NumberOptOption::SetValue(const uint32_t value) {
    if (value < optValueMin || value > optValueMax) {
        SK_LOGW("OptionName: %s, set value is invalid, value is %u, valid range is [%u, %u],"
            " the process will use default value: %u",
            optionName.c_str(), value, optValueMin, optValueMax, optValue);
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
        SK_LOGW("OptionName: %s, set value is invalid, value is empty", optionName.c_str());
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
        SK_LOGW("OptionName: %s, set value is invalid, value is empty", optionName.c_str());
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
        SK_LOGW("OptionName:%s, set value is invalid, value is empty", optionName.c_str());
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
        SK_LOGW("OptionName:%s, already exists", option->GetName().c_str());
        return;
    }
    optionMap[option->GetType()] = std::move(option);
}

OptOptionBase* SuperKernelOptionsManager::GetOption(aclskOptionType optType) {
    auto iter = optionMap.find(optType);
    if (iter == optionMap.end()) {
        return nullptr;
    }
    return iter->second.get();
}

const OptOptionBase* SuperKernelOptionsManager::GetOption(aclskOptionType optType) const {
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

bool SuperKernelOptionsManager::JudgeUbufLockIgnoreKernel(const aclskAggressiveOptStrategies& aggressiveOpts,
                                                          const std::string& opName) const {
    const size_t kernelCnt = static_cast<size_t>(aggressiveOpts.ubufLockIgnoreKernelCnt);
    if (kernelCnt > 0 && aggressiveOpts.ubufLockIgnoreKernel == nullptr) {
        SK_LOGW("ubufLockIgnoreKernel is nullptr while ubufLockIgnoreKernelCnt is %zu", kernelCnt);
        return false;
    }
    for (size_t i = 0; i < kernelCnt; i++) {
        if (aggressiveOpts.ubufLockIgnoreKernel[i] == nullptr) {
            SK_LOGW("ubufLockIgnoreKernel[%zu] is nullptr, skip", i);
            continue;
        }
        if (MatchRegex(aggressiveOpts.ubufLockIgnoreKernel[i], opName)) {
            SK_LOGI("op: %s match ubuf lock ignore kernel option: %s, op will ignore mix kernel split",
                opName.c_str(), aggressiveOpts.ubufLockIgnoreKernel[i]);
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
    auto iterSyncAll = optionMap.find(aclskOptionType::DEBUG_SYNC_ALL);
    const bool enableSyncAll =
        (iterSyncAll != optionMap.end() && iterSyncAll->second != nullptr && iterSyncAll->second->GetIntValue() == 1);
    if (enableSyncAll) {
        SK_LOGI("debug mode enabled");
        return true;
    }
    return false;
}

std::string SuperKernelOptionsManager::GetSocName() const
{
    SK_LOGI("Init socName");
    const char* socNameTmp = aclrtGetSocName();
    if (socNameTmp == nullptr) {
        SK_LOGW("Failed to get soc name");
        return "";
    }
    std::string socName(socNameTmp);
    SK_LOGI("Soc name: %s", socName.c_str());
    return socName;
}

bool SuperKernelOptionsManager::EnableMixKernelSplit() const
{
    const std::string socName = GetSocName();
    const bool enableMixKernelSplit = socName.find("Ascend950") != std::string::npos;
    SK_LOGI("Mix kernel split default by soc: socName=%s, enable=%d",
            socName.c_str(), static_cast<int>(enableMixKernelSplit));
    return enableMixKernelSplit;
}

void SuperKernelOptionsManager::SetOptOptionValue(const aclskOption* option) {
    if (option == nullptr) {
        SK_LOGW("sub aclskOption is nullptr");
        return;
    }
    switch (option->optionType) {
        case aclskOptionType::PRELOAD_CODE:
            {
                AddOption(std::make_unique<NumberOptOption>("preload_code", option->optionType, 1, 0, 2));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->preload.preloadMode);
                }
                break;
            }
        case aclskOptionType::SPLIT_MODE:
            {
                AddOption(std::make_unique<NumberOptOption>("split_mode", option->optionType, 4, 1, 4));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->splitMode.splitCnt);
                }
                break;
            }
        case aclskOptionType::DCCI_DISABLE_ON_KERNEL:
            {
                AddOption(std::make_unique<StringListOptOption>("dcci_disable_on_kernel", option->optionType));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    std::vector<std::string> vecValue;
                    const size_t kernelCnt = static_cast<size_t>(option->disableKernelDcci.kernelCnt);
                    if (kernelCnt > 0 && option->disableKernelDcci.kernelNames == nullptr) {
                        SK_LOGW("OptionName:%s, kernelNames is nullptr while kernelCnt is %zu",
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
        case aclskOptionType::DCCI_BEFORE_KERNEL_START:
            {
                AddOption(std::make_unique<StringListOptOption>("dcci_before_kernel_start", option->optionType));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    std::vector<std::string> vecValue;
                    const size_t kernelCnt = static_cast<size_t>(option->dcciBeforeKernelStart.kernelCnt);
                    if (kernelCnt > 0 && option->dcciBeforeKernelStart.kernelNames == nullptr) {
                        SK_LOGW("OptionName:%s, kernelNames is nullptr while kernelCnt is %zu",
                            subOption->GetName().c_str(), kernelCnt);
                        break;
                    }
                    vecValue.reserve(kernelCnt);
                    for (size_t i = 0; i < kernelCnt; i++) {
                        if (option->dcciBeforeKernelStart.kernelNames[i] == nullptr) {
                            SK_LOGW("OptionName:%s, kernelNames[%zu] is nullptr, skip",
                                subOption->GetName().c_str(), i);
                            continue;
                        }
                        vecValue.push_back(std::string(option->dcciBeforeKernelStart.kernelNames[i]));
                    }
                    subOption->SetValue(vecValue);
                }
                break;
            }
        case aclskOptionType::DCCI_AFTER_KERNEL_END:
            {
                AddOption(std::make_unique<StringListOptOption>("dcci_after_kernel_end", option->optionType));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    std::vector<std::string> vecValue;
                    const size_t kernelCnt = static_cast<size_t>(option->dcciAfterKernelEnd.kernelCnt);
                    if (kernelCnt > 0 && option->dcciAfterKernelEnd.kernelNames == nullptr) {
                        SK_LOGW("OptionName:%s, kernelNames is nullptr while kernelCnt is %zu",
                            subOption->GetName().c_str(), kernelCnt);
                        break;
                    }
                    vecValue.reserve(kernelCnt);
                    for (size_t i = 0; i < kernelCnt; i++) {
                        if (option->dcciAfterKernelEnd.kernelNames[i] == nullptr) {
                            SK_LOGW("OptionName:%s, kernelNames[%zu] is nullptr, skip",
                                subOption->GetName().c_str(), i);
                            continue;
                        }
                        vecValue.push_back(std::string(option->dcciAfterKernelEnd.kernelNames[i]));
                    }
                    subOption->SetValue(vecValue);
                }
                break;
            }
        case aclskOptionType::AGGRESSIVE_OPT_STRATEGIES:
            {
                AddOption(std::make_unique<AggressiveOptStrategiesOption>(
                    "aggressive_opt_strategies", option->optionType));
                auto subOption = static_cast<AggressiveOptStrategiesOption*>(GetOption(option->optionType));
                if (subOption == nullptr) {
                    break;
                }
                auto aggressiveOpts = option->aggressiveOpts;
                aggressiveOpts.eventBreakerBypass = GetValidatedUintValue(
                    "event_breaker_bypass",
                    option->aggressiveOpts.eventBreakerBypass,
                    0,
                    0,
                    std::numeric_limits<decltype(option->aggressiveOpts.eventBreakerBypass)>::max());
                aggressiveOpts.valueBreakerBypass = GetValidatedUintValue(
                    "value_breaker_bypass",
                    option->aggressiveOpts.valueBreakerBypass,
                    ACLSK_VALUE_BREAKER_BYPASS_NONE,
                    ACLSK_VALUE_BREAKER_BYPASS_NONE,
                    std::numeric_limits<decltype(option->aggressiveOpts.valueBreakerBypass)>::max());
                aggressiveOpts.taskBreakerBypass = GetValidatedUintValue(
                    "task_breaker_bypass", option->aggressiveOpts.taskBreakerBypass, 0, 0, 1);
                SK_LOGI("Aggressive opt strategies set: eventBreakerBypass=%u, valueBreakerBypass=%u, "
                        "taskBreakerBypass=%u",
                        aggressiveOpts.eventBreakerBypass, aggressiveOpts.valueBreakerBypass,
                        aggressiveOpts.taskBreakerBypass);
                subOption->SetValue(aggressiveOpts);
                break;
            }
        case aclskOptionType::DEBUG_SYNC_ALL:
            {
                AddOption(std::make_unique<NumberOptOption>("debug_sync_all", option->optionType, 0, 0, 1));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->debugSync.debugSyncAll);
                }
                break;
            }
        case aclskOptionType::OPT_EXTEND_OPTION:
            {
                AddOption(std::make_unique<MapOptOption>("opt_extend_option", option->optionType));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    std::unordered_map<std::string, std::vector<std::string>> parsedValue;
                    if (ParseAndValidateExtendOptionValue(
                        option->optExtend.value, subOption->GetName(), parsedValue)) {
                        subOption->SetValue(parsedValue);
                    }
                }
                break;
            }
        case aclskOptionType::DEBUG_EXTEND_OPTION:
            {
                AddOption(std::make_unique<MapOptOption>("debug_extend_option", option->optionType));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    std::unordered_map<std::string, std::vector<std::string>> parsedValue;
                    if (ParseAndValidateExtendOptionValue(
                        option->debugExtend.value, subOption->GetName(), parsedValue)) {
                        subOption->SetValue(parsedValue);
                    }
                }
                break;
            }
        case aclskOptionType::STREAM_FUSION:
            {
                AddOption(std::make_unique<NumberOptOption>("stream_fusion", option->optionType, 1, 0, 1));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->streamFusion.streamFusion);
                }
                break;
            }
        case aclskOptionType::CONSTANT_CODEGEN:
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
        case aclskOptionType::AUTO_OP_PARALLEL:
            {
                AddOption(std::make_unique<NumberOptOption>("auto_op_parallel", option->optionType, 0, 0, 1));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->autoOpParallel.enableAutoOpParallel);
                }
                SK_LOGI("Auto op parallel option set: enable=%u", option->autoOpParallel.enableAutoOpParallel);
                break;
            }
        case aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK:
            {
                AddOption(std::make_unique<NumberOptOption>("debug_cross_core_sync_check", option->optionType, 0, 0, 1));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->debugCrossCoreSyncCheck.enableCrossCoreSyncCheck);
                }
                SK_LOGI("Debug cross-core sync check option set: enable=%u",
                    option->debugCrossCoreSyncCheck.enableCrossCoreSyncCheck);
                break;
            }
        case aclskOptionType::DEBUG_OP_EXEC_TRACE:
            {
                AddOption(std::make_unique<NumberOptOption>("debug_op_exec_trace", option->optionType, 0, 0, 1));
                auto subOption = GetOption(option->optionType);
                if (subOption != nullptr) {
                    subOption->SetValue(option->debugOpExecTrace.enableOpExecTrace);
                }
                SK_LOGI("Debug op exec trace option set: enable=%u",
                    option->debugOpExecTrace.enableOpExecTrace);
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

nlohmann::ordered_json SuperKernelOptionsManager::ToJson() const
{
    nlohmann::ordered_json optionsJson = nlohmann::ordered_json::object();

    for (int32_t i = 0; i < static_cast<int32_t>(aclskOptionType::SK_OPTION_MAX); ++i) {
        auto type = static_cast<aclskOptionType>(i);
        const auto iter = optionMap.find(type);
        if (iter == optionMap.end()) {
            continue;
        }

        const OptOptionBase* opt = iter->second.get();
        if (opt == nullptr) {
            continue;
        }

        nlohmann::ordered_json optJson;
        optJson["name"] = opt->GetName();
        optJson["type"] = static_cast<int>(type);

        switch (type) {
            case aclskOptionType::PRELOAD_CODE:
            case aclskOptionType::SPLIT_MODE:
            case aclskOptionType::DEBUG_SYNC_ALL:
            case aclskOptionType::STREAM_FUSION:
            case aclskOptionType::CONSTANT_CODEGEN:
            case aclskOptionType::AUTO_OP_PARALLEL:
            case aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK:
            case aclskOptionType::DEBUG_OP_EXEC_TRACE:
                optJson["value"] = opt->GetIntValue();
                break;

            case aclskOptionType::DCCI_DISABLE_ON_KERNEL:
            case aclskOptionType::DCCI_BEFORE_KERNEL_START:
            case aclskOptionType::DCCI_AFTER_KERNEL_END:
                optJson["value"] = opt->GetStringListValue();
                break;

            case aclskOptionType::OPT_EXTEND_OPTION:
            case aclskOptionType::DEBUG_EXTEND_OPTION:
                optJson["value"] = opt->GetMapValue();
                break;

            case aclskOptionType::AGGRESSIVE_OPT_STRATEGIES:
                {
                    const auto* aggressiveOpt = static_cast<const AggressiveOptStrategiesOption*>(opt);
                    const auto& value = aggressiveOpt->GetValue();
                    nlohmann::ordered_json ubufLockIgnoreKernel = nlohmann::ordered_json::array();
                    const size_t kernelCnt = static_cast<size_t>(value.ubufLockIgnoreKernelCnt);
                    if (kernelCnt > 0 && value.ubufLockIgnoreKernel != nullptr) {
                        for (size_t kernelIdx = 0; kernelIdx < kernelCnt; kernelIdx++) {
                            if (value.ubufLockIgnoreKernel[kernelIdx] != nullptr) {
                                ubufLockIgnoreKernel.push_back(value.ubufLockIgnoreKernel[kernelIdx]);
                            }
                        }
                    }
                    optJson["value"] = {
                        {"eventBreakerBypass", value.eventBreakerBypass},
                        {"valueBreakerBypass", value.valueBreakerBypass},
                        {"taskBreakerBypass", value.taskBreakerBypass},
                        {"ubufLockIgnoreKernel", ubufLockIgnoreKernel}
                    };
                    break;
                }

            default:
                optJson["value"] = nullptr;
                break;
        }

        optionsJson[opt->GetName()] = optJson;
    }

    return optionsJson;
}
