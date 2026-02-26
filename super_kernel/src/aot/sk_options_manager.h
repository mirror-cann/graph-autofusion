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
 * \file sk_options_manager.h
 * \brief
 */

#ifndef SK_OPTIONS_MANAGER_H
#define SK_OPTIONS_MANAGER_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include <stdexcept>
#include <regex>
#include "super_kernel.h"


class OptOptionBase {
public:
    explicit OptOptionBase(std::string name, aclskOtionType optionTypeIn)
        : optionName(std::move(name)), optionType(optionTypeIn)
    {}

    virtual ~OptOptionBase() = default;

    virtual const std::string& GetName() const {
        return optionName;
    }

    virtual aclskOtionType GetType() const {
        return optionType;
    }

    virtual void SetValue(const uint32_t value) {}
    virtual void SetValue(const std::string& value) {}
    virtual void SetValue(const std::vector<std::string>& value) {}
    virtual void SetValue(const std::unordered_map<std::string, std::vector<std::string>>& value) {}

    virtual uint32_t GetIntValue() const {
        return 0;
    }

    virtual std::string GetStringValue() const {
        return "";
    }

    virtual std::unordered_map<std::string, std::vector<std::string>> GetMapValue() const {
        return {};
    }

    virtual std::vector<std::string> GetStringListValue() const {
        return {};
    }

protected:
    std::string optionName;
    aclskOtionType optionType = aclskOtionType::SK_OPTION_MAX;
};

class NumberOptOption : public OptOptionBase {
public:
    NumberOptOption(std::string name, aclskOtionType optionTypeIn,
        uint32_t defaultValue = 0, uint32_t minValue = 0, uint32_t maxValue = 0xffffffff) :
        OptOptionBase(std::move(name), optionTypeIn),
        optValue(defaultValue),
        optValueMin(minValue),
        optValueMax(maxValue)
    {}

    void SetValue(const uint32_t value) override;

    uint32_t GetIntValue() const override;
private:
    uint32_t optValue = 0;
    uint32_t optValueMin = 0;
    uint32_t optValueMax = 0;
};

class StringOptOption : public OptOptionBase {
public:
    StringOptOption(std::string name, aclskOtionType optionTypeIn,
        std::string defaultValue = {}) :
        OptOptionBase(std::move(name), optionTypeIn),
        optValue(defaultValue)
    {}

    void SetValue(const std::string& value) override;

    std::string GetStringValue() const override;

private:
    std::string optValue;
};

class StringListOptOption : public OptOptionBase {
public:
    StringListOptOption(std::string name, aclskOtionType optionTypeIn,
        std::vector<std::string> defaultValue = {}) :
        OptOptionBase(std::move(name), optionTypeIn),
        optValue(defaultValue)
    {}

    void SetValue(const std::vector<std::string>& value) override;

    std::vector<std::string> GetStringListValue() const override;

private:
    std::vector<std::string> optValue;
};

class MapOptOption : public OptOptionBase {
public:
    MapOptOption(std::string name, aclskOtionType optionTypeIn,
        std::unordered_map<std::string, std::vector<std::string>> defaultValue = {}) :
        OptOptionBase(std::move(name), optionTypeIn),
        optValue(defaultValue)
    {}

    void SetValue(const std::unordered_map<std::string, std::vector<std::string>>& value) override;

    std::unordered_map<std::string, std::vector<std::string>> GetMapValue() const override;

private:
    std::unordered_map<std::string, std::vector<std::string>> optValue;
};

class SuperKernelOptionsManager {
public:
    SuperKernelOptionsManager() = default;
    ~SuperKernelOptionsManager() = default;

    SuperKernelOptionsManager(const SuperKernelOptionsManager&) = default;
    SuperKernelOptionsManager& operator=(const SuperKernelOptionsManager&) = default;

    void AddOption(std::unique_ptr<OptOptionBase> option);

    OptOptionBase* GetOption(const aclskOtionType optType) const;

    bool JudgeDisableKernelDcci(std::vector<std::string>& dcciOps, const std::string& opName) const;

    bool EnableDebug() const;

    void SetOptOptionValue(aclskOption* option);

    void ParseOptions(aclskOptions* options);

private:
    std::unordered_map<aclskOtionType, std::unique_ptr<OptOptionBase>> optionMap;
};

#endif
