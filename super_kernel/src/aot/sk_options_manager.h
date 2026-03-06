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


/*!
 * \brief Base class for optimization options, providing common interface for different option types.
 *
 * This abstract base class defines the interface for all optimization option types, including
 * name, type, and value accessors. Concrete classes (NumberOptOption, StringOptOption, etc.)
 * implement type-specific value storage and retrieval logic.
 */
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

/*!
 * \brief Integer-type optimization option with value range validation.
 *
 * Stores numeric options with optional minimum and maximum bounds.
 * Values outside the specified range will be rejected during setting.
 */
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

/*!
 * \brief String-type optimization option.
 *
 * Stores string-based options such as file paths, operation names, or configuration strings.
 */
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

/*!
 * \brief String list-type optimization option.
 *
 * Stores collections of string values, useful for multi-valued options
 * such as operation lists, include directories, or configuration sets.
 */
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

/*!
 * \brief Map-type optimization option with string keys and string list values.
 *
 * Stores key-value pairs where each key maps to a list of strings.
 * Suitable for complex configuration options like operation attributes or named parameter lists.
 */
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

/*!
 * \brief Manager class for super kernel optimization options.
 *
 * This class manages a collection of optimization options of various types (integer, string,
 * string list, map). It provides methods to add, retrieve, and configure options, as well as
 * utility functions for option validation and filtering.
 *
 * Key features:
 * - Type-safe option storage using polymorphic OptOptionBase hierarchy
 * - Dynamic option registration and retrieval by type
 * - DCCI (disable kernel) operation filtering based on configured patterns
 * - Debug mode control
 * - Automatic parsing and validation of aclskOptions structures
 */
class SuperKernelOptionsManager {
public:
    SuperKernelOptionsManager() = default;
    ~SuperKernelOptionsManager() = default;

    SuperKernelOptionsManager(const SuperKernelOptionsManager&) = default;
    SuperKernelOptionsManager& operator=(const SuperKernelOptionsManager&) = default;

    /*!
     * \brief Add a new option to the manager
     * \param option Unique pointer to the option to add (ownership transferred)
     */
    void AddOption(std::unique_ptr<OptOptionBase> option);

    /*!
     * \brief Get an option by its type
     * \param optType The option type to retrieve
     * \return Pointer to the option, or nullptr if not found
     */
    OptOptionBase* GetOption(const aclskOtionType optType) const;

    /*!
     * \brief Judge whether a kernel should be disabled based on DCCI patterns
     * \param dcciOps List of DCCI operation patterns (may be modified by function)
     * \param opName The operation name to check against patterns
     * \return True if the kernel should be disabled, false otherwise
     */
    bool JudgeDisableKernelDcci(std::vector<std::string>& dcciOps, const std::string& opName) const;

    /*!
     * \brief Simple regex-like pattern matching without std::regex
     * \param pattern The regex pattern (supports: . *)
     * \param text The text to match against
     * \return True if text matches the pattern, false otherwise
     */
    static bool MatchRegex(const std::string& pattern, const std::string& text);

    /*!
     * \brief Check if debug mode is enabled
     * \return True if debug mode is enabled, false otherwise
     */
    bool EnableDebug() const;

    /*!
     * \brief Set option value from an aclskOption structure
     * \param option Pointer to the aclskOption structure containing value to set
     */
    void SetOptOptionValue(aclskOption* option);

    /*!
     * \brief Parse and populate options from an aclskOptions structure
     * \param options Pointer to the aclskOptions structure containing all options
     */
    void ParseOptions(aclskOptions* options);

private:
    std::unordered_map<aclskOtionType, std::unique_ptr<OptOptionBase>> optionMap;
};

#endif
