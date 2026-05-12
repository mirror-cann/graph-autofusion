/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <memory>
#define private public
#define protected public
#include "sk_options_manager.h"

/**
 * @brief 测试 fixture 类，用于 SuperKernelOptionsManager 单元测试
 */
class SuperKernelOptionsManagerTest : public testing::Test {
protected:
    void SetUp() override {
        // 每个测试用例前的初始化
        opts_test = std::make_unique<SuperKernelOptionsManager>();
    }

    void TearDown() override {
        // 每个测试用例后的清理
        GlobalMockObject::verify();
        opts_test.reset();
    }

    std::unique_ptr<SuperKernelOptionsManager> opts_test;
};

// ==================== OptOptionBase 基类测试 ====================

TEST_F(SuperKernelOptionsManagerTest, OptOptionBase_GetName)
{
    auto option = std::make_unique<OptOptionBase>("test_option", aclskOptionType::PRELOAD_CODE);
    EXPECT_EQ(option->GetName(), "test_option");
}

TEST_F(SuperKernelOptionsManagerTest, OptOptionBase_GetType)
{
    auto option = std::make_unique<OptOptionBase>("test_option", aclskOptionType::PRELOAD_CODE);
    EXPECT_EQ(option->GetType(), aclskOptionType::PRELOAD_CODE);
}

// ==================== NumberOptOption 测试 ====================

TEST_F(SuperKernelOptionsManagerTest, NumberOptOption_GetIntValue_Default)
{
    auto option = std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE);
    EXPECT_EQ(option->GetIntValue(), 0); // 默认值
}

TEST_F(SuperKernelOptionsManagerTest, NumberOptOption_GetIntValue_WithDefault)
{
    auto option = std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1);
    EXPECT_EQ(option->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, NumberOptOption_SetValue_Valid)
{
    auto option = std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 0, 0, 2);
    option->SetValue(1);
    EXPECT_EQ(option->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, NumberOptOption_SetValue_OutOfRangeLow)
{
    auto option = std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 0, 0, 2);
    option->SetValue(0xFFFFFFFF); // 超出范围
    EXPECT_EQ(option->GetIntValue(), 0); // 值不应改变
}

TEST_F(SuperKernelOptionsManagerTest, NumberOptOption_SetValue_OutOfRangeHigh)
{
    auto option = std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 0, 0, 2);
    option->SetValue(3); // 超出范围
    EXPECT_EQ(option->GetIntValue(), 0); // 值不应改变
}

TEST_F(SuperKernelOptionsManagerTest, NumberOptOption_SetValue_BoundaryMin)
{
    auto option = std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 0, 1, 10);
    option->SetValue(1); // 边界值
    EXPECT_EQ(option->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, NumberOptOption_SetValue_BoundaryMax)
{
    auto option = std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 0, 1, 10);
    option->SetValue(10); // 边界值
    EXPECT_EQ(option->GetIntValue(), 10);
}

// ==================== StringOptOption 测试 ====================

TEST_F(SuperKernelOptionsManagerTest, StringOptOption_GetStringValue_Default)
{
    auto option = std::make_unique<StringOptOption>("string_option", aclskOptionType::PRELOAD_CODE);
    EXPECT_EQ(option->GetStringValue(), "");
}

TEST_F(SuperKernelOptionsManagerTest, StringOptOption_GetStringValue_WithDefault)
{
    auto option = std::make_unique<StringOptOption>("string_option", aclskOptionType::PRELOAD_CODE, "default_value");
    EXPECT_EQ(option->GetStringValue(), "default_value");
}

TEST_F(SuperKernelOptionsManagerTest, StringOptOption_SetValue_Valid)
{
    auto option = std::make_unique<StringOptOption>("string_option", aclskOptionType::PRELOAD_CODE);
    option->SetValue("test_value");
    EXPECT_EQ(option->GetStringValue(), "test_value");
}

TEST_F(SuperKernelOptionsManagerTest, StringOptOption_SetValue_Empty)
{
    auto option = std::make_unique<StringOptOption>("string_option", aclskOptionType::PRELOAD_CODE);
    option->SetValue("initial");
    option->SetValue(""); // 空字符串
    EXPECT_EQ(option->GetStringValue(), "initial"); // 值不应改变
}

// ==================== StringListOptOption 测试 ====================

TEST_F(SuperKernelOptionsManagerTest, StringListOptOption_GetStringListValue_Default)
{
    auto option = std::make_unique<StringListOptOption>("list_option", aclskOptionType::PRELOAD_CODE);
    EXPECT_EQ(option->GetStringListValue().size(), 0); // 默认为空
}

TEST_F(SuperKernelOptionsManagerTest, StringListOptOption_GetStringListValue_WithDefault)
{
    std::vector<std::string> defaultVal = {"a", "b", "c"};
    auto option = std::make_unique<StringListOptOption>("list_option", aclskOptionType::PRELOAD_CODE, defaultVal);
    auto result = option->GetStringListValue();
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "a");
    EXPECT_EQ(result[1], "b");
    EXPECT_EQ(result[2], "c");
}

TEST_F(SuperKernelOptionsManagerTest, StringListOptOption_SetValue_Valid)
{
    auto option = std::make_unique<StringListOptOption>("list_option", aclskOptionType::PRELOAD_CODE);
    std::vector<std::string> val = {"op1", "op2", "op3"};
    option->SetValue(val);
    auto result = option->GetStringListValue();
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "op1");
    EXPECT_EQ(result[1], "op2");
    EXPECT_EQ(result[2], "op3");
}

TEST_F(SuperKernelOptionsManagerTest, StringListOptOption_SetValue_Empty)
{
    auto option = std::make_unique<StringListOptOption>("list_option", aclskOptionType::PRELOAD_CODE);
    std::vector<std::string> val = {"initial"};
    option->SetValue(val);
    option->SetValue(std::vector<std::string>()); // 空列表
    EXPECT_EQ(option->GetStringListValue().size(), 1); // 值不应改变
}

// ==================== MapOptOption 测试 ====================

TEST_F(SuperKernelOptionsManagerTest, MapOptOption_GetMapValue_Default)
{
    auto option = std::make_unique<MapOptOption>("map_option", aclskOptionType::PRELOAD_CODE);
    EXPECT_EQ(option->GetMapValue().size(), 0); // 默认为空
}

TEST_F(SuperKernelOptionsManagerTest, MapOptOption_GetMapValue_WithDefault)
{
    std::unordered_map<std::string, std::vector<std::string>> defaultVal = {
        {"key1", {"val1", "val2"}},
        {"key2", {"val3"}}
    };
    auto option = std::make_unique<MapOptOption>("map_option", aclskOptionType::PRELOAD_CODE, defaultVal);
    auto result = option->GetMapValue();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result["key1"].size(), 2);
    EXPECT_EQ(result["key2"].size(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, MapOptOption_SetValue_Valid)
{
    auto option = std::make_unique<MapOptOption>("map_option", aclskOptionType::PRELOAD_CODE);
    std::unordered_map<std::string, std::vector<std::string>> val = {
        {"op1", {"sub1", "sub2"}},
        {"op2", {"sub3"}}
    };
    option->SetValue(val);
    auto result = option->GetMapValue();
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result["op1"].size(), 2);
    EXPECT_EQ(result["op2"].size(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, MapOptOption_SetValue_Empty)
{
    auto option = std::make_unique<MapOptOption>("map_option", aclskOptionType::PRELOAD_CODE);
    std::unordered_map<std::string, std::vector<std::string>> val = {{"initial", {"val"}}};
    option->SetValue(val);
    option->SetValue(std::unordered_map<std::string, std::vector<std::string>>()); // 空 map
    EXPECT_EQ(option->GetMapValue().size(), 1); // 值不应改变
}

// ==================== SuperKernelOptionsManager::AddOption 测试 ====================

TEST_F(SuperKernelOptionsManagerTest, AddOption_Valid)
{
    auto option = std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1);
    opts_test->AddOption(std::move(option));
    auto result = opts_test->GetOption(aclskOptionType::PRELOAD_CODE);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetName(), "preload_code");
}

TEST_F(SuperKernelOptionsManagerTest, AddOption_Nullptr)
{
    opts_test->AddOption(nullptr);
    // 不应崩溃
    auto result = opts_test->GetOption(aclskOptionType::PRELOAD_CODE);
    EXPECT_EQ(result, nullptr);
}

TEST_F(SuperKernelOptionsManagerTest, AddOption_Duplicate)
{
    auto option1 = std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1);
    auto option2 = std::make_unique<NumberOptOption>("preload_code2", aclskOptionType::PRELOAD_CODE, 2);
    
    opts_test->AddOption(std::move(option1));
    opts_test->AddOption(std::move(option2));
    
    auto result = opts_test->GetOption(aclskOptionType::PRELOAD_CODE);
    ASSERT_NE(result, nullptr);
    // 第一个添加的值应该被保留
    EXPECT_EQ(static_cast<NumberOptOption*>(result)->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, AddOption_Multiple)
{
    opts_test->AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1));
    opts_test->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 4));
    opts_test->AddOption(std::make_unique<StringOptOption>("string_opt", aclskOptionType::STREAM_FUSION, "value"));
    
    ASSERT_NE(opts_test->GetOption(aclskOptionType::PRELOAD_CODE), nullptr);
    ASSERT_NE(opts_test->GetOption(aclskOptionType::SPLIT_MODE), nullptr);
    ASSERT_NE(opts_test->GetOption(aclskOptionType::STREAM_FUSION), nullptr);
}

// ==================== SuperKernelOptionsManager::GetOption 测试 ====================

TEST_F(SuperKernelOptionsManagerTest, GetOption_Found)
{
    opts_test->AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1));
    auto result = opts_test->GetOption(aclskOptionType::PRELOAD_CODE);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), aclskOptionType::PRELOAD_CODE);
}

TEST_F(SuperKernelOptionsManagerTest, GetOption_NotFound)
{
    auto result = opts_test->GetOption(aclskOptionType::PRELOAD_CODE);
    EXPECT_EQ(result, nullptr);
}

// ==================== SuperKernelOptionsManager::JudgeDisableKernelDcci 测试 ====================

TEST_F(SuperKernelOptionsManagerTest, JudgeDisableKernelDcci_Match)
{
    std::vector<std::string> dcciOps = {"Add", "Mul.*", ".*Op"};
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "Add"));
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "Mul"));
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "SomeOp"));
}

TEST_F(SuperKernelOptionsManagerTest, JudgeDisableKernelDcci_NoMatch)
{
    std::vector<std::string> dcciOps = {"Add", "Mul"};
    EXPECT_FALSE(opts_test->JudgeDisableKernelDcci(dcciOps, "Sub"));
    EXPECT_FALSE(opts_test->JudgeDisableKernelDcci(dcciOps, "Div"));
    EXPECT_FALSE(opts_test->JudgeDisableKernelDcci(dcciOps, "Conv"));
}

TEST_F(SuperKernelOptionsManagerTest, JudgeDisableKernelDcci_EmptyList)
{
    std::vector<std::string> dcciOps;
    EXPECT_FALSE(opts_test->JudgeDisableKernelDcci(dcciOps, "Add"));
}

TEST_F(SuperKernelOptionsManagerTest, JudgeDisableKernelDcci_InvalidRegex)
{
    std::vector<std::string> dcciOps = {"[invalid(regex"}; // 无效的正则表达式
    EXPECT_FALSE(opts_test->JudgeDisableKernelDcci(dcciOps, "Add")); // 应返回 false 而不是崩溃
}

TEST_F(SuperKernelOptionsManagerTest, JudgeDisableKernelDcci_ComplexPattern)
{
    // 注意：正则只支持 . 和 *，不支持 ^ 和 $
    // 使用完全匹配语义，模式需要匹配整个算子名称
    std::vector<std::string> dcciOps = {"Conv2D", "MatMul.*", ".*BatchNorm.*"};
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "Conv2D"));
    EXPECT_FALSE(opts_test->JudgeDisableKernelDcci(dcciOps, "Conv2DBackward")); // 不匹配 Conv2D
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "MatMul"));
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "MatMulV2"));
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "BatchNorm"));
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "SomeBatchNormOp"));
}

// ==================== SuperKernelOptionsManager::EnableDebug 测试 ====================

TEST_F(SuperKernelOptionsManagerTest, EnableDebug_WithDebugSyncAll)
{
    opts_test->AddOption(std::make_unique<NumberOptOption>("debug_sync_all", aclskOptionType::DEBUG_SYNC_ALL, 1));
    EXPECT_TRUE(opts_test->EnableDebug());
}

TEST_F(SuperKernelOptionsManagerTest, EnableDebug_WithDisableKernelDcci)
{
    opts_test->AddOption(std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL));
    EXPECT_FALSE(opts_test->EnableDebug());
}

TEST_F(SuperKernelOptionsManagerTest, EnableDebug_WithDcciBeforeKernelStart)
{
    opts_test->AddOption(std::make_unique<StringListOptOption>(
        "dcci_before_kernel_start", aclskOptionType::DCCI_BEFORE_KERNEL_START));
    EXPECT_FALSE(opts_test->EnableDebug());
}

TEST_F(SuperKernelOptionsManagerTest, EnableDebug_WithDcciAfterKernelEnd)
{
    opts_test->AddOption(std::make_unique<StringListOptOption>(
        "dcci_after_kernel_end", aclskOptionType::DCCI_AFTER_KERNEL_END));
    EXPECT_FALSE(opts_test->EnableDebug());
}

TEST_F(SuperKernelOptionsManagerTest, EnableDebug_WithoutDebugOptions)
{
    opts_test->AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1));
    EXPECT_FALSE(opts_test->EnableDebug());
}

TEST_F(SuperKernelOptionsManagerTest, EnableDebug_WithBothDebugOptions)
{
    opts_test->AddOption(std::make_unique<NumberOptOption>("debug_sync_all", aclskOptionType::DEBUG_SYNC_ALL, 1));
    opts_test->AddOption(std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL));
    EXPECT_TRUE(opts_test->EnableDebug());
}

TEST_F(SuperKernelOptionsManagerTest, EnableDebug_WithDebugSyncAllZero)
{
    opts_test->AddOption(std::make_unique<NumberOptOption>("debug_sync_all", aclskOptionType::DEBUG_SYNC_ALL, 0));
    EXPECT_FALSE(opts_test->EnableDebug());
}

// ==================== SuperKernelOptionsManager::SetOptOptionValue 测试 ====================

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_PreloadCode)
{
    aclskOption option;
    option.optionType = aclskOptionType::PRELOAD_CODE;
    option.preload.preloadMode = 1;
    
    opts_test->SetOptOptionValue(&option);
    
    auto result = opts_test->GetOption(aclskOptionType::PRELOAD_CODE);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(static_cast<NumberOptOption*>(result)->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_SplitMode)
{
    aclskOption option;
    option.optionType = aclskOptionType::SPLIT_MODE;
    option.splitMode.splitCnt = 3;
    
    opts_test->SetOptOptionValue(&option);
    
    auto result = opts_test->GetOption(aclskOptionType::SPLIT_MODE);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(static_cast<NumberOptOption*>(result)->GetIntValue(), 3);
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugDcciDisable)
{
    aclskOption option;
    option.optionType = aclskOptionType::DCCI_DISABLE_ON_KERNEL;
    
    const char* kernelNames[] = {"Add", "Mul", ".*Op"};
    option.disableKernelDcci.kernelNames = const_cast<char**>(kernelNames);
    option.disableKernelDcci.kernelCnt = 3;
    
    opts_test->SetOptOptionValue(&option);
    
    auto result = opts_test->GetOption(aclskOptionType::DCCI_DISABLE_ON_KERNEL);
    ASSERT_NE(result, nullptr);
    auto strList = static_cast<StringListOptOption*>(result)->GetStringListValue();
    EXPECT_EQ(strList.size(), 3);
    EXPECT_EQ(strList[0], "Add");
    EXPECT_EQ(strList[1], "Mul");
    EXPECT_EQ(strList[2], ".*Op");
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugDcciDisable_NullKernelNames)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DCCI_DISABLE_ON_KERNEL;
    option.disableKernelDcci.kernelNames = nullptr;
    option.disableKernelDcci.kernelCnt = 2;

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DCCI_DISABLE_ON_KERNEL);
    ASSERT_NE(result, nullptr);
    auto strList = static_cast<StringListOptOption*>(result)->GetStringListValue();
    EXPECT_TRUE(strList.empty());
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugDcciDisable_WithNullEntry)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DCCI_DISABLE_ON_KERNEL;

    char name0[] = "Add";
    char name2[] = "Mul";
    char* kernelNames[] = {name0, nullptr, name2};
    option.disableKernelDcci.kernelNames = kernelNames;
    option.disableKernelDcci.kernelCnt = 3;

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DCCI_DISABLE_ON_KERNEL);
    ASSERT_NE(result, nullptr);
    auto strList = static_cast<StringListOptOption*>(result)->GetStringListValue();
    EXPECT_EQ(strList.size(), 2);
    EXPECT_EQ(strList[0], "Add");
    EXPECT_EQ(strList[1], "Mul");
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugSyncAll)
{
    aclskOption option;
    option.optionType = aclskOptionType::DEBUG_SYNC_ALL;
    option.debugSync.debugSyncAll = 1;
    
    opts_test->SetOptOptionValue(&option);
    
    auto result = opts_test->GetOption(aclskOptionType::DEBUG_SYNC_ALL);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(static_cast<NumberOptOption*>(result)->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DcciBeforeKernelStart)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DCCI_BEFORE_KERNEL_START;

    const char* kernelNames[] = {"Add", "Mul", ".*Op"};
    option.dcciBeforeKernelStart.kernelNames = const_cast<char**>(kernelNames);
    option.dcciBeforeKernelStart.kernelCnt = 3;

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DCCI_BEFORE_KERNEL_START);
    ASSERT_NE(result, nullptr);
    auto strList = static_cast<StringListOptOption*>(result)->GetStringListValue();
    EXPECT_EQ(strList.size(), 3);
    EXPECT_EQ(strList[0], "Add");
    EXPECT_EQ(strList[1], "Mul");
    EXPECT_EQ(strList[2], ".*Op");
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DcciBeforeKernelStart_NullKernelNames)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DCCI_BEFORE_KERNEL_START;
    option.dcciBeforeKernelStart.kernelNames = nullptr;
    option.dcciBeforeKernelStart.kernelCnt = 2;

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DCCI_BEFORE_KERNEL_START);
    ASSERT_NE(result, nullptr);
    auto strList = static_cast<StringListOptOption*>(result)->GetStringListValue();
    EXPECT_TRUE(strList.empty());
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DcciAfterKernelEnd)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DCCI_AFTER_KERNEL_END;

    const char* kernelNames[] = {"Add", "Mul", ".*Op"};
    option.dcciAfterKernelEnd.kernelNames = const_cast<char**>(kernelNames);
    option.dcciAfterKernelEnd.kernelCnt = 3;

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DCCI_AFTER_KERNEL_END);
    ASSERT_NE(result, nullptr);
    auto strList = static_cast<StringListOptOption*>(result)->GetStringListValue();
    EXPECT_EQ(strList.size(), 3);
    EXPECT_EQ(strList[0], "Add");
    EXPECT_EQ(strList[1], "Mul");
    EXPECT_EQ(strList[2], ".*Op");
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DcciAfterKernelEnd_NullKernelNames)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DCCI_AFTER_KERNEL_END;
    option.dcciAfterKernelEnd.kernelNames = nullptr;
    option.dcciAfterKernelEnd.kernelCnt = 2;

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DCCI_AFTER_KERNEL_END);
    ASSERT_NE(result, nullptr);
    auto strList = static_cast<StringListOptOption*>(result)->GetStringListValue();
    EXPECT_TRUE(strList.empty());
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_OptExtendOption_TrimmedValid)
{
    aclskOption option {};
    option.optionType = aclskOptionType::OPT_EXTEND_OPTION;
    std::string rawValue = "  key1=value1 : key2=value2,value3  ";
    option.optExtend.value = const_cast<char*>(rawValue.c_str());

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::OPT_EXTEND_OPTION);
    ASSERT_NE(result, nullptr);
    auto valueMap = static_cast<MapOptOption*>(result)->GetMapValue();
    ASSERT_EQ(valueMap.size(), 2);
    EXPECT_EQ(valueMap["key1"].size(), 1);
    EXPECT_EQ(valueMap["key1"][0], "value1");
    EXPECT_EQ(valueMap["key2"].size(), 2);
    EXPECT_EQ(valueMap["key2"][0], "value2");
    EXPECT_EQ(valueMap["key2"][1], "value3");
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_OptExtendOption_InvalidDangerousChar)
{
    aclskOption option {};
    option.optionType = aclskOptionType::OPT_EXTEND_OPTION;
    std::string rawValue = "key1=value1:key2=rm -rf /";
    option.optExtend.value = const_cast<char*>(rawValue.c_str());

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::OPT_EXTEND_OPTION);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(static_cast<MapOptOption*>(result)->GetMapValue().empty());
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_OptExtendOption_EmptyAfterTrim)
{
    aclskOption option {};
    option.optionType = aclskOptionType::OPT_EXTEND_OPTION;
    std::string rawValue = "   ";
    option.optExtend.value = const_cast<char*>(rawValue.c_str());

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::OPT_EXTEND_OPTION);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(static_cast<MapOptOption*>(result)->GetMapValue().empty());
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_AggressiveOptStrategies)
{
    aclskOption option {};
    option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
    option.aggressiveOpts.eventBreakerBypass = 7;
    option.aggressiveOpts.valueBreakerBypass = ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT;
    option.aggressiveOpts.taskBreakerBypass = 1;

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::AGGRESSIVE_OPT_STRATEGIES);
    ASSERT_NE(result, nullptr);
    const auto& aggressiveOpts = static_cast<AggressiveOptStrategiesOption*>(result)->GetValue();
    EXPECT_EQ(aggressiveOpts.eventBreakerBypass, 7U);
    EXPECT_EQ(aggressiveOpts.valueBreakerBypass, ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT);
    EXPECT_EQ(aggressiveOpts.taskBreakerBypass, 1);
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugExtendOption_Nullptr)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DEBUG_EXTEND_OPTION;
    option.debugExtend.value = nullptr;

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DEBUG_EXTEND_OPTION);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(static_cast<MapOptOption*>(result)->GetMapValue().empty());
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugExtendOption_EmptyString)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DEBUG_EXTEND_OPTION;
    std::string rawValue = "";
    option.debugExtend.value = const_cast<char*>(rawValue.c_str());

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DEBUG_EXTEND_OPTION);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(static_cast<MapOptOption*>(result)->GetMapValue().empty());
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugExtendOption_TooLong)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DEBUG_EXTEND_OPTION;
    std::string rawValue(1025, 'a');
    option.debugExtend.value = const_cast<char*>(rawValue.c_str());

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DEBUG_EXTEND_OPTION);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(static_cast<MapOptOption*>(result)->GetMapValue().empty());
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugExtendOption_DuplicateKey)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DEBUG_EXTEND_OPTION;
    std::string rawValue = "key1=value1:key1=value2";
    option.debugExtend.value = const_cast<char*>(rawValue.c_str());

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DEBUG_EXTEND_OPTION);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(static_cast<MapOptOption*>(result)->GetMapValue().empty());
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugExtendOption_EmptySubValue)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DEBUG_EXTEND_OPTION;
    std::string rawValue = "key1=value1,,value3";
    option.debugExtend.value = const_cast<char*>(rawValue.c_str());

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DEBUG_EXTEND_OPTION);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(static_cast<MapOptOption*>(result)->GetMapValue().empty());
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_Unsupported)
{
    aclskOption option;
    option.optionType = static_cast<aclskOptionType>(100); // 不支持的类型
    
    // 不应崩溃
    opts_test->SetOptOptionValue(&option);
}

// ==================== SuperKernelOptionsManager::ParseOptions 测试 ====================

TEST_F(SuperKernelOptionsManagerTest, ParseOptions_Nullptr)
{
    // 不应崩溃
    opts_test->ParseOptions(nullptr);
}

TEST_F(SuperKernelOptionsManagerTest, ParseOptions_SingleOption)
{
    aclskOption options[1];
    options[0].optionType = aclskOptionType::PRELOAD_CODE;
    options[0].preload.preloadMode = 1;
    
    aclskOptions optList;
    optList.options = options;
    optList.numOptions = 1;
    
    opts_test->ParseOptions(&optList);
    
    auto result = opts_test->GetOption(aclskOptionType::PRELOAD_CODE);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(static_cast<NumberOptOption*>(result)->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, ParseOptions_MultipleOptions)
{
    aclskOption options[3];
    
    options[0].optionType = aclskOptionType::PRELOAD_CODE;
    options[0].preload.preloadMode = 1;
    
    options[1].optionType = aclskOptionType::SPLIT_MODE;
    options[1].splitMode.splitCnt = 4;
    
    options[2].optionType = aclskOptionType::DEBUG_SYNC_ALL;
    options[2].debugSync.debugSyncAll = 1;
    
    aclskOptions optList;
    optList.options = options;
    optList.numOptions = 3;
    
    opts_test->ParseOptions(&optList);
    
    auto opt1 = opts_test->GetOption(aclskOptionType::PRELOAD_CODE);
    auto opt2 = opts_test->GetOption(aclskOptionType::SPLIT_MODE);
    auto opt3 = opts_test->GetOption(aclskOptionType::DEBUG_SYNC_ALL);
    
    ASSERT_NE(opt1, nullptr);
    ASSERT_NE(opt2, nullptr);
    ASSERT_NE(opt3, nullptr);
    
    EXPECT_EQ(static_cast<NumberOptOption*>(opt1)->GetIntValue(), 1);
    EXPECT_EQ(static_cast<NumberOptOption*>(opt2)->GetIntValue(), 4);
    EXPECT_EQ(static_cast<NumberOptOption*>(opt3)->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, ParseOptions_DuplicateOption)
{
    aclskOption options[2];

    options[0].optionType = aclskOptionType::PRELOAD_CODE;
    options[0].preload.preloadMode = 1;

    options[1].optionType = aclskOptionType::PRELOAD_CODE; // 重复
    options[1].preload.preloadMode = 2;

    aclskOptions optList;
    optList.options = options;
    optList.numOptions = 2;

    opts_test->ParseOptions(&optList);

    auto result = opts_test->GetOption(aclskOptionType::PRELOAD_CODE);
    ASSERT_NE(result, nullptr);
    // 第一个值应该被保留
    EXPECT_EQ(static_cast<NumberOptOption*>(result)->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, ParseOptions_AllOptionTypes)
{
    // 测试所有支持的不同选项类型
    aclskOption options[4];
    
    options[0].optionType = aclskOptionType::PRELOAD_CODE;
    options[0].preload.preloadMode = 1;
    
    options[1].optionType = aclskOptionType::SPLIT_MODE;
    options[1].splitMode.splitCnt = 3;
    
    const char* dcciKernels[] = {"Add", "Mul"};
    options[2].optionType = aclskOptionType::DCCI_DISABLE_ON_KERNEL;
    options[2].disableKernelDcci.kernelNames = const_cast<char**>(dcciKernels);
    options[2].disableKernelDcci.kernelCnt = 2;
    
    options[3].optionType = aclskOptionType::DEBUG_SYNC_ALL;
    options[3].debugSync.debugSyncAll = 1;
    
    aclskOptions optList;
    optList.options = options;
    optList.numOptions = 4;
    
    opts_test->ParseOptions(&optList);
    
    // 验证所有选项都已正确解析
    auto opt1 = opts_test->GetOption(aclskOptionType::PRELOAD_CODE);
    auto opt2 = opts_test->GetOption(aclskOptionType::SPLIT_MODE);
    auto opt3 = opts_test->GetOption(aclskOptionType::DCCI_DISABLE_ON_KERNEL);
    auto opt4 = opts_test->GetOption(aclskOptionType::DEBUG_SYNC_ALL);
    
    ASSERT_NE(opt1, nullptr);
    ASSERT_NE(opt2, nullptr);
    ASSERT_NE(opt3, nullptr);
    ASSERT_NE(opt4, nullptr);
    
    EXPECT_EQ(static_cast<NumberOptOption*>(opt1)->GetIntValue(), 1);
    EXPECT_EQ(static_cast<NumberOptOption*>(opt2)->GetIntValue(), 3);
    
    auto dcciList = static_cast<StringListOptOption*>(opt3)->GetStringListValue();
    EXPECT_EQ(dcciList.size(), 2);
    EXPECT_EQ(dcciList[0], "Add");
    EXPECT_EQ(dcciList[1], "Mul");
    
    EXPECT_EQ(static_cast<NumberOptOption*>(opt4)->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, ParseOptions_ExtendOptions)
{
    aclskOption options[2] {};
    std::string optValue = "  level=2,3  ";
    std::string debugValue = "trace=on:path=/tmp/sk_meta";

    options[0].optionType = aclskOptionType::OPT_EXTEND_OPTION;
    options[0].optExtend.value = const_cast<char*>(optValue.c_str());

    options[1].optionType = aclskOptionType::DEBUG_EXTEND_OPTION;
    options[1].debugExtend.value = const_cast<char*>(debugValue.c_str());

    aclskOptions optList;
    optList.options = options;
    optList.numOptions = 2;

    opts_test->ParseOptions(&optList);

    auto optExtend = opts_test->GetOption(aclskOptionType::OPT_EXTEND_OPTION);
    auto debugExtend = opts_test->GetOption(aclskOptionType::DEBUG_EXTEND_OPTION);

    ASSERT_NE(optExtend, nullptr);
    ASSERT_NE(debugExtend, nullptr);
    auto optValueMap = static_cast<MapOptOption*>(optExtend)->GetMapValue();
    auto debugValueMap = static_cast<MapOptOption*>(debugExtend)->GetMapValue();
    ASSERT_EQ(optValueMap.size(), 1);
    ASSERT_EQ(debugValueMap.size(), 2);
    EXPECT_EQ(optValueMap["level"].size(), 2);
    EXPECT_EQ(optValueMap["level"][0], "2");
    EXPECT_EQ(optValueMap["level"][1], "3");
    EXPECT_EQ(debugValueMap["trace"].size(), 1);
    EXPECT_EQ(debugValueMap["trace"][0], "on");
    EXPECT_EQ(debugValueMap["path"].size(), 1);
    EXPECT_EQ(debugValueMap["path"][0], "/tmp/sk_meta");
}

TEST_F(SuperKernelOptionsManagerTest, ParseOptions_WithPreExistingOptions)
{
    // 先添加一些选项
    opts_test->AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 0));
    
    // 然后解析包含相同类型选项的配置
    aclskOption options[2];
    options[0].optionType = aclskOptionType::PRELOAD_CODE;
    options[0].preload.preloadMode = 1;
    options[1].optionType = aclskOptionType::SPLIT_MODE;
    options[1].splitMode.splitCnt = 4;
    
    aclskOptions optList;
    optList.options = options;
    optList.numOptions = 2;
    
    opts_test->ParseOptions(&optList);
    
    // 验证新选项没有被覆盖（因为已存在）
    auto preloadOpt = opts_test->GetOption(aclskOptionType::PRELOAD_CODE);
    auto splitOpt = opts_test->GetOption(aclskOptionType::SPLIT_MODE);
    
    ASSERT_NE(preloadOpt, nullptr);
    ASSERT_NE(splitOpt, nullptr);
    
    // PRELOAD_CODE 应该保持原值（因为已存在）
    EXPECT_EQ(static_cast<NumberOptOption*>(preloadOpt)->GetIntValue(), 0);
    // SPLIT_MODE 应该是新值（因为不存在）
    EXPECT_EQ(static_cast<NumberOptOption*>(splitOpt)->GetIntValue(), 4);
}

TEST_F(SuperKernelOptionsManagerTest, ParseOptions_MixedValidAndInvalid)
{
    // 测试混合有效和无效的选项
    aclskOption options[3];
    
    options[0].optionType = aclskOptionType::PRELOAD_CODE;
    options[0].preload.preloadMode = 1;
    
    options[1].optionType = aclskOptionType::SPLIT_MODE;
    options[1].splitMode.splitCnt = 2;
    
    options[2].optionType = aclskOptionType::DEBUG_SYNC_ALL;
    options[2].debugSync.debugSyncAll = 1;
    
    aclskOptions optList;
    optList.options = options;
    optList.numOptions = 3;
    
    opts_test->ParseOptions(&optList);
    
    // 验证所有有效选项都已解析
    ASSERT_NE(opts_test->GetOption(aclskOptionType::PRELOAD_CODE), nullptr);
    ASSERT_NE(opts_test->GetOption(aclskOptionType::SPLIT_MODE), nullptr);
    ASSERT_NE(opts_test->GetOption(aclskOptionType::DEBUG_SYNC_ALL), nullptr);
    
    EXPECT_EQ(static_cast<NumberOptOption*>(opts_test->GetOption(aclskOptionType::PRELOAD_CODE))->GetIntValue(), 1);
    EXPECT_EQ(static_cast<NumberOptOption*>(opts_test->GetOption(aclskOptionType::SPLIT_MODE))->GetIntValue(), 2);
    EXPECT_EQ(static_cast<NumberOptOption*>(opts_test->GetOption(aclskOptionType::DEBUG_SYNC_ALL))->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, ParseOptions_EmptyOptionsList)
{
    // 测试空选项列表
    aclskOptions optList;
    optList.options = nullptr;
    optList.numOptions = 0;
    
    // 不应崩溃
    opts_test->ParseOptions(&optList);
    
    // 验证没有选项被添加
    EXPECT_EQ(opts_test->GetOption(aclskOptionType::PRELOAD_CODE), nullptr);
}

TEST_F(SuperKernelOptionsManagerTest, ParseOptions_DebugSyncAllZero_NotEnableDebug)
{
    aclskOption options[1];
    options[0].optionType = aclskOptionType::DEBUG_SYNC_ALL;
    options[0].debugSync.debugSyncAll = 0;

    aclskOptions optList;
    optList.options = options;
    optList.numOptions = 1;

    opts_test->ParseOptions(&optList);
    EXPECT_FALSE(opts_test->EnableDebug());
}

TEST_F(SuperKernelOptionsManagerTest, ParseOptions_LargeNumberOptions)
{
    // 测试大量选项的解析
    const int numOptions = 10;
    aclskOption* options = new aclskOption[numOptions];
    
    for (int i = 0; i < numOptions; i++) {
        if (i % 4 == 0) {
            options[i].optionType = aclskOptionType::PRELOAD_CODE;
            options[i].preload.preloadMode = i;
        } else if (i % 4 == 1) {
            options[i].optionType = aclskOptionType::SPLIT_MODE;
            options[i].splitMode.splitCnt = i;
        } else if (i % 4 == 2) {
            const char* kernelNames[] = {"TestOp"};
            options[i].optionType = aclskOptionType::DCCI_DISABLE_ON_KERNEL;
            options[i].disableKernelDcci.kernelNames = const_cast<char**>(kernelNames);
            options[i].disableKernelDcci.kernelCnt = 1;
        } else {
            options[i].optionType = aclskOptionType::DEBUG_SYNC_ALL;
            options[i].debugSync.debugSyncAll = i % 2;
        }
    }
    
    aclskOptions optList;
    optList.options = options;
    optList.numOptions = numOptions;
    
    opts_test->ParseOptions(&optList);
    
    // 验证不同类型的选项数量
    int preloadCount = 0;
    int splitCount = 0;
    int dcciCount = 0;
    int syncCount = 0;
    
    for (int i = 0; i < numOptions; i++) {
        if (i % 4 == 0 && opts_test->GetOption(aclskOptionType::PRELOAD_CODE)) preloadCount++;
        else if (i % 4 == 1 && opts_test->GetOption(aclskOptionType::SPLIT_MODE)) splitCount++;
        else if (i % 4 == 2 && opts_test->GetOption(aclskOptionType::DCCI_DISABLE_ON_KERNEL)) dcciCount++;
        else if (i % 4 == 3 && opts_test->GetOption(aclskOptionType::DEBUG_SYNC_ALL)) syncCount++;
    }
    
    EXPECT_GT(preloadCount, 0);
    EXPECT_GT(splitCount, 0);
    EXPECT_GT(dcciCount, 0);
    EXPECT_GT(syncCount, 0);
    
    delete[] options;
}

// ==================== MatchRegex 单元测试 ====================
// 仅支持 "." 和 "*" 的正则表达式匹配（完全匹配，非部分匹配）
//   .   - 匹配任意单个字符
//   *   - 匹配前一个字符0次或多次

TEST_F(SuperKernelOptionsManagerTest, MatchRegex_ExactMatch)
{
    // 精确匹配（无特殊字符）
    EXPECT_TRUE(opts_test->MatchRegex("abc", "abc"));
    EXPECT_FALSE(opts_test->MatchRegex("abc", "abcd"));
    EXPECT_FALSE(opts_test->MatchRegex("abc", "ab"));
    EXPECT_FALSE(opts_test->MatchRegex("abc", "ABC"));
}

TEST_F(SuperKernelOptionsManagerTest, MatchRegex_DotMatch)
{
    // . 匹配任意单个字符
    EXPECT_TRUE(opts_test->MatchRegex("a.c", "abc"));
    EXPECT_TRUE(opts_test->MatchRegex("a.c", "aXc"));
    EXPECT_TRUE(opts_test->MatchRegex("a.c", "a1c"));
    EXPECT_FALSE(opts_test->MatchRegex("a.c", "ac"));
    EXPECT_FALSE(opts_test->MatchRegex("a.c", "abcc"));
    EXPECT_TRUE(opts_test->MatchRegex(".", "a"));
    EXPECT_TRUE(opts_test->MatchRegex(".", "b"));
    EXPECT_FALSE(opts_test->MatchRegex(".", ""));
    EXPECT_TRUE(opts_test->MatchRegex("..", "ab"));
    EXPECT_FALSE(opts_test->MatchRegex("..", "a"));
}

TEST_F(SuperKernelOptionsManagerTest, MatchRegex_StarMatch)
{
    // * 匹配前一个字符0次或多次
    EXPECT_TRUE(opts_test->MatchRegex("ab*c", "ac"));      // b出现0次
    EXPECT_TRUE(opts_test->MatchRegex("ab*c", "abc"));     // b出现1次
    EXPECT_TRUE(opts_test->MatchRegex("ab*c", "abbc"));    // b出现2次
    EXPECT_TRUE(opts_test->MatchRegex("ab*c", "abbbbbbc")); // b出现多次
    EXPECT_FALSE(opts_test->MatchRegex("ab*c", "axc"));
    EXPECT_TRUE(opts_test->MatchRegex("a*", ""));          // a出现0次
    EXPECT_TRUE(opts_test->MatchRegex("a*", "a"));         // a出现1次
    EXPECT_TRUE(opts_test->MatchRegex("a*", "aaaaa"));     // a出现多次
}

TEST_F(SuperKernelOptionsManagerTest, MatchRegex_DotStar)
{
    // .* 匹配任意字符0次或多次
    EXPECT_TRUE(opts_test->MatchRegex(".*", "anything"));
    EXPECT_TRUE(opts_test->MatchRegex(".*", ""));
    EXPECT_TRUE(opts_test->MatchRegex("a.*b", "ab"));
    EXPECT_TRUE(opts_test->MatchRegex("a.*b", "axxxb"));
    EXPECT_TRUE(opts_test->MatchRegex("a.*b", "a123456789b"));
    EXPECT_FALSE(opts_test->MatchRegex("a.*b", "abc"));    // 没有结尾的b
}

TEST_F(SuperKernelOptionsManagerTest, MatchRegex_ComplexPatterns)
{
    // 复杂模式测试
    EXPECT_TRUE(opts_test->MatchRegex(".*DequantSwigluQuant.*", "DequantSwigluQuant"));
    EXPECT_TRUE(opts_test->MatchRegex(".*DequantSwigluQuant.*", "abcDequantSwigluQuant"));
    EXPECT_TRUE(opts_test->MatchRegex(".*DequantSwigluQuant.*", "DequantSwigluQuantxyz"));
    EXPECT_TRUE(opts_test->MatchRegex(".*DequantSwigluQuant.*", "abcDequantSwigluQuantxyz"));
    EXPECT_FALSE(opts_test->MatchRegex(".*DequantSwigluQuant.*", "DequantSwigluQuan"));
    
    EXPECT_TRUE(opts_test->MatchRegex("Conv.*", "Conv"));
    EXPECT_TRUE(opts_test->MatchRegex("Conv.*", "Conv2D"));
    EXPECT_TRUE(opts_test->MatchRegex("Conv.*", "Convolution"));
    EXPECT_TRUE(opts_test->MatchRegex("Conv.*", "ConvolutionBackward"));
    EXPECT_FALSE(opts_test->MatchRegex("Conv.*", "NotConv"));
}

TEST_F(SuperKernelOptionsManagerTest, MatchRegex_EmptyPattern)
{
    // 空模式测试
    EXPECT_TRUE(opts_test->MatchRegex("", ""));
    EXPECT_FALSE(opts_test->MatchRegex("", "abc"));
}

TEST_F(SuperKernelOptionsManagerTest, MatchRegex_InvalidLeadingStar)
{
    EXPECT_FALSE(opts_test->MatchRegex("*abc", "abc"));
    EXPECT_FALSE(opts_test->MatchRegex("*", ""));
}

TEST_F(SuperKernelOptionsManagerTest, MatchRegex_MixedPatterns)
{
    // 混合模式测试
    EXPECT_TRUE(opts_test->MatchRegex("a.*b", "ab"));
    EXPECT_TRUE(opts_test->MatchRegex("a.*b", "a123b"));
    EXPECT_TRUE(opts_test->MatchRegex("a.b", "aXb"));
    EXPECT_TRUE(opts_test->MatchRegex("a.b", "a1b"));
    EXPECT_TRUE(opts_test->MatchRegex("a.*b.*c", "abc"));
    EXPECT_TRUE(opts_test->MatchRegex("a.*b.*c", "aXXbYYc"));
    // 完全匹配：模式 "a.*b.*c" 只能匹配到 'c'，文本后面还有 "ZZZ"，无法完全匹配
    EXPECT_FALSE(opts_test->MatchRegex("a.*b.*c", "aXXXbYYYcZZZ"));
    // 若要完全匹配，模式应为 "a.*b.*c.*"
    EXPECT_TRUE(opts_test->MatchRegex("a.*b.*c.*", "aXXXbYYYcZZZ"));
    EXPECT_FALSE(opts_test->MatchRegex("a.*b.*c", "aXXXc"));
}

TEST_F(SuperKernelOptionsManagerTest, MatchRegex_DcciScenario)
{
    // DCCI 场景测试
    std::vector<std::string> dcciOps = {"Add", "Mul.*", ".*Op"};
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "Add"));
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "Mul"));
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "MulV2"));
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "SomeOp"));
    EXPECT_FALSE(opts_test->JudgeDisableKernelDcci(dcciOps, "Sub"));
    EXPECT_FALSE(opts_test->JudgeDisableKernelDcci(dcciOps, "Conv"));
}

// ==================== SetOptOptionValue: DEBUG_OP_EXEC_TRACE 测试 ====================

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugOpExecTrace_Enable)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DEBUG_OP_EXEC_TRACE;
    option.debugOpExecTrace.enableOpExecTrace = 1;

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DEBUG_OP_EXEC_TRACE);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(static_cast<NumberOptOption*>(result)->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugOpExecTrace_Disable)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DEBUG_OP_EXEC_TRACE;
    option.debugOpExecTrace.enableOpExecTrace = 0;

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DEBUG_OP_EXEC_TRACE);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(static_cast<NumberOptOption*>(result)->GetIntValue(), 0);
}

// ==================== SetOptOptionValue: DEBUG_CROSS_CORE_SYNC_CHECK 测试 ====================

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugCrossCoreSyncCheck_Enable)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK;
    option.debugCrossCoreSyncCheck.enableCrossCoreSyncCheck = 1;

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(static_cast<NumberOptOption*>(result)->GetIntValue(), 1);
}

TEST_F(SuperKernelOptionsManagerTest, SetOptOptionValue_DebugCrossCoreSyncCheck_Disable)
{
    aclskOption option {};
    option.optionType = aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK;
    option.debugCrossCoreSyncCheck.enableCrossCoreSyncCheck = 0;

    opts_test->SetOptOptionValue(&option);

    auto result = opts_test->GetOption(aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(static_cast<NumberOptOption*>(result)->GetIntValue(), 0);
}

TEST_F(SuperKernelOptionsManagerTest, EnableMixKernelSplit_DefaultBySocName)
{
    EXPECT_EQ(opts_test->GetSocName(), "Ascend910B");
    EXPECT_FALSE(opts_test->EnableMixKernelSplit());
}

// ==================== 综合测试 ====================

TEST_F(SuperKernelOptionsManagerTest, CompleteWorkflow)
{
    // 添加多个选项
    opts_test->AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1));
    opts_test->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 4));
    opts_test->AddOption(std::make_unique<NumberOptOption>("debug_sync_all", aclskOptionType::DEBUG_SYNC_ALL, 1));
    
    // 验证所有选项都已添加
    EXPECT_NE(opts_test->GetOption(aclskOptionType::PRELOAD_CODE), nullptr);
    EXPECT_NE(opts_test->GetOption(aclskOptionType::SPLIT_MODE), nullptr);
    EXPECT_NE(opts_test->GetOption(aclskOptionType::DEBUG_SYNC_ALL), nullptr);
    
    // 验证 debug 模式已启用
    EXPECT_TRUE(opts_test->EnableDebug());
    
    // 验证 DCCI 判断
    std::vector<std::string> dcciOps = {"Add", "Mul.*"};
    EXPECT_TRUE(opts_test->JudgeDisableKernelDcci(dcciOps, "Add"));
    EXPECT_FALSE(opts_test->JudgeDisableKernelDcci(dcciOps, "Sub"));
}

// ==================== SuperKernelOptionsManager::ToJson Tests ====================

TEST_F(SuperKernelOptionsManagerTest, ToJson_EmptyOptionsManager)
{
    nlohmann::ordered_json json = opts_test->ToJson();
    EXPECT_TRUE(json.is_object());
    EXPECT_EQ(json.size(), 0);
}

TEST_F(SuperKernelOptionsManagerTest, ToJson_SingleIntegerOption)
{
    opts_test->AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1));
    
    nlohmann::ordered_json json = opts_test->ToJson();
    EXPECT_EQ(json.size(), 1);
    EXPECT_TRUE(json.contains("preload_code"));
    EXPECT_EQ(json["preload_code"]["name"], "preload_code");
    EXPECT_EQ(json["preload_code"]["type"], static_cast<int>(aclskOptionType::PRELOAD_CODE));
    EXPECT_EQ(json["preload_code"]["value"], 1);
}

TEST_F(SuperKernelOptionsManagerTest, ToJson_SingleStringListOption)
{
    std::vector<std::string> kernels = {"Add", "Mul", "Conv"};
    opts_test->AddOption(std::make_unique<StringListOptOption>(
        "dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL, kernels));
    
    nlohmann::ordered_json json = opts_test->ToJson();
    EXPECT_EQ(json.size(), 1);
    EXPECT_TRUE(json.contains("dcci_disable"));
    EXPECT_EQ(json["dcci_disable"]["name"], "dcci_disable");
    EXPECT_EQ(json["dcci_disable"]["type"], static_cast<int>(aclskOptionType::DCCI_DISABLE_ON_KERNEL));
    EXPECT_EQ(json["dcci_disable"]["value"].size(), 3);
    EXPECT_EQ(json["dcci_disable"]["value"][0], "Add");
    EXPECT_EQ(json["dcci_disable"]["value"][1], "Mul");
    EXPECT_EQ(json["dcci_disable"]["value"][2], "Conv");
}

TEST_F(SuperKernelOptionsManagerTest, ToJson_SingleMapOption)
{
    std::unordered_map<std::string, std::vector<std::string>> mapValue = {
        {"key1", {"val1", "val2"}},
        {"key2", {"val3"}}
    };
    opts_test->AddOption(std::make_unique<MapOptOption>(
        "opt_extend", aclskOptionType::OPT_EXTEND_OPTION, mapValue));
    
    nlohmann::ordered_json json = opts_test->ToJson();
    EXPECT_EQ(json.size(), 1);
    EXPECT_TRUE(json.contains("opt_extend"));
    EXPECT_EQ(json["opt_extend"]["name"], "opt_extend");
    EXPECT_EQ(json["opt_extend"]["type"], static_cast<int>(aclskOptionType::OPT_EXTEND_OPTION));
    EXPECT_TRUE(json["opt_extend"]["value"].is_object());
    EXPECT_EQ(json["opt_extend"]["value"]["key1"].size(), 2);
    EXPECT_EQ(json["opt_extend"]["value"]["key1"][0], "val1");
    EXPECT_EQ(json["opt_extend"]["value"]["key1"][1], "val2");
    EXPECT_EQ(json["opt_extend"]["value"]["key2"].size(), 1);
    EXPECT_EQ(json["opt_extend"]["value"]["key2"][0], "val3");
}

TEST_F(SuperKernelOptionsManagerTest, ToJson_MultipleMixedOptions)
{
    opts_test->AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 2));
    opts_test->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 4));
    opts_test->AddOption(std::make_unique<NumberOptOption>("debug_sync_all", aclskOptionType::DEBUG_SYNC_ALL, 1));
    
    std::vector<std::string> dcciKernels = {"Add", "Mul"};
    opts_test->AddOption(std::make_unique<StringListOptOption>(
        "dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL, dcciKernels));
    
    nlohmann::ordered_json json = opts_test->ToJson();
    EXPECT_EQ(json.size(), 4);
    
    EXPECT_TRUE(json.contains("preload_code"));
    EXPECT_EQ(json["preload_code"]["value"], 2);
    
    EXPECT_TRUE(json.contains("split_mode"));
    EXPECT_EQ(json["split_mode"]["value"], 4);
    
    EXPECT_TRUE(json.contains("debug_sync_all"));
    EXPECT_EQ(json["debug_sync_all"]["value"], 1);
    
    EXPECT_TRUE(json.contains("dcci_disable"));
    EXPECT_EQ(json["dcci_disable"]["value"].size(), 2);
}

TEST_F(SuperKernelOptionsManagerTest, ToJson_AfterParseOptions)
{
    aclskOption options[3];
    options[0].optionType = aclskOptionType::PRELOAD_CODE;
    options[0].preload.preloadMode = 1;
    
    options[1].optionType = aclskOptionType::SPLIT_MODE;
    options[1].splitMode.splitCnt = 3;
    
    options[2].optionType = aclskOptionType::DEBUG_SYNC_ALL;
    options[2].debugSync.debugSyncAll = 1;
    
    aclskOptions optList;
    optList.options = options;
    optList.numOptions = 3;
    
    opts_test->ParseOptions(&optList);
    
    nlohmann::ordered_json json = opts_test->ToJson();
    EXPECT_EQ(json.size(), 3);
    EXPECT_TRUE(json.contains("preload_code"));
    EXPECT_TRUE(json.contains("split_mode"));
    EXPECT_TRUE(json.contains("debug_sync_all"));
}
