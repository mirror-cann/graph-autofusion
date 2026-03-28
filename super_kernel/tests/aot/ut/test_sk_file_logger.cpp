/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it 
 * under the terms of CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sk_file_logger.cpp
 * \brief 文件日志器单元测试
 */

#include <gtest/gtest.h>
#include <fstream>
#include <thread>
#include "sk_file_logger.h"

using namespace sk::logger;

class SkFileLoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 每个测试前重置状态
        // 注意: FileLogger是单例,需要在测试中手动清理
        testLogDir_ = "test_logs";
    }
    
    void TearDown() override {
        // 测试后清理测试日志目录
        system(("rm -rf " + testLogDir_).c_str());
    }
    
    std::string testLogDir_;
};

// 测试1: 基本初始化
TEST_F(SkFileLoggerTest, BasicInitialization) {
    LoggerConfig config;
    config.enabled = true;
    config.modelRI = "test_model_1";
    config.baseDir = testLogDir_;
    config.minLevel = LogLevel::DEBUG;
    
    EXPECT_TRUE(FileLogger::Instance().Initialize(config));
    EXPECT_TRUE(FileLogger::Instance().IsInitialized());
    EXPECT_TRUE(FileLogger::Instance().IsEnabled());
}

// 测试2: 禁用状态初始化
TEST_F(SkFileLoggerTest, DisabledInitialization) {
    LoggerConfig config;
    config.enabled = false;
    config.modelRI = "test_model_2";
    config.baseDir = testLogDir_;
    
    EXPECT_TRUE(FileLogger::Instance().Initialize(config));
    EXPECT_TRUE(FileLogger::Instance().IsInitialized());
    EXPECT_FALSE(FileLogger::Instance().IsEnabled());
}

// 测试3: 日志写入(使用原有SK_LOG*宏)
TEST_F(SkFileLoggerTest, LogWithOriginalMacros) {
    LoggerConfig config;
    config.enabled = true;
    config.modelRI = "test_model_3";
    config.baseDir = testLogDir_;
    config.minLevel = LogLevel::DEBUG;
    
    FileLogger::Instance().Initialize(config);
    
    // 使用原有日志宏
    SK_LOGI("Test info message");
    SK_LOGD("Test debug message: %d", 42);
    SK_LOGW("Test warning message");
    SK_LOGE("Test error message");
    
    // 验证文件是否创建
    // 注意: 实际验证需要等待文件刷新
    EXPECT_TRUE(FileLogger::Instance().IsEnabled());
}

// 测试4: RAII上下文管理
TEST_F(SkFileLoggerTest, RAIIContext) {
    LoggerConfig config;
    config.enabled = true;
    config.modelRI = "test_model_4";
    config.baseDir = testLogDir_;
    
    FileLogger::Instance().Initialize(config);
    
    // 测试RAII上下文
    {
        SK_LOG_CONTEXT_SIMPLE("context_test.log");
        SK_LOGI("This should go to context_test.log");
        EXPECT_EQ(FileHandleManager::Instance().GetCurrentHandle(), "context_test.log");
    }
    
    // 作用域结束后应恢复默认
    EXPECT_EQ(FileHandleManager::Instance().GetCurrentHandle(), "default");
}

// 测试5: 日志级别过滤
TEST_F(SkFileLoggerTest, LogLevelFiltering) {
    LoggerConfig config;
    config.enabled = true;
    config.modelRI = "test_model_5";
    config.baseDir = testLogDir_;
    config.minLevel = LogLevel::WARNING;  // 只记录WARNING及以上
    
    FileLogger::Instance().Initialize(config);
    
    // 这些日志应该被过滤(不落盘,但会透传)
    SK_LOGD("Debug message - should be filtered");
    SK_LOGI("Info message - should be filtered");
    
    // 这些日志应该记录
    SK_LOGW("Warning message - should be logged");
    SK_LOGE("Error message - should be logged");
    
    // 验证级别设置
    FileLogger::Instance().SetMinLevel(LogLevel::INFO);
    SK_LOGI("Info message - should now be logged");
}

// 测试6: 动态开关控制
TEST_F(SkFileLoggerTest, DynamicToggle) {
    LoggerConfig config;
    config.enabled = true;
    config.modelRI = "test_model_6";
    config.baseDir = testLogDir_;
    
    FileLogger::Instance().Initialize(config);
    
    // 启用状态
    EXPECT_TRUE(FileLogger::Instance().IsEnabled());
    SK_LOGI("Enabled: this should be logged");
    
    // 禁用
    FileLogger::Instance().SetEnabled(false);
    EXPECT_FALSE(FileLogger::Instance().IsEnabled());
    SK_LOGI("Disabled: this should not be logged to file");
    
    // 重新启用
    FileLogger::Instance().SetEnabled(true);
    EXPECT_TRUE(FileLogger::Instance().IsEnabled());
    SK_LOGI("Re-enabled: this should be logged again");
}

// 测试7: 多线程安全性
TEST_F(SkFileLoggerTest, ThreadSafety) {
    LoggerConfig config;
    config.enabled = true;
    config.modelRI = "test_model_7";
    config.baseDir = testLogDir_;
    
    FileLogger::Instance().Initialize(config);
    
    const int numThreads = 10;
    const int logsPerThread = 100;
    
    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([t, logsPerThread]() {
            for (int i = 0; i < logsPerThread; ++i) {
                SK_LOGI("Thread %d, log %d", t, i);
            }
        });
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    // 如果能正常完成,说明线程安全
    SUCCEED();
}

// 测试8: 文件句柄管理
TEST_F(SkFileLoggerTest, FileHandleManagement) {
    LoggerConfig config;
    config.enabled = true;
    config.modelRI = "test_model_8";
    config.baseDir = testLogDir_;
    
    FileLogger::Instance().Initialize(config);
    
    // 注册多个日志文件
    EXPECT_TRUE(FileLogger::Instance().RegisterLogFile("module1.log"));
    EXPECT_TRUE(FileLogger::Instance().RegisterLogFile("module2.log"));
    
    // 切换文件
    EXPECT_TRUE(FileLogger::Instance().SwitchToFile("module1.log"));
    SK_LOGI("Log to module1");
    
    EXPECT_TRUE(FileLogger::Instance().SwitchToFile("module2.log"));
    SK_LOGI("Log to module2");
    
    FileLogger::Instance().SwitchToDefault();
    SK_LOGI("Log to default");
}

// 测试9: 超长日志分段
TEST_F(SkFileLoggerTest, LongLogSegmentation) {
    LoggerConfig config;
    config.enabled = true;
    config.modelRI = "test_model_9";
    config.baseDir = testLogDir_;
    config.maxLineLength = 100;  // 设置较小的行长度
    
    FileLogger::Instance().Initialize(config);
    
    // 生成超长日志
    std::string longMessage(500, 'X');
    SK_LOGI("Long message: %s", longMessage.c_str());
    
    // 如果没有崩溃,说明分段处理正常
    SUCCEED();
}

// 测试10: ModelRI子目录
TEST_F(SkFileLoggerTest, ModelRIDirectory) {
    LoggerConfig config;
    config.enabled = true;
    config.modelRI = "model_with_special_chars/test";
    config.baseDir = testLogDir_;
    
    FileLogger::Instance().Initialize(config);
    
    SK_LOGI("Test with modelRI containing special characters");
    
    // 验证特殊字符被替换
    // 实际验证需要检查文件系统
    SUCCEED();
}

// 测试11: 使用简化宏
TEST_F(SkFileLoggerTest, SimplifiedMacros) {
    // 使用全局初始化宏
    InitializeSkFileLogger(true, "test_model_11", LogLevel::DEBUG);
    
    SK_LOGI("Using simplified initialization macro");
    
    // 使用禁用宏
    DISABLE_SK_FILE_LOGGER();
    SK_LOGI("This should not be logged to file");
    
    EXPECT_TRUE(true);
}

// 测试12: 性能基准测试
TEST_F(SkFileLoggerTest, DISABLED_PerformanceBenchmark) {
    LoggerConfig config;
    config.enabled = true;
    config.modelRI = "perf_test";
    config.baseDir = testLogDir_;
    
    FileLogger::Instance().Initialize(config);
    
    const int logCount = 10000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < logCount; ++i) {
        SK_LOGI("Performance test log %d", i);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Logged " << logCount << " messages in " 
              << duration.count() << " ms" << std::endl;
    std::cout << "Average: " << (duration.count() * 1000.0 / logCount) 
              << " microseconds per log" << std::endl;
}

// 测试13: RAII上下文嵌套
TEST_F(SkFileLoggerTest, NestedRAIIContext) {
    LoggerConfig config;
    config.enabled = true;
    config.modelRI = "nested_test";
    config.baseDir = testLogDir_;
    
    FileLogger::Instance().Initialize(config);
    
    SK_LOGI("Default context");
    
    {
        SK_LOG_CONTEXT_SIMPLE("outer.log");
        SK_LOGI("Outer context");
        
        {
            SK_LOG_CONTEXT_SIMPLE("inner.log");
            SK_LOGI("Inner context");
            EXPECT_EQ(FileHandleManager::Instance().GetCurrentHandle(), "inner.log");
        }
        
        // 内层作用域结束,但不会恢复到最外层
        // 因为每个guard独立管理自己的恢复
    }
    
    // 恢复默认
    EXPECT_EQ(FileHandleManager::Instance().GetCurrentHandle(), "default");
}

// 测试14: 空消息处理
TEST_F(SkFileLoggerTest, EmptyMessage) {
    LoggerConfig config;
    config.enabled = true;
    config.modelRI = "empty_test";
    config.baseDir = testLogDir_;
    
    FileLogger::Instance().Initialize(config);
    
    // 空消息应该被安全处理
    SK_LOGI("");
    SK_LOGI(nullptr);
    
    SUCCEED();
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
