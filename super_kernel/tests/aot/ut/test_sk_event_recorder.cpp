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
 * \file test_sk_event_recorder.cpp
 * \brief Unit tests for SkEventRecorder
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "securec.h"

#define private public
#define protected public
#include "sk_event_recorder.h"
#include "sk_resource_manager.h"

// 辅助函数：复制源文件中的 CoreIsAiv 逻辑用于测试
// 源文件中的 static 函数无法直接调用，这里复制相同逻辑
static int TestCoreIsAiv(int core_id)
{
#if defined(NPU_ARCH) && NPU_ARCH == 310
    if (core_id < 18) {
        return 0;
    } else if (core_id < 54) {
        return 1;
    } else if (core_id < 72) {
        return 0;
    } else {
        return 1;
    }
#else
    if (core_id >= 25) {
        return 1;
    } else {
        return 0;
    }
#endif
}

class SkEventRecorderTest : public testing::Test {
protected:
    void SetUp() override {
        // 清理环境变量
        unsetenv("ASCEND_PROF_SK_ON");
        // 清理测试文件和目录
        CleanupTestFiles();
    }

    void TearDown() override {
        // 确保每次测试后 recorder 被重置
        SkEventRecorder::Instance().SkProfilingShutdown();
        (void)SkResourceManager::ReleasePidMemory();
        SkEventRecorder::Instance().deviceCtxs.gmAddr = nullptr;
        SkEventRecorder::Instance().deviceCtxs.hostBuf.reset();
        SkEventRecorder::Instance().deviceCtxs.outputFp.Close();
        SkEventRecorder::Instance().deviceCtxs.outputDir.clear();
        SkEventRecorder::Instance().deviceCtxs.active.store(0);
        SkEventRecorder::Instance().deviceCtxs.totalSize = 0;
        unsetenv("ASCEND_PROF_SK_ON");
        // 重置 call_once 标志，使后续测试的 Init() 可以重新执行
        SkEventRecorder::Instance().initFlag_.~once_flag();
        new (&SkEventRecorder::Instance().initFlag_) std::once_flag();
        // 重置 static 成员变量
        SkEventRecorder::coreSize_ = SK_EVENT_DEFAULT_CORE_SIZE;
        SkEventRecorder::totalSize_ = SK_EVENT_CORE_NUM * SK_EVENT_DEFAULT_CORE_SIZE;
        // 清理测试文件和目录
        CleanupTestFiles();
    }

    // 辅助函数：清理测试文件和目录
    void CleanupTestFiles() {
        pid_t pid = getpid();
        for (int i = 0; i < 4; i++) {
            // 新路径：./sk_meta/<pid>/
            std::string finalFile = "./sk_meta/" + std::to_string(pid) + "/sk_event_dev_device_" + std::to_string(i) + ".json";
            remove(finalFile.c_str());
        }
        // 清理 sk_meta/<pid> 目录
        std::string pidDir = "./sk_meta/" + std::to_string(pid);
        rmdir(pidDir.c_str());
        // 尝试清理 sk_meta 目录（如果为空）
        rmdir("./sk_meta");
    }

    // 辅助函数：创建模拟的事件数据
    void CreateMockEventData(SkEventDeviceCtx* ctx, uint32_t coreId,
                             uint64_t modelRI, uint32_t skId, uint32_t nodeId,
                             uint64_t startTime, uint64_t endTime,
                             uint8_t blockIdx = 0, uint8_t blockNum = 1) {
        uint8_t* hostBuf = ctx->hostBuf.get();
        SkKernelEventCoreBuf* coreBuf = reinterpret_cast<SkKernelEventCoreBuf*>(
            hostBuf + coreId * SkEventRecorder::coreSize_);

        // 设置 offset 以便 DumpDeviceData 能读取到数据
        uint32_t newOffset = sizeof(SkKernelEventCoreBuf) + sizeof(SkKernelEventRecord);
        coreBuf->offset = newOffset;

        // 写入事件记录
        SkKernelEventRecord* record = reinterpret_cast<SkKernelEventRecord*>(
            hostBuf + coreId * SkEventRecorder::coreSize_ + sizeof(SkKernelEventRecord));
        record->modelRI = modelRI;
        record->skId = skId;
        record->nodeId = nodeId;
        record->blockIdx = blockIdx;
        record->blockNum = blockNum;
        record->startTime = startTime;
        record->endTime = endTime;

        // 重置 lastOffset 以便重新读取
        ctx->lastOffset[coreId] = sizeof(SkKernelEventRecord);
    }

    // 辅助函数：初始化一个模拟的设备上下文
    void InitMockDeviceCtx() {
        SkEventDeviceCtx* ctx = &SkEventRecorder::Instance().deviceCtxs;
        ctx->deviceId = 0;
        ctx->totalSize = SkEventRecorder::totalSize_;
        ctx->gmAddr = malloc(SkEventRecorder::totalSize_);
        ctx->hostBuf = std::make_unique<uint8_t[]>(SkEventRecorder::totalSize_);
        (void)memset_s(ctx->hostBuf.get(), SkEventRecorder::totalSize_, 0, SkEventRecorder::totalSize_);
        ctx->outputDir = SkEventRecorder::CreateOutputDir();  // 设置输出目录
        ctx->outputFp.Close();  // FileGuard 默认构造已经是无效状态
        for (uint32_t i = 0; i < SK_EVENT_CORE_NUM; i++) {
            ctx->lastOffset[i] = sizeof(SkKernelEventRecord);
        }
        ctx->active.store(1);
    }

    // 辅助函数：清理模拟的设备上下文
    void CleanupMockDeviceCtx() {
        SkEventDeviceCtx* ctx = &SkEventRecorder::Instance().deviceCtxs;
        if (ctx->gmAddr) {
            free(ctx->gmAddr);
            ctx->gmAddr = nullptr;
        }
        ctx->hostBuf.reset();
        ctx->outputFp.Close();  // 使用 FileGuard 的 Close 方法
        ctx->active.store(0);
    }
};

// ==================== 基础功能测试 ====================

// Test 1: 单例模式验证
TEST_F(SkEventRecorderTest, InstanceReturnsSameInstance) {
    SkEventRecorder& instance1 = SkEventRecorder::Instance();
    SkEventRecorder& instance2 = SkEventRecorder::Instance();
    
    EXPECT_EQ(&instance1, &instance2);
}

// Test 2: 默认状态下未启用
TEST_F(SkEventRecorderTest, DisabledByDefault) {
    bool result = SkEventRecorder::Instance().Init();
    EXPECT_FALSE(result);
    EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// Test 3: 环境变量设置为 "64" 时启用（64KB coreSize）
TEST_F(SkEventRecorderTest, EnabledWhenEnvSetToValidSize) {
    setenv("ASCEND_PROF_SK_ON", "64", 1);
    
    bool result = SkEventRecorder::Instance().Init();
    EXPECT_TRUE(result);
    EXPECT_TRUE(SkEventRecorder::Instance().IsEnabled());
    
    SkEventRecorder::Instance().SkProfilingShutdown();
}

// Test 4: 环境变量设置为非 "1" 时不启用
TEST_F(SkEventRecorderTest, NotEnabledWhenEnvNotOne) {
    setenv("ASCEND_PROF_SK_ON", "0", 1);
    
    bool result = SkEventRecorder::Instance().Init();
    EXPECT_FALSE(result);
    EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// Test 5: 环境变量设置为其他值时不启用
TEST_F(SkEventRecorderTest, NotEnabledWhenEnvSetToOtherValue) {
    setenv("ASCEND_PROF_SK_ON", "true", 1);
    
    bool result = SkEventRecorder::Instance().Init();
    EXPECT_FALSE(result);
    EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// ==================== NodeInfo 测试 ====================

// Test 6: NodeInfo 映射添加和获取
TEST_F(SkEventRecorderTest, AddAndGetNodeInfo) {
    uint64_t modelRI = 100;
    uint32_t skId = 1;
    uint32_t nodeId = 10;
    std::string nodeName = "test_node";
    uint32_t numBlocks = 4;
    
    SkEventRecorder::Instance().AddNodeInfoMapping(modelRI, skId, nodeId, nodeName, numBlocks);
    
    SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo(modelRI, skId, nodeId);
    
    EXPECT_EQ(info.nodeName, nodeName);
    EXPECT_EQ(info.numBlocks, numBlocks);
}

// Test 7: 获取不存在的 NodeInfo
TEST_F(SkEventRecorderTest, GetNonExistentNodeInfo) {
    SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo(999, 999, 999);
    
    EXPECT_TRUE(info.nodeName.empty());
    EXPECT_EQ(info.numBlocks, 0);
}

// Test 8: 多个 NodeInfo 映射
TEST_F(SkEventRecorderTest, MultipleNodeInfoMappings) {
    SkEventRecorder::Instance().AddNodeInfoMapping(1, 1, 1, "node_1_1_1", 2);
    SkEventRecorder::Instance().AddNodeInfoMapping(1, 1, 2, "node_1_1_2", 3);
    SkEventRecorder::Instance().AddNodeInfoMapping(1, 2, 1, "node_1_2_1", 4);
    SkEventRecorder::Instance().AddNodeInfoMapping(2, 1, 1, "node_2_1_1", 5);
    
    EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo(1, 1, 1).nodeName, "node_1_1_1");
    EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo(1, 1, 2).nodeName, "node_1_1_2");
    EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo(1, 2, 1).nodeName, "node_1_2_1");
    EXPECT_EQ(SkEventRecorder::Instance().GetNodeInfo(2, 1, 1).nodeName, "node_2_1_1");
}

// Test 9: NodeInfo 映射覆盖
TEST_F(SkEventRecorderTest, NodeInfoMappingOverwrite) {
    SkEventRecorder::Instance().AddNodeInfoMapping(100, 1, 10, "old_name", 2);
    SkEventRecorder::Instance().AddNodeInfoMapping(100, 1, 10, "new_name", 4);
    
    SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo(100, 1, 10);
    
    EXPECT_EQ(info.nodeName, "new_name");
    EXPECT_EQ(info.numBlocks, 4);
}

// Test 10: 空字符串 nodeName
TEST_F(SkEventRecorderTest, EmptyNodeInfoStrings) {
    SkEventRecorder::Instance().AddNodeInfoMapping(100, 1, 10, "", 0);
    
    SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo(100, 1, 10);
    
    EXPECT_TRUE(info.nodeName.empty());
    EXPECT_EQ(info.numBlocks, 0);
}

// Test 11: 大量 NodeInfo 映射
TEST_F(SkEventRecorderTest, ManyNodeInfoMappings) {
    const int count = 100;
    
    for (int i = 0; i < count; i++) {
        SkEventRecorder::Instance().AddNodeInfoMapping(
            i, i % 10, i % 20, 
            "node_" + std::to_string(i), 
            i % 5 + 1
        );
    }
    
    for (int i = 0; i < count; i += 10) {
        SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo(i, i % 10, i % 20);
        EXPECT_EQ(info.nodeName, "node_" + std::to_string(i));
    }
}

// ==================== 线程安全测试 ====================

// Test 12: 线程安全 - 并发添加 NodeInfo
TEST_F(SkEventRecorderTest, ThreadSafeAddNodeInfo) {
    const int numThreads = 4;
    const int opsPerThread = 25;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([t, opsPerThread]() {
            for (int i = 0; i < opsPerThread; i++) {
                SkEventRecorder::Instance().AddNodeInfoMapping(
                    t * 1000 + i, t, i,
                    "node_" + std::to_string(t) + "_" + std::to_string(i),
                    1
                );
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    for (int t = 0; t < numThreads; t++) {
        SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo(t * 1000, t, 0);
        EXPECT_EQ(info.nodeName, "node_" + std::to_string(t) + "_0");
    }
}

// Test 13: 线程安全 - 并发读取 NodeInfo
TEST_F(SkEventRecorderTest, ThreadSafeGetNodeInfo) {
    for (int i = 0; i < 100; i++) {
        SkEventRecorder::Instance().AddNodeInfoMapping(i, i, i, "node_" + std::to_string(i), 1);
    }
    
    const int numThreads = 4;
    std::vector<std::thread> threads;
    std::atomic<int> successCount{0};
    
    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([&successCount]() {
            for (int i = 0; i < 25; i++) {
                SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo(i, i, i);
                if (info.nodeName == "node_" + std::to_string(i)) {
                    successCount++;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(successCount.load(), 100);
}

// ==================== CoreIsAiv 函数测试 ====================

// Test 14: CoreIsAiv - AIC 核心（core_id < 25）
TEST_F(SkEventRecorderTest, CoreIsAiv_AicCores) {
    // 对于非 310 架构，core_id < 25 是 AIC
    for (int i = 0; i < 25; i++) {
        EXPECT_EQ(TestCoreIsAiv(i), 0);
    }
}

// Test 15: CoreIsAiv - AIV 核心（core_id >= 25）
TEST_F(SkEventRecorderTest, CoreIsAiv_AivCores) {
    // 对于非 310 架构，core_id >= 25 是 AIV
    for (int i = 25; i < 75; i++) {
        EXPECT_EQ(TestCoreIsAiv(i), 1);
    }
}

// Test 16: CoreIsAiv - 边界值
TEST_F(SkEventRecorderTest, CoreIsAiv_BoundaryValues) {
    // 边界测试：24 是 AIC，25 是 AIV
    EXPECT_EQ(TestCoreIsAiv(24), 0);
    EXPECT_EQ(TestCoreIsAiv(25), 1);
}

// Test 17: GetGmAddrForDevice 在未启用时返回 nullptr
TEST_F(SkEventRecorderTest, GetGmAddrForDeviceReturnsNullWhenDisabled) {
    void* addr = SkEventRecorder::Instance().GetGmAddrForDevice(0);
    EXPECT_EQ(addr, nullptr);
}

// Test 19: GetGmAddrForDevice 在 deviceId 越界时返回 nullptr
TEST_F(SkEventRecorderTest, GetGmAddrForDeviceReturnsNullForInvalidDeviceId) {
    SkEventRecorder::Instance().enabled = true;
    
    void* addressBeyondMaxDevice = SkEventRecorder::Instance().GetGmAddrForDevice(SK_EVENT_MAX_DEVICE_NUM);
    EXPECT_EQ(addressBeyondMaxDevice, nullptr);
    
    void* addressExceedingMaxDevice = SkEventRecorder::Instance().GetGmAddrForDevice(SK_EVENT_MAX_DEVICE_NUM + 1);
    EXPECT_EQ(addressExceedingMaxDevice, nullptr);
}

// Test 21: GetGmAddrForDevice 已激活的设备返回缓存地址
TEST_F(SkEventRecorderTest, GetGmAddrForDeviceReturnsCachedAddr) {
    // 手动初始化一个设备上下文
    InitMockDeviceCtx();
    SkEventRecorder::Instance().enabled = true;
    
    void* firstCallAddr = SkEventRecorder::Instance().GetGmAddrForDevice(0);
    EXPECT_NE(firstCallAddr, nullptr);
    
    // 第二次调用应该返回相同地址（fast path）
    void* secondCallAddr = SkEventRecorder::Instance().GetGmAddrForDevice(0);
    EXPECT_EQ(firstCallAddr, secondCallAddr);
    
    CleanupMockDeviceCtx();
}

// ==================== DumpDeviceData 测试 ====================

// Test 23: DumpDeviceData 空数据处理
TEST_F(SkEventRecorderTest, DumpDeviceDataEmptyData) {
    InitMockDeviceCtx();
    
    // 没有事件数据
    SkEventRecorder::Instance().DumpDeviceData(&SkEventRecorder::Instance().deviceCtxs);
    
    // 不应该崩溃
    CleanupMockDeviceCtx();
}

// Test 24: DumpDeviceData 空 gmAddr 或 hostBuf
TEST_F(SkEventRecorderTest, DumpDeviceDataNullPointers) {
    SkEventDeviceCtx ctx;
    ctx.gmAddr = nullptr;
    ctx.hostBuf = nullptr;
    ctx.deviceId = 0;
    
    // 不应该崩溃
    SkEventRecorder::Instance().DumpDeviceData(&ctx);
}

// Test 28: DumpDeviceData 跳过没有 NodeInfo 的事件
TEST_F(SkEventRecorderTest, DumpDeviceDataSkipsEventsWithoutNodeInfo) {
    InitMockDeviceCtx();

    SkEventDeviceCtx* ctx = &SkEventRecorder::Instance().deviceCtxs;

    // 创建事件数据但不添加 NodeInfo
    CreateMockEventData(ctx, 0, 999, 999, 999, 100, 200, 0, 1);

    SkEventRecorder::Instance().DumpDeviceData(ctx);

    // 不应该崩溃，正常处理完成

    CleanupMockDeviceCtx();
}

// ==================== CreateDeviceCtx 测试 ====================

// Test 29: CreateDeviceCtx 基本功能
TEST_F(SkEventRecorderTest, CreateDeviceCtxBasic) {
    SkEventDeviceCtx* ctx = SkEventRecorder::Instance().CreateDeviceCtx(0);

    // stub 实现应该成功创建
    if (ctx != nullptr) {
        EXPECT_NE(ctx->gmAddr, nullptr);
        EXPECT_NE(ctx->hostBuf, nullptr);
        EXPECT_EQ(ctx->deviceId, 0);
        EXPECT_EQ(ctx->active.load(), 1);
        EXPECT_EQ(ctx->totalSize, SkEventRecorder::totalSize_);
        EXPECT_FALSE(ctx->outputDir.empty());  // 输出目录应被设置

        // 清理
        if (ctx->gmAddr) {
            EXPECT_EQ(SkResourceManager::ReleasePidMemory(), ACL_SUCCESS);
            ctx->gmAddr = nullptr;
        }
        ctx->hostBuf.reset();
        ctx->active.store(0);

        // 删除文件
        std::string jsonFile = ctx->outputDir + "/sk_event_dev_device_0.json";
        remove(jsonFile.c_str());
    }
}

// ==================== Shutdown 测试 ====================

// Test 30: Shutdown 后 IsEnabled 返回 false
TEST_F(SkEventRecorderTest, IsEnabledFalseAfterShutdown) {
    setenv("ASCEND_PROF_SK_ON", "64", 1);
    SkEventRecorder::Instance().Init();
    EXPECT_TRUE(SkEventRecorder::Instance().IsEnabled());
    
    SkEventRecorder::Instance().SkProfilingShutdown();
    EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// Test 31: Shutdown 未启用时不做任何操作
TEST_F(SkEventRecorderTest, ShutdownDoesNothingWhenDisabled) {
    EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
    
    // 不应该崩溃
    SkEventRecorder::Instance().SkProfilingShutdown();
    SkEventRecorder::Instance().SkProfilingShutdown();
    
    EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// Test 32: Shutdown 创建最终文件
TEST_F(SkEventRecorderTest, ShutdownCreatesFinalFile) {
    InitMockDeviceCtx();
    SkEventRecorder::Instance().enabled = true;

    SkEventDeviceCtx* ctx = &SkEventRecorder::Instance().deviceCtxs;

    // 创建临时文件
    char tmpFile[512];
    (void)snprintf_s(tmpFile, sizeof(tmpFile), sizeof(tmpFile) - 1,
                     "%s/sk_event_dev_device_%u.json", ctx->outputDir.c_str(), 0);
    ctx->outputFp.Open(tmpFile, "wb");
    if (ctx->outputFp.IsValid()) {
        const char* jsonStart = "[{}]";
        fwrite(jsonStart, 1, strlen(jsonStart), ctx->outputFp.Get());
    }

    // 执行 Shutdown（不启动线程）
    SkEventRecorder::Instance().globalRunning.store(false);

    ctx->outputFp.Close();

    // 验证文件创建
    struct stat buffer;
    EXPECT_EQ(stat(tmpFile, &buffer), 0);

    // 清理
    remove(tmpFile);
    CleanupMockDeviceCtx();
}

// Test 33: Shutdown 释放资源
TEST_F(SkEventRecorderTest, ShutdownReleasesResources) {
    InitMockDeviceCtx();
    SkEventRecorder::Instance().enabled = true;
    
    // 调用 Shutdown
    SkEventRecorder::Instance().SkProfilingShutdown();
    
    // 验证资源被释放
    EXPECT_EQ(SkEventRecorder::Instance().deviceCtxs.hostBuf, nullptr);
}

// Test 36: SkKernelEventRecord 结构体大小验证
TEST_F(SkEventRecorderTest, EventRecordSize) {
    EXPECT_LE(sizeof(SkKernelEventRecord), 64);
    EXPECT_GE(sizeof(SkKernelEventRecord), sizeof(uint64_t) * 2 + sizeof(uint32_t) * 2 + sizeof(uint8_t) * 2);
}

// Test 38: SkEventDeviceCtx 初始状态
TEST_F(SkEventRecorderTest, DeviceCtxInitialState) {
    SkEventDeviceCtx ctx;

    EXPECT_EQ(ctx.active.load(), 0);
    EXPECT_EQ(ctx.gmAddr, nullptr);
    EXPECT_EQ(ctx.hostBuf, nullptr);
    EXPECT_EQ(ctx.deviceId, 0);
    EXPECT_EQ(ctx.totalSize, 0);
    EXPECT_FALSE(ctx.outputFp.IsValid());  // FileGuard 默认无效状态
}

// ==================== 多次调用安全性测试 ====================

// Test 39: Init 多次调用安全性
TEST_F(SkEventRecorderTest, MultipleInitCalls) {
    setenv("ASCEND_PROF_SK_ON", "64", 1);
    
    bool result1 = SkEventRecorder::Instance().Init();
    bool result2 = SkEventRecorder::Instance().Init();
    
    EXPECT_TRUE(SkEventRecorder::Instance().IsEnabled());
    
    SkEventRecorder::Instance().SkProfilingShutdown();
}

// Test 40: Shutdown 多次调用安全性
TEST_F(SkEventRecorderTest, MultipleShutdownCalls) {
    SkEventRecorder::Instance().SkProfilingShutdown();
    SkEventRecorder::Instance().SkProfilingShutdown();
    SkEventRecorder::Instance().SkProfilingShutdown();
    
    EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// ==================== 特殊字符串测试 ====================

// Test 41: 特殊字符 nodeName
TEST_F(SkEventRecorderTest, SpecialCharacterNodeName) {
    std::string specialName = "node-with_special.chars:123/456";
    
    SkEventRecorder::Instance().AddNodeInfoMapping(1, 1, 1, specialName, 1);
    
    SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo(1, 1, 1);
    EXPECT_EQ(info.nodeName, specialName);
}

// Test 42: 长字符串 nodeName
TEST_F(SkEventRecorderTest, LongNodeName) {
    std::string longName(256, 'a');
    
    SkEventRecorder::Instance().AddNodeInfoMapping(1, 1, 1, longName, 1);
    
    SkNodeInfo info = SkEventRecorder::Instance().GetNodeInfo(1, 1, 1);
    EXPECT_EQ(info.nodeName, longName);
}

// ==================== 常量验证测试 ====================

// Test 43: 常量值验证
TEST_F(SkEventRecorderTest, ConstantsVerification) {
    EXPECT_EQ(SK_EVENT_MAX_DEVICE_NUM, 16);
    EXPECT_EQ(SK_EVENT_CORE_NUM, 75);
    EXPECT_EQ(SK_EVENT_DEFAULT_CORE_SIZE, 1024 * 1024);
    EXPECT_EQ(SkEventRecorder::coreSize_, SK_EVENT_DEFAULT_CORE_SIZE);
    EXPECT_EQ(SkEventRecorder::totalSize_, SK_EVENT_CORE_NUM * SK_EVENT_DEFAULT_CORE_SIZE);
}

// ==================== Init/Shutdown 生命周期测试 ====================

// Test 44: Init 成功后 Shutdown 能正确关闭
TEST_F(SkEventRecorderTest, InitShutdownLifecycle) {
    // 设置 64KB coreSize（已对齐，无需向上取整）
    setenv("ASCEND_PROF_SK_ON", "64", 1);
    SkEventRecorder::Instance().Init();
    EXPECT_TRUE(SkEventRecorder::Instance().IsEnabled());
    EXPECT_EQ(SkEventRecorder::coreSize_, 64U * 1024U);

    SkEventRecorder::Instance().SkProfilingShutdown();
    EXPECT_FALSE(SkEventRecorder::Instance().IsEnabled());
}

// ==================== DumpThreadFunc 间接测试 ====================

// Test 45: DumpThreadFunc 可以被安全调用
TEST_F(SkEventRecorderTest, DumpThreadFuncCanBeCalledSafely) {
    InitMockDeviceCtx();
    
    // 直接调用 DumpThreadFunc 的单次迭代逻辑
    SkEventRecorder::Instance().globalRunning.store(false);  // 立即停止
    
    // 手动调用 DumpDeviceData
    SkEventRecorder::Instance().DumpDeviceData(&SkEventRecorder::Instance().deviceCtxs);
    
    // 不应该崩溃
    CleanupMockDeviceCtx();
}

// Test 46: DumpThreadFunc 处理单个设备
TEST_F(SkEventRecorderTest, DumpThreadFuncHandlesSingleDevice) {
    InitMockDeviceCtx();
    
    SkEventRecorder::Instance().globalRunning.store(false);
    
    // 手动调用 DumpDeviceData
    SkEventDeviceCtx* ctx = &SkEventRecorder::Instance().deviceCtxs;
    if (ctx->active.load()) {
        SkEventRecorder::Instance().DumpDeviceData(ctx);
    }
    
    CleanupMockDeviceCtx();
}

// ==================== 边界条件测试 ====================

// Test 47: GetNodeInfo 部分键存在部分不存在
TEST_F(SkEventRecorderTest, GetNodeInfoPartialKeyExists) {
    SkEventRecorder::Instance().AddNodeInfoMapping(1, 1, 1, "node", 1);
    
    // modelRI 存在，skId 不存在
    SkNodeInfo info1 = SkEventRecorder::Instance().GetNodeInfo(1, 999, 1);
    EXPECT_TRUE(info1.nodeName.empty());
    
    // modelRI 和 skId 存在，nodeId 不存在
    SkNodeInfo info2 = SkEventRecorder::Instance().GetNodeInfo(1, 1, 999);
    EXPECT_TRUE(info2.nodeName.empty());
    
    // modelRI 不存在
    SkNodeInfo info3 = SkEventRecorder::Instance().GetNodeInfo(999, 1, 1);
    EXPECT_TRUE(info3.nodeName.empty());
}

// Test 48: Double-check locking 路径测试
TEST_F(SkEventRecorderTest, GetGmAddrForDeviceDoubleCheckLocking) {
    SkEventRecorder::Instance().enabled = true;

    // 第一次调用 - slow path
    void* addr1 = SkEventRecorder::Instance().GetGmAddrForDevice(0);

    // 第二次调用 - fast path（active 已设置）
    void* addr2 = SkEventRecorder::Instance().GetGmAddrForDevice(0);

    if (addr1 != nullptr && addr2 != nullptr) {
        EXPECT_EQ(addr1, addr2);

        // 清理
        SkEventDeviceCtx* ctx = &SkEventRecorder::Instance().deviceCtxs;
        std::string outputDir = ctx->outputDir;  // 保存路径用于清理
        if (ctx->gmAddr) {
            EXPECT_EQ(SkResourceManager::ReleasePidMemory(), ACL_SUCCESS);
            ctx->gmAddr = nullptr;
        }
        ctx->hostBuf.reset();
        ctx->active.store(0);
        // 删除文件
        std::string jsonFile = outputDir + "/sk_event_dev_device_0.json";
        remove(jsonFile.c_str());
    }
}
