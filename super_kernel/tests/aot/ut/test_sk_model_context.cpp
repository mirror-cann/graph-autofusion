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
 * \file test_sk_model_context.cpp
 * \brief Unit tests for sk_model_context.h (per-model identity, sk_meta layout, SkModelContext)
 */

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <atomic>
#include <fstream>
#include <string>
#include <set>
#include <mutex>

#include "sk_log.h"
#include "sk_model_context.h"

namespace {

constexpr const char* MODEL_LABEL_PREFIX = "model_";
constexpr const char* UNKNOWN_MODEL_ID = "unknown";

std::string UnknownModelLabel()
{
    return std::string(MODEL_LABEL_PREFIX) + UNKNOWN_MODEL_ID;
}

// 测试用的 aclmdlRI 模拟句柄
inline aclmdlRI MakeFakeModel(uintptr_t v)
{
    return reinterpret_cast<aclmdlRI>(v);
}

std::string ExpectedModelId(uint32_t rtsModelId, uint64_t callCount)
{
    return std::to_string(rtsModelId) + "_" + std::to_string(callCount);
}

std::string ExpectedModelLabel(uint32_t rtsModelId, uint64_t callCount)
{
    return "model_" + ExpectedModelId(rtsModelId, callCount);
}

// 清理 sk_meta 测试目录（递归）
void CleanupSkMetaDirs()
{
    pid_t pid = getpid();
    std::string pidDir = "sk_meta/" + std::to_string(pid);
    // best-effort 清理；遗留的子目录留给后续测试或人工处理
    rmdir(pidDir.c_str());
    rmdir("sk_meta");
}

}  // namespace

// ==================== 测试 Fixture ====================

class TestSkModelContext : public testing::Test {
protected:
    void SetUp() override
    {
        CleanupSkMetaDirs();
    }

    void TearDown() override
    {
        CleanupSkMetaDirs();
    }
};

// ==================== Public entry counter behavior ====================

TEST_F(TestSkModelContext, SkModelContext_RepeatedInvocationBumpsCounter)
{
    constexpr uint32_t modelId = 0xA001;
    aclmdlRI model = MakeFakeModel(modelId);

    {
        SkModelContext firstGuard(model);
        EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(modelId, 1));
        EXPECT_EQ(GetCurrentModelLabel(), ExpectedModelLabel(modelId, 1));
    }

    {
        SkModelContext secondGuard(model);
        EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(modelId, 2));
        EXPECT_EQ(GetCurrentModelLabel(), ExpectedModelLabel(modelId, 2));
    }

    EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
}

TEST_F(TestSkModelContext, SkModelContext_SameRtsModelIdDifferentAddressSharesCounter)
{
    constexpr uint32_t modelId = 0xA002;
    aclmdlRI firstModel = MakeFakeModel(0x100000000ULL | modelId);
    aclmdlRI secondModel = MakeFakeModel(0x200000000ULL | modelId);

    {
        SkModelContext firstGuard(firstModel);
        EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(modelId, 1));
    }
    {
        SkModelContext secondGuard(secondModel);
        EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(modelId, 2));
    }
}

TEST_F(TestSkModelContext, SkModelContext_ConcurrentInvocationsProduceUniqueIds)
{
    constexpr uint32_t modelId = 0xA003;
    aclmdlRI model = MakeFakeModel(modelId);
    const int numThreads = 4;
    const int guardCountPerThread = 25;
    const int totalGuardCount = numThreads * guardCountPerThread;
    std::vector<std::thread> threads;
    std::set<std::string> observedIds;
    std::mutex observedIdsMutex;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([model, guardCountPerThread, &observedIds, &observedIdsMutex]() {
            for (int i = 0; i < guardCountPerThread; ++i) {
                SkModelContext guard(model);
                std::lock_guard<std::mutex> lock(observedIdsMutex);
                observedIds.insert(GetCurrentModelId());
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(observedIds.size(), static_cast<size_t>(totalGuardCount));
    EXPECT_NE(observedIds.find(ExpectedModelId(modelId, 1)), observedIds.end());
    EXPECT_NE(observedIds.find(ExpectedModelId(modelId, totalGuardCount)), observedIds.end());
}

// ==================== Thread-local frozen id ====================

TEST_F(TestSkModelContext, CurrentModelContext_DefaultValue)
{
    EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
    EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());
}

TEST_F(TestSkModelContext, CurrentModelContext_ThreadLocalIsolated)
{
    constexpr uint32_t mainModelId = 0xA006;
    constexpr uint32_t workerModelId = 0xA007;
    SkModelContext mainGuard(MakeFakeModel(mainModelId));
    std::string workerBeforeGuard;
    std::string workerInsideGuard;
    std::thread worker([&workerBeforeGuard, &workerInsideGuard]() {
        workerBeforeGuard = GetCurrentModelLabel();
        SkModelContext workerGuard(MakeFakeModel(workerModelId));
        workerInsideGuard = GetCurrentModelLabel();
    });
    worker.join();
    EXPECT_EQ(workerBeforeGuard, UnknownModelLabel());
    EXPECT_EQ(workerInsideGuard, ExpectedModelLabel(workerModelId, 1));
    EXPECT_EQ(GetCurrentModelLabel(), ExpectedModelLabel(mainModelId, 1));
}

// ==================== SkModelContext ====================

TEST_F(TestSkModelContext, SkModelContext_FreezesUniqueIdAndBumpsCounter)
{
    constexpr uint32_t modelId = 0xA008;
    aclmdlRI model = MakeFakeModel(modelId);
    EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
    EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());

    {
        SkModelContext guard(model);
        EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(modelId, 1));
        EXPECT_EQ(GetCurrentModelLabel(), ExpectedModelLabel(modelId, 1));
    }

    // RAII 析构后恢复为默认上下文
    EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
    EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());
}

TEST_F(TestSkModelContext, SkModelContext_NestedScopeDoesNotRestoreOuterContext)
{
    constexpr uint32_t outerModelId = 0xA009;
    constexpr uint32_t innerModelId = 0xA00A;
    aclmdlRI outer = MakeFakeModel(outerModelId);
    aclmdlRI inner = MakeFakeModel(innerModelId);

    {
        SkModelContext outerGuard(outer);
        EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(outerModelId, 1));
        EXPECT_EQ(GetCurrentModelLabel(), ExpectedModelLabel(outerModelId, 1));
        {
            SkModelContext innerGuard(inner);
            EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(innerModelId, 1));
            EXPECT_EQ(GetCurrentModelLabel(), ExpectedModelLabel(innerModelId, 1));
        }
        EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
        EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());
    }
    EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
    EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());
}

TEST_F(TestSkModelContext, SkModelContext_RepeatedInvocationDisambiguates)
{
    constexpr uint32_t modelId = 0xA00B;
    aclmdlRI model = MakeFakeModel(modelId);
    std::string first;
    std::string second;
    {
        SkModelContext g1(model);
        first = GetCurrentModelLabel();
    }
    {
        SkModelContext g2(model);
        second = GetCurrentModelLabel();
    }
    EXPECT_EQ(first, ExpectedModelLabel(modelId, 1));
    EXPECT_EQ(second, ExpectedModelLabel(modelId, 2));
    EXPECT_NE(first, second);  // 反复调用同一 handle，id 不冲突
}

TEST_F(TestSkModelContext, SkModelContext_SameRtsModelIdDifferentAddressDisambiguates)
{
    constexpr uint32_t modelId = 0xA00C;
    aclmdlRI firstModel = MakeFakeModel(0x100000000ULL | modelId);
    aclmdlRI secondModel = MakeFakeModel(0x200000000ULL | modelId);
    std::string first;
    std::string second;
    {
        SkModelContext guard(firstModel);
        first = GetCurrentModelId();
    }
    {
        SkModelContext guard(secondModel);
        second = GetCurrentModelId();
    }
    EXPECT_EQ(first, ExpectedModelId(modelId, 1));
    EXPECT_EQ(second, ExpectedModelId(modelId, 2));
}

TEST_F(TestSkModelContext, SkModelContext_NullModelStillSetsFrozenId)
{
    {
        SkModelContext guard(nullptr);
        EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
        // nullptr 不递增 counter，但仍设置一个稳定的 frozen id
        EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());
    }
    EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
    EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());
}

// ==================== SanitizePathComponent ====================

TEST_F(TestSkModelContext, SanitizePathComponent_NoInvalidCharsUnchanged)
{
    EXPECT_EQ(SanitizePathComponent("model_42_1"), "model_42_1");
    EXPECT_EQ(SanitizePathComponent("simple_name"), "simple_name");
    EXPECT_EQ(SanitizePathComponent(""), "");
}

TEST_F(TestSkModelContext, SanitizePathComponent_ReplacesAllInvalidChars)
{
    // 9 个非法字符全替换为下划线
    EXPECT_EQ(SanitizePathComponent("a/b\\c:d*e?f\"g<h>i|j"),
              "a_b_c_d_e_f_g_h_i_j");
}

TEST_F(TestSkModelContext, SanitizePathComponent_PreservesNormalChars)
{
    std::string s = "abc-123.XYZ_test";
    EXPECT_EQ(SanitizePathComponent(s), s);
}

TEST_F(TestSkModelContext, SanitizePathComponent_AllInvalidProducesAllUnderscores)
{
    EXPECT_EQ(SanitizePathComponent("/\\:*?\"<>|"), "_________");
}

// ==================== GetSkMetaBasePath / GetSkMetaPath ====================

TEST_F(TestSkModelContext, GetSkMetaBasePath_HasPidSuffix)
{
    std::string base = GetSkMetaBasePath();
    std::string expected = "sk_meta/" + std::to_string(getpid());
    EXPECT_EQ(base, expected);
}

TEST_F(TestSkModelContext, GetSkMetaPath_UnknownModelUsesUnknownLabel)
{
    std::string path = GetSkMetaPath(UnknownModelLabel());
    EXPECT_EQ(path, GetSkMetaBasePath() + "/" + UnknownModelLabel());
}

TEST_F(TestSkModelContext, GetSkMetaPath_UsesFrozenIdWhenActive)
{
    constexpr uint32_t modelId = 0xA010;
    aclmdlRI model = MakeFakeModel(modelId);
    SkModelContext guard(model);
    EXPECT_EQ(GetSkMetaPath(GetCurrentModelLabel()), GetSkMetaBasePath() + "/" + ExpectedModelLabel(modelId, 1));
}

TEST_F(TestSkModelContext, GetSkMetaPath_UsesExplicitModelLabel)
{
    constexpr uint32_t modelId = 0xA011;
    EXPECT_EQ(GetSkMetaPath(ExpectedModelLabel(modelId, 0)),
              GetSkMetaBasePath() + "/" + ExpectedModelLabel(modelId, 0));
}

// ==================== CreateDirectoryRecursive ====================

TEST_F(TestSkModelContext, CreateDirectoryRecursive_EmptyPathReturnsFalse)
{
    EXPECT_FALSE(CreateDirectoryRecursive(""));
}

TEST_F(TestSkModelContext, CreateDirectoryRecursive_CreatesNestedPath)
{
    std::string base = "sk_meta/" + std::to_string(getpid()) + "/nested_a/nested_b";
    EXPECT_TRUE(CreateDirectoryRecursive(base));

    struct stat st;
    EXPECT_EQ(stat(base.c_str(), &st), 0);
    EXPECT_TRUE(S_ISDIR(st.st_mode));

    // 清理
    rmdir(base.c_str());
    std::string parent = "sk_meta/" + std::to_string(getpid()) + "/nested_a";
    rmdir(parent.c_str());
}

TEST_F(TestSkModelContext, CreateDirectoryRecursive_IdempotentOnExisting)
{
    std::string path = "sk_meta/" + std::to_string(getpid()) + "/idem";
    EXPECT_TRUE(CreateDirectoryRecursive(path));
    // 二次调用应仍成功
    EXPECT_TRUE(CreateDirectoryRecursive(path));
    rmdir(path.c_str());
}

// ==================== CreateSkMetaDirectory ====================

TEST_F(TestSkModelContext, CreateSkMetaDirectory_UnknownModelCreatesUnknownSubdir)
{
    std::string path = CreateSkMetaDirectory(UnknownModelLabel());
    EXPECT_FALSE(path.empty());
    EXPECT_EQ(path, GetSkMetaBasePath() + "/" + UnknownModelLabel());

    struct stat st;
    EXPECT_EQ(stat(path.c_str(), &st), 0);
    EXPECT_TRUE(S_ISDIR(st.st_mode));

    rmdir(path.c_str());
}

TEST_F(TestSkModelContext, CreateSkMetaDirectory_UsesFrozenIdWhenActive)
{
    constexpr uint32_t modelId = 0xA012;
    aclmdlRI model = MakeFakeModel(modelId);
    SkModelContext guard(model);

    std::string path = CreateSkMetaDirectory(GetCurrentModelLabel());
    EXPECT_EQ(path, GetSkMetaBasePath() + "/" + ExpectedModelLabel(modelId, 1));

    struct stat st;
    EXPECT_EQ(stat(path.c_str(), &st), 0);
    EXPECT_TRUE(S_ISDIR(st.st_mode));

    rmdir(path.c_str());
}

TEST_F(TestSkModelContext, CreateSkMetaDirectory_IdempotentOnSameModelHandle)
{
    aclmdlRI model = MakeFakeModel(0xA013);
    SkModelContext guard(model);

    std::string first = CreateSkMetaDirectory(GetCurrentModelLabel());
    std::string second = CreateSkMetaDirectory(GetCurrentModelLabel());
    // 同一个 guard 范围内，frozen id 不变，路径相同且都成功
    EXPECT_EQ(first, second);
    EXPECT_FALSE(first.empty());

    rmdir(first.c_str());
}

// ==================== 端到端：guard + 路径一致性 ====================

TEST_F(TestSkModelContext, GuardedScope_PathStaysStableAcrossCalls)
{
    aclmdlRI model = MakeFakeModel(0xA014);
    SkModelContext guard(model);

    std::string id1 = GetCurrentModelLabel();
    std::string path1 = GetSkMetaPath(id1);
    std::string id2 = GetCurrentModelLabel();
    std::string path2 = GetSkMetaPath(id2);

    EXPECT_EQ(id1, id2);
    EXPECT_EQ(path1, path2);
}

TEST_F(TestSkModelContext, LogContextUsesExplicitModelLabel)
{
    constexpr uint32_t modelId = 0xA016;
    aclmdlRI model = MakeFakeModel(modelId);
    sk::logger::FileLogger::Instance().SetEnabled(true);

    {
        SkModelContext guard(model);
        sk::logger::LoggerConfig config;
        config.enabled = true;
        config.modelLabel = GetCurrentModelLabel();
        ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));

        const std::string modelLabel = GetCurrentModelLabel();
        SK_LOG_CONTEXT("explicit_label.log", modelLabel);
        SK_LOGI("log with explicit model label");

        std::string path = GetSkMetaBasePath() + "/" + modelLabel + "/explicit_label.log";
        std::ifstream file(path);
        EXPECT_TRUE(file.good());
    }

    sk::logger::FileLogger::Instance().SetEnabled(false);
}
