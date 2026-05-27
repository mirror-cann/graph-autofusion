/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#define private public
#include "sk_constant_codegen.h"
#undef private

namespace {

bool Contains(const std::string& value, const std::string& expected)
{
    return value.find(expected) != std::string::npos;
}

} // namespace

TEST(ConstantCodeGeneratorTest, GenerateTaskExecutionForSplit_EmitsEventValueAndFlag)
{
    ConstantCodeGenOptions options;
    ConstantCodeGenerator generator(options);

    SkTask task;
    ASSERT_TRUE(task.Init(4));
    TaskQue* taskQue = task.GetTaskQue();
    ASSERT_NE(taskQue, nullptr);

    TaskInfo& notify = taskQue->taskInfos[taskQue->taskCnt++];
    notify = {};
    notify.type = SkTaskType::TYPE_EVENT_NOTIFY;
    SetEventTaskArgs(notify, 0x1000, 0x55, SK_DEFAULT_WRITE_FLAG);

    TaskInfo& wait = taskQue->taskInfos[taskQue->taskCnt++];
    wait = {};
    wait.type = SkTaskType::TYPE_EVENT_WAIT;
    SetEventTaskArgs(wait, 0x2000, 0x66, static_cast<uint32_t>(SkMemoryWaitFlag::AND));

    TaskInfo& reset = taskQue->taskInfos[taskQue->taskCnt++];
    reset = {};
    reset.type = SkTaskType::TYPE_EVENT_RESET;
    SetEventTaskArgs(reset, 0x3000, SK_DEFAULT_RESET_VALUE, SK_DEFAULT_WRITE_FLAG);

    const std::string notifyCode = generator.GenerateTaskExecutionForSplit(taskQue, 0, true, 0);
    EXPECT_TRUE(Contains(notifyCode,
        "AscendC::NotifyFunc<true>(0x0000000000001000ULL, 0x0000000000000055ULL)"));
    EXPECT_TRUE(Contains(notifyCode,
        "AscendC::NotifyFunc<false>(0x0000000000001000ULL, 0x0000000000000055ULL)"));

    const std::string waitCode = generator.GenerateTaskExecutionForSplit(taskQue, 1, true, 0);
    EXPECT_TRUE(Contains(waitCode,
        "AscendC::WaitFunc<true>(0x0000000000002000ULL, 0x0000000000000066ULL, 2U)"));
    EXPECT_TRUE(Contains(waitCode,
        "AscendC::WaitFunc<false>(0x0000000000002000ULL, 0x0000000000000066ULL, 2U)"));

    const std::string resetCode = generator.GenerateTaskExecutionForSplit(taskQue, 2, true, 0);
    EXPECT_TRUE(Contains(resetCode,
        "AscendC::ResetFunc<true>(0x0000000000003000ULL, 0x0000000000000000ULL)"));
    EXPECT_TRUE(Contains(resetCode,
        "AscendC::ResetFunc<false>(0x0000000000003000ULL, 0x0000000000000000ULL)"));
}

TEST(ConstantCodeGeneratorTest, GenerateTaskExecutionForSplit_EmitsEarlyStartFuncAndSyncConfig)
{
    ConstantCodeGenOptions options;
    ConstantCodeGenerator generator(options);

    SkTask task;
    ASSERT_TRUE(task.Init(4));
    TaskQue* taskQue = task.GetTaskQue();
    ASSERT_NE(taskQue, nullptr);

    TaskInfo& func = taskQue->taskInfos[taskQue->taskCnt++];
    func = {};
    func.type = SkTaskType::TYPE_FUNC;
    func.numBlocks = 8;
    func.entryCnt = 1;
    func.args = 0x4000;
    func.entry[0] = 0x5000;
    func.reserved = static_cast<uint64_t>(SkEarlyStartMask::AIC_TO_AIV_SET);

    TaskInfo& sync = taskQue->taskInfos[taskQue->taskCnt++];
    sync = {};
    sync.type = SkTaskType::TYPE_SYNC;
    sync.args = static_cast<uint64_t>(SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC);
    sync.numBlocks = 7;
    sync.reserved = static_cast<uint64_t>(SkEarlyStartMask::AIC_TO_AIC_SET);

    const std::string funcCode = generator.GenerateTaskExecutionForSplit(taskQue, 0, true, 0);
    EXPECT_TRUE(Contains(funcCode, "sysArgs.skTaskSyncCfg = static_cast<uint16_t>(4ULL);"));

    const std::string syncCode = generator.GenerateTaskExecutionForSplit(taskQue, 1, true, 0);
    EXPECT_TRUE(Contains(syncCode,
        "AscendC::AutoCoreSyncImpl<aic, aiv>(static_cast<SkCoreSyncType>(1), static_cast<uint8_t>(7), "
        "0x0000000000000001ULL);"));
}

TEST(ConstantCodeGeneratorTest, GenerateCombinedSource_IncludesEarlyStartRuntimeHelpers)
{
    ConstantCodeGenOptions options;
    ConstantCodeGenerator generator(options);

    SkTask aicTask;
    SkTask aivTask;
    ASSERT_TRUE(aicTask.Init(1));
    ASSERT_TRUE(aivTask.Init(1));

    const std::string source = generator.GenerateCombinedSource(aicTask, aivTask, SkKernelType::MIX_AIC_1_1);
    EXPECT_TRUE(Contains(source, "enum class SkEarlyStartMask : uint32_t"));
    EXPECT_TRUE(Contains(source,
        "__aicore__ inline void AutoCoreSyncImpl(SkCoreSyncType syncType, uint8_t numBlocks, uint64_t syncConfig)"));
    EXPECT_TRUE(Contains(source, "AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(AscendC::SYNC_AIC_FLAG);"));
    EXPECT_TRUE(Contains(source, "AscendC::CrossCoreSetFlag<0x02, PIPE_FIX>(AscendC::SYNC_AIC_AIV_FLAG);"));
}
