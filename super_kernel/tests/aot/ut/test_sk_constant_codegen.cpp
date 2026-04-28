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

    const std::string notifyCode =
        generator.GenerateTaskExecutionForSplit(taskQue, 0, true, SkKernelType::DEFAULT, 0);
    EXPECT_TRUE(Contains(notifyCode,
        "AscendC::NotifyFunc<true>(0x0000000000001000ULL, 0x0000000000000055ULL)"));
    EXPECT_TRUE(Contains(notifyCode,
        "AscendC::NotifyFunc<false>(0x0000000000001000ULL, 0x0000000000000055ULL)"));

    const std::string waitCode =
        generator.GenerateTaskExecutionForSplit(taskQue, 1, true, SkKernelType::DEFAULT, 0);
    EXPECT_TRUE(Contains(waitCode,
        "AscendC::WaitFunc<true>(0x0000000000002000ULL, 0x0000000000000066ULL, 2U)"));
    EXPECT_TRUE(Contains(waitCode,
        "AscendC::WaitFunc<false>(0x0000000000002000ULL, 0x0000000000000066ULL, 2U)"));

    const std::string resetCode =
        generator.GenerateTaskExecutionForSplit(taskQue, 2, true, SkKernelType::DEFAULT, 0);
    EXPECT_TRUE(Contains(resetCode,
        "AscendC::ResetFunc<true>(0x0000000000003000ULL, 0x0000000000000000ULL)"));
    EXPECT_TRUE(Contains(resetCode,
        "AscendC::ResetFunc<false>(0x0000000000003000ULL, 0x0000000000000000ULL)"));
}
