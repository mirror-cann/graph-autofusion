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
#include <memory>
#include <cstring>
#include <map>
#include <unordered_map>

#define private public
#define protected public
#include "sk_node.h"
#include "sk_scope_launch.h"
#include "sk_common.h"
#include "securec.h"

class SkNodeTest : public testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
        GlobalMockObject::verify();
    }
};

struct JudgeTaskKernelInfo {
    bool isBegin = true;
    bool isFuseEnable = true;
    std::unique_ptr<char[]> scopeName;
};

extern bool IsScopeKernel(aclmdlRIKernelTaskParams params, JudgeTaskKernelInfo* info);

TEST_F(SkNodeTest, IsScopeKernel_GetFunctionName_Failed)
{
    aclmdlRIKernelTaskParams params{};
    params.funcHandle = nullptr;
    JudgeTaskKernelInfo info;
    bool ret = IsScopeKernel(params, &info);
    EXPECT_EQ(ret, false);
}

int Fake_aclrtGetFunctionNameBegin(void* funcHandle, size_t size, char* name) 
{
    const char* src = "sk_scope_kernel_begin";
    snprintf_s(name, size, size, "%s", src);
    return 0;
}

int Fake_aclrtMemcpy(void* dst, size_t dstSize, const void* src, size_t count, aclrtMemcpyKind kind) 
{
    ScopeKernelArgs fakeArgs;
    const char* defaultName = "default_sk_scope_name";
    snprintf_s(fakeArgs.name, sizeof(fakeArgs.name), sizeof(fakeArgs.name), "%s", defaultName);
    fakeArgs.name[MAX_SCOPE_NAME_LEN - 1] = '\0';
    memcpy_s(dst, sizeof(ScopeKernelArgs), &fakeArgs, sizeof(ScopeKernelArgs));
    return 0;
}

TEST_F(SkNodeTest, IsScopeKernel_Normal_ScopeName)
{
    aclmdlRIKernelTaskParams params{};
    params.funcHandle = nullptr;
    JudgeTaskKernelInfo info;
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionNameBegin));
    MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy));
    bool ret = IsScopeKernel(params, &info);
    EXPECT_EQ(ret, true);
}

TEST_F(SkNodeTest, IsScopeKernel_ScopeName_Isnullptr)
{
    aclmdlRIKernelTaskParams params{};
    params.funcHandle = nullptr;
    JudgeTaskKernelInfo info;
    MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionNameBegin));
    MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy));
    bool ret = IsScopeKernel(params, &info);
    EXPECT_EQ(ret, true);
}
