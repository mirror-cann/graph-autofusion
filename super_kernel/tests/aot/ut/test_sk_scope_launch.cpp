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

#define private public
#define protected public
#include "sk_scope_launch.h"
#include "super_kernel.h"
#include "securec.h"

class SkScopeLaunchTest : public testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
        GlobalMockObject::verify();
    }
};

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_Begin_Success)
{
    const char* scopeName = "test_scope";
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(scopeName, stream, true);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_End_Success)
{
    const char* scopeName = "test_scope";
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(scopeName, stream, false);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_NullScopeName_Begin)
{
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(nullptr, stream, true);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_NullScopeName_End)
{
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(nullptr, stream, false);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_EmptyScopeName)
{
    const char* scopeName = "";
    aclrtStream stream = nullptr;
    aclError ret = aclskScopeBegin(scopeName, stream);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = aclskScopeEnd(scopeName, stream);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_MaxLengthScopeName)
{
    char scopeName[MAX_SCOPE_NAME_LENN];
    (void)memset_s(scopeName, sizeof(scopeName), 'a', MAX_SCOPE_NAME_LENN - 1);
    scopeName[MAX_SCOPE_NAME_LENN - 1] = '\0';
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(scopeName, stream, true);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_ExceedMaxLengthScopeName)
{
    char scopeName[MAX_SCOPE_NAME_LENN + 10];
    (void)memset_s(scopeName, sizeof(scopeName), 'a', MAX_SCOPE_NAME_LENN + 9);
    scopeName[MAX_SCOPE_NAME_LENN + 9] = '\0';
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(scopeName, stream, true);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_BeginEndConsistency)
{
    const char* scopeName = "consistency_test";
    aclrtStream stream = nullptr;
    
    aclError retBegin = LaunchScopeKernel(scopeName, stream, true);
    aclError retEnd = LaunchScopeKernel(scopeName, stream, false);
    
    EXPECT_EQ(retBegin, retEnd);
    EXPECT_EQ(retBegin, ACL_SUCCESS);
}
