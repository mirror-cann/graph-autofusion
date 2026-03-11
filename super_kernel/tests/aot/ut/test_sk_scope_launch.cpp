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
#include <gmock/gmock.h>
#include <memory>
#include <cstring>

#define private public
#define protected public
#include "sk_scope_launch.h"
#include "securec.h"

// ==================== Test Fixture ====================

class SkScopeLaunchTest : public testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// ==================== LaunchScopeKernel Basic Tests ====================

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

// ==================== Null ScopeName Tests ====================

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

// ==================== ScopeName Length Boundary Tests ====================

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_EmptyScopeName)
{
    const char* scopeName = "";
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(scopeName, stream, true);
    EXPECT_EQ(ret, ACL_SUCCESS);
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

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_ExactlyMaxLengthMinusOne)
{
    char scopeName[MAX_SCOPE_NAME_LENN];
    (void)memset_s(scopeName, sizeof(scopeName), 'b', MAX_SCOPE_NAME_LENN - 2);
    scopeName[MAX_SCOPE_NAME_LENN - 2] = '\0';
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(scopeName, stream, true);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

// ==================== Various ScopeName Content Tests ====================

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_SingleCharScopeName)
{
    const char* scopeName = "a";
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(scopeName, stream, true);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_SpecialCharsScopeName)
{
    const char* scopeName = "test_scope_!@#$%";
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(scopeName, stream, true);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_UnderscoreScopeName)
{
    const char* scopeName = "_test_scope_name_";
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(scopeName, stream, true);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_NumericScopeName)
{
    const char* scopeName = "123456789";
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(scopeName, stream, true);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

// ==================== Multiple Calls Tests ====================

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_MultipleCalls_SameName)
{
    const char* scopeName = "repeat_scope";
    aclrtStream stream = nullptr;
    
    for (int i = 0; i < 10; ++i) {
        aclError ret = LaunchScopeKernel(scopeName, stream, true);
        EXPECT_EQ(ret, ACL_SUCCESS);
    }
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_MultipleCalls_DifferentNames)
{
    aclrtStream stream = nullptr;
    
    for (int i = 0; i < 10; ++i) {
        char scopeName[32];
        (void)snprintf_s(scopeName, sizeof(scopeName), sizeof(scopeName) - 1, "scope_%d", i);
        aclError ret = LaunchScopeKernel(scopeName, stream, true);
        EXPECT_EQ(ret, ACL_SUCCESS);
    }
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_AlternatingBeginEnd)
{
    const char* scopeName = "alternating_scope";
    aclrtStream stream = nullptr;
    
    for (int i = 0; i < 5; ++i) {
        aclError retBegin = LaunchScopeKernel(scopeName, stream, true);
        EXPECT_EQ(retBegin, ACL_SUCCESS);
        
        aclError retEnd = LaunchScopeKernel(scopeName, stream, false);
        EXPECT_EQ(retEnd, ACL_SUCCESS);
    }
}

// ==================== Large Scale Tests ====================

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_LargeScale)
{
    aclrtStream stream = nullptr;
    
    for (int i = 0; i < 100; ++i) {
        char scopeName[64];
        (void)snprintf_s(scopeName, sizeof(scopeName), sizeof(scopeName) - 1, "large_scale_test_scope_%d", i);
        
        aclError retBegin = LaunchScopeKernel(scopeName, stream, true);
        EXPECT_EQ(retBegin, ACL_SUCCESS);
        
        aclError retEnd = LaunchScopeKernel(scopeName, stream, false);
        EXPECT_EQ(retEnd, ACL_SUCCESS);
    }
}

// ==================== Long Name Near Boundary Tests ====================

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_NearMaxLength_100)
{
    char scopeName[101];
    (void)memset_s(scopeName, sizeof(scopeName), 'c', 100);
    scopeName[100] = '\0';
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(scopeName, stream, true);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_NearMaxLength_200)
{
    char scopeName[201];
    (void)memset_s(scopeName, sizeof(scopeName), 'd', 200);
    scopeName[200] = '\0';
    aclrtStream stream = nullptr;
    aclError ret = LaunchScopeKernel(scopeName, stream, true);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

// ==================== Consistency Tests ====================

TEST_F(SkScopeLaunchTest, LaunchScopeKernel_BeginEndConsistency)
{
    const char* scopeName = "consistency_test";
    aclrtStream stream = nullptr;
    
    aclError retBegin = LaunchScopeKernel(scopeName, stream, true);
    aclError retEnd = LaunchScopeKernel(scopeName, stream, false);
    
    EXPECT_EQ(retBegin, retEnd);
    EXPECT_EQ(retBegin, ACL_SUCCESS);
}
