/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "acl/acl.h"

/**
 * @brief Test fixture class for runtime metadata stub functions
 */
class RuntimeMetadataStubTest : public testing::Test {
protected:
    void SetUp() override {
        binHdl = reinterpret_cast<void*>(0x1000);
    }

    void TearDown() override {
    }

    void* binHdl;
};

// ==================== rtBinaryGetMetaNum Tests ====================
TEST_F(RuntimeMetadataStubTest, rtBinaryGetMetaNum_WithValidHandle)
{
    size_t metaNum = 0;
    int ret = rtBinaryGetMetaNum(binHdl, RT_BINARY_TYPE_SK_INFO, &metaNum);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(metaNum, 0);
}

TEST_F(RuntimeMetadataStubTest, rtBinaryGetMetaNum_WithNullHandle)
{
    size_t metaNum = 0;
    int ret = rtBinaryGetMetaNum(nullptr, RT_BINARY_TYPE_SK_INFO, &metaNum);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(metaNum, 0);
}

TEST_F(RuntimeMetadataStubTest, rtBinaryGetMetaNum_WithDifferentType)
{
    size_t metaNum = 0;
    int ret = rtBinaryGetMetaNum(binHdl, 2, &metaNum);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(metaNum, 0);
}

TEST_F(RuntimeMetadataStubTest, rtBinaryGetMetaNum_WithNullMetaNumPtr)
{
    int ret = rtBinaryGetMetaNum(binHdl, RT_BINARY_TYPE_SK_INFO, nullptr);

    EXPECT_EQ(ret, 0);
}

// ==================== rtBinaryGetMetaData Tests ====================

TEST_F(RuntimeMetadataStubTest, rtBinaryGetMetaData_WithZeroCount)
{
    size_t metaNum = 0;
    void* data_list[] = {nullptr};
    size_t size_list[] = {0};

    int ret = rtBinaryGetMetaData(binHdl, RT_BINARY_TYPE_SK_INFO, metaNum, data_list, size_list);

    EXPECT_EQ(ret, 0);
}

TEST_F(RuntimeMetadataStubTest, rtBinaryGetMetaData_WithNullHandle)
{
    size_t metaNum = 0;
    void* data_list[] = {nullptr};
    size_t size_list[] = {0};

    int ret = rtBinaryGetMetaData(nullptr, RT_BINARY_TYPE_SK_INFO, metaNum, data_list, size_list);

    EXPECT_EQ(ret, 0);
}

TEST_F(RuntimeMetadataStubTest, rtBinaryGetMetaData_WithNullDataList)
{
    size_t metaNum = 0;

    int ret = rtBinaryGetMetaData(binHdl, RT_BINARY_TYPE_SK_INFO, metaNum, nullptr, nullptr);

    EXPECT_EQ(ret, 0);
}

TEST_F(RuntimeMetadataStubTest, rtBinaryGetMetaData_WithDifferentType)
{
    size_t metaNum = 0;
    void* data_list[] = {nullptr};
    size_t size_list[] = {0};

    int ret = rtBinaryGetMetaData(binHdl, 3, metaNum, data_list, size_list);

    EXPECT_EQ(ret, 0);
}

// ==================== rtBinaryGetMetaInfo Tests ====================

TEST_F(RuntimeMetadataStubTest, rtBinaryGetMetaInfo_WithZeroCount)
{
    size_t metaNum = 0;
    void* data_list[] = {nullptr};
    size_t size_list[] = {0};

    int ret = rtBinaryGetMetaInfo(binHdl, RT_BINARY_TYPE_SK_INFO, metaNum, data_list, size_list);

    EXPECT_EQ(ret, 0);
}

TEST_F(RuntimeMetadataStubTest, rtBinaryGetMetaInfo_WithNullHandle)
{
    size_t metaNum = 0;
    void* data_list[] = {nullptr};
    size_t size_list[] = {0};

    int ret = rtBinaryGetMetaInfo(nullptr, RT_BINARY_TYPE_SK_INFO, metaNum, data_list, size_list);

    EXPECT_EQ(ret, 0);
}

TEST_F(RuntimeMetadataStubTest, rtBinaryGetMetaInfo_WithNullDataList)
{
    size_t metaNum = 0;

    int ret = rtBinaryGetMetaInfo(binHdl, RT_BINARY_TYPE_SK_INFO, metaNum, nullptr, nullptr);

    EXPECT_EQ(ret, 0);
}

TEST_F(RuntimeMetadataStubTest, rtBinaryGetMetaInfo_WithDifferentType)
{
    size_t metaNum = 0;
    void* data_list[] = {nullptr};
    size_t size_list[] = {0};

    int ret = rtBinaryGetMetaInfo(binHdl, 4, metaNum, data_list, size_list);

    EXPECT_EQ(ret, 0);
}

// ==================== Combined Usage Tests ====================

TEST_F(RuntimeMetadataStubTest, CombinedUsage_GetNumThenGetData)
{
    size_t metaNum = 0;
    int ret1 = rtBinaryGetMetaNum(binHdl, RT_BINARY_TYPE_SK_INFO, &metaNum);

    void* data_list[] = {nullptr, nullptr};
    size_t size_list[] = {0, 0};
    int ret2 = rtBinaryGetMetaData(binHdl, RT_BINARY_TYPE_SK_INFO, metaNum, data_list, size_list);

    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(metaNum, 0);
    EXPECT_EQ(ret2, 0);
}

TEST_F(RuntimeMetadataStubTest, CombinedUsage_GetNumThenGetInfo)
{
    size_t metaNum = 0;
    int ret1 = rtBinaryGetMetaNum(binHdl, RT_BINARY_TYPE_SK_INFO, &metaNum);

    void* data_list[] = {nullptr, nullptr};
    size_t size_list[] = {0, 0};
    int ret2 = rtBinaryGetMetaInfo(binHdl, RT_BINARY_TYPE_SK_INFO, metaNum, data_list, size_list);

    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(metaNum, 0);
    EXPECT_EQ(ret2, 0);
}

// ==================== Boundary Tests ====================

TEST_F(RuntimeMetadataStubTest, Boundary_LargeHandleValue)
{
    void* largeHandle = reinterpret_cast<void*>(0xFFFFFFFF);
    size_t metaNum = 0;
    int ret = rtBinaryGetMetaNum(largeHandle, RT_BINARY_TYPE_SK_INFO, &metaNum);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(metaNum, 0);
}

TEST_F(RuntimeMetadataStubTest, Boundary_ZeroType)
{
    size_t metaNum = 0;
    int ret = rtBinaryGetMetaNum(binHdl, 0, &metaNum);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(metaNum, 0);
}

TEST_F(RuntimeMetadataStubTest, Boundary_LargeTypeValue)
{
    size_t metaNum = 0;
    int ret = rtBinaryGetMetaNum(binHdl, 9999, &metaNum);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(metaNum, 0);
}
