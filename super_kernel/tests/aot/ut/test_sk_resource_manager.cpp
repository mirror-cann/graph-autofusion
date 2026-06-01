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

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define private public
#include "sk_resource_manager.h"
#undef private
#include "sk_model_context.h"
#include "stub/ut_common_stubs.h"

class SkResourceManagerTest : public testing::Test {
protected:
    void SetUp() override
    {
        SkUtResetTestControls();
        SkResourceManager::SetCurrentModel(reinterpret_cast<aclmdlRI>(0x1));
    }

    void TearDown() override
    {
        SkResourceManager::SetCurrentModel(nullptr);
        SkUtResetTestControls();
    }
};

TEST_F(SkResourceManagerTest, ValueMemory_InvalidInputsOrNullModel_ReturnInvalidParam)
{
    void* addr = nullptr;

    EXPECT_EQ(SkResourceManager::ValueMemory(nullptr), ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(SkResourceManager::ValueMemory(&addr, 0U), ACL_ERROR_INVALID_PARAM);

    SkResourceManager::SetCurrentModel(nullptr);
    EXPECT_EQ(SkResourceManager::ValueMemory(&addr), ACL_ERROR_INVALID_PARAM);
}

TEST_F(SkResourceManagerTest, ValueMemory_UnregisteredCallback_ReturnFailure)
{
    const aclmdlRI model = reinterpret_cast<aclmdlRI>(0x101);
    SkModelContext guard(model);
    SkResourceManager::SetCurrentModel(model);

    void* addr = nullptr;
    EXPECT_EQ(SkResourceManager::ValueMemory(&addr), ACL_ERROR_FAILURE);
    EXPECT_EQ(addr, nullptr);
    EXPECT_EQ(SkUtGetDestroyRegisterCallbackCallCount(), 0U);
}

TEST_F(SkResourceManagerTest, CallbackRegisterOrMallocFail_ReturnFailure)
{
    const aclmdlRI model = reinterpret_cast<aclmdlRI>(0x102);
    SkModelContext guard(model);
    SkResourceManager::SetCurrentModel(model);

    void* addr = nullptr;

    SkUtSetAclmdlRIDestroyRegisterCallbackRet(ACL_ERROR_FAILURE);
    EXPECT_EQ(SkResourceManager::CallbackRegister(model), ACL_ERROR_FAILURE);
    EXPECT_TRUE(SkResourceManager::callbackDataLabels_.empty());
    EXPECT_TRUE(SkResourceManager::destroyContexts_.empty());
    EXPECT_TRUE(SkResourceManager::registeredModelLabels_.empty());

    SkUtSetAclmdlRIDestroyRegisterCallbackRet(ACL_SUCCESS);
    EXPECT_EQ(SkResourceManager::CallbackRegister(model), ACL_SUCCESS);
    EXPECT_EQ(SkResourceManager::callbackDataLabels_.size(), 1U);
    EXPECT_EQ(SkResourceManager::destroyContexts_.size(), 1U);
    EXPECT_EQ(SkResourceManager::registeredModelLabels_.size(), 1U);
    SkUtSetAclrtMallocRet(ACL_ERROR_FAILURE);
    EXPECT_EQ(SkResourceManager::ValueMemory(&addr), ACL_ERROR_FAILURE);

    EXPECT_EQ(SkUtInvokeModelDestroyCallback(model), ACL_SUCCESS);
}

TEST_F(SkResourceManagerTest, ValueMemory_DestroyCallback_ReleasesTrackedMemory)
{
    const aclmdlRI model = reinterpret_cast<aclmdlRI>(0x202);
    {
        SkModelContext guard(model);
        SkResourceManager::SetCurrentModel(model);

        void* addrA = nullptr;
        void* addrB = nullptr;
        EXPECT_EQ(SkResourceManager::CallbackRegister(model), ACL_SUCCESS);
        EXPECT_EQ(SkResourceManager::ValueMemory(&addrA), ACL_SUCCESS);
        EXPECT_EQ(SkResourceManager::ValueMemory(&addrB), ACL_SUCCESS);
        ASSERT_NE(addrA, nullptr);
        ASSERT_NE(addrB, nullptr);
        EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 1U);

        EXPECT_EQ(SkUtInvokeModelDestroyCallback(model), ACL_SUCCESS);
        EXPECT_EQ(SkUtInvokeModelDestroyCallback(model), ACL_ERROR_INVALID_PARAM);
    }

    const aclmdlRI model2 = reinterpret_cast<aclmdlRI>(0x303);
    {
        SkModelContext guard(model2);
        SkResourceManager::SetCurrentModel(model2);
        void* addrC = nullptr;
        EXPECT_EQ(SkResourceManager::CallbackRegister(model2), ACL_SUCCESS);
        EXPECT_EQ(SkResourceManager::ValueMemory(&addrC), ACL_SUCCESS);
        ASSERT_NE(addrC, nullptr);

        SkUtSetAclrtFreeRet(ACL_ERROR_FAILURE);
        EXPECT_EQ(SkUtInvokeModelDestroyCallback(model2), ACL_SUCCESS);
    }
}

TEST_F(SkResourceManagerTest, OnModelDestroy_RepeatedSameUserData_IgnoredAfterFirstRelease)
{
    const aclmdlRI model = reinterpret_cast<aclmdlRI>(0x304);
    void* userData = nullptr;
    {
        SkModelContext guard(model);
        SkResourceManager::SetCurrentModel(model);
        void* addr = nullptr;
        EXPECT_EQ(SkResourceManager::CallbackRegister(model), ACL_SUCCESS);
        EXPECT_EQ(SkResourceManager::ValueMemory(&addr), ACL_SUCCESS);
        ASSERT_NE(addr, nullptr);
        ASSERT_EQ(SkResourceManager::callbackDataLabels_.size(), 1U);
        userData = SkResourceManager::callbackDataLabels_.begin()->first;
    }

    ASSERT_NE(userData, nullptr);
    SkResourceManager::OnModelDestroy(userData);
    EXPECT_TRUE(SkResourceManager::callbackDataLabels_.empty());
    EXPECT_TRUE(SkResourceManager::destroyContexts_.empty());
    EXPECT_TRUE(SkResourceManager::registeredModelLabels_.empty());
    EXPECT_TRUE(SkResourceManager::modelResources_.empty());

    SkResourceManager::OnModelDestroy(userData);
    EXPECT_TRUE(SkResourceManager::callbackDataLabels_.empty());
    EXPECT_TRUE(SkResourceManager::destroyContexts_.empty());
    EXPECT_TRUE(SkResourceManager::registeredModelLabels_.empty());
    EXPECT_TRUE(SkResourceManager::modelResources_.empty());
}

TEST_F(SkResourceManagerTest, CallbackRegister_ConcurrentSameHandleDifferentContexts_RegisterSeparately)
{
    const aclmdlRI model = reinterpret_cast<aclmdlRI>(0x404);
    constexpr uint32_t kThreadNum = 8;
    std::vector<aclError> results(kThreadNum, ACL_ERROR_FAILURE);
    std::vector<std::thread> workers;
    workers.reserve(kThreadNum);
    std::atomic<bool> startFlag(false);

    SkUtSetDestroyRegisterCallbackDelayUs(50000U);

    for (uint32_t i = 0; i < kThreadNum; ++i) {
        workers.emplace_back([&, i]() {
            while (!startFlag.load(std::memory_order_acquire)) {
            }
            SkModelContext guard(model);
            results[i] = SkResourceManager::CallbackRegister(model);
        });
    }

    startFlag.store(true, std::memory_order_release);
    for (auto& worker : workers) {
        worker.join();
    }

    for (uint32_t i = 0; i < kThreadNum; ++i) {
        EXPECT_EQ(results[i], ACL_SUCCESS);
    }
    EXPECT_EQ(SkUtGetDestroyRegisterCallbackCallCount(), kThreadNum);
    EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), kThreadNum);

    for (uint32_t i = 0; i < kThreadNum; ++i) {
        EXPECT_EQ(SkUtInvokeModelDestroyCallback(model), ACL_SUCCESS);
    }
    EXPECT_EQ(SkUtInvokeModelDestroyCallback(model), ACL_ERROR_INVALID_PARAM);
}

TEST_F(SkResourceManagerTest, CallbackRegister_NullModel_ReturnInvalidParam)
{
    EXPECT_EQ(SkResourceManager::CallbackRegister(nullptr), ACL_ERROR_INVALID_PARAM);
}

TEST_F(SkResourceManagerTest, ReleaseRecord_NullAddrOrUnknownKind_CoverBranches)
{
    const aclmdlRI model = reinterpret_cast<aclmdlRI>(0x606);

    SkResourceManager::ResourceRecord nullRecord;
    nullRecord.kind = SkResourceManager::ResourceKind::kDeviceMemory;
    nullRecord.addr = nullptr;
    nullRecord.bytes = 0U;
    EXPECT_EQ(SkResourceManager::ReleaseRecord(nullRecord), ACL_SUCCESS);

    SkResourceManager::ResourceRecord unknownRecord;
    unknownRecord.kind = static_cast<SkResourceManager::ResourceKind>(255);
    unknownRecord.addr = reinterpret_cast<void*>(0x1);
    unknownRecord.bytes = 8U;
    EXPECT_EQ(SkResourceManager::ReleaseRecord(unknownRecord), ACL_ERROR_FAILURE);
}

TEST_F(SkResourceManagerTest, CallbackRegister_AfterDestroyRegistersAgain)
{
    const aclmdlRI model = reinterpret_cast<aclmdlRI>(0x707);
    SkModelContext guard(model);
    EXPECT_EQ(SkResourceManager::CallbackRegister(model), ACL_SUCCESS);
    EXPECT_EQ(SkUtInvokeModelDestroyCallback(model), ACL_SUCCESS);

    // Destroy removed the registered modelLabel, so the next optimize for the same
    // fallback identity should register a fresh callback.
    const uint32_t registerCallCountBefore = SkUtGetDestroyRegisterCallbackCallCount();
    EXPECT_EQ(SkResourceManager::CallbackRegister(model), ACL_SUCCESS);
    EXPECT_EQ(SkUtGetDestroyRegisterCallbackCallCount(), registerCallCountBefore + 1U);
    EXPECT_EQ(SkUtInvokeModelDestroyCallback(model), ACL_SUCCESS);
}

TEST_F(SkResourceManagerTest, CallbackRegister_SameHandleDifferentOptimizeIdsRegisterSeparately)
{
    const aclmdlRI model = reinterpret_cast<aclmdlRI>(0x808);

    {
        SkModelContext firstGuard(model);
        SkResourceManager::SetCurrentModel(model);
        void* addr = nullptr;
        EXPECT_EQ(SkResourceManager::CallbackRegister(model), ACL_SUCCESS);
        EXPECT_EQ(SkResourceManager::ValueMemory(&addr), ACL_SUCCESS);
        ASSERT_NE(addr, nullptr);
    }

    {
        SkModelContext secondGuard(model);
        SkResourceManager::SetCurrentModel(model);
        EXPECT_EQ(SkResourceManager::CallbackRegister(model), ACL_SUCCESS);
    }

    EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 2U);
    EXPECT_EQ(SkUtInvokeModelDestroyCallback(model), ACL_SUCCESS);
    EXPECT_EQ(SkUtInvokeModelDestroyCallback(model), ACL_SUCCESS);
}
