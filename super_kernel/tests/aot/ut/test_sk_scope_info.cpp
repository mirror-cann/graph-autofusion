/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "sk_scope_info.h"

class ScopeIdGeneratorTest : public testing::Test {};

TEST_F(ScopeIdGeneratorTest, NextId_IncrementSequentially) {
  uint16_t id0 = ScopeIdGenerator::Instance().NextId();
  uint16_t id1 = ScopeIdGenerator::Instance().NextId();
  uint16_t id2 = ScopeIdGenerator::Instance().NextId();

  EXPECT_EQ(id1, id0 + 1);
  EXPECT_EQ(id2, id1 + 1);
}

class SuperKernelScopeInfoIdTest : public testing::Test {};

TEST_F(SuperKernelScopeInfoIdTest, ScopeId_AssignedAtCreation) {
  uint16_t prevId = ScopeIdGenerator::Instance().NextId();
  SuperKernelScopeInfo scope1;
  SuperKernelScopeInfo scope2;
  SuperKernelScopeInfo scope3;

  EXPECT_EQ(scope1.GetScopeId(), prevId + 1);
  EXPECT_EQ(scope2.GetScopeId(), prevId + 2);
  EXPECT_EQ(scope3.GetScopeId(), prevId + 3);
}

TEST_F(SuperKernelScopeInfoIdTest, ScopeId_NotRecycledOnDestroy) {
  uint16_t prevId = ScopeIdGenerator::Instance().NextId();

  {
    SuperKernelScopeInfo scope1;
    EXPECT_EQ(scope1.GetScopeId(), prevId + 1);
  }

  SuperKernelScopeInfo scope2;
  EXPECT_EQ(scope2.GetScopeId(), prevId + 2);
}

TEST_F(SuperKernelScopeInfoIdTest, ScopeId_MovedScopeKeepsOriginalId) {
  uint16_t prevId = ScopeIdGenerator::Instance().NextId();

  SuperKernelScopeInfo scope1;
  uint16_t originalId = scope1.GetScopeId();
  EXPECT_EQ(originalId, prevId + 1);

  SuperKernelScopeInfo scope2 = std::move(scope1);
  EXPECT_EQ(scope2.GetScopeId(), originalId);
}

TEST_F(SuperKernelScopeInfoIdTest, ScopeId_AssignmentMoveKeepsId) {
  uint16_t prevId = ScopeIdGenerator::Instance().NextId();

  SuperKernelScopeInfo scope1;
  uint16_t originalId1 = scope1.GetScopeId();

  SuperKernelScopeInfo scope2;
  uint16_t originalId2 = scope2.GetScopeId();

  scope2 = std::move(scope1);
  EXPECT_EQ(scope2.GetScopeId(), originalId1);
  EXPECT_NE(scope2.GetScopeId(), originalId2);
}

TEST_F(SuperKernelScopeInfoIdTest, ScopeId_PersistAcrossSessions) {
  uint16_t prevId = ScopeIdGenerator::Instance().NextId();
  SuperKernelScopeInfo scope1;
  uint16_t id1 = scope1.GetScopeId();
  EXPECT_EQ(id1, prevId + 1);

  SuperKernelScopeInfo scope2;
  uint16_t id2 = scope2.GetScopeId();
  EXPECT_EQ(id2, id1 + 1);
}