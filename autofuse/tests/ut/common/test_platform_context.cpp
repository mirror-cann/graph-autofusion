/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "common/platform_context.h"
#include "tests/depends/runtime/src/runtime_stub.h"

class PlatformContextTest : public testing::Test {
 protected:
  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
  }
};

TEST_F(PlatformContextTest, TryGetInitializedPlatformInfoReturnsFalseAfterReset) {
  ge::PlatformContext::GetInstance().Reset();

  ge::PlatformInfo info;
  EXPECT_FALSE(ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(info));
  EXPECT_EQ(info.ub_size, 0);
}

TEST_F(PlatformContextTest, TryGetInitializedPlatformInfoReturnsInjectedInfo) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo injected;
  injected.soc_ver = "mock_soc";
  injected.aiv_num = 20;
  injected.ub_size = 262144;
  ge::PlatformContext::GetInstance().SetPlatformInfo(injected);

  ge::PlatformInfo actual;
  EXPECT_TRUE(ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(actual));
  EXPECT_EQ(actual.soc_ver, "mock_soc");
  EXPECT_EQ(actual.aiv_num, 20);
  EXPECT_EQ(actual.ub_size, 262144);
}

TEST_F(PlatformContextTest, SetPlatformInfoWithEmptySocVerStoresUbSizeOverride) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo injected;
  injected.ub_size = 262144;
  ge::PlatformContext::GetInstance().SetPlatformInfo(injected);

  ge::PlatformInfo actual;
  EXPECT_FALSE(ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(actual));

  int64_t ub_size = 0;
  EXPECT_TRUE(ge::PlatformContext::GetInstance().TryGetUbSizeOverride(ub_size));
  EXPECT_EQ(ub_size, 262144);
}

TEST_F(PlatformContextTest, SetUbSizeOverrideDoesNotInitializePlatformInfo) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformContext::GetInstance().SetUbSizeOverride(2048);

  ge::PlatformInfo actual;
  int64_t ub_size = 0;
  EXPECT_FALSE(ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(actual));
  EXPECT_TRUE(ge::PlatformContext::GetInstance().TryGetUbSizeOverride(ub_size));
  EXPECT_EQ(ub_size, 2048);
}

TEST_F(PlatformContextTest, ResetClearsUbSizeOverride) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformContext::GetInstance().SetUbSizeOverride(262144);
  ge::PlatformContext::GetInstance().Reset();

  int64_t ub_size = 0;
  EXPECT_FALSE(ge::PlatformContext::GetInstance().TryGetUbSizeOverride(ub_size));
  EXPECT_EQ(ub_size, 0);
}

TEST_F(PlatformContextTest, GetPlatformInfoInitializesUbSizeFromRuntime) {
  ge::RuntimeStub::Reset();

  ge::PlatformInfo actual;
  EXPECT_EQ(ge::PlatformContext::GetInstance().GetPlatformInfo(actual), af::SUCCESS);
  EXPECT_EQ(actual.soc_ver, "2201");
  EXPECT_EQ(actual.aiv_num, 48);
  EXPECT_EQ(actual.ub_size, 245760);
}
