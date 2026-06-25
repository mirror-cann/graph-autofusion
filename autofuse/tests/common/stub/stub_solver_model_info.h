/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STUB_MODEL_INFO_H_
#define STUB_MODEL_INFO_H_
#include <vector>
#include "base/model_info.h"
#include "graph/symbolizer/symbolic.h"
namespace att {
ModelInfo CreateModelInfo(const uint32_t m_align = 1u,
                          const ge::ExprType expr_type = ge::ExprType::kExprVariable);

class TestTilingScheduleConfigTable : public TilingScheduleConfigTable {
 public:
  uint32_t cache_line_size_{128U};
  bool enable_cache_line_check_{true};

  [[nodiscard]] bool IsEnableBlockLoopAutoTune() const override { return false; }
  [[nodiscard]] bool IsEnableCacheLineCheck() const override { return enable_cache_line_check_; }
  [[nodiscard]] TradeOffConfig GetTradeOffConfig() const override { return {}; }
  [[nodiscard]] double GetUbThresholdPerfValEffect() const override { return 0.0; }
  [[nodiscard]] TilingScheduleConfig GetModelTilingScheduleConfig() const override {
    TilingScheduleConfig config;
    config.cache_line_size = cache_line_size_;
    config.vector_len_size = GetVectorLenSize();
    return config;
  }
  [[nodiscard]] uint32_t GetCacheLineSize() const override { return cache_line_size_; }
  [[nodiscard]] bool IsCoreNumThresholdPenaltyEnable() const override { return false; }
};

const TilingScheduleConfigTable *RegisterTestScheduleTable(
    uint32_t cache_line_size, bool enable_cache_line_check = true);

ModelInfo CreateGroupParallelCacheLineModelInfo(
    size_t group_id, uint32_t tiling_case_id, const Expr &cache_line_expr,
    uint32_t cache_line_size, CacheLineDirection direction,
    const ge::ExprType expr_type = ge::ExprType::kExprVariable);
}
#endif
