/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V1_ALIGNMENT_STRATEGY_H
#define OPTIMIZE_PLATFORM_V1_ALIGNMENT_STRATEGY_H

#include "platform/common/base_alignment_strategy.h"
namespace optimize {
class AlignmentStrategy : public BaseAlignmentStrategy {
 public:
  ~AlignmentStrategy() override = default;
  AlignmentStrategy() = default;

 protected:
  AlignmentType GetDefaultAlignmentType() override;

  af::Status DefaultAlignmentInferFunc(const af::AscNodePtr &node) override;
  af::Status BroadcastAlignmentInferFunc(const af::AscNodePtr &node) override;
  af::Status ConcatAlignmentInferFunc(const af::AscNodePtr &node) override;
  af::Status EleWiseAlignmentInferFunc(const af::AscNodePtr &node) override;
  af::Status LoadAlignmentInferFunc(const af::AscNodePtr &node) override;
  af::Status StoreAlignmentInferFunc(const af::AscNodePtr &node) override;
};

}  // namespace optimize
#endif  // OPTIMIZE_PLATFORM_V1_ALIGNMENT_STRATEGY_H
