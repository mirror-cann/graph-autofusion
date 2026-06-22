/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "defalut_reg_func.h"

namespace af {
namespace ascir {
std::vector<std::unique_ptr<TmpBufDesc>> CalcArgmaxWithValueTmpSize(const AscNode &node) {
  // 固定分配 2048 字节的临时buffer
  // 内存布局：8个临时tensor，每个256B，总计2048B
  (void)node;
  const Expression exp = Symbol(2048);
  return GetTmpBuffer(exp);
}
}  // namespace ascir
}  // namespace af
