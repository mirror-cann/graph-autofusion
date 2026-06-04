/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "expression_convert_struct.h"
#include "codegen/codegen_kernel.h"

namespace codegen {

// 辅助函数实现，调用 Tiler 的 Size 和 ActualSize 方法
std::string TilerSize(const Tiler& tiler, const ge::Expression& expr) {
  // 空指针保护：检查 Expression 的 Str() 返回值
  auto str_ptr = expr.Str(af::StrType::kStrCpp);
  if (!str_ptr) {
    return "";  // 空指针时返回空字符串作为默认值
  }
  return tiler.Size(expr);
}

std::string TilerActualSize(const Tiler& tiler, const ge::Expression& expr) {
  // 空指针保护：检查 Expression 的 Str() 返回值
  auto str_ptr = expr.Str(af::StrType::kStrCpp);
  if (!str_ptr) {
    return "";  // 空指针时返回空字符串作为默认值
  }
  return tiler.ActualSize(expr);
}

}  // namespace codegen