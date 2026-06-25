/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_BASE_ARCH_PARAM_H_
#define ATT_BASE_ARCH_PARAM_H_

#include <cstdint>

namespace att {
namespace arch_param {
struct ArchHardwareConfig {
  uint32_t cache_line_size;
  uint32_t vector_len_size;
};

inline constexpr ArchHardwareConfig kV1ArchHardwareConfig{512U, 256U};
inline constexpr ArchHardwareConfig kV2ArchHardwareConfig{128U, 256U};
inline constexpr uint32_t kDefaultCacheLineSize = kV2ArchHardwareConfig.cache_line_size;
inline constexpr uint32_t kDefaultVectorLenSize = kV2ArchHardwareConfig.vector_len_size;
}  // namespace arch_param
}  // namespace att

#endif  // ATT_BASE_ARCH_PARAM_H_
