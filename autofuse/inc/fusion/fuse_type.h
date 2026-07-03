/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_INC_FUSION_FUSE_TYPE_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_INC_FUSION_FUSE_TYPE_H_

#include <cstdint>
#include <string>

namespace af {
namespace loop {

enum class FuseType : int32_t {
  kDefault = 0,
  kPointwise,
  kReduction,
  kConcat,
  kSplit,
  kSliceSplit,
  kGather,
  kTranspose,
  kCube,
  kReshape,
  kExtern
};

inline std::string FuseTypeToString(FuseType type) {
  switch (type) {
    case FuseType::kDefault:
      return "default";
    case FuseType::kPointwise:
      return "pointwise";
    case FuseType::kReduction:
      return "reduce";
    case FuseType::kConcat:
      return "concat";
    case FuseType::kSplit:
      return "split";
    case FuseType::kSliceSplit:
      return "slice";
    case FuseType::kGather:
      return "gather";
    case FuseType::kTranspose:
      return "transpose";
    case FuseType::kCube:
      return "cube";
    case FuseType::kReshape:
      return "reshape";
    default:
      return "extern";
  }
}

}  // namespace loop
}  // namespace af

namespace ge {
namespace loop {
using af::loop::FuseType;
using af::loop::FuseTypeToString;
}  // namespace loop
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_INC_FUSION_FUSE_TYPE_H_
