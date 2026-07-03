/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_BASE_COMMON_AUTOFUSE_BACKEND_SPEC_API_H_
#define ASCGEN_DEV_BASE_COMMON_AUTOFUSE_BACKEND_SPEC_API_H_

#include <cstdint>
#include <memory>

namespace ge {
struct AutofuseGatherSpec {
  bool enable_non_tail_gather = false;
  bool enable_reduce_gather_fusion = false;
  bool enable_gather_concat_fusion = false;
  bool enable_gather_broadcast_fusion = false;
  bool enable_gather_elementwise_forward_fusion = false;
};

struct AutofuseSliceSplitSpec {
  bool split_lowered_to_split = false;
  bool slice_fuse_with_end_dim_1 = false;
  bool enable_split_flatten = false;
};

enum class AutofuseTransposeMode : uint32_t {
  TRANSPOSE_MODE_NORMAL = 0,
  TRANSPOSE_MODE_UNNORMAL = 1,
};

struct AutofuseBackendSpec {
  uint32_t concat_max_input_num = 0U;
  int32_t concat_alg = 0;
  AutofuseGatherSpec gather_spec;
  AutofuseSliceSplitSpec slice_split_spec;
  uint32_t max_load_num = 0U;
  uint32_t max_input_nums_after_fuse = 8U;
  uint32_t transpose_mode = static_cast<uint32_t>(AutofuseTransposeMode::TRANSPOSE_MODE_NORMAL);
  bool enable_matmul_lowering_to_matmul = false;
};

std::unique_ptr<AutofuseBackendSpec> GetAutofuseBackendSpec();
}  // namespace ge

#endif  // ASCGEN_DEV_BASE_COMMON_AUTOFUSE_BACKEND_SPEC_API_H_
