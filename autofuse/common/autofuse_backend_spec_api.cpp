/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/autofuse_backend_spec_api.h"

#include "backend/backend_spec.h"

namespace ge {
std::unique_ptr<AutofuseBackendSpec> GetAutofuseBackendSpec() {
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  if (backend_spec == nullptr) {
    return nullptr;
  }

  auto spec = std::make_unique<AutofuseBackendSpec>();
  spec->concat_max_input_num = backend_spec->concat_max_input_num;
  spec->concat_alg = backend_spec->concat_alg;
  spec->gather_spec = {
      backend_spec->gather_spec.enable_non_tail_gather, backend_spec->gather_spec.enable_reduce_gather_fusion,
      backend_spec->gather_spec.enable_gather_concat_fusion, backend_spec->gather_spec.enable_gather_broadcast_fusion,
      backend_spec->gather_spec.enable_gather_elementwise_forward_fusion};
  spec->slice_split_spec = {backend_spec->slice_split_spec.split_lowered_to_split,
                            backend_spec->slice_split_spec.slice_fuse_with_end_dim_1,
                            backend_spec->slice_split_spec.enable_split_flatten};
  spec->max_load_num = backend_spec->max_load_num;
  spec->max_input_nums_after_fuse = backend_spec->max_input_nums_after_fuse;
  spec->transpose_mode = backend_spec->transpose_mode;
  spec->enable_matmul_lowering_to_matmul = backend_spec->enable_matmul_lowering_to_matmul;
  return spec;
}
}  // namespace ge
