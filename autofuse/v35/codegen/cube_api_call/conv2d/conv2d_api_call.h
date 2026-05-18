/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_CONV2D_API_CALL_H__
#define __AUTOFUSE_CONV2D_API_CALL_H__
#include "codegen_kernel.h"
#include "common_utils.h"

namespace codegen {
class Conv2DApiCall final : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit Conv2DApiCall(const std::string &api_name) : ApiCall(api_name) {}
  Status ParseAttr(const ascir::NodeView &node) override;
  Status PreProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                    const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                    std::string &result) const override;
  Status GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler, std::stringstream &ss) const override;
  Status PostProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                     const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                     std::string &result) const override;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  std::string &result) const override;
  Status GenerateMacro(std::string &result) const override;
  ~Conv2DApiCall() final = default;

 private:
  ascgen_utils::Conv2DAttr conv_attr_data_;
};
}  // namespace codegen
#endif  // __AUTOFUSE_CONV2D_API_CALL_H__
