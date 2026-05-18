/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "micro_api_call_factory.h"
#include "ascir_ops.h"

#include "micro_where_api_call.h"

namespace codegen {
Status MicroWhereApiCall::Generate(const codegen::TensorManager &tensor_mng, [[maybe_unused]] const TPipe &tpipe,
                                   CallParam &param, string &result) {
  std::stringstream ss;

  // 第一个输入是 mask 输入，检查是否需要转换
  auto mask_tensor = tensor_mng.GetTensor(this->inputs_[0].second);
  GE_ASSERT_NOTNULL(mask_tensor);
  bool is_mask_reg = mask_tensor->init_as_mask_reg_;

  // mask 输入使用的名称（实际 MaskReg 或临时 MaskReg）
  std::string mask_name = is_mask_reg ? mask_tensor->name : (mask_tensor->name + "_temp_mask");

  // 如果不是 MaskReg，CompareScalar 转换 RegTensor 到临时 MaskReg
  if (!is_mask_reg) {
    std::string dtype_name;
    Tensor::DtypeName(mask_tensor->dtype_, dtype_name);
    ss << "AscendC::Reg::CompareScalar<" << dtype_name << ", AscendC::CMPMODE::NE>(" << mask_name << ", "
       << *mask_tensor << ", static_cast<" << dtype_name << ">(0), " << param.p_reg << ");" << std::endl;
  }

  // Where/Select 调用：Select(output, T1, T2, mask)
  // mask 输入在 AscendC::MicroAPI::Select 接口中排在最后一个参数位置
  ss << "AscendC::MicroAPI::" << this->api_name_ << "(";
  for (auto out_arg : this->outputs_) {
    GE_ASSERT_NOTNULL(tensor_mng.GetTensor(out_arg.second));
    ss << *(tensor_mng.GetTensor(out_arg.second)) << ", ";
  }
  // 其他输入（T1, T2）
  for (size_t i = 1; i < this->inputs_.size(); ++i) {
    GE_ASSERT_NOTNULL(tensor_mng.GetTensor(this->inputs_[i].second));
    ss << *(tensor_mng.GetTensor(this->inputs_[i].second)) << ", ";
  }
  // mask 输入排在最后
  ss << mask_name << ");" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

static MicroApiCallRegister<MicroWhereApiCall> register_micro_where_api_call("MicroWhereApiCall");
}  // namespace codegen
