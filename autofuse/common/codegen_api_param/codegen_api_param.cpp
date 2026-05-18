/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_api_param.h"

using namespace codegen;

const char *const kCodegenApiParam = "CodegenApiParam";

ge::Status CodegenApiParam::Register(af::AscNodePtr node, CodegenApiParamPtr api_param) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kCodegenApiParam, api_param), "Graph:%s, Node:%s SetExtAttr failed",
                 node->GetOwnerComputeGraph()->GetName().c_str(), node->GetNamePtr());
  return ge::SUCCESS;
}

CodegenApiParamPtr CodegenApiParam::GetNodeApiParam(af::AscNodePtr node) {
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  CodegenApiParamPtr api_param = nullptr;
  api_param = op_desc->TryGetExtAttr(kCodegenApiParam, api_param);
  GE_ASSERT_NOTNULL(api_param, "Graph:%s, Node:%s api_param is null", node->GetOwnerComputeGraph()->GetName().c_str(),
                    node->GetNamePtr());
  return api_param;
}
