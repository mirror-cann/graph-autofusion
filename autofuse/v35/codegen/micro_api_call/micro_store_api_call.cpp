/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "micro_api_call_factory.h"
#include "ascir_ops.h"

#include "micro_store_api_call.h"
#include "micro_dtype_utils.h"

namespace {
using codegen::DTYPE_SIZE_1BYTE;
using codegen::DTYPE_SIZE_2BYTE;
using codegen::DTYPE_SIZE_4BYTE;
using codegen::DTYPE_SIZE_8BYTE;
using codegen::GetDtypeSizeByName;
using codegen::GetUintDtypeNameBySize;

// StoreDist PACK 模式转换能力：
// DIST_PACK_B16:  uint16 → uint8  (2→1)
// DIST_PACK_B32:  uint32 → uint16 (4→2)
// DIST_PACK_B64:  uint64 → uint32 (8→4)
// DIST_PACK4_B32: uint32 → uint8  (4→1)
// FIRST_ELEMENT:  不做 dtype 转换

// 根据 dtype 转换比例确定 StoreDist 模式和 Pack 序列
// reg_dtype_size: RegTensor 的 dtype 大小（来自 max_dtype_size）
// output_dtype_size: UB tensor 的 dtype 大小（Store 输出目标）
void DetermineStoreDistAndPackSequence(uint32_t reg_dtype_size, uint32_t output_dtype_size, std::string &dist,
                                       int &pack_times, std::vector<std::string> &pack_sequence,
                                       std::string &current_dtype) {
  // RegTensor 当前 dtype（初始值）
  current_dtype = GetUintDtypeNameBySize(reg_dtype_size);

  // 不需要转换
  if (output_dtype_size >= reg_dtype_size) {
    pack_times = 0;
    return;
  }

  // dist 已有值（非PACK类型，不做dtype转换，全部通过Pack完成）
  if (!dist.empty()) {
    pack_times = 0;
    if (reg_dtype_size > output_dtype_size) {
      // 全部通过 Pack 完成 dtype 转换：从 reg_dtype_size 到 output_dtype_size
      uint32_t current_size = reg_dtype_size;
      while (current_size > output_dtype_size) {
        current_size /= 2;
        pack_times++;
        pack_sequence.push_back(GetUintDtypeNameBySize(current_size));
      }
    }
    return;
  }

  // 根据输出 dtype 大小选择最优 StoreDist 模式（优先使用能一步到位的模式）
  uint32_t store_dist_input_size = reg_dtype_size;

  if (output_dtype_size == DTYPE_SIZE_1BYTE) {
    // 目标是 uint8
    if (reg_dtype_size >= DTYPE_SIZE_4BYTE) {
      // uint32/uint64 → uint8：Pack 到 uint32，用 DIST_PACK4_B32 一步到位（最优）
      dist = "DIST_PACK4_B32";
      store_dist_input_size = DTYPE_SIZE_4BYTE;
    } else if (reg_dtype_size == DTYPE_SIZE_2BYTE) {
      // uint16 → uint8：用 DIST_PACK_B16
      dist = "DIST_PACK_B16";
      store_dist_input_size = DTYPE_SIZE_2BYTE;
    }
  } else if (output_dtype_size == DTYPE_SIZE_2BYTE) {
    // 目标是 uint16：用 DIST_PACK_B32 (32→16)
    dist = "DIST_PACK_B32";
    store_dist_input_size = DTYPE_SIZE_4BYTE;
  } else if (output_dtype_size == DTYPE_SIZE_4BYTE && reg_dtype_size == DTYPE_SIZE_8BYTE) {
    // 目标是 uint32，输入是 uint64：用 DIST_PACK_B64 (64→32) 一步到位
    dist = "DIST_PACK_B64";
    store_dist_input_size = DTYPE_SIZE_8BYTE;
  }

  // 计算 Pack 序列：从 reg_dtype_size 到 store_dist_input_size
  pack_times = 0;
  uint32_t current_size = reg_dtype_size;
  while (current_size > store_dist_input_size) {
    current_size /= 2;
    pack_times++;
    pack_sequence.push_back(GetUintDtypeNameBySize(current_size));
  }
}
}  // namespace

namespace codegen {
Status MicroStoreApiCall::Generate(const codegen::TensorManager &tensor_mng, const TPipe &tpipe, CallParam &param,
                                   string &result) {
  std::stringstream ss;
  auto tensor_id = GetInputTensorIdByIndex(0);
  GE_ASSERT_NOTNULL(tensor_mng.GetTensor(tensor_id));

  auto tensor_ptr = tensor_mng.GetTensor(tensor_id);

  const Tensor *ub_tensor_ptr = tpipe.GetTensor(this->GetOutputTensorIdByIndex(0));
  std::string dtype_name;
  Tensor::DtypeName(ub_tensor_ptr->dtype, dtype_name);
  uint32_t output_dtype_size = ge::GetSizeByDataType(ub_tensor_ptr->dtype);
  uint32_t reg_dtype_size = param.max_dtype_size.empty() ? output_dtype_size : GetDtypeSizeByName(param.max_dtype_size);

  int pack_times = 0;
  std::vector<std::string> pack_sequence;
  std::string current_dtype;

  DetermineStoreDistAndPackSequence(reg_dtype_size, output_dtype_size, this->dist_, pack_times, pack_sequence,
                                    current_dtype);

  // Pack 需要申请临时 MaskReg，并同步进行 MaskPack
  std::string p_reg_for_store = param.p_reg;
  if (pack_times > 0) {
    // 声明并初始化临时 MaskReg
    ss << "AscendC::MicroAPI::MaskReg " << tensor_ptr->name << "_temp = " << param.p_reg << ";" << std::endl;
    p_reg_for_store = tensor_ptr->name + "_temp";

    // Pack: 同时处理 RegTensor 和 MaskReg
    for (int i = 0; i < pack_times && i < static_cast<int>(pack_sequence.size()); ++i) {
      ss << "AscendC::Reg::Pack<" << pack_sequence[i] << ", " << current_dtype << ">((AscendC::Reg::RegTensor<"
         << pack_sequence[i] << ">&)" << tensor_ptr->name << ", (AscendC::Reg::RegTensor<" << current_dtype << ">&)"
         << tensor_ptr->name << ");" << std::endl;
      // MaskReg 同步压缩
      ss << "AscendC::Reg::MaskPack(" << p_reg_for_store << ", " << p_reg_for_store << ");" << std::endl;
      current_dtype = pack_sequence[i];
    }
  }

  // Store
  std::string ub_tensor_name = ub_tensor_ptr->name;
  std::string store_template_params = "";
  if (!this->dist_.empty()) {
    store_template_params = "<" + dtype_name + ", AscendC::MicroAPI::StoreDist::" + this->dist_ + ">";
  }
  ss << "AscendC::MicroAPI::StoreAlign" << store_template_params << "(" << ub_tensor_name << " + " << param.offset
     << ", " << tensor_ptr->name << ", " << p_reg_for_store << ");" << std::endl;

  result = ss.str();
  return ge::SUCCESS;
}

Status MicroStoreApiCall::Init(const ascir::NodeView &node) {
  this->dist_ = "";
  auto in_node = std::dynamic_pointer_cast<af::AscNode>(node->inputs[0].anchor.GetOwnerNode());
  auto out_dtype_size = ge::GetSizeByDataType(in_node->outputs[0].attr.dtype);
  bool is_output_out_dtype_scalar = std::all_of(node->outputs[0].attr.vectorized_strides.begin(),
                                 node->outputs[0].attr.vectorized_strides.end(), [](const ascir::SizeExpr &stride) {
                                   return af::SymbolicUtils::StaticCheckEq(stride.Simplify(), af::sym::kSymbolZero) ==
                                          af::TriBool::kTrue;
                                 });
  if (is_output_out_dtype_scalar) {
    if (out_dtype_size == DTYPE_SIZE_1BYTE) {
      this->dist_ = "DIST_FIRST_ELEMENT_B8";
    } else if (out_dtype_size == DTYPE_SIZE_2BYTE) {
      this->dist_ = "DIST_FIRST_ELEMENT_B16";
    } else if (out_dtype_size == DTYPE_SIZE_4BYTE) {
      this->dist_ = "DIST_FIRST_ELEMENT_B32";
    }
  }
  return ge::SUCCESS;
}
static MicroApiCallRegister<MicroStoreApiCall> register_micro_store_api_call("MicroStoreApiCall");
}  // namespace codegen