/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __CODEGEN_API_PARAM_H__
#define __CODEGEN_API_PARAM_H__

#include <variant>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include "ge_common/ge_api_types.h"
#include "ascir.h"

namespace codegen {

struct MergeAxesInfo {
  std::vector<std::string> repeats;
  std::vector<std::string> gm_strides;
  std::vector<std::string> ub_strides;
};
struct DataCopyBaseParams {
  std::string block_count;
  std::string block_len;
  std::string src_stride;
  std::string dst_stride;
};
struct DataCopyLoopModeParams {
  // loop1Size, loop2Size, loop1是内层循环, loop2是外层循环
  std::vector<std::string> loop_sizes;
  // loop1SrcStride, loop2SrcStride 单位: 数字个数，在拼接参数时，会转换成字节数
  std::vector<std::string> loop_src_strides;
  // loop1DrcStride, loop2DstStride单位: 数字个数，在拼接参数时，会转换成字节数
  std::vector<std::string> loop_dst_strides;
};
struct DmaSpecificParams {
  MergeAxesInfo merge_axes_info;
  DataCopyBaseParams data_copy_params;
  DataCopyLoopModeParams loop_mode_params;
};
struct ReduceSpecificParams {
  std::string reduce_type;
};
struct BroadcastSpecificParams {
  std::string broadcast_type;
};
struct TransposeSpecificParams {
  // Transpose api内部处理的vectorized output dims
  std::vector<std::string> output_dims;
  // Transpose api内部处理的vectorized input strides，根据输出tensor的vectorized axis的顺序进行重排
  std::vector<std::string> input_strides;
  // Transpose api内部处理的vectorized output strides
  std::vector<std::string> output_strides;
};

struct CodegenApiParam;
using CodegenApiParamPtr = std::shared_ptr<CodegenApiParam>;
struct CodegenApiParam {
  static ge::Status Register(af::AscNodePtr node, CodegenApiParamPtr api_param);
  static CodegenApiParamPtr GetNodeApiParam(af::AscNodePtr node);
  std::string api_name;
  std::vector<std::string> template_params;

  struct TensorParam {
    TensorParam() = default;
    TensorParam(const std::string& name, bool is_tensor, const std::string& offset)
                : name(name), is_tensor(is_tensor), offset(offset) {}
    std::string name;
    bool is_tensor = true;
    std::string offset;
  };

  std::vector<std::string> outer_loop_axes;
  std::vector<std::string> api_pre_process;
  std::vector<std::string> api_post_process;

  std::vector<TensorParam> input_params;
  std::vector<TensorParam> output_params;

  std::string tmp_buf_name;
  std::string cal_count;

  using AnySpecificParams = std::variant<
    std::monostate,
    DmaSpecificParams,
    ReduceSpecificParams,
    BroadcastSpecificParams,
    TransposeSpecificParams
  >;
  AnySpecificParams specific_params;
};
} // namespace codegen
#endif
