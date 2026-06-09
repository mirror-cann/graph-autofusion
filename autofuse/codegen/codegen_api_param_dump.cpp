/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_api_param/codegen_api_param.h"
#include "codegen/codegen_kernel.h"
#include "codegen/expression_convert_struct.h"

#include <fstream>
#include <iomanip>
#include <sstream>

using namespace codegen;

thread_local uint64_t g_api_param_dump_index = 0UL;
namespace {
constexpr int kDumpIndexWidth = 5;

std::string SanitizeFileName(const std::string &name) {
  std::string result;
  result.reserve(name.length());
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.' || c == ' ') {
      result += c;
    } else {
      result += '_';
    }
  }
  return result;
}

void DumpStringList(std::stringstream &ss, const std::string &label, const std::vector<std::string> &items,
                    const std::string &indent) {
  if (items.empty()) return;
  ss << indent << "." << label << " = {";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << items[i];
  }
  ss << "}" << std::endl;
}

void DumpExpressionList(std::stringstream &ss, const std::string &label,
                        const std::vector<CombinedExpression> &items, const std::string &indent,
                        const Tiler &tiler) {
  if (items.empty()) return;
  ss << indent << "." << label << " = {";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << items[i].ToStr(tiler);
  }
  ss << "}" << std::endl;
}

void DumpTensorParamList(std::stringstream &ss, const std::string &label,
                         const std::vector<CodegenApiParam::TensorParam> &params, const std::string &indent,
                         const Tiler &tiler) {
  if (params.empty()) return;
  ss << indent << "." << label << " = {" << std::endl;
  for (size_t i = 0; i < params.size(); ++i) {
    const auto &p = params[i];
    ss << indent << "  [" << i << "] " << p.name;
    if (!p.is_tensor) ss << " (scalar)";
    if (!p.offset.IsEmpty()) ss << ", offset=" << p.offset.ToStr(tiler);
    ss << std::endl;
  }
  ss << indent << "}" << std::endl;
}

void DumpMergeAxesInfo(std::stringstream &ss, const MergeAxesInfo &m, const std::string &indent) {
  ss << indent << ".merge_axes_info = {" << std::endl;
  auto dump_expr_vec = [&](const std::string &lbl, const std::vector<ge::Expression> &v) {
    if (v.empty()) return;
    ss << indent << "  ." << lbl << " = {";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i > 0) ss << ", ";
      auto s = v[i].Str(af::StrType::kStrCpp);
      ss << (s ? s.get() : "");
    }
    ss << "}" << std::endl;
  };
  dump_expr_vec("repeats", m.repeats);
  dump_expr_vec("gm_strides", m.gm_strides);
  dump_expr_vec("ub_strides", m.ub_strides);
  ss << indent << "}" << std::endl;
}

void DumpDataCopyBaseParams(std::stringstream &ss, const DataCopyBaseParams &p, const std::string &indent,
                            const Tiler &tiler) {
  ss << indent << ".data_copy_params = {" << std::endl;
  ss << indent << "  .block_count = " << p.block_count.ToStr(tiler) << std::endl;
  ss << indent << "  .block_len = " << p.block_len.ToStr(tiler) << std::endl;
  ss << indent << "  .src_stride = " << p.src_stride.ToStr(tiler) << std::endl;
  ss << indent << "  .dst_stride = " << p.dst_stride.ToStr(tiler) << std::endl;
  ss << indent << "}" << std::endl;
}

void DumpDataCopyLoopModeParams(std::stringstream &ss, const DataCopyLoopModeParams &p, const std::string &indent,
                                const Tiler &tiler) {
  ss << indent << ".loop_mode_params = {" << std::endl;
  DumpExpressionList(ss, "loop_sizes", p.loop_sizes, indent + "  ", tiler);
  DumpExpressionList(ss, "loop_src_strides", p.loop_src_strides, indent + "  ", tiler);
  DumpExpressionList(ss, "loop_dst_strides", p.loop_dst_strides, indent + "  ", tiler);
  ss << indent << "}" << std::endl;
}

void DumpSpecificParams(std::stringstream &ss, const CodegenApiParam::AnySpecificParams &sp,
                        const std::string &indent, const Tiler &tiler) {
  if (std::holds_alternative<std::monostate>(sp)) return;
  ss << indent << ".specific_params = {" << std::endl;
  if (std::holds_alternative<DmaSpecificParams>(sp)) {
    const auto &dma = std::get<DmaSpecificParams>(sp);
    ss << indent << "  .type = DmaSpecific" << std::endl;
    DumpMergeAxesInfo(ss, dma.merge_axes_info, indent + "  ");
    DumpDataCopyBaseParams(ss, dma.data_copy_params, indent + "  ", tiler);
    DumpDataCopyLoopModeParams(ss, dma.loop_mode_params, indent + "  ", tiler);
  } else if (std::holds_alternative<ReduceSpecificParams>(sp)) {
    const auto &reduce = std::get<ReduceSpecificParams>(sp);
    ss << indent << "  .type = ReduceSpecific" << std::endl;
    ss << indent << "  .reduce_type = " << reduce.reduce_type << std::endl;
  } else if (std::holds_alternative<BroadcastSpecificParams>(sp)) {
    const auto &brc = std::get<BroadcastSpecificParams>(sp);
    ss << indent << "  .type = BroadcastSpecific" << std::endl;
    ss << indent << "  .broadcast_type = " << brc.broadcast_type << std::endl;
  } else if (std::holds_alternative<TransposeSpecificParams>(sp)) {
    const auto &tp = std::get<TransposeSpecificParams>(sp);
    ss << indent << "  .type = TransposeSpecific" << std::endl;
    DumpExpressionList(ss, "output_dims", tp.output_dims, indent + "  ", tiler);
    DumpExpressionList(ss, "input_strides", tp.input_strides, indent + "  ", tiler);
    DumpExpressionList(ss, "output_strides", tp.output_strides, indent + "  ", tiler);
  }
  ss << indent << "}" << std::endl;
}

std::string ApiParamToTxtString(const CodegenApiParam &p, const Tiler &tiler) {
  std::stringstream ss;
  std::string indent = "    ";
  ss << indent << ".api_name = " << p.api_name << std::endl;
  DumpStringList(ss, "template_params", p.template_params, indent);
  DumpExpressionList(ss, "outer_loop_axes", p.outer_loop_axes, indent, tiler);
  DumpStringList(ss, "api_pre_process", p.api_pre_process, indent);
  DumpStringList(ss, "api_post_process", p.api_post_process, indent);
  DumpTensorParamList(ss, "input_params", p.input_params, indent, tiler);
  DumpTensorParamList(ss, "output_params", p.output_params, indent, tiler);
  if (!p.tmp_buf_name.empty()) {
    ss << indent << ".tmp_buf_name = " << p.tmp_buf_name << std::endl;
  }
  if (!p.cal_count.IsEmpty()) {
    ss << indent << ".cal_count = " << p.cal_count.ToStr(tiler) << std::endl;
  }
  DumpSpecificParams(ss, p.specific_params, indent, tiler);
  return ss.str();
}
}  // namespace

ge::Status CodegenApiParam::DumpGraphApiParams(const ascir::ImplGraph &graph, const Tiler &tiler,
                                               const std::string &prefix) {
  std::ostringstream name_ss;
  name_ss << "api_param_" << std::setw(kDumpIndexWidth) << std::setfill('0') << g_api_param_dump_index
          << "_" << graph.GetName() << ".txt";
  std::string file_path = prefix + SanitizeFileName(name_ss.str());
  ++g_api_param_dump_index;
  std::ofstream ofs(file_path);
  if (!ofs.is_open()) {
    GELOGE(ge::FAILED, "[DumpCodegenApiParam] open file failed: %s", file_path.c_str());
    return ge::FAILED;
  }

  ofs << "================================================================================" << std::endl;
  ofs << "Graph: " << graph.GetName() << std::endl;
  ofs << "================================================================================" << std::endl;
  ofs << std::endl;
  ofs << "ApiParams:" << std::endl;

  size_t idx = 0UL;
  for (const auto &node : graph.GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    CodegenApiParamPtr api_param = nullptr;
    if (op_desc != nullptr) {
      api_param = op_desc->TryGetExtAttr(kCodegenApiParam, api_param);
    }
    ofs << "  [" << idx << "] " << node->GetName() << " : " << node->GetType();
    if (api_param == nullptr) {
      ofs << std::endl;
    } else {
      ofs << " {" << std::endl;
      ofs << ApiParamToTxtString(*api_param, tiler);
      ofs << "  }" << std::endl;
    }
    ++idx;
  }

  ofs << "================================================================================" << std::endl;
  ofs << "End of Dump" << std::endl;
  ofs << "================================================================================" << std::endl;

  ofs.close();
  GELOGI("[DumpCodegenApiParam] dumped %zu nodes of graph %s to %s", idx, graph.GetName().c_str(),
         file_path.c_str());
  return ge::SUCCESS;
}
