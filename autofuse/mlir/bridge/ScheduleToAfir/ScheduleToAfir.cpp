/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ScheduleToAfir.cpp - FusedScheduledResult → AFIR (in-process) *- C++ -*-===//

#include "ScheduleToAfir/ScheduleToAfir.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>
#include <unordered_map>

#include "AfirBuilder/AfirBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"

namespace af_demo {
namespace {

// dtype → AfirBuilder dtype string. AfirBuilder::parseDtype consumes these exact spellings.
std::string DTypeStr(ge::DataType dtype) {
  switch (dtype) {
    case ge::DT_FLOAT:
      return "float32";
    case ge::DT_FLOAT16:
      return "float16";
    case ge::DT_BF16:
      return "bfloat16";
    case ge::DT_INT8:
      return "int8";
    case ge::DT_INT16:
      return "int16";
    case ge::DT_INT32:
      return "int32";
    case ge::DT_INT64:
      return "int64";
    case ge::DT_UINT8:
      return "uint8";
    case ge::DT_UINT16:
      return "uint16";
    case ge::DT_UINT32:
      return "uint32";
    case ge::DT_UINT64:
      return "uint64";
    default:
      return "float16";
  }
}

// axis type → AfirBuilder axis-type string.
std::string AxisTypeStr(af::Axis::Type type) {
  switch (type) {
    case af::Axis::Type::kAxisTypeOriginal:
      return "original";
    case af::Axis::Type::kAxisTypeBlockOuter:
      return "block_outer";
    case af::Axis::Type::kAxisTypeBlockInner:
      return "block_inner";
    case af::Axis::Type::kAxisTypeTileOuter:
      return "tile_outer";
    case af::Axis::Type::kAxisTypeTileInner:
      return "tile_inner";
    case af::Axis::Type::kAxisTypeMerged:
      return "merged";
    default:
      return "original";
  }
}

// Expression (symbolic shape dim) → static int64. Non-constant dims become
// ShapedType::kDynamic (-9223372036854775808 sentinel used by MLIR); the demo
// keeps them dynamic rather than doing full symengine evaluation.
int64_t DimToStatic(const ge::Expression &e) {
  int64_t v = 0;
  if (e.IsValid() && e.IsConstExpr() && e.GetConstValue<int64_t>(v)) {
    return v;
  }
  return INT64_MIN;  // mlir::ShapedType::kDynamic
}

// Build a tensor's shape directly from the scheduler's `repeats` — this faithfully
// carries the POST-tiling shape the scheduler produced (our ScheduleToAfir runs
// AFTER the scheduler). A load/store reshapes GM (logical rank) ↔ merged/tiled
// local, so operand and result ranks legitimately differ; the AFIR load/store
// verifier permits that. Tiling decisions ride on the func axes + per-op sched_axis
// for the M2 loop-building lowering.
TensorInfo ToTensorInfo(const af::AscTensorAttr &attr) {
  TensorInfo t;
  t.dtype = DTypeStr(attr.dtype);
  t.tensor_id = attr.mem.tensor_id;
  t.position = static_cast<int64_t>(attr.mem.position);  // Position enum order == AfirBuilder convention
  t.axis_ids = attr.axis;
  t.vectorized_axis = attr.vectorized_axis;  // API vector-length axes (for M2 loop-builder)
  for (const auto &rep : attr.repeats) t.shape.push_back(DimToStatic(rep));
  return t;
}

std::string ExprStr(const ge::Expression &e) {
  if (!e.IsValid()) return "<invalid>";
  auto v = e.Serialize();
  return v.get() == nullptr ? "<null>" : v.get();
}

std::string ExprListStr(const std::vector<ge::Expression> &es) {
  std::string s = "[";
  for (size_t i = 0; i < es.size(); ++i) {
    if (i) s += ", ";
    s += ExprStr(es[i]);
  }
  return s + "]";
}

std::string IntListStr(const std::vector<int64_t> &v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(v[i]);
  }
  return s + "]";
}

void DumpTensorLine(std::ostream &os, const char *tag, const af::AscTensorAttr &attr) {
  os << "    " << tag << " tensor_id=" << attr.mem.tensor_id << " position=" << static_cast<int>(attr.mem.position)
     << " axis=" << IntListStr(attr.axis) << " repeats=" << ExprListStr(attr.repeats)
     << " vectorized_axis=" << IntListStr(attr.vectorized_axis) << "\n";
}

template <typename GraphT>
void DumpGraph(const GraphT &graph, std::ostream &os) {
  os << "-- impl_graph: " << graph.GetName() << " tiling_key=" << graph.GetTilingKey() << "\n";
  os << "  AXES:\n";
  for (const auto &axis : graph.GetAllAxis()) {
    os << "    id=" << axis->id << " name=" << axis->name << " type=" << AxisTypeStr(axis->type)
       << " size=" << ExprStr(axis->size) << " from=" << IntListStr(axis->from)
       << " split_pair_other_id=" << axis->split_pair_other_id << " bind_block=" << axis->bind_block << "\n";
  }
  os << "  NODES:\n";
  for (const auto &node : graph.GetAllNodes()) {
    os << "  node " << node->GetName() << " [" << node->GetType() << "]"
       << " sched.axis=" << IntListStr(node->attr.sched.axis) << " loop_axis=" << node->attr.sched.loop_axis << "\n";
    for (const auto &input : node->inputs()) DumpTensorLine(os, "in ", input->attr);
    for (const auto &output : node->outputs()) DumpTensorLine(os, "out", output->attr);
  }
}

// M0 diagnostic: dump the RAW scheduler structure (axes with type/from/split,
// per-node repeats/vectorized_axis/position/sched.axis). Runs BEFORE the AFIR
// build (which aborts on tiled rank mismatch), gated by AF_MLIR_AFIR_DEBUG_DIR.
void DumpRawScheduleStructure(const ascir::FusedScheduledResult &result, std::ostream &os) {
  os << "=== RAW SCHEDULE STRUCTURE: " << result.fused_graph_name.GetString() << " ===\n";
  os << "input_nodes=" << result.input_nodes.size() << " output_nodes=" << result.output_nodes.size() << "\n";
  for (const auto &scheduled_results : result.node_idx_to_scheduled_results) {
    for (const auto &sr : scheduled_results) {
      for (const auto &group : sr.schedule_groups) {
        for (const auto &graph : group.impl_graphs) {
          DumpGraph(graph, os);
        }
      }
    }
  }
  os << "=== END ===\n";
}

IoTensor ToIoTensor(const af::AscNodePtr &node) {
  IoTensor io;
  io.name = node->GetName();
  auto outputs = node->outputs();
  if (!outputs.empty()) io.info = ToTensorInfo(outputs[0]->attr);
  return io;
}

template <typename GraphT>
GraphInfo ToGraphInfo(const GraphT &graph) {
  GraphInfo info;
  info.name = graph.GetName();
  info.tiling_key = graph.GetTilingKey();
  for (const auto &axis : graph.GetAllAxis()) {
    int64_t size = DimToStatic(axis->size);
    info.axes.push_back(
        {axis->id, axis->name, AxisTypeStr(axis->type), size == INT64_MIN ? axis->name : std::to_string(size)});
  }
  for (const auto &sizeVar : graph.GetAllSizeVar()) info.size_vars.push_back(sizeVar->name);

  std::unordered_map<int64_t, std::string> producerByTensor;
  for (const auto &node : graph.GetAllNodes()) {
    NodeInfo nodeInfo;
    nodeInfo.name = node->GetName();
    nodeInfo.type = node->GetType();
    nodeInfo.sched_axis = node->attr.sched.axis;
    nodeInfo.loop_axis = node->attr.sched.loop_axis;
    for (const auto &input : node->inputs()) {
      auto producer = producerByTensor.find(input->attr.mem.tensor_id);
      nodeInfo.input_names.push_back(producer != producerByTensor.end() ? producer->second : "");
    }
    auto outputs = node->outputs();
    if (!outputs.empty()) {
      nodeInfo.output = ToTensorInfo(outputs[0]->attr);
      producerByTensor[outputs[0]->attr.mem.tensor_id] = nodeInfo.name;
    }
    info.nodes.push_back(std::move(nodeInfo));
  }
  return info;
}

}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> ScheduleResultToAfir(mlir::MLIRContext &context,
                                                       const ascir::FusedScheduledResult &result) {
  KernelInfo kernel;
  kernel.name = result.fused_graph_name.GetString();
  for (const auto &node : result.input_nodes) kernel.inputs.push_back(ToIoTensor(node));
  for (const auto &node : result.output_nodes) kernel.outputs.push_back(ToIoTensor(node));

  // Walk schedule → results → groups → impl_graphs; each impl_graph → one func.
  for (const auto &scheduled_results : result.node_idx_to_scheduled_results) {
    for (const auto &sr : scheduled_results) {
      for (const auto &group : sr.schedule_groups) {
        for (const auto &graph : group.impl_graphs) {
          kernel.graphs.push_back(ToGraphInfo(graph));
        }
      }
    }
  }

  return BuildAfirModule(context, kernel);
}

void MaybeDumpAfirFromSchedule(const std::string &stage, const ascir::FusedScheduledResult &result) {
  // M0 diagnostic: dump raw scheduler structure first (survives an AFIR-build
  // abort on tiled rank mismatch). Gated separately by AF_MLIR_AFIR_DEBUG_DIR.
  if (const char *dbg = std::getenv("AF_MLIR_AFIR_DEBUG_DIR"); dbg && dbg[0]) {
    try {
      static std::atomic<uint64_t> dbgIdx{0};
      (void)mkdir(dbg, 0755);
      std::string path = std::string(dbg) + "/rawsched_" + std::to_string(dbgIdx.fetch_add(1)) + "_" + stage + "_" +
                         result.fused_graph_name.GetString() + ".txt";
      std::ofstream out(path);
      if (out.is_open()) DumpRawScheduleStructure(result, out);
    } catch (...) {
    }
  }

  const char *dir = std::getenv("AF_MLIR_AFIR_DUMP_DIR");
  if (dir == nullptr || dir[0] == '\0') {
    return;
  }
  try {
    static std::atomic<uint64_t> idx{0};
    mlir::MLIRContext context;
    auto module = ScheduleResultToAfir(context, result);
    if (!module) {
      return;
    }
    (void)mkdir(dir, 0755);
    std::string path = std::string(dir) + "/afir_" + std::to_string(idx.fetch_add(1)) + "_" + stage + "_" +
                       result.fused_graph_name.GetString() + ".mlir";
    std::ofstream out(path);
    if (!out.is_open()) {
      return;
    }
    std::string text;
    llvm::raw_string_ostream os(text);
    module->print(os);
    os.flush();
    out << text << "\n";
  } catch (...) {
    // Never propagate into the codegen path.
  }
}

}  // namespace af_demo
