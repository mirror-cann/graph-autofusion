/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "test_fa_ascir_graph.h"
#include "base/base_types.h"

namespace att {
using namespace att;
using namespace af::ascir_op;
namespace {
template <typename T>
void InitGraphNode(T &node, int32_t &exec_order, std::initializer_list<int64_t> axis, af::DataType dtype,
                   std::initializer_list<Expr> repeats, std::initializer_list<Expr> strides) {
  node.attr.sched.exec_order = exec_order++;
  node.attr.sched.axis = axis;
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

struct QueBufInfo {
  const char *name;
  int32_t alloc_type;
  int32_t hardware;
  int32_t position;
  int32_t res_id;
  int32_t que_depth;
  bool reuse_as_tensor;
  bool zero_reuse;
};

void SetNodeMemAttr(af::AscNodePtr node, int32_t &tensor_id, const QueBufInfo &info) {
  auto &out = node->outputs[0];
  if (info.alloc_type == 0) {
    out.attr.mem.hardware = af::MemHardware::kMemHardwareGM;
    out.attr.mem.position = af::Position::kPositionGM;
    return;
  }
  out.attr.mem.tensor_id = tensor_id++;
  af::MemHardware hw = (info.hardware == 0) ? af::MemHardware::kMemHardwareGM : af::MemHardware::kMemHardwareUB;
  af::Position pos;
  if (info.position == 0) {
    pos = af::Position::kPositionGM;
  } else if (info.position == 1) {
    pos = af::Position::kPositionVecIn;
  } else {
    pos = af::Position::kPositionVecCalc;
  }
  out.attr.mem.hardware = hw;
  out.attr.mem.position = pos;
  out.attr.mem.reuse_id = info.reuse_as_tensor ? out.attr.mem.tensor_id : (info.zero_reuse ? 0 : af::kIdNone);
  out.attr.opt.ref_tensor = info.zero_reuse ? 0 : af::kIdNone;
  if (info.alloc_type == 1) {
    out.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    out.attr.buf.id = af::kIdNone;
    out.attr.que.id = info.res_id;
    out.attr.que.depth = info.que_depth;
    out.attr.que.buf_num = info.que_depth;
  } else if (info.alloc_type == 2) {
    out.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
    out.attr.buf.id = info.res_id;
    out.attr.que.id = af::kIdNone;
    out.attr.que.depth = af::kIdNone;
    out.attr.que.buf_num = af::kIdNone;
  } else {
    out.attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    out.attr.buf.id = af::kIdNone;
    out.attr.que.id = info.res_id;
    out.attr.que.depth = info.que_depth;
    out.attr.que.buf_num = info.que_depth;
  }
}

void ApplyBrcSched(af::AscGraph &graph, int64_t z1T, int64_t z1t, int64_t z2T, int64_t z2t, int64_t z0z2T,
                   int64_t z0z2TB, int64_t z0z2Tb, const char *name, const std::vector<int64_t> &vec_axis) {
  auto node = graph.FindNode(name);
  graph.ApplySplit(node, z1T, z1t);
  graph.ApplySplit(node, z2T, z2t);
  graph.ApplyMerge(node, z0z2T);
  graph.ApplySplit(node, z0z2TB, z0z2Tb);
  graph.ApplyReorder(node, {z0z2TB, z0z2Tb, z1T, z1t, z2t});
  node->attr.sched.loop_axis = z1T;
  node->outputs[0].attr.vectorized_axis = vec_axis;
}
}  // namespace

void BrcBufBeforeAutoFuse1(af::AscGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;
  auto Z0 = af::Symbol("Z0");
  auto Z1 = af::Symbol("Z1");
  auto Z2 = af::Symbol("Z2");
  auto z0 = graph.CreateAxis("z0", Z0);
  auto z1 = graph.CreateAxis("z1", Z1);
  auto z2 = graph.CreateAxis("z2", Z2);

  auto axis = {z0.id, z1.id, z2.id};
  auto normalR = std::initializer_list<Expr>{Z0, Z1, Z2};
  auto normalS = std::initializer_list<Expr>{Z1 * Z2, Z2, ONE};
  auto beforeR = std::initializer_list<Expr>{Z0, ONE, Z2};
  auto beforeS = std::initializer_list<Expr>{Z2, ZERO, ONE};
  int32_t exec_order = 0;

  Data input_data("input_data", graph);
  InitGraphNode(input_data, exec_order, axis, af::DT_FLOAT16, beforeR, beforeS);
  Load load("load");
  load.x = input_data.y;
  InitGraphNode(load, exec_order, axis, af::DT_FLOAT16, beforeR, beforeS);
  Cast cast0("cast0");
  cast0.x = load.y;
  InitGraphNode(cast0, exec_order, axis, af::DT_FLOAT, beforeR, beforeS);
  Broadcast broadcast("broadcast");
  broadcast.x = cast0.y;
  InitGraphNode(broadcast, exec_order, axis, af::DT_FLOAT, normalR, normalS);
  Sum sum("sum");
  sum.x = broadcast.y;
  InitGraphNode(sum, exec_order, axis, af::DT_FLOAT, beforeR, beforeS);
  Cast cast1("cast1");
  cast1.x = sum.y;
  InitGraphNode(cast1, exec_order, axis, af::DT_FLOAT16, beforeR, beforeS);
  Store store("store");
  store.x = cast1.y;
  InitGraphNode(store, exec_order, axis, af::DT_FLOAT16, beforeR, beforeS);
  Output output_data("output_data");
  output_data.x = store.y;
  output_data.attr.sched.exec_order = exec_order++;
}

void BrcBufAfterScheduler1(af::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;
  auto z1 = graph.GetAllAxis()[1]->id;
  auto z2 = graph.GetAllAxis()[2]->id;

  auto split = graph.TileSplit(z1);
  auto z1T = *(std::get<0>(split));
  auto z1t = *(std::get<1>(split));
  split = graph.TileSplit(z2);
  auto z2T = *(std::get<0>(split));
  auto z2t = *(std::get<1>(split));
  auto z0z2T = *graph.MergeAxis({z0, z2T.id});
  split = graph.BlockSplit(z0z2T.id);
  auto z0z2TB = *(std::get<0>(split));
  auto z0z2Tb = *(std::get<1>(split));

  std::vector<int64_t> vec_axis{z1t.id, z2t.id};
  const char *nodes[] = {"load", "cast0", "broadcast", "sum", "cast1", "store"};
  for (auto name : nodes) {
    ApplyBrcSched(graph, z1T.id, z1t.id, z2T.id, z2t.id, z0z2T.id, z0z2TB.id, z0z2Tb.id, name, vec_axis);
  }
}

void BrcBufAfterQueBufAlloc1(af::AscGraph &graph) {
  int32_t tensorID = 0;
  int32_t queID = 0;
  int32_t bufID = 0;
  int32_t loadQue = queID++;
  int32_t cast0Buf = bufID++;
  int32_t broadcastBuf = bufID++;
  int32_t sumBuf = bufID++;
  int32_t cast1Buf = bufID++;

  QueBufInfo nodes[] = {
      {"input_data", 0, 0, 0, -1, -1, false, false},  {"load", 1, 1, 1, loadQue, 2, false, false},
      {"cast0", 2, 1, 2, cast0Buf, -1, false, false}, {"broadcast", 2, 1, 2, broadcastBuf, -1, false, false},
      {"sum", 2, 1, 2, sumBuf, -1, false, false},     {"cast1", 2, 1, 2, cast1Buf, -1, false, false},
      {"store", 3, 0, 0, -1, -1, false, true},
  };
  for (const auto &info : nodes) {
    SetNodeMemAttr(graph.FindNode(info.name), tensorID, info);
  }
}

void BrcBufBeforeAutoFuse2(af::AscGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;
  auto Z0 = af::Symbol("Z0");
  auto Z1 = af::Symbol("Z1");
  auto Z2 = af::Symbol("Z2");
  auto z0 = graph.CreateAxis("z0", Z0);
  auto z1 = graph.CreateAxis("z1", Z1);
  auto z2 = graph.CreateAxis("z2", Z2);

  auto axis = {z0.id, z1.id, z2.id};
  auto normalR = std::initializer_list<Expr>{Z0, Z1, Z2};
  auto normalS = std::initializer_list<Expr>{Z1 * Z2, Z2, ONE};
  auto beforeR = std::initializer_list<Expr>{Z0, ONE, Z2};
  auto beforeS = std::initializer_list<Expr>{Z2, ZERO, ONE};
  auto reduceR = std::initializer_list<Expr>{Z0, Z1, ONE};
  auto reduceS = std::initializer_list<Expr>{Z1, ONE, ZERO};
  int32_t exec_order = 0;

  Data input_data("input_data", graph);
  InitGraphNode(input_data, exec_order, axis, af::DT_FLOAT16, beforeR, beforeS);
  Load load("load");
  load.x = input_data.y;
  InitGraphNode(load, exec_order, axis, af::DT_FLOAT16, beforeR, beforeS);
  Cast cast0("cast0");
  cast0.x = load.y;
  InitGraphNode(cast0, exec_order, axis, af::DT_FLOAT, beforeR, beforeS);
  Broadcast broadcast("broadcast");
  broadcast.x = cast0.y;
  InitGraphNode(broadcast, exec_order, axis, af::DT_FLOAT, normalR, normalS);
  Sum sum("sum");
  sum.x = broadcast.y;
  InitGraphNode(sum, exec_order, axis, af::DT_FLOAT, reduceR, reduceS);
  Cast cast1("cast1");
  cast1.x = sum.y;
  InitGraphNode(cast1, exec_order, axis, af::DT_FLOAT16, reduceR, reduceS);
  Store store("store");
  store.x = cast1.y;
  InitGraphNode(store, exec_order, axis, af::DT_FLOAT16, reduceR, reduceS);
  Output output_data("output_data");
  output_data.x = store.y;
  output_data.attr.sched.exec_order = exec_order++;
}

void BrcBufBeforeAutoFuse3(af::AscGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;
  auto Z0 = af::Symbol("Z0");
  auto Z1 = af::Symbol("Z1");
  auto Z2 = af::Symbol("Z2");
  auto z0 = graph.CreateAxis("z0", Z0);
  auto z1 = graph.CreateAxis("z1", Z1);
  auto z2 = graph.CreateAxis("z2", Z2);

  auto axis = {z0.id, z1.id, z2.id};
  auto normalR = std::initializer_list<Expr>{Z0, Z1, Z2};
  auto normalS = std::initializer_list<Expr>{Z1 * Z2, Z2, ONE};
  auto beforeR = std::initializer_list<Expr>{Z0, ONE, Z2};
  auto beforeS = std::initializer_list<Expr>{Z2, ZERO, ONE};
  int32_t exec_order = 0;

  Data input_data("input_data", graph);
  InitGraphNode(input_data, exec_order, axis, af::DT_FLOAT16, beforeR, beforeS);
  Load load("load");
  load.x = input_data.y;
  InitGraphNode(load, exec_order, axis, af::DT_FLOAT16, beforeR, beforeS);
  Cast cast0("cast0");
  cast0.x = load.y;
  InitGraphNode(cast0, exec_order, axis, af::DT_FLOAT, beforeR, beforeS);
  Broadcast broadcast("broadcast");
  broadcast.x = cast0.y;
  InitGraphNode(broadcast, exec_order, axis, af::DT_FLOAT, normalR, normalS);
  Cast cast1("cast1");
  cast1.x = broadcast.y;
  InitGraphNode(cast1, exec_order, axis, af::DT_FLOAT16, normalR, normalS);
  Store store("store");
  store.x = cast1.y;
  InitGraphNode(store, exec_order, axis, af::DT_FLOAT16, normalR, normalS);
  Output output_data("output_data");
  output_data.x = store.y;
  output_data.attr.sched.exec_order = exec_order++;
}

void BrcBufAfterScheduler3(af::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;
  auto z1 = graph.GetAllAxis()[1]->id;
  auto z2 = graph.GetAllAxis()[2]->id;

  auto split = graph.TileSplit(z1);
  auto z1T = *(std::get<0>(split));
  auto z1t = *(std::get<1>(split));
  split = graph.TileSplit(z2);
  auto z2T = *(std::get<0>(split));
  auto z2t = *(std::get<1>(split));
  auto z0z2T = *graph.MergeAxis({z0, z2T.id});
  split = graph.BlockSplit(z0z2T.id);
  auto z0z2TB = *(std::get<0>(split));
  auto z0z2Tb = *(std::get<1>(split));

  std::vector<int64_t> vec_axis{z1t.id, z2t.id};
  const char *nodes[] = {"load", "cast0", "broadcast", "cast1", "store"};
  for (auto name : nodes) {
    ApplyBrcSched(graph, z1T.id, z1t.id, z2T.id, z2t.id, z0z2T.id, z0z2TB.id, z0z2Tb.id, name, vec_axis);
  }
}

void BrcBufAfterQueBufAlloc3(af::AscGraph &graph) {
  int32_t tensorID = 0;
  int32_t queID = 0;
  int32_t bufID = 0;
  int32_t loadQue = queID++;
  int32_t cast0Buf = bufID++;
  int32_t broadcastBuf = bufID++;
  int32_t sumBuf = bufID++;
  int32_t cast1Buf = bufID++;

  QueBufInfo nodes[] = {
      {"input_data", 0, 0, 0, -1, -1, false, false},  {"load", 1, 1, 1, loadQue, 2, false, false},
      {"cast0", 2, 1, 2, cast0Buf, -1, false, false}, {"broadcast", 2, 1, 2, broadcastBuf, -1, false, false},
      {"cast1", 2, 1, 2, cast1Buf, -1, false, false}, {"store", 3, 0, 0, -1, -1, false, true},
  };
  for (const auto &info : nodes) {
    SetNodeMemAttr(graph.FindNode(info.name), tensorID, info);
  }
}

void BrcBufBeforeAutoFuse4(af::AscGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;
  auto Z0 = af::Symbol("Z0");
  auto Z1 = af::Symbol("Z1");
  auto Z2 = af::Symbol("Z2");
  auto z0 = graph.CreateAxis("z0", Z0);
  auto z1 = graph.CreateAxis("z1", Z1);
  auto z2 = graph.CreateAxis("z2", Z2);

  auto axis = {z0.id, z1.id, z2.id};
  auto normalR = std::initializer_list<Expr>{Z0, Z1, Z2};
  auto normalS = std::initializer_list<Expr>{Z1 * Z2, Z2, ONE};
  auto beforeR = std::initializer_list<Expr>{Z0, ONE, Z2};
  auto beforeS = std::initializer_list<Expr>{Z2, ZERO, ONE};
  int32_t exec_order = 0;

  Data input_data("input_data", graph);
  InitGraphNode(input_data, exec_order, axis, af::DT_FLOAT16, beforeR, beforeS);
  Load load("load");
  load.x = input_data.y;
  InitGraphNode(load, exec_order, axis, af::DT_FLOAT16, beforeR, beforeS);
  Cast cast0("cast0");
  cast0.x = load.y;
  InitGraphNode(cast0, exec_order, axis, af::DT_FLOAT, beforeR, beforeS);
  Sum sum("sum");
  sum.x = cast0.y;
  InitGraphNode(sum, exec_order, axis, af::DT_FLOAT, beforeR, beforeS);
  Cast cast1("cast1");
  cast1.x = sum.y;
  InitGraphNode(cast1, exec_order, axis, af::DT_FLOAT16, beforeR, beforeS);
  Store store("store");
  store.x = cast1.y;
  InitGraphNode(store, exec_order, axis, af::DT_FLOAT16, beforeR, beforeS);
  Output output_data("output_data");
  output_data.x = store.y;
  output_data.attr.sched.exec_order = exec_order++;
}

void BrcBufAfterScheduler4(af::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;
  auto z1 = graph.GetAllAxis()[1]->id;
  auto z2 = graph.GetAllAxis()[2]->id;

  auto split = graph.TileSplit(z1);
  auto z1T = *(std::get<0>(split));
  auto z1t = *(std::get<1>(split));
  split = graph.TileSplit(z2);
  auto z2T = *(std::get<0>(split));
  auto z2t = *(std::get<1>(split));
  auto z0z2T = *graph.MergeAxis({z0, z2T.id});
  split = graph.BlockSplit(z0z2T.id);
  auto z0z2TB = *(std::get<0>(split));
  auto z0z2Tb = *(std::get<1>(split));

  std::vector<int64_t> vec_axis{z1t.id, z2t.id};
  const char *nodes[] = {"load", "cast0", "sum", "cast1", "store"};
  for (auto name : nodes) {
    ApplyBrcSched(graph, z1T.id, z1t.id, z2T.id, z2t.id, z0z2T.id, z0z2TB.id, z0z2Tb.id, name, vec_axis);
  }
}

void BrcBufAfterQueBufAlloc4(af::AscGraph &graph) {
  int32_t tensorID = 0;
  int32_t queID = 0;
  int32_t bufID = 0;
  int32_t loadQue = queID++;
  int32_t cast0Buf = bufID++;
  int32_t broadcastBuf = bufID++;
  int32_t sumBuf = bufID++;
  int32_t cast1Buf = bufID++;

  QueBufInfo nodes[] = {
      {"input_data", 0, 0, 0, -1, -1, false, false},  {"load", 1, 1, 1, loadQue, 2, false, false},
      {"cast0", 2, 1, 2, cast0Buf, -1, false, false}, {"sum", 2, 1, 2, sumBuf, -1, false, false},
      {"cast1", 2, 1, 2, cast1Buf, -1, false, false}, {"store", 3, 0, 0, -1, -1, false, true},
  };
  for (const auto &info : nodes) {
    SetNodeMemAttr(graph.FindNode(info.name), tensorID, info);
  }
}
}  // namespace att
