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
void InitDataNode(Data &node, int32_t &exec_order, std::initializer_list<int64_t> axis, af::DataType dtype,
                  std::initializer_list<Expr> repeats, std::initializer_list<Expr> strides) {
  node.attr.sched.exec_order = exec_order++;
  node.attr.sched.axis = axis;
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

void InitLoadNode(Load &node, int32_t &exec_order, std::initializer_list<int64_t> axis, af::DataType dtype,
                  std::initializer_list<Expr> repeats, std::initializer_list<Expr> strides) {
  node.attr.sched.exec_order = exec_order++;
  node.attr.sched.axis = axis;
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

void InitStoreNode(Store &node, int32_t &exec_order, std::initializer_list<int64_t> axis, af::DataType dtype,
                   std::initializer_list<Expr> repeats, std::initializer_list<Expr> strides) {
  node.attr.sched.exec_order = exec_order++;
  node.attr.sched.axis = axis;
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

void InitOutputNode(Output &node, int32_t &exec_order, std::initializer_list<int64_t> axis, af::DataType dtype,
                    std::initializer_list<Expr> repeats, std::initializer_list<Expr> strides) {
  node.attr.sched.exec_order = exec_order++;
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

template <typename T>
void InitUnaryNode(T &node, int32_t &exec_order, std::initializer_list<int64_t> axis, af::DataType dtype,
                   std::initializer_list<Expr> repeats, std::initializer_list<Expr> strides) {
  node.attr.sched.exec_order = exec_order++;
  node.attr.sched.axis = axis;
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

struct SchedAxis {
  int64_t s1T, s1t, mc, mcB, mcb, s2T, s2t, d, bl;
};

struct SchedConfig {
  std::initializer_list<int64_t> reorder;
  int64_t loop_axis;
  std::vector<int64_t> vec_axis;
};

void ApplyBmmSched(af::AscGraph &g, const SchedAxis &a, const char *name, const SchedConfig &cfg) {
  auto node = g.FindNode(name);
  g.ApplySplit(node, a.s1T, a.s1t);
  g.ApplyMerge(node, a.mc);
  g.ApplySplit(node, a.mcB, a.mcb);
  g.ApplySplit(node, a.s2T, a.s2t);
  g.ApplyReorder(node, cfg.reorder);
  node->attr.sched.loop_axis = cfg.loop_axis;
  node->outputs[0].attr.vectorized_axis = cfg.vec_axis;
}

void ApplyVecSched(af::AscGraph &g, const SchedAxis &a, int64_t s1tT, int64_t s1tt,
                   const char *name, const SchedConfig &cfg) {
  auto node = g.FindNode(name);
  g.ApplySplit(node, a.s1T, a.s1t);
  g.ApplyMerge(node, a.mc);
  g.ApplySplit(node, a.mcB, a.mcb);
  g.ApplySplit(node, a.s2T, a.s2t);
  g.ApplySplit(node, s1tT, s1tt);
  g.ApplyReorder(node, cfg.reorder);
  node->attr.sched.loop_axis = cfg.loop_axis;
  node->outputs[0].attr.vectorized_axis = cfg.vec_axis;
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
    pos = af::Position::kPositionVecOut;
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

void SetApiAttr(af::AscNodePtr node, af::ApiType apiType, af::ComputeUnit unit) {
  node->attr.api.type = apiType;
  node->attr.api.unit = unit;
}

void InitQueBufIds(int32_t &queID, int32_t &bufID, int32_t ids[8]) {
  ids[0] = queID++;
  ids[1] = queID++;
  ids[2] = bufID++;
  ids[3] = bufID++;
  ids[4] = bufID++;
  ids[5] = bufID++;
  ids[6] = queID++;
  ids[7] = queID++;
}

struct ApiInfo {
  const char *name;
  af::ApiType type;
  af::ComputeUnit unit;
};

void ApplyBmmNodesSched(af::AscGraph &graph, const SchedAxis &a) {
  auto bmmReorder = {a.mcB, a.mcb, a.s2T, a.s1t, a.s2t, a.d, a.bl};
  ApplyBmmSched(graph, a, "query", {{a.mcB, a.mcb, a.s2T, a.s1t, a.s2t, a.d, a.bl}, a.s2T, {a.s1t, a.s2t, a.d}});
  ApplyBmmSched(graph, a, "key", {{a.mcB, a.mcb, a.s2T, a.s1t, a.d, a.s2t, a.bl}, a.s2T, {a.s1t, a.d, a.s2t}});
  SchedConfig bmmCfg{bmmReorder, a.s2T, {a.s1t, a.s2t, a.d}};
  ApplyBmmSched(graph, a, "bmm1", bmmCfg);
  ApplyBmmSched(graph, a, "value", bmmCfg);
  ApplyBmmSched(graph, a, "bmm2", bmmCfg);
}

void ApplyVec1NodesSched(af::AscGraph &graph, const SchedAxis &a, int64_t s1tT, int64_t s1tt) {
  auto vec1Reorder = {a.mcB, a.mcb, a.s2T, s1tT, s1tt, a.s2t, a.d, a.bl};
  vector<int64_t> vec1{s1tt, a.s2t, a.d};
  SchedConfig vec1Cfg{vec1Reorder, s1tT, vec1};
  const char *vec1Nodes[] = {"load1", "loadPse", "castPse", "add1", "mul1", "loadAttenMask",
                             "select", "loadDropMask", "dropout", "castVec1Res",
                             "softmaxApiTmpBuf", "flashSoftmax"};
  for (auto name : vec1Nodes) {
    ApplyVecSched(graph, a, s1tT, s1tt, name, vec1Cfg);
  }
  ApplyVecSched(graph, a, s1tT, s1tt, "storeVec1Res", {vec1Reorder, s1tT, {s1tT, s1tt, a.s2t, a.d}});
  SchedConfig expCfg{vec1Reorder, s1tT, {s1tT, s1tt, a.s2t, a.d, a.bl}};
  ApplyVecSched(graph, a, s1tT, s1tt, "softmaxExp", expCfg);
  ApplyVecSched(graph, a, s1tT, s1tt, "storeSoftmaxMax", expCfg);
}

void ApplyVec2NodesSched(af::AscGraph &graph, const SchedAxis &a) {
  auto split = graph.TileSplit(a.s1t, "s1tT2", "s1tt2");
  auto s1Vec2tT = *(std::get<0>(split));
  auto s1Vec2tt = *(std::get<1>(split));
  auto vec2Reorder = {a.mcB, a.mcb, a.s2T, s1Vec2tT.id, s1Vec2tt.id, a.s2t, a.d, a.bl};
  SchedConfig vec2Cfg{vec2Reorder, s1Vec2tT.id, {s1Vec2tt.id, a.d, a.s2t}};
  const char *vec2Nodes[] = {"load2", "addResOut", "loadAddResOut", "mulRes", "addRes", "div",
                             "castBmm2Res", "store"};
  for (auto name : vec2Nodes) {
    ApplyVecSched(graph, a, s1Vec2tT.id, s1Vec2tt.id, name, vec2Cfg);
  }
}
}  // namespace

void FaBeforeAutoFuseBmm1(af::AscGraph &graph, int32_t &exec_order,
                           const std::initializer_list<int64_t> &bmmAxis,
                           const std::initializer_list<Expr> &vec1R,
                           const std::initializer_list<Expr> &vec1S,
                           const Expr &B, const Expr &N, const Expr &G,
                           const Expr &S1, const Expr &S2, const Expr &D,
                           const Expr &ONE, const Expr &ZERO,
                           Add &out_mul1) {
  Data query("query", graph);
  InitDataNode(query, exec_order, bmmAxis, af::DT_FLOAT16, {B, N, G, S1, ONE, D, ONE},
               {N * G * S1 * D, G * S1 * D, S1 * D, D, ZERO, ONE, ZERO});
  Data key("key", graph);
  InitDataNode(key, exec_order, bmmAxis, af::DT_FLOAT16, {B, N, G, ONE, S2, D, ONE},
               {N * S1 * D, S2 * D, S2 * D, ZERO, D, ONE, ZERO});
  Add bmm1("bmm1");
  bmm1.x1 = query.y;
  bmm1.x2 = key.y;
  InitUnaryNode(bmm1, exec_order, bmmAxis, af::DT_FLOAT, vec1R, vec1S);
  Load load1("load1");
  load1.x = bmm1.y;
  InitLoadNode(load1, exec_order, bmmAxis, af::DT_FLOAT, vec1R, vec1S);
  Data pse("pse", graph);
  InitDataNode(pse, exec_order, bmmAxis, af::DT_FLOAT16, vec1R, vec1S);
  Load loadPse("loadPse");
  loadPse.x = pse.y;
  InitLoadNode(loadPse, exec_order, bmmAxis, af::DT_FLOAT16, vec1R, vec1S);
  Cast castPse("castPse");
  castPse.x = loadPse.y;
  InitUnaryNode(castPse, exec_order, bmmAxis, af::DT_FLOAT, vec1R, vec1S);
  Add add1("add1");
  add1.x1 = load1.y;
  add1.x2 = castPse.y;
  InitUnaryNode(add1, exec_order, bmmAxis, af::DT_FLOAT, vec1R, vec1S);
  Data scaleValue("scaleValue", graph);
  scaleValue.attr.sched.exec_order = exec_order++;
  scaleValue.y.dtype = af::DT_FLOAT;
  out_mul1.x1 = add1.y;
  out_mul1.x2 = scaleValue.y;
  InitUnaryNode(out_mul1, exec_order, bmmAxis, af::DT_FLOAT, vec1R, vec1S);
}

void FaBeforeAutoFuseVec1(af::AscGraph &graph, int32_t &exec_order,
                           const std::initializer_list<int64_t> &bmmAxis,
                           const std::initializer_list<Expr> &vec1R,
                           const std::initializer_list<Expr> &vec1S,
                           const std::initializer_list<Expr> &reduceR,
                           const std::initializer_list<Expr> &reduceS,
                           const Expr &B, const Expr &N, const Expr &G,
                           const Expr &S1, const Expr &S2, const Expr &BL,
                           const Expr &ONE, const Expr &ZERO,
                           Add &mul1, Data &out_softmaxExp, Concat &out_flashSoftmax,
                           Store &out_storeVec1Res) {
  auto maskR = std::initializer_list<Expr>{B, ONE, ONE, S1, S2, ONE, ONE};
  auto maskS = std::initializer_list<Expr>{S1 * S2, S1 * S2, S1 * S2, S2, ONE, ZERO, ZERO};
  Data attenMask("attenMask", graph);
  InitDataNode(attenMask, exec_order, bmmAxis, af::DT_UINT8, maskR, maskS);
  Load loadAttenMask("loadAttenMask");
  loadAttenMask.x = attenMask.y;
  InitLoadNode(loadAttenMask, exec_order, bmmAxis, af::DT_UINT8, maskR, maskS);
  Select select("select");
  select.x1 = mul1.y;
  select.x2 = loadAttenMask.y;
  InitUnaryNode(select, exec_order, bmmAxis, af::DT_FLOAT, vec1R, vec1S);
  InitDataNode(out_softmaxExp, exec_order, bmmAxis, af::DT_FLOAT, reduceR, reduceS);
  Data softmaxApiTmpBuf("softmaxApiTmpBuf", graph);
  InitDataNode(softmaxApiTmpBuf, exec_order, bmmAxis, af::DT_FLOAT,
               {ONE, ONE, ONE, S1, S2, ONE, ONE}, {ZERO, ZERO, ZERO, S2, ONE, ZERO, ZERO});
  out_flashSoftmax.x = {select.y, out_softmaxExp.y, softmaxApiTmpBuf.y};
  InitUnaryNode(out_flashSoftmax, exec_order, bmmAxis, af::DT_FLOAT, vec1R, vec1S);
  Store storeSoftmaxMax("storeSoftmaxMax");
  storeSoftmaxMax.x = out_flashSoftmax.y;
  InitStoreNode(storeSoftmaxMax, exec_order, bmmAxis, af::DT_FLOAT, {B, N, G, S1, S2, ONE, BL},
               {N * G * S1 * BL, G * S1 * BL, S1 * BL, BL, ZERO, ZERO, ONE});
  Output softmaxMax("softmaxMax");
  softmaxMax.x = storeSoftmaxMax.y;
  softmaxMax.attr.sched.exec_order = exec_order++;
  Data dropMask("dropMask", graph);
  InitDataNode(dropMask, exec_order, bmmAxis, af::DT_UINT8, vec1R, vec1S);
  Load loadDropMask("loadDropMask");
  loadDropMask.x = dropMask.y;
  InitLoadNode(loadDropMask, exec_order, bmmAxis, af::DT_UINT8, vec1R, vec1S);
  Add dropout("dropout");
  dropout.x1 = out_flashSoftmax.y;
  dropout.x2 = loadDropMask.y;
  InitUnaryNode(dropout, exec_order, bmmAxis, af::DT_FLOAT, vec1R, vec1S);
  Cast castVec1Res("castVec1Res");
  castVec1Res.x = dropout.y;
  InitUnaryNode(castVec1Res, exec_order, bmmAxis, af::DT_FLOAT16, vec1R, vec1S);
  out_storeVec1Res.x = castVec1Res.y;
  InitStoreNode(out_storeVec1Res, exec_order, bmmAxis, af::DT_FLOAT16, vec1R, vec1R);
}

void FaBeforeAutoFuseBmm2(af::AscGraph &graph, int32_t &exec_order,
                           const std::initializer_list<int64_t> &bmmAxis,
                           const std::initializer_list<Expr> &vec2R,
                           const std::initializer_list<Expr> &vec2S,
                           const Expr &B, const Expr &N, const Expr &G,
                           const Expr &S2, const Expr &D,
                           const Expr &ONE, const Expr &ZERO,
                           Store &storeVec1Res, Data &softmaxExp, Concat &flashSoftmax) {
  Data value("value", graph);
  InitDataNode(value, exec_order, bmmAxis, af::DT_FLOAT16, {B, N, G, ONE, S2, D, ONE},
               {N * S2 * D, S2 * D, S2 * D, ZERO, D, ONE, ZERO});
  Add bmm2("bmm2");
  bmm2.x1 = storeVec1Res.y;
  bmm2.x2 = value.y;
  InitUnaryNode(bmm2, exec_order, bmmAxis, af::DT_FLOAT, vec2R, vec2S);
  Load load2("load2");
  load2.x = bmm2.y;
  InitLoadNode(load2, exec_order, bmmAxis, af::DT_FLOAT, vec2R, vec2S);
  Workspace addResOut("addResOut");
  addResOut.attr.sched.exec_order = exec_order++;
  addResOut.attr.sched.axis = bmmAxis;
  addResOut.x = load2.y;
  addResOut.y.dtype = af::DT_FLOAT;
  *addResOut.y.axis = bmmAxis;
  *addResOut.y.repeats = vec2R;
  *addResOut.y.strides = vec2S;
  Load loadAddResOut("loadAddResOut");
  loadAddResOut.x = addResOut.y;
  InitLoadNode(loadAddResOut, exec_order, bmmAxis, af::DT_FLOAT, vec2R, vec2S);
  Mul mulRes("mulRes");
  mulRes.x1 = loadAddResOut.y;
  mulRes.x2 = softmaxExp.y;
  InitUnaryNode(mulRes, exec_order, bmmAxis, af::DT_FLOAT, vec2R, vec2S);
  Add addRes("addRes");
  addRes.x1 = load2.y;
  addRes.x2 = mulRes.y;
  InitUnaryNode(addRes, exec_order, bmmAxis, af::DT_FLOAT, vec2R, vec2S);
  Div div("div");
  div.x1 = addRes.y;
  div.x2 = flashSoftmax.y;
  InitUnaryNode(div, exec_order, bmmAxis, af::DT_FLOAT, vec2R, vec2S);
  Cast castBmm2Res("castBmm2Res");
  castBmm2Res.x = div.y;
  InitUnaryNode(castBmm2Res, exec_order, bmmAxis, af::DT_FLOAT16, vec2R, vec2S);
  Store store("store");
  store.x = castBmm2Res.y;
  InitStoreNode(store, exec_order, bmmAxis, af::DT_FLOAT16, vec2R, vec2S);
  Output buf("buf");
  buf.x = store.y;
  InitOutputNode(buf, exec_order, bmmAxis, af::DT_FLOAT16, vec2R, vec2S);
}

void FaBeforeAutoFuse(af::AscGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;
  auto B = af::Symbol("B");
  auto N = af::Symbol("N");
  auto G = af::Symbol("G");
  auto S1 = af::Symbol("S1");
  auto S2 = af::Symbol("S2");
  auto D = af::Symbol("D");
  auto BL = af::Symbol(8, "BL");
  auto b = graph.CreateAxis("b", B);
  auto n = graph.CreateAxis("n", N);
  auto g = graph.CreateAxis("g", G);
  auto s1 = graph.CreateAxis("s1", S1);
  auto s2 = graph.CreateAxis("s2", S2);
  auto d = graph.CreateAxis("d", D);
  auto bl = graph.CreateAxis("l", BL);
  auto bmmAxis = {b.id, n.id, g.id, s1.id, s2.id, d.id, bl.id};
  auto vec1R = std::initializer_list<Expr>{B, N, G, S1, S2, ONE, ONE};
  auto vec1S = std::initializer_list<Expr>{N * G * S1 * S2, G * S1 * S2, S1 * S2, S2, ONE, ZERO, ZERO};
  auto vec2R = std::initializer_list<Expr>{B, N, G, S1, ONE, D, ONE};
  auto vec2S = std::initializer_list<Expr>{N * G * S1 * D, G * S1 * D, S1 * D, D, ZERO, ONE, ZERO};
  auto reduceR = std::initializer_list<Expr>{ONE, ONE, ONE, S1, ONE, ONE, BL};
  auto reduceS = std::initializer_list<Expr>{ZERO, ZERO, ZERO, BL, ZERO, ZERO, ONE};
  int32_t exec_order = 0;
  Add mul1("mul1");
  FaBeforeAutoFuseBmm1(graph, exec_order, bmmAxis, vec1R, vec1S, B, N, G, S1, S2, D, ONE, ZERO, mul1);
  Data softmaxExp("softmaxExp", graph);
  Concat flashSoftmax("flashSoftmax");
  Store storeVec1Res("storeVec1Res");
  FaBeforeAutoFuseVec1(graph, exec_order, bmmAxis, vec1R, vec1S, reduceR, reduceS,
                       B, N, G, S1, S2, BL, ONE, ZERO, mul1, softmaxExp, flashSoftmax, storeVec1Res);
  FaBeforeAutoFuseBmm2(graph, exec_order, bmmAxis, vec2R, vec2S, B, N, G, S2, D, ONE, ZERO,
                       storeVec1Res, softmaxExp, flashSoftmax);
}

void FaAfterApiInfo(af::AscGraph &graph) {
  ApiInfo infos[] = {
    {"query", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"key", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"bmm1", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitCube},
    {"load1", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitNone},
    {"pse", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"loadPse", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"castPse", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitVector},
    {"add1", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitVector},
    {"scaleValue", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"mul1", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitVector},
    {"attenMask", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"loadAttenMask", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitNone},
    {"select", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitVector},
    {"softmaxExp", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"flashSoftmax", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitVector},
    {"softmaxSum", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"storeSoftmaxMax", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitNone},
    {"softmaxMax", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"dropMask", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"loadDropMask", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitNone},
    {"dropout", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"castVec1Res", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitVector},
    {"storeVec1Res", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitNone},
    {"value", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"bmm2", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitCube},
    {"load2", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitNone},
    {"addResOut", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
    {"loadAddResOut", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitNone},
    {"mulRes", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitVector},
    {"addRes", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitVector},
    {"div", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitVector},
    {"castBmm2Res", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitVector},
    {"store", af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitNone},
    {"buf", af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone},
  };
  for (const auto &info : infos) {
    auto node = graph.FindNode(info.name);
    node->attr.api.type = info.type;
    node->attr.api.unit = info.unit;
  }
}

void FaAfterScheduler(af::AscGraph &graph) {
  auto b = graph.GetAllAxis()[0]->id;
  auto n = graph.GetAllAxis()[1]->id;
  auto g = graph.GetAllAxis()[2]->id;
  auto s1 = graph.GetAllAxis()[3]->id;
  auto s2 = graph.GetAllAxis()[4]->id;
  auto d = graph.GetAllAxis()[5]->id;
  auto bl = graph.GetAllAxis()[6]->id;

  std::tuple<af::AxisPtr, af::AxisPtr> split = graph.TileSplit(s1);
  auto s1T = *(std::get<0>(split));
  auto s1t = *(std::get<1>(split));
  graph.FindAxis(s1t.id)->align = af::Symbol(128);
  auto mcAxis = *graph.MergeAxis({b, n, g, s1T.id});
  split = graph.BlockSplit(mcAxis.id);
  auto mcAxisB = *(std::get<0>(split));
  auto mcAxisb = *(std::get<1>(split));

  split = graph.TileSplit(s2);
  auto s2T = *(std::get<0>(split));
  auto s2t = *(std::get<1>(split));
  graph.FindAxis(s2t.id)->align = af::Symbol(256);

  split = graph.TileSplit(s1t.id);
  auto s1tT = *(std::get<0>(split));
  auto s1tt = *(std::get<1>(split));
  graph.FindAxis(s1tt.id)->align = af::Symbol(8);
  graph.FindAxis(s2t.id)->allow_unaligned_tail = false;
  graph.FindAxis(s1tt.id)->allow_unaligned_tail = false;

  SchedAxis a{s1T.id, s1t.id, mcAxis.id, mcAxisB.id, mcAxisb.id, s2T.id, s2t.id, d, bl};
  ApplyBmmNodesSched(graph, a);
  ApplyVec1NodesSched(graph, a, s1tT.id, s1tt.id);
  ApplyVec2NodesSched(graph, a);
}

void FaAfterQueBufAlloc(af::AscGraph &graph) {
  int32_t tensorID = 0, queID = 0, bufID = 0;
  int32_t ids[8];
  InitQueBufIds(queID, bufID, ids);
  int32_t stage2Buf = bufID++;
  int32_t mm2ResQueue = queID++;
  int32_t vec2ResQueue = queID++;

  QueBufInfo nodes[] = {
    {"query", 0, 0, 0, -1, -1, false, false},
    {"key", 0, 0, 0, -1, -1, false, false},
    {"value", 0, 0, 0, -1, -1, false, false},
    {"bmm1", 1, 0, 0, ids[0], 2, false, false},
    {"load1", 1, 1, 1, ids[1], 2, false, false},
    {"loadPse", 2, 1, 1, ids[2], -1, false, false},
    {"castPse", 2, 1, 2, ids[3], -1, false, false},
    {"add1", 1, 1, 2, ids[1], 2, false, false},
    {"mul1", 1, 1, 2, ids[1], 2, false, false},
    {"select", 1, 1, 2, ids[1], 2, false, false},
    {"softmaxExp", 1, 1, 1, ids[6], 2, false, false},
    {"softmaxApiTmpBuf", 2, 1, 1, ids[3], -1, false, false},
    {"flashSoftmax", 1, 1, 2, ids[1], 2, false, false},
    {"loadAttenMask", 2, 1, 1, ids[4], -1, false, false},
    {"loadDropMask", 2, 1, 1, ids[5], -1, false, false},
    {"dropout", 1, 1, 2, ids[1], 2, false, false},
    {"castVec1Res", 2, 1, 2, ids[2], -1, false, false},
    {"storeVec1Res", 1, 0, 0, ids[7], 2, false, false},
    {"bmm2", 1, 0, 0, mm2ResQueue, 2, false, false},
    {"load2", 2, 1, 1, ids[3], -1, false, false},
    {"addResOut", 3, 0, 0, vec2ResQueue, 2, false, false},
    {"loadAddResOut", 2, 1, 1, stage2Buf, -1, false, false},
    {"mulRes", 2, 1, 2, stage2Buf, -1, false, false},
    {"addRes", 2, 1, 2, stage2Buf, -1, false, false},
    {"div", 2, 1, 2, stage2Buf, -1, false, false},
    {"castBmm2Res", 2, 1, 2, stage2Buf, -1, true, false},
    {"store", 3, 0, 0, -1, -1, false, true},
    {"storeSoftmaxMax", 3, 0, 0, -1, -1, false, true},
  };
  for (const auto &info : nodes) {
    SetNodeMemAttr(graph.FindNode(info.name), tensorID, info);
  }
}

void UnknownGraph(af::AscGraph &graph) {
  int32_t tensorID = 0;
  auto ONE = af::sym::kSymbolOne;
  auto X = af::Symbol("X");
  auto Y = af::Symbol("Y");
  auto x = graph.CreateAxis("x", X);
  auto y = graph.CreateAxis("y", Y);
  auto resAxis = {x.id, y.id};
  auto resRepeat = std::initializer_list<Expr>{X, Y};
  auto resStride = std::initializer_list<Expr>{Y, ONE};
  int32_t exec_order = 0;

  Data input("input", graph);
  InitDataNode(input, exec_order, resAxis, af::DT_UINT32, resRepeat, resStride);
  SetApiAttr(graph.FindNode("input"), af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone);
  SetNodeMemAttr(graph.FindNode("input"), tensorID, {"input", 0, 0, 0, -1, -1, false, false});

  Load load("load");
  load.x = input.y;
  InitLoadNode(load, exec_order, resAxis, af::DT_UINT32, resRepeat, resStride);
  auto load_node = graph.FindNode("load");
  SetApiAttr(load_node, af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitNone);
  load_node->attr.sched.loop_axis = x.id;
  SetNodeMemAttr(load_node, tensorID, {"load", 2, 1, 1, 0, -1, false, false});
  load_node->outputs[0].attr.vectorized_axis = {y.id};

  Nop unknown("unknown");
  unknown.x = load.y;
  InitUnaryNode(unknown, exec_order, resAxis, af::DT_UINT32, resRepeat, resStride);
  auto unknown_node = graph.FindNode("unknown");
  SetApiAttr(unknown_node, af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitVector);
  unknown_node->attr.sched.loop_axis = x.id;
  SetNodeMemAttr(unknown_node, tensorID, {"unknown", 2, 1, 2, 0, -1, false, false});
  unknown_node->outputs[0].attr.vectorized_axis = {y.id};

  Store store("store");
  store.x = unknown.y;
  InitStoreNode(store, exec_order, resAxis, af::DT_UINT32, resRepeat, resStride);
  auto store_node = graph.FindNode("store");
  SetApiAttr(store_node, af::ApiType::kAPITypeCompute, af::ComputeUnit::kUnitNone);
  store_node->attr.sched.loop_axis = x.id;
  store_node->outputs[0].attr.vectorized_axis = {y.id};
  SetNodeMemAttr(store_node, tensorID, {"store", 3, 0, 0, -1, -1, false, true});

  Output output("output");
  output.x = store.y;
  InitOutputNode(output, exec_order, resAxis, af::DT_UINT32, resRepeat, resStride);
  SetApiAttr(graph.FindNode("output"), af::ApiType::kAPITypeBuffer, af::ComputeUnit::kUnitNone);
  SetNodeMemAttr(graph.FindNode("output"), tensorID, {"output", 0, 0, 0, -1, -1, false, false});
}
}  // namespace att
