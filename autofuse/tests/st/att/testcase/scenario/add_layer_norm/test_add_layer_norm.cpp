/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "gtest/gtest.h"
#include "base/att_const_values.h"
#include "gen_model_info.h"
#include "ascir_ops.h"
#include "tiling_code_generator.h"
#include "api_tiling_gen/gen_api_tiling.h"
#include "autofuse_config/auto_fuse_config.h"
#include "gen_tiling_impl.h"
#include "graph_construct_utils.h"
#include "result_checker_utils.h"
#include "common/st_scenario_utils.h"
#include "test_common_utils.h"

using namespace af::ascir_op;
namespace ascir {
constexpr int64_t ID_NONE = -1;  // 取多少？
using namespace ge;
using HintGraph = AscGraph;
}  // namespace ascir

using namespace att;

namespace {
template <typename NodeT>
void SetNodeScheduleAndTensor(NodeT &node, std::initializer_list<int64_t> axis, ge::DataType dtype,
                              std::initializer_list<af::Expression> repeats,
                              std::initializer_list<af::Expression> strides) {
  node.attr.sched.axis = axis;
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

template <typename NodeT, typename InputT>
void InitInputNode(NodeT &node, const InputT &input, std::initializer_list<int64_t> axis, ge::DataType dtype,
                   std::initializer_list<af::Expression> repeats, std::initializer_list<af::Expression> strides) {
  node.x = input;
  SetNodeScheduleAndTensor(node, axis, dtype, repeats, strides);
}

template <typename NodeT>
void SetGmOutputNode(const NodeT &node) {
  node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  node->outputs[0].attr.mem.position = af::Position::kPositionGM;
}

template <typename NodeT>
void SetQueueNode(const NodeT &node, int &tensor_id, int queue_id, af::Position position) {
  node->outputs[0].attr.mem.tensor_id = tensor_id++;
  node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  node->outputs[0].attr.mem.position = position;
  node->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  node->outputs[0].attr.buf.id = ascir::ID_NONE;
  node->outputs[0].attr.que.id = queue_id;
  node->outputs[0].attr.que.depth = 1;
  node->outputs[0].attr.que.buf_num = 1;
  node->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;
}

void CreateDataAndLoad(Load &load, ascir::HintGraph &graph, const char *data_name, std::initializer_list<int64_t> axis,
                       ge::DataType dtype, std::initializer_list<af::Expression> repeats,
                       std::initializer_list<af::Expression> strides) {
  Data data(data_name, graph);
  SetNodeScheduleAndTensor(data, axis, dtype, repeats, strides);
  InitInputNode(load, data.y, axis, dtype, repeats, strides);
}

template <typename NodeT>
void SetBufferNode(const NodeT &node, int &tensor_id, int buf_id) {
  node->outputs[0].attr.mem.tensor_id = tensor_id++;
  node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  node->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  node->outputs[0].attr.buf.id = buf_id;
  node->outputs[0].attr.que.id = ascir::ID_NONE;
  node->outputs[0].attr.que.depth = ascir::ID_NONE;
  node->outputs[0].attr.que.buf_num = ascir::ID_NONE;
  node->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;
}

void ApplySplitAndVectorize(ascir::HintGraph &graph, const char *node_name, int64_t outer_axis,
                            int64_t inner_axis_outer, int64_t inner_axis_inner, int64_t loop_axis,
                            std::initializer_list<int64_t> vectorized_axis,
                            af::ComputeUnit compute_unit = af::ComputeUnit::kUnitInvalid) {
  auto node = graph.FindNode(node_name);
  if (compute_unit != af::ComputeUnit::kUnitInvalid) {
    node->attr.api.unit = compute_unit;
  }
  graph.ApplySplit(node, outer_axis, inner_axis_outer);
  graph.ApplySplit(node, loop_axis, inner_axis_inner);
  node->attr.sched.loop_axis = loop_axis;
  node->outputs[0].attr.vectorized_axis = vectorized_axis;
}
}  // namespace

class TestGenAddLayerNormalModelInfo : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  }
};

void Add_Layer_Norm_Normal_BeforeAutofuseConstInput(ascir::HintGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;
  auto A = af::Symbol(128, "A");
  auto R = af::Symbol("R");
  auto BL = af::Symbol(8, "BL");

  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);
  const std::initializer_list<int64_t> axes = {a.id, r.id, bl.id};
  Load x1Local("x1Local");
  CreateDataAndLoad(x1Local, graph, "x1", axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});
  Load x2Local("x2Local");
  CreateDataAndLoad(x2Local, graph, "x2", axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});
  Load biasLocal("biasLocal");
  CreateDataAndLoad(biasLocal, graph, "bias", axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});

  Concat mean("mean");
  mean.x = {x1Local.y, x2Local.y, biasLocal.y};
  SetNodeScheduleAndTensor(mean, axes, ge::DT_FLOAT, {A, R, ONE}, {R, ONE, ZERO});

  Store x_out("x_out");
  InitInputNode(x_out, mean.y, axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});
  Store mean_out("mean_out");
  InitInputNode(mean_out, mean.y, axes, ge::DT_FLOAT, {A, ONE, ONE}, {ONE, ZERO, ZERO});

  Data one("one", graph);
  SetNodeScheduleAndTensor(one, axes, ge::DT_FLOAT, {ONE, ONE, BL}, {ZERO, ZERO, ONE});

  Concat rstd("rstd");
  rstd.x = {mean.y, mean.y, one.y};
  SetNodeScheduleAndTensor(rstd, axes, ge::DT_FLOAT, {A, R, ONE}, {R, ONE, ZERO});

  Store rstd_out("rstd_out");
  InitInputNode(rstd_out, rstd.y, axes, ge::DT_FLOAT, {A, ONE, ONE}, {ONE, ZERO, ZERO});

  Load betaLocal("betaLocal");
  CreateDataAndLoad(betaLocal, graph, "beta", axes, ge::DT_FLOAT16, {ONE, R, ONE}, {ZERO, ONE, ZERO});
  Load gammaLocal("gammaLocal");
  CreateDataAndLoad(gammaLocal, graph, "gamma", axes, ge::DT_FLOAT16, {ONE, R, ONE}, {ZERO, ONE, ZERO});

  Concat y("y");
  y.attr.api.unit = af::ComputeUnit::kUnitVector;
  y.x = {rstd.y, betaLocal.y, gammaLocal.y, rstd.y};
  SetNodeScheduleAndTensor(y, axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});

  Store y_out("y_out");
  InitInputNode(y_out, y.y, axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});

  Output buf1("buf1");
  InitInputNode(buf1, x_out.y, axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});
  Output buf2("buf2");
  InitInputNode(buf2, mean_out.y, axes, ge::DT_FLOAT, {A, ONE, ONE}, {ONE, ZERO, ZERO});
  Output buf3("buf3");
  InitInputNode(buf3, rstd_out.y, axes, ge::DT_FLOAT, {A, ONE, ONE}, {ONE, ZERO, ZERO});
  Output buf("buf");
  InitInputNode(buf, y_out.y, axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});
}

void Add_Layer_Norm_Normal_BeforeAutofuse(ascir::HintGraph &graph, const std::string &ident = "") {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;
  std::string axis_name1("A");
  axis_name1.append(ident);
  std::string axis_name2("R");
  axis_name2.append(ident);
  std::string axis_name3("BL");
  axis_name3.append(ident);
  auto A = af::Symbol(axis_name1.c_str());
  auto R = af::Symbol(axis_name2.c_str());
  auto BL = af::Symbol(8, axis_name3.c_str());

  auto a = graph.CreateAxis(axis_name1, A);
  auto r = graph.CreateAxis(axis_name2, R);
  auto bl = graph.CreateAxis(axis_name3, BL);
  const std::initializer_list<int64_t> axes = {a.id, r.id, bl.id};
  Load x1Local("x1Local");
  CreateDataAndLoad(x1Local, graph, "x1", axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});
  Load x2Local("x2Local");
  CreateDataAndLoad(x2Local, graph, "x2", axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});
  Load biasLocal("biasLocal");
  CreateDataAndLoad(biasLocal, graph, "bias", axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});

  Concat mean("mean");
  mean.x = {x1Local.y, x2Local.y, biasLocal.y};
  SetNodeScheduleAndTensor(mean, axes, ge::DT_FLOAT, {A, ONE, ONE}, {ONE, ZERO, ZERO});

  Store x_out("x_out");
  InitInputNode(x_out, mean.y, axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});
  Store mean_out("mean_out");
  InitInputNode(mean_out, mean.y, axes, ge::DT_FLOAT, {A, ONE, ONE}, {ONE, ZERO, ZERO});

  Data one("one", graph);
  SetNodeScheduleAndTensor(one, axes, ge::DT_FLOAT, {ONE, ONE, BL}, {ZERO, ZERO, ONE});

  Concat rstd("rstd");
  rstd.x = {mean.y, mean.y, one.y};
  SetNodeScheduleAndTensor(rstd, axes, ge::DT_FLOAT, {A, R, ONE}, {R, ONE, ZERO});

  Store rstd_out("rstd_out");
  InitInputNode(rstd_out, rstd.y, axes, ge::DT_FLOAT, {A, ONE, ONE}, {ONE, ZERO, ZERO});

  Load betaLocal("betaLocal");
  CreateDataAndLoad(betaLocal, graph, "beta", axes, ge::DT_FLOAT16, {ONE, R, ONE}, {ZERO, ONE, ZERO});
  Load gammaLocal("gammaLocal");
  CreateDataAndLoad(gammaLocal, graph, "gamma", axes, ge::DT_FLOAT16, {ONE, R, ONE}, {ZERO, ONE, ZERO});

  Concat y("y");
  y.attr.api.unit = af::ComputeUnit::kUnitVector;
  y.x = {rstd.y, betaLocal.y, gammaLocal.y, rstd.y};
  SetNodeScheduleAndTensor(y, axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});

  Store y_out("y_out");
  InitInputNode(y_out, y.y, axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});

  Output buf1("buf1");
  InitInputNode(buf1, x_out.y, axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});
  Output buf2("buf2");
  InitInputNode(buf2, mean_out.y, axes, ge::DT_FLOAT, {A, ONE, ONE}, {ONE, ZERO, ZERO});
  Output buf3("buf3");
  InitInputNode(buf3, rstd_out.y, axes, ge::DT_FLOAT, {A, ONE, ONE}, {ONE, ZERO, ZERO});
  Output buf("buf");
  InitInputNode(buf, y_out.y, axes, ge::DT_FLOAT16, {A, R, ONE}, {R, ONE, ZERO});
}

/*
for aBO
  for aBIO
    for aBII
      for r
        load x1
        load x2
        load bias
        CalcMean
        CalcRstd
        Store X
        Store mean
        Load beta
        Load gamma
        CalcRstd
        Store rstd
        CalcY
        Store y
*/

void Add_Layer_Norm_Normal_AfterScheduler(ascir::HintGraph &graph, const std::string &ident = "") {
  auto a = graph.FindAxis(0)->id;
  auto r = graph.FindAxis(1)->id;

  auto [aBO, aBI] = graph.BlockSplit(a, "nbi" + ident, "nbo" + ident);         // AB Ab
  auto [aBIO, aBII] = graph.TileSplit(aBI->id, "nii" + ident, "nio" + ident);  // AbT Abt

  ApplySplitAndVectorize(graph, "x1", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r});
  ApplySplitAndVectorize(graph, "x2", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r});
  ApplySplitAndVectorize(graph, "bias", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r});
  ApplySplitAndVectorize(graph, "x1Local", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r});
  ApplySplitAndVectorize(graph, "x2Local", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r});
  ApplySplitAndVectorize(graph, "biasLocal", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r});
  ApplySplitAndVectorize(graph, "mean", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r},
                         af::ComputeUnit::kUnitVector);
  ApplySplitAndVectorize(graph, "x_out", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r});
  ApplySplitAndVectorize(graph, "mean_out", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r});
  ApplySplitAndVectorize(graph, "rstd", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r},
                         af::ComputeUnit::kUnitVector);
  ApplySplitAndVectorize(graph, "rstd_out", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r},
                         af::ComputeUnit::kUnitVector);
  ApplySplitAndVectorize(graph, "betaLocal", aBO->id, aBI->id, aBII->id, aBIO->id, {r});
  ApplySplitAndVectorize(graph, "gammaLocal", aBO->id, aBI->id, aBII->id, aBIO->id, {r});
  ApplySplitAndVectorize(graph, "y", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r});
  ApplySplitAndVectorize(graph, "y_out", aBO->id, aBI->id, aBII->id, aBIO->id, {aBII->id, r});
}

void SetGmOutputNodeByName(ascir::HintGraph &graph, const char *name) {
  auto node = graph.FindNode(name);
  SetGmOutputNode(node);
}

void SetQueueNodeByName(ascir::HintGraph &graph, const char *name, int &tensor_id, int queue_id,
                        af::Position position) {
  auto node = graph.FindNode(name);
  SetQueueNode(node, tensor_id, queue_id, position);
}

void Add_Layer_Norm_Normal_AfterQueBufAlloc(ascir::HintGraph &graph) {
  int tensorID = 0;
  int queID = 0;
  int bufID = 0;
  int x1Que = queID++;
  int x2Que = queID++;
  int biasQue = queID++;
  int gammaQue = queID++;
  int betaQue = queID++;
  int meanQue = queID++;
  queID++;  // rstdQue unused
  int yQue = queID++;
  int xQue = queID++;
  queID++;  // x32Queue unused
  int oneTBuf = bufID++;

  SetGmOutputNodeByName(graph, "x1");
  SetGmOutputNodeByName(graph, "x2");
  SetGmOutputNodeByName(graph, "bias");
  SetQueueNodeByName(graph, "x1Local", tensorID, x1Que, af::Position::kPositionVecIn);
  SetQueueNodeByName(graph, "x2Local", tensorID, x2Que, af::Position::kPositionVecIn);
  SetQueueNodeByName(graph, "biasLocal", tensorID, biasQue, af::Position::kPositionVecIn);
  SetQueueNodeByName(graph, "mean", tensorID, meanQue, af::Position::kPositionVecOut);
  SetGmOutputNodeByName(graph, "x_out");
  SetGmOutputNodeByName(graph, "mean_out");

  auto one = graph.FindNode("one");
  SetBufferNode(one, tensorID, oneTBuf);

  SetQueueNodeByName(graph, "rstd", tensorID, yQue, af::Position::kPositionVecOut);
  SetGmOutputNodeByName(graph, "rstd_out");
  SetGmOutputNodeByName(graph, "beta");
  SetQueueNodeByName(graph, "betaLocal", tensorID, betaQue, af::Position::kPositionVecIn);
  SetGmOutputNodeByName(graph, "gamma");
  SetQueueNodeByName(graph, "gammaLocal", tensorID, gammaQue, af::Position::kPositionVecIn);
  SetQueueNodeByName(graph, "y", tensorID, yQue, af::Position::kPositionVecOut);
  SetGmOutputNodeByName(graph, "y_out");
}

void Add_Layer_Norm_Slice_BeforeAutofuse(ascir::HintGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;
  auto A = af::Symbol("A");
  auto R = af::Symbol("R");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  const std::initializer_list<int64_t> axes = {a.id, r.id};
  Load x1Local("x1Local");
  CreateDataAndLoad(x1Local, graph, "x1", axes, ge::DT_FLOAT16, {A, R}, {R, ONE});
  Load x2Local("x2Local");
  CreateDataAndLoad(x2Local, graph, "x2", axes, ge::DT_FLOAT16, {A, R}, {R, ONE});
  Load biasLocal("biasLocal");
  CreateDataAndLoad(biasLocal, graph, "bias", axes, ge::DT_FLOAT16, {A, R}, {R, ONE});

  Concat mean("mean");
  mean.attr.api.unit = af::ComputeUnit::kUnitVector;
  mean.x = {x1Local.y, x2Local.y, biasLocal.y};
  SetNodeScheduleAndTensor(mean, axes, ge::DT_FLOAT, {A, ONE}, {ONE, ONE});

  Store x_out("x_out");
  InitInputNode(x_out, mean.y, axes, ge::DT_FLOAT16, {A, R}, {R, ONE});

  Concat rstd("rstd");
  rstd.attr.api.unit = af::ComputeUnit::kUnitVector;
  rstd.x = {mean.y, mean.y};
  SetNodeScheduleAndTensor(rstd, axes, ge::DT_FLOAT, {A, R}, {R, ONE});

  Store mean_out("mean_out");
  InitInputNode(mean_out, mean.y, axes, ge::DT_FLOAT, {A, ONE}, {ONE, ONE});
  Store rstd_out("rstd_out");
  InitInputNode(rstd_out, rstd.y, axes, ge::DT_FLOAT, {A, ONE}, {ONE, ONE});

  Load betaLocal("betaLocal");
  CreateDataAndLoad(betaLocal, graph, "beta", axes, ge::DT_FLOAT16, {ONE, R}, {ZERO, ONE});
  Load gammaLocal("gammaLocal");
  CreateDataAndLoad(gammaLocal, graph, "gamma", axes, ge::DT_FLOAT16, {ONE, R}, {ZERO, ONE});

  Concat y("y");
  y.attr.api.unit = af::ComputeUnit::kUnitVector;
  y.x = {rstd.y, betaLocal.y, gammaLocal.y, rstd.y};
  SetNodeScheduleAndTensor(y, axes, ge::DT_FLOAT16, {A, R}, {R, ONE});
  Store y_out("y_out");
  InitInputNode(y_out, y.y, axes, ge::DT_FLOAT16, {A, R}, {R, ONE});

  Output buf1("buf1");
  InitInputNode(buf1, x_out.y, axes, ge::DT_FLOAT16, {A, R}, {R, ONE});
  Output buf2("buf2");
  InitInputNode(buf2, mean_out.y, axes, ge::DT_FLOAT, {A, ONE}, {ONE, ONE});
  Output buf3("buf3");
  InitInputNode(buf3, rstd_out.y, axes, ge::DT_FLOAT, {A, ONE}, {ONE, ONE});
  Output buf("buf");
  InitInputNode(buf, y_out.y, axes, ge::DT_FLOAT16, {A, R}, {R, ONE});
}

/*
for aBO
  for aBI
    for rO
      for rI
        load x1
        load x2
        load bias
        CalcMean
        Store X
        Load beta
        Load gamma
        CalcRstd
        Store mean
        Store rstd
        CalcY
        Store y
*/

void Add_Layer_Norm_Slice_AfterScheduler(ascir::HintGraph &graph) {
  auto a = graph.FindAxis(0)->id;
  auto r = graph.FindAxis(1)->id;

  auto [aBO, aBI] = graph.BlockSplit(a, "sbi", "sbo");
  auto [rO, rI] = graph.TileSplit(r, "sii", "sio");
  rI->align = af::Symbol(16);
  ApplySplitAndVectorize(graph, "x1", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "x2", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "bias", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "x1Local", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "x2Local", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "biasLocal", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "mean", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "x_out", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "mean_out", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "rstd", aBO->id, aBI->id, rI->id, rO->id, {rO->id, rI->id});
  ApplySplitAndVectorize(graph, "rstd_out", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "betaLocal", aBO->id, aBI->id, rI->id, rO->id, {rO->id, rI->id});
  ApplySplitAndVectorize(graph, "gammaLocal", aBO->id, aBI->id, rI->id, rO->id, {rO->id, rI->id});
  ApplySplitAndVectorize(graph, "y", aBO->id, aBI->id, rI->id, rO->id, {rO->id, rI->id});
  ApplySplitAndVectorize(graph, "y_out", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
}

void Add_Layer_Norm_Slice_AfterQueBufAlloc(ascir::HintGraph &graph) {
  int tensorID = 0;
  int queID = 0;
  int bufID = 0;
  (void)bufID;
  int x1Que = queID++;
  int x2Que = queID++;
  int biasQue = queID++;
  int xQue = queID++;
  int yQue = queID++;
  int betaQue = queID++;
  int gammaQue = queID++;
  int meanQue = queID++;
  queID++;  // rstdQue unused

  auto x1 = graph.FindNode("x1");
  SetGmOutputNode(x1);

  auto x2 = graph.FindNode("x2");
  SetGmOutputNode(x2);

  auto bias = graph.FindNode("bias");
  SetGmOutputNode(bias);

  auto x1Local = graph.FindNode("x1Local");
  SetQueueNode(x1Local, tensorID, x1Que, af::Position::kPositionVecIn);

  auto x2Local = graph.FindNode("x2Local");
  SetQueueNode(x2Local, tensorID, x2Que, af::Position::kPositionVecIn);

  auto biasLocal = graph.FindNode("biasLocal");
  SetQueueNode(biasLocal, tensorID, biasQue, af::Position::kPositionVecIn);

  auto mean = graph.FindNode("mean");
  SetQueueNode(mean, tensorID, meanQue, af::Position::kPositionVecOut);

  auto x_out = graph.FindNode("x_out");
  SetGmOutputNode(x_out);

  auto mean_out = graph.FindNode("mean_out");
  SetGmOutputNode(mean_out);

  auto rstd = graph.FindNode("rstd");
  SetQueueNode(rstd, tensorID, yQue, af::Position::kPositionVecOut);

  auto rstd_out = graph.FindNode("rstd_out");
  SetGmOutputNode(rstd_out);

  auto beta = graph.FindNode("beta");
  SetGmOutputNode(beta);

  auto betaLocal = graph.FindNode("betaLocal");
  SetQueueNode(betaLocal, tensorID, betaQue, af::Position::kPositionVecIn);

  auto gamma = graph.FindNode("gamma");
  SetGmOutputNode(gamma);

  auto gammaLocal = graph.FindNode("gammaLocal");
  SetQueueNode(gammaLocal, tensorID, gammaQue, af::Position::kPositionVecIn);

  auto y = graph.FindNode("y");
  SetQueueNode(y, tensorID, yQue, af::Position::kPositionVecOut);

  auto y_out = graph.FindNode("y_out");
  SetGmOutputNode(y_out);
}

void Add_Layer_Norm_Welford_BeforeAutofuse(ascir::HintGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;
  auto A = af::Symbol("A");
  auto R = af::Symbol("R");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  const std::initializer_list<int64_t> axes = {a.id, r.id};
  Load x1Local("x1Local");
  CreateDataAndLoad(x1Local, graph, "x1", axes, ge::DT_FLOAT16, {A, R}, {R, ONE});
  Load x2Local("x2Local");
  CreateDataAndLoad(x2Local, graph, "x2", axes, ge::DT_FLOAT16, {A, R}, {R, ONE});
  Load biasLocal("biasLocal");
  CreateDataAndLoad(biasLocal, graph, "bias", axes, ge::DT_FLOAT16, {A, R}, {R, ONE});

  Concat part1("part1");
  part1.attr.api.unit = af::ComputeUnit::kUnitVector;
  part1.x = {x1Local.y, x2Local.y, biasLocal.y};
  SetNodeScheduleAndTensor(part1, axes, ge::DT_FLOAT16, {A, R}, {R, ONE});

  Store x_out("x_out");
  InitInputNode(x_out, part1.y, axes, ge::DT_FLOAT16, {A, R}, {R, ONE});
  Store x_fp32_out("x_fp32_out");
  InitInputNode(x_fp32_out, part1.y, axes, ge::DT_FLOAT, {A, R}, {R, ONE});

  Concat part1Final("part1Final");
  part1Final.attr.api.unit = af::ComputeUnit::kUnitVector;
  part1Final.x = {part1.y, part1.y};
  SetNodeScheduleAndTensor(part1Final, axes, ge::DT_FLOAT, {A, ONE}, {ONE, ONE});

  Store mean_out("mean_out");
  InitInputNode(mean_out, part1Final.y, axes, ge::DT_FLOAT, {A, ONE}, {ONE, ONE});
  Store rstd_out("rstd_out");
  InitInputNode(rstd_out, part1Final.y, axes, ge::DT_FLOAT, {A, ONE}, {ONE, ONE});

  Load x32("x32");
  InitInputNode(x32, x_fp32_out.y, axes, ge::DT_FLOAT, {A, R}, {R, ONE});
  Load betaLocal("betaLocal");
  CreateDataAndLoad(betaLocal, graph, "beta", axes, ge::DT_FLOAT16, {ONE, R}, {ZERO, ONE});
  Load gammaLocal("gammaLocal");
  CreateDataAndLoad(gammaLocal, graph, "gamma", axes, ge::DT_FLOAT16, {ONE, R}, {ZERO, ONE});

  Concat y("y");
  y.attr.api.unit = af::ComputeUnit::kUnitVector;
  y.x = {x32.y, betaLocal.y, gammaLocal.y, x32.y};
  SetNodeScheduleAndTensor(y, axes, ge::DT_FLOAT16, {A, R}, {R, ONE});
  Store y_out("y_out");
  InitInputNode(y_out, y.y, axes, ge::DT_FLOAT16, {A, R}, {R, ONE});

  Output buf1("buf1");
  InitInputNode(buf1, x_out.y, axes, ge::DT_FLOAT16, {A, R}, {R, ONE});
  Output buf2("buf2");
  InitInputNode(buf2, mean_out.y, axes, ge::DT_FLOAT, {A, ONE}, {ONE, ONE});
  Output buf3("buf3");
  InitInputNode(buf3, rstd_out.y, axes, ge::DT_FLOAT, {A, ONE}, {ONE, ONE});
  Output buf("buf");
  InitInputNode(buf, y_out.y, axes, ge::DT_FLOAT16, {A, R}, {R, ONE});
}

void Add_Layer_Norm_Welford_AfterScheduler(ascir::HintGraph &graph) {
  auto a = graph.FindAxis(0)->id;
  auto r = graph.FindAxis(1)->id;

  auto [aBO, aBI] = graph.BlockSplit(a, "wbi", "wbo");
  auto [rO, rI] = graph.TileSplit(r, "wii", "wio");
  ApplySplitAndVectorize(graph, "x1", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "x2", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "bias", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "x1Local", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "x2Local", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "biasLocal", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "part1", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "x_out", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "x_fp32_out", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "mean_out", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "part1Final", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "rstd_out", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "x32", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "betaLocal", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "gammaLocal", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "y", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
  ApplySplitAndVectorize(graph, "y_out", aBO->id, aBI->id, rI->id, rO->id, {rI->id});
}

void Add_Layer_Norm_Welford_AfterQueBufAlloc(ascir::HintGraph &graph);

void BuildAddLayerNormGraphs(std::vector<ascir::AscGraph> &graphs, bool use_const_input) {
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  if (use_const_input) {
    Add_Layer_Norm_Normal_BeforeAutofuseConstInput(graph_normal);
  } else {
    Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  }
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);

  ascir::AscGraph graph_slice("graph_slice");
  graph_slice.SetTilingKey(1111u);
  Add_Layer_Norm_Slice_BeforeAutofuse(graph_slice);
  Add_Layer_Norm_Slice_AfterScheduler(graph_slice);
  Add_Layer_Norm_Slice_AfterQueBufAlloc(graph_slice);
  graphs.emplace_back(graph_slice);

  ascir::AscGraph graph_welford("graph_welford");
  graph_welford.SetTilingKey(1151u);
  Add_Layer_Norm_Welford_BeforeAutofuse(graph_welford);
  Add_Layer_Norm_Welford_AfterScheduler(graph_welford);
  Add_Layer_Norm_Welford_AfterQueBufAlloc(graph_welford);
  graphs.emplace_back(graph_welford);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);
}

void BuildAddLayerNormBinary() {
  auto ret = std::system(
      std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_main_add_layer_norm.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/stub/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);
  ret = std::system(
      "g++ tiling_func_main_add_layer_norm.cpp AddLayerNorm_*_tiling_func.cpp -o tiling_func_main_add_layer_norm "
      "-I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
}

void Add_Layer_Norm_Welford_AfterQueBufAlloc(ascir::HintGraph &graph) {
  int tensorID = 0;
  int queID = 0;
  int bufID = 0;
  (void)bufID;
  int x1Que = queID++;
  int x2Que = queID++;
  int biasQue = queID++;
  int xQue = queID++;
  int yQue = queID++;
  queID++;  // x32Que unused
  int vQue = queID++;
  int meanQue = queID++;
  queID++;  // rstdQue unused

  SetGmOutputNodeByName(graph, "x1");
  SetGmOutputNodeByName(graph, "x2");
  SetGmOutputNodeByName(graph, "bias");
  SetQueueNodeByName(graph, "x1Local", tensorID, x1Que, af::Position::kPositionVecIn);
  SetQueueNodeByName(graph, "x2Local", tensorID, x2Que, af::Position::kPositionVecIn);
  SetQueueNodeByName(graph, "biasLocal", tensorID, biasQue, af::Position::kPositionVecIn);
  SetQueueNodeByName(graph, "part1", tensorID, xQue, af::Position::kPositionVecOut);
  SetGmOutputNodeByName(graph, "x_out");
  SetGmOutputNodeByName(graph, "x_fp32_out");
  SetQueueNodeByName(graph, "part1Final", tensorID, meanQue, af::Position::kPositionVecOut);
  SetGmOutputNodeByName(graph, "mean_out");
  SetGmOutputNodeByName(graph, "rstd_out");
  SetQueueNodeByName(graph, "x32", tensorID, vQue, af::Position::kPositionVecIn);
  SetGmOutputNodeByName(graph, "beta");
  SetQueueNodeByName(graph, "betaLocal", tensorID, x2Que, af::Position::kPositionVecIn);
  SetGmOutputNodeByName(graph, "gamma");
  SetQueueNodeByName(graph, "gammaLocal", tensorID, biasQue, af::Position::kPositionVecIn);
  SetQueueNodeByName(graph, "y", tensorID, yQue, af::Position::kPositionVecOut);
  SetGmOutputNodeByName(graph, "y_out");
}

TEST_F(TestGenAddLayerNormalModelInfo, case0) {
  std::vector<ascir::AscGraph> graphs;
  BuildAddLayerNormGraphs(graphs, false);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["solver_type"] = "HighPerf";
  EXPECT_EQ(GenTilingImpl("AddLayerNorm", graphs, options), true);

  BuildAddLayerNormBinary();
  auto ret = std::system("./tiling_func_main_add_layer_norm");
}

TEST_F(TestGenAddLayerNormalModelInfo, case_axes_reorder) {
  std::vector<ascir::AscGraph> graphs;
  BuildAddLayerNormGraphs(graphs, true);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("AddLayerNorm", graphs, options), true);

  BuildAddLayerNormBinary();
  auto ret = std::system("./tiling_func_main_add_layer_norm");
}

TEST_F(TestGenAddLayerNormalModelInfo, case_axes_reorder_replace) {
  std::vector<ascir::AscGraph> graphs;
  BuildAddLayerNormGraphs(graphs, true);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["duration_level"] = "1";
  options["do_variable_replace"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("AddLayerNorm", graphs, options), true);
}
