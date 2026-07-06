
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"

#include "pattern_fusion/pattern_fusion.h"
#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/asc_ir_lowerer.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "can_fuse/backend/fusion_decider_registry.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "post_process/asc_backend_post_processor.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"
#include "utils/auto_fuse_config.h"
#include "backend/backend_spec.h"
#include "ascgen_log.h"

#include "common/util/mem_utils.h"
#include "expression/testcase/source_stub.h"
#include "op_creator_register.h"
#include "all_ops_cpp.h"
#include "compliant_op_desc_builder.h"
#include "esb_graph.h"
#include "platform_context.h"
#include "base/att_const_values.h"
#include "depends/runtime/src/runtime_stub.h"

using namespace std;
using namespace testing;

namespace af {
using namespace autofuse;
namespace {
struct ScopedEnv {
  explicit ScopedEnv(const char *k, const char *v) : key_(k) {
    old_ = std::getenv(k);
    setenv(k, v, 1);
  }
  ~ScopedEnv() {
    if (old_)
      setenv(key_, old_, 1);
    else
      unsetenv(key_);
  }

 private:
  const char *key_;
  const char *old_;
};

template <typename T>
es::Tensor CreateConst(es::Graph &graph, ge::DataType dtype, const std::vector<int64_t> &dims, std::vector<T> value) {
  auto result = es::FileConstant(graph, dims, dtype);
  GeTensorDesc desc(GeShape(dims), ge::FORMAT_ND, dtype);
  GeTensorPtr tensor =
      std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(value.data()), sizeof(T) * value.size());
  AttrUtils::SetTensor(result.GetEsbTensor()->GetProducer()->GetOpDesc(), "value", tensor);
  result.GetEsbTensor()->GetProducer()->GetOpDesc()->SetType(ge::CONSTANT);
  return result;
}

static std::vector<int64_t> RepeatsToInt64Vec(const std::vector<af::Expression> &repeats) {
  std::vector<int64_t> result;
  for (const auto &expr : repeats) {
    int64_t val = 0;
    if (expr.IsConstExpr()) {
      (void)expr.GetConstValue(val);
    }
    result.push_back(val);
  }
  return result;
}

static void BuildConv2DGraphWithBiasAndAttr(es::Graph &graph, const std::string &data_format,
                                            const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                                            const std::vector<int64_t> &dilations, int64_t groups, int64_t offset_x,
                                            const std::string &pad_mode, bool enable_hf32, bool has_add_input,
                                            std::initializer_list<const char *> output_shape,
                                            std::initializer_list<const char *> add_input_shape) {
  auto data0 = graph.CreateInput(0, "data0", nullptr);
  data0.SetSymbolShape({"1", "224", "224", "3"});

  auto filter = graph.CreateInput(1, "filter", nullptr);
  filter.SetSymbolShape({"3", "3", "3", "64"});

  auto bias = graph.CreateInput(2, "bias", nullptr);
  bias.SetSymbolShape({"64"});

  auto conv2d =
      es::Conv2D(data0, filter, bias, nullptr, strides, pads, dilations, groups, data_format.c_str(), offset_x);
  conv2d.SetSymbolShape(output_shape);

  if (!pad_mode.empty()) {
    (void)af::AttrUtils::SetStr(conv2d.GetEsbTensor()->GetProducer()->GetOpDesc(), "padding", pad_mode);
    (void)af::AttrUtils::SetBool(conv2d.GetEsbTensor()->GetProducer()->GetOpDesc(), "enable_hf32", enable_hf32);
  }

  if (has_add_input) {
    auto data1 = graph.CreateInput(3, "data1", nullptr);
    data1.SetSymbolShape(add_input_shape);
    auto add = es::Add(conv2d, data1);
    add.SetSymbolShape(add_input_shape);
    graph.SetOutput(add, 0);
  } else {
    graph.SetOutput(conv2d, 0);
  }
}

static void BuildConv2DNoBiasGraph(es::Graph &graph) {
  auto data0 = graph.CreateInput(0, "data0", nullptr);
  data0.SetSymbolShape({"1", "224", "224", "3"});

  auto filter = graph.CreateInput(1, "filter", nullptr);
  filter.SetSymbolShape({"3", "3", "3", "64"});

  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  std::vector<int64_t> dilations = {1, 1, 1, 1};
  int64_t groups = 1;
  std::string data_format = "NHWC";
  int64_t offset_x = 0;

  auto conv2d =
      es::Conv2D(data0, filter, nullptr, nullptr, strides, pads, dilations, groups, data_format.c_str(), offset_x);
  conv2d.SetSymbolShape({"1", "224", "224", "64"});

  graph.SetOutput(conv2d, 0);
}

static void BuildConv2DWithBiasGraph(es::Graph &graph) {
  auto data0 = graph.CreateInput(0, "data0", nullptr);
  data0.SetSymbolShape({"1", "224", "224", "3"});

  auto filter = graph.CreateInput(1, "filter", nullptr);
  filter.SetSymbolShape({"3", "3", "3", "64"});

  auto bias = graph.CreateInput(2, "bias", nullptr);
  bias.SetSymbolShape({"64"});

  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  std::vector<int64_t> dilations = {1, 1, 1, 1};
  int64_t groups = 1;
  std::string data_format = "NHWC";
  int64_t offset_x = 0;

  auto conv2d =
      es::Conv2D(data0, filter, bias, nullptr, strides, pads, dilations, groups, data_format.c_str(), offset_x);
  conv2d.SetSymbolShape({"1", "224", "224", "64"});

  graph.SetOutput(conv2d, 0);
}

static void BuildConv2DWithOffsetWGraph(es::Graph &graph, bool has_bias) {
  auto data0 = graph.CreateInput(0, "data0", nullptr);
  data0.SetSymbolShape({"1", "224", "224", "3"});

  auto filter = graph.CreateInput(1, "filter", nullptr);
  filter.SetSymbolShape({"3", "3", "3", "64"});

  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  std::vector<int64_t> dilations = {1, 1, 1, 1};
  int64_t groups = 1;
  std::string data_format = "NHWC";
  int64_t offset_x = 0;

  if (has_bias) {
    auto bias = graph.CreateInput(2, "bias", nullptr);
    bias.SetSymbolShape({"64"});
    auto offset_w = graph.CreateInput(3, "offset_w", nullptr);
    offset_w.SetSymbolShape({"64"});
    (void)offset_w.GetEsbTensor()->GetProducer()->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT8);

    auto conv2d =
        es::Conv2D(data0, filter, bias, offset_w, strides, pads, dilations, groups, data_format.c_str(), offset_x);
    conv2d.SetSymbolShape({"1", "224", "224", "64"});
    graph.SetOutput(conv2d, 0);
  } else {
    auto offset_w = graph.CreateInput(2, "offset_w", nullptr);
    offset_w.SetSymbolShape({"64"});
    (void)offset_w.GetEsbTensor()->GetProducer()->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT8);

    auto conv2d =
        es::Conv2D(data0, filter, nullptr, offset_w, strides, pads, dilations, groups, data_format.c_str(), offset_x);
    conv2d.SetSymbolShape({"1", "224", "224", "64"});
    graph.SetOutput(conv2d, 0);
  }
}

static void SetupShapeEnv(ShapeEnvAttr &shape_env) {
  (void)shape_env.CreateSymbol(1, MakeShared<GraphInputShapeSourceStub>(0, 0));
  (void)shape_env.CreateSymbol(224, MakeShared<GraphInputShapeSourceStub>(0, 1));
  (void)shape_env.CreateSymbol(224, MakeShared<GraphInputShapeSourceStub>(0, 2));
  (void)shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 3));
  (void)shape_env.CreateSymbol(64, MakeShared<GraphInputShapeSourceStub>(0, 4));
}

static void SetupShapeEnvWithMoreSymbols(ShapeEnvAttr &shape_env) {
  (void)shape_env.CreateSymbol(1, MakeShared<GraphInputShapeSourceStub>(0, 0));
  (void)shape_env.CreateSymbol(224, MakeShared<GraphInputShapeSourceStub>(0, 1));
  (void)shape_env.CreateSymbol(224, MakeShared<GraphInputShapeSourceStub>(0, 2));
  (void)shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 3));
  (void)shape_env.CreateSymbol(64, MakeShared<GraphInputShapeSourceStub>(0, 4));
  (void)shape_env.CreateSymbol(112, MakeShared<GraphInputShapeSourceStub>(0, 5));
  (void)shape_env.CreateSymbol(74, MakeShared<GraphInputShapeSourceStub>(0, 6));
  (void)shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 7));
}

static bool HasAscNodeType(const ComputeGraphPtr &cg, const std::string &target_type) {
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() != "AscBackend") continue;
    const auto attr = node->GetOpDesc()->GetAttrsGroup<af::AutoFuseAttrs>();
    if (attr == nullptr) continue;
    const auto &asc_graph = attr->GetAscGraph();
    if (asc_graph == nullptr) continue;
    for (const auto &asc_node : asc_graph->GetAllNodes()) {
      if (asc_node->GetType() == target_type) return true;
    }
  }
  return false;
}

static bool FindConv2DInAscGraph(const ComputeGraphPtr &cg, const std::vector<std::string> &node_types) {
  for (const auto &node_type : node_types) {
    if (HasAscNodeType(cg, node_type)) return true;
  }
  return false;
}

static void CheckConv2DOffsetX(const NodePtr &asc_node, const std::string &node_type, int64_t expected_offset_x) {
  auto op_desc = asc_node->GetOpDesc();
  const auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
  if (node_attr == nullptr || node_attr->ir_attr == nullptr) return;

  int64_t offset_x_attr = 0;
  if (node_type == "Conv2DBias") {
    auto conv2d_attr = node_attr->ir_attr->DownCastTo<af::ascir_op::Conv2DBias::AscConv2DBiasIrAttrDef>();
    if (conv2d_attr != nullptr) {
      (void)conv2d_attr->GetOffset_x(offset_x_attr);
      EXPECT_EQ(offset_x_attr, expected_offset_x);
    }
  } else if (node_type == "Conv2D") {
    auto conv2d_attr = node_attr->ir_attr->DownCastTo<af::ascir_op::Conv2D::AscConv2DIrAttrDef>();
    if (conv2d_attr != nullptr) {
      (void)conv2d_attr->GetOffset_x(offset_x_attr);
      EXPECT_EQ(offset_x_attr, expected_offset_x);
    }
  }
}

static bool FindAndCheckConv2DAttr(const ComputeGraphPtr &cg, const std::vector<std::string> &node_types,
                                   int64_t expected_offset_x) {
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() != "AscBackend") continue;
    const auto attr = node->GetOpDesc()->GetAttrsGroup<af::AutoFuseAttrs>();
    if (attr == nullptr) continue;
    const auto &asc_graph = attr->GetAscGraph();
    if (asc_graph == nullptr) continue;
    for (const auto &asc_node : asc_graph->GetAllNodes()) {
      for (const auto &node_type : node_types) {
        if (asc_node->GetType() == node_type) {
          CheckConv2DOffsetX(asc_node, node_type, expected_offset_x);
          return true;
        }
      }
    }
  }
  return false;
}

static void VerifyConv2DAttrGeneric(const ComputeGraphPtr &cg, const std::vector<std::string> &node_types,
                                    int64_t expected_offset_x) {
  bool found = FindAndCheckConv2DAttr(cg, node_types, expected_offset_x);
  EXPECT_TRUE(found) << "Conv2D node not found in AscGraph";
}

static bool VerifyConv2DIrAttr(const NodePtr &asc_node, const std::string &node_type) {
  auto op_desc = asc_node->GetOpDesc();
  const auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
  if (node_attr == nullptr || node_attr->ir_attr == nullptr) return false;

  std::vector<int64_t> strides, pads, dilations;
  int64_t groups = 0;

  if (node_type == "Conv2D") {
    auto conv_attr = node_attr->ir_attr->DownCastTo<af::ascir_op::Conv2D::AscConv2DIrAttrDef>();
    if (conv_attr == nullptr) return false;
    (void)conv_attr->GetStrides(strides);
    (void)conv_attr->GetPads(pads);
    (void)conv_attr->GetDilations(dilations);
    (void)conv_attr->GetGroups(groups);
  } else if (node_type == "Conv2DBias") {
    auto conv_attr = node_attr->ir_attr->DownCastTo<af::ascir_op::Conv2DBias::AscConv2DBiasIrAttrDef>();
    if (conv_attr == nullptr) return false;
    (void)conv_attr->GetStrides(strides);
    (void)conv_attr->GetPads(pads);
    (void)conv_attr->GetDilations(dilations);
    (void)conv_attr->GetGroups(groups);
  } else if (node_type == "Conv2DOffset") {
    auto conv_attr = node_attr->ir_attr->DownCastTo<af::ascir_op::Conv2DOffset::AscConv2DOffsetIrAttrDef>();
    if (conv_attr == nullptr) return false;
    (void)conv_attr->GetStrides(strides);
    (void)conv_attr->GetPads(pads);
    (void)conv_attr->GetDilations(dilations);
    (void)conv_attr->GetGroups(groups);
  } else if (node_type == "Conv2DOffsetBias") {
    auto conv_attr = node_attr->ir_attr->DownCastTo<af::ascir_op::Conv2DOffsetBias::AscConv2DOffsetBiasIrAttrDef>();
    if (conv_attr == nullptr) return false;
    (void)conv_attr->GetStrides(strides);
    (void)conv_attr->GetPads(pads);
    (void)conv_attr->GetDilations(dilations);
    (void)conv_attr->GetGroups(groups);
  } else {
    return false;
  }

  EXPECT_EQ(strides, (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_EQ(pads, (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_EQ(dilations, (std::vector<int64_t>{1, 1, 1, 1}));
  EXPECT_EQ(groups, 1);
  return true;
}

static void VerifyConv2DRepeats(const ComputeGraphPtr &cg, const std::string &node_type,
                                const std::vector<int64_t> &expected_input0,
                                const std::vector<int64_t> &expected_input1,
                                const std::vector<int64_t> &expected_output, size_t min_inputs, size_t min_outputs) {
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() != "AscBackend") continue;
    const auto attr = node->GetOpDesc()->GetAttrsGroup<af::AutoFuseAttrs>();
    if (attr == nullptr) continue;
    const auto &asc_graph = attr->GetAscGraph();
    if (asc_graph == nullptr) continue;
    for (const auto &asc_node : asc_graph->GetAllNodes()) {
      if (asc_node->GetType() != node_type) continue;
      auto op_desc = asc_node->GetOpDesc();
      if (!VerifyConv2DIrAttr(asc_node, node_type)) continue;
      ASSERT_GE(op_desc->GetInputsSize(), min_inputs);
      ASSERT_GE(op_desc->GetOutputsSize(), min_outputs);
      EXPECT_EQ(RepeatsToInt64Vec(asc_node->inputs[0].attr.repeats), expected_input0);
      EXPECT_EQ(RepeatsToInt64Vec(asc_node->inputs[1].attr.repeats), expected_input1);
      EXPECT_EQ(RepeatsToInt64Vec(asc_node->outputs[0].attr.repeats), expected_output);
      return;
    }
  }
  EXPECT_TRUE(false) << node_type << " node not found in AscGraph";
}

static ComputeGraphPtr ProcessGraphWithPipeline(const Graph &graph) {
  auto cg = GraphUtilsEx::GetComputeGraph(graph);
  ge::AscIrLowerer lowerer;
  if (lowerer.Lowering(cg) != GRAPH_SUCCESS) {
    return nullptr;
  }
  if (asc_adapt::GeFallback(cg) != GRAPH_SUCCESS) {
    return nullptr;
  }
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  if (fusion_strategy_solver.Fuse(cg) != af::SUCCESS) {
    return nullptr;
  }
  if (lowerer.Lifting(cg) != GRAPH_SUCCESS) {
    return nullptr;
  }
  AscBackendPostProcessor post_processor;
  if (post_processor.Do(cg) != af::SUCCESS) {
    return nullptr;
  }
  return cg;
}
}  // namespace

class UTestLoweringAndCanfuseV2_2 : public testing::Test {
 public:
 protected:
  void SetUp() override {
    AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fusion_size = 64U;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_split = true;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_matmul = true;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_conv = true;
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  std::unique_ptr<es::Graph> es_graph_;
};

TEST_F(UTestLoweringAndCanfuseV2_2, Conv2DAttrPostProcessPropagation) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  std::vector<int64_t> strides = {1, 2, 3, 1};
  std::vector<int64_t> pads = {1, 2, 3, 4};
  std::vector<int64_t> dilations = {1, 2, 1, 2};
  BuildConv2DGraphWithBiasAndAttr(*es_graph_, "NHWC", strides, pads, dilations, 2, 5, "SPECIFIC", true, true,
                                  {"1", "112", "74", "32"}, {"1", "112", "74", "32"});

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnvWithMoreSymbols(shape_env);

  auto cg = ProcessGraphWithPipeline(*es_graph_->Build());
  ASSERT_NE(cg, nullptr);

  bool found = FindConv2DInAscGraph(cg, {"Conv2D", "Conv2DBias"});
  EXPECT_TRUE(found) << "Conv2D node not found in AscGraph";

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2_2, Conv2DAttrBoundaryTest) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilations = {1, 1, 1, 1};
  BuildConv2DGraphWithBiasAndAttr(*es_graph_, "NCHW", strides, pads, dilations, 1, 0, "", false, true,
                                  {"1", "224", "224", "64"}, {"1", "224", "224", "64"});

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnv(shape_env);

  auto cg = ProcessGraphWithPipeline(*es_graph_->Build());
  ASSERT_NE(cg, nullptr);

  bool found = FindConv2DInAscGraph(cg, {"Conv2D", "Conv2DBias"});
  EXPECT_TRUE(found) << "Conv2D node not found in AscGraph";

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2_2, Conv2DAttrEdgeBoundaryTest) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  std::vector<int64_t> strides = {1, 63, 63, 1};
  std::vector<int64_t> pads = {0, 255, 255, 0};
  std::vector<int64_t> dilations = {1, 255, 255, 1};
  BuildConv2DGraphWithBiasAndAttr(*es_graph_, "NCHW", strides, pads, dilations, 1, 127, "SAME", false, true,
                                  {"1", "3", "3", "1"}, {"1", "3", "3", "1"});

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnv(shape_env);

  auto cg = ProcessGraphWithPipeline(*es_graph_->Build());
  ASSERT_NE(cg, nullptr);

  bool found = FindConv2DInAscGraph(cg, {"Conv2D", "Conv2DBias"});
  EXPECT_TRUE(found) << "Conv2D node not found in AscGraph";

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2_2, Conv2DAttrOffsetXNegativeTest) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilations = {1, 1, 1, 1};
  BuildConv2DGraphWithBiasAndAttr(*es_graph_, "NCHW", strides, pads, dilations, 1, -128, "VALID", false, true,
                                  {"1", "224", "224", "64"}, {"1", "224", "224", "64"});

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnv(shape_env);

  auto cg = ProcessGraphWithPipeline(*es_graph_->Build());
  ASSERT_NE(cg, nullptr);

  VerifyConv2DAttrGeneric(cg, {"Conv2DBias", "Conv2D"}, -128);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2_2, Conv2DNoBiasTest) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  BuildConv2DNoBiasGraph(*es_graph_);

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnv(shape_env);

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);

  VerifyConv2DRepeats(cg, "Conv2D", {1, 224, 224, 3}, {3, 3, 3, 64}, {1, 224, 224, 64}, 2, 1);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2_2, Conv2DWithBiasTest) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  BuildConv2DWithBiasGraph(*es_graph_);

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnv(shape_env);

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);

  VerifyConv2DRepeats(cg, "Conv2DBias", {1, 224, 224, 3}, {3, 3, 3, 64}, {1, 224, 224, 64}, 3, 1);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2_2, Conv2DWithOffsetWTest) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  BuildConv2DWithOffsetWGraph(*es_graph_, false);

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnv(shape_env);

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  for (auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Conv2D") {
      auto op_desc = node->GetOpDesc();
      for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
        auto input_desc = op_desc->MutableInputDesc(i);
        if (input_desc != nullptr && input_desc->GetShape().GetDims().size() == 1) {
          input_desc->SetDataType(DT_INT8);
        }
      }
    }
  }

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);

  VerifyConv2DRepeats(cg, "Conv2DOffset", {1, 224, 224, 3}, {3, 3, 3, 64}, {1, 224, 224, 64}, 3, 1);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2_2, Conv2DWithBiasAndOffsetWTest) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  BuildConv2DWithOffsetWGraph(*es_graph_, true);

  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  SetupShapeEnv(shape_env);

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  for (auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Conv2D") {
      auto op_desc = node->GetOpDesc();
      for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
        auto input_desc = op_desc->MutableInputDesc(i);
        if (input_desc != nullptr && input_desc->GetShape().GetDims().size() == 1) {
          input_desc->SetDataType(DT_INT8);
        }
      }
    }
  }

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);

  VerifyConv2DRepeats(cg, "Conv2DOffsetBias", {1, 224, 224, 3}, {3, 3, 3, 64}, {1, 224, 224, 64}, 4, 1);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}
}  // namespace af
